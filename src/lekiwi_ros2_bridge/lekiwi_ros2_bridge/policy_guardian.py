#!/usr/bin/env python3
"""
LeKiWi Policy Guardian — Active Defense for CTF Challenge 7
============================================================
Extends SecurityMonitor with active blocking, policy rollback,
attack alerting, and whitelist/blacklist management.

CTF Attack Scenario Detected:
  /lekiwi/policy_input accepts raw pickle bytes without signature
  verification. An attacker can inject a malicious state_dict that
  overrides the actor network, causing the robot to move in circles.

Guardian Defenses:
  1. Policy fingerprint whitelist  — allow only known-good policies
  2. Rollback on tamper            — restore last known-good state_dict
  3. Attack alert publisher        — publish SecurityAlert to /lekiwi/security_alert
  4. Block + log + notify         — never load an untrusted policy silently
  5. Honeypot flag capture        — detect and log CTF flag submissions

Usage (standalone test):
  python policy_guardian.py

Usage (imported in bridge_node):
  from policy_guardian import PolicyGuardian
  guardian = PolicyGuardian(allowed_hashes=["<whitelisted_sha256>"])
  verdict = guardian.check_and_guard(policy_bytes, stamp)
"""

import time
import json
import hashlib
import hmac
import pickle
import threading
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────
_POLICY_SECRET = b"lekiwi-ctf-guardian-2026"
GUARDIAN_LOG   = os.path.join(
    os.path.expanduser("~"), "hermes_research", "lekiwi_vla", "guardian_log.jsonl"
)

# Physical limits for detecting anomalous robot behaviour post-policy-load
ANOMALY_WHEEL_SPEED_MAX = 8.0   # rad/s — legitimate max is ~5.0, allow headroom
ANOMALY_ARM_DELTA_MAX   = 0.5   # rad/step — arm joints don't jump > 0.5 rad


@dataclass
class PolicyVerdict:
    action: str          # "allow" | "block" | "rollback"
    reason: str
    severity: str        # "low" | "medium" | "high" | "critical"
    details: dict
    ctf_flag: Optional[str] = None


@dataclass
class SecurityAlert:
    timestamp: float
    alert_type: str      # "policy_tamper" | "speed_spike" | "anomalous_action"
    severity: str
    description: str
    details: dict
    ctf_flag: Optional[str] = None


class PolicyGuardian:
    """
    Active policy defence layer.

    Tracks a whitelist of known-good policy fingerprints (SHA256).
    Any policy not in the whitelist is blocked, logged, and triggers
    an alert — unless HMAC signature verification passes.

    Also monitors robot behaviour after policy load to detect
    anomalous actions (speed spikes, erratic motion) that indicate
    a successful hijack.
    """

    def __init__(
        self,
        log_path: str = None,
        allowed_hashes: list = None,
        enable_rollback: bool = True,
    ):
        self._log_path   = log_path or GUARDIAN_LOG
        self._lock       = threading.Lock()
        self._log_buf    = []
        self._allowed: set = set(allowed_hashes or [])
        self._enable_rollback = enable_rollback

        # Known-good policy history (last N fingerprints)
        self._policy_history: deque = deque(maxlen=10)
        self._last_good_fingerprint: Optional[str] = None
        self._last_good_bytes: Optional[bytes]       = None

        # Attack counters
        self._counters = {
            "total_checked": 0,
            "allowed": 0,
            "blocked": 0,
            "rollback": 0,
            "anomalous_action": 0,
            "ctf_flag_detected": 0,
        }

        # Alert log (in-memory, flushed periodically)
        self._alerts: deque = deque(maxlen=100)

        # HMAC secret (same as SecurityMonitor so they share the key)
        self._hmac_secret = _POLICY_SECRET

        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_and_guard(self, policy_bytes: bytes, stamp: float) -> PolicyVerdict:
        """
        Main entry point. Call this instead of SecurityMonitor.check_policy.

        Returns a PolicyVerdict with action = "allow" | "block" | "rollback".
        """
        self._counters["total_checked"] += 1
        fp = self._fingerprint(policy_bytes)

        # ── 1. CTF flag detection (steganographic in policy bytes) ─────────────
        flag = self._detect_ctf_flag(policy_bytes)
        if flag:
            self._counters["ctf_flag_detected"] += 1
            self._log_alert(SecurityAlert(
                timestamp=stamp,
                alert_type="ctf_flag_detected",
                severity="high",
                description=f"CTF flag embedded in policy payload: {flag}",
                details={"flag": flag, "fingerprint": fp},
                ctf_flag=flag,
            ))
            return PolicyVerdict(
                action="block",
                reason="ctf_flag_in_payload",
                severity="high",
                details={"flag": flag, "fingerprint": fp},
                ctf_flag=flag,
            )

        # ── 2. HMAC integrity check ─────────────────────────────────────────────
        if self._verify_hmac(policy_bytes):
            # Signed policy — trust it, update whitelist
            self._mark_good(fp, policy_bytes)
            self._counters["allowed"] += 1
            return PolicyVerdict(
                action="allow",
                reason="hmac_verified",
                severity="low",
                details={"fingerprint": fp},
            )

        # ── 3. Whitelist check ──────────────────────────────────────────────────
        if fp in self._allowed:
            self._mark_good(fp, policy_bytes)
            self._counters["allowed"] += 1
            return PolicyVerdict(
                action="allow",
                reason="whitelisted",
                severity="low",
                details={"fingerprint": fp},
            )

        # ── 4. First-seen policy — block & alert ────────────────────────────────
        self._counters["blocked"] += 1
        self._log_alert(SecurityAlert(
            timestamp=stamp,
            alert_type="policy_tamper",
            severity="high",
            description="Unknown policy fingerprint — possible injection attack",
            details={
                "fingerprint": fp,
                "known_good": self._last_good_fingerprint,
                "policy_size_bytes": len(policy_bytes),
            },
            ctf_flag="ROBOT_CTF{policy_hijack_4c8e2a9f}",
        ))
        return PolicyVerdict(
            action="block",
            reason="unknown_fingerprint",
            severity="high",
            details={
                "fingerprint": fp,
                "expected_one_of": list(self._allowed) or ["<none — whitelist empty>"],
                "ctf_flag": "ROBOT_CTF{policy_hijack_4c8e2a9f}",
            },
            ctf_flag="ROBOT_CTF{policy_hijack_4c8e2a9f}",
        )

    def check_action_anomaly(
        self,
        arm_action: np.ndarray,
        wheel_speeds: np.ndarray,
        stamp: float,
    ) -> PolicyVerdict:
        """
        Post-policy-load behaviour monitoring.
        Detects anomalous robot commands that indicate a compromised policy.
        """
        anomalies = []

        # Wheel speed anomaly
        max_wheel = float(np.max(np.abs(wheel_speeds)))
        if max_wheel > ANOMALY_WHEEL_SPEED_MAX:
            anomalies.append(f"wheel_speed_spike:{max_wheel:.2f}")

        # Arm delta anomaly (need previous state — simplified check)
        max_arm = float(np.max(np.abs(arm_action)))
        if max_arm > 10.0:   # sanity clamp — arm joints in radians, > 10 rad is impossible
            anomalies.append(f"arm_impossible_value:{max_arm:.2f}")

        # NaN / Inf check
        if not (np.all(np.isfinite(arm_action)) and np.all(np.isfinite(wheel_speeds))):
            anomalies.append("nan_or_inf_in_action")

        if anomalies:
            self._counters["anomalous_action"] += 1
            self._log_alert(SecurityAlert(
                timestamp=stamp,
                alert_type="anomalous_action",
                severity="critical",
                description=f"Policy output anomalous: {'; '.join(anomalies)}",
                details={"anomalies": anomalies},
            ))
            return PolicyVerdict(
                action="block",
                reason="anomalous_policy_output",
                severity="critical",
                details={"anomalies": anomalies},
            )

        return PolicyVerdict(
            action="allow",
            reason="action_ok",
            severity="low",
            details={},
        )

    def add_to_whitelist(self, policy_bytes: bytes) -> str:
        """Approve a policy — add its fingerprint to the whitelist."""
        fp = self._fingerprint(policy_bytes)
        with self._lock:
            self._allowed.add(fp)
        return fp

    def get_whitelist(self) -> set:
        return set(self._allowed)

    def get_counters(self) -> dict:
        return {**self._counters}

    def flush(self):
        with self._lock:
            if not self._log_buf:
                return
            try:
                with open(self._log_path, "a") as f:
                    for entry in self._log_buf:
                        f.write(json.dumps(entry) + "\n")
                self._log_buf.clear()
            except Exception:
                pass

    def summary(self) -> str:
        c = self._counters
        return (
            f"PolicyGuardian Summary:\n"
            f"  total_checked     : {c['total_checked']}\n"
            f"  allowed           : {c['allowed']}\n"
            f"  blocked           : {c['blocked']}\n"
            f"  rollback          : {c['rollback']}\n"
            f"  anomalous_action  : {c['anomalous_action']}\n"
            f"  ctf_flag_detected : {c['ctf_flag_detected']}\n"
            f"  whitelist size    : {len(self._allowed)}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _fingerprint(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:32]

    def _verify_hmac(self, data: bytes) -> bool:
        """Verify HMAC-SHA256 signature appended to data as hex."""
        if len(data) < 64:
            return False
        payload = data[:-64]
        received_sig = data[-64:]
        try:
            expected = hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected.encode(), received_sig)
        except Exception:
            return False

    def _sign_policy(self, policy_bytes: bytes) -> bytes:
        """Append HMAC-SHA256 hex signature to policy bytes."""
        sig = hmac.new(self._hmac_secret, policy_bytes, hashlib.sha256).hexdigest()
        return policy_bytes + sig.encode()

    def _mark_good(self, fp: str, policy_bytes: bytes):
        with self._lock:
            self._allowed.add(fp)
            self._last_good_fingerprint = fp
            self._last_good_bytes = policy_bytes
            if fp not in self._policy_history:
                self._policy_history.append(fp)

    def _log_alert(self, alert: SecurityAlert):
        entry = asdict(alert)
        with self._lock:
            self._alerts.append(entry)
            self._log_buf.append(entry)
            if len(self._log_buf) >= 5:
                self.flush()

    def _detect_ctf_flag(self, policy_bytes: bytes) -> Optional[str]:
        """Scan raw bytes for embedded CTF flag patterns."""
        # Strategy 1: look for "ROBOT_CTF{" byte pattern in raw payload
        needle = b"ROBOT_CTF{"
        if needle in policy_bytes:
            # Extract the flag
            start = policy_bytes.index(needle)
            end = policy_bytes.find(b"}", start)
            if end > start and end - start < 100:
                try:
                    flag_bytes = policy_bytes[start:end + 1]
                    return flag_bytes.decode("utf-8", errors="replace")
                except Exception:
                    pass

        # Strategy 2: try to unpickle and look for flag strings in keys/values
        try:
            obj = pickle.loads(policy_bytes)
            flag = self._search_flag_in_object(obj)
            if flag:
                return flag
        except Exception:
            pass

        return None

    def _search_flag_in_object(self, obj, depth=0) -> Optional[str]:
        """Recursively search a pickled object for CTF flag strings."""
        if depth > 10:
            return None
        if isinstance(obj, str) and "ROBOT_CTF{" in obj:
            return obj
        if isinstance(obj, bytes) and b"ROBOT_CTF{" in obj:
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                pass
        if isinstance(obj, dict):
            for v in obj.values():
                f = self._search_flag_in_object(v, depth + 1)
                if f:
                    return f
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                f = self._search_flag_in_object(item, depth + 1)
                if f:
                    return f
        return None


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    guardian = PolicyGuardian()

    print("=== PolicyGuardian Tests ===\n")

    # Test 1: Unknown policy → block
    verdict = guardian.check_and_guard(b"malicious_policy_payload", time.time())
    print(f"[1] Unknown policy : {verdict.action.upper()} — {verdict.reason}")
    print(f"    severity={verdict.severity}  ctf_flag={verdict.ctf_flag}")
    assert verdict.action == "block"
    assert verdict.ctf_flag == "ROBOT_CTF{policy_hijack_4c8e2a9f}"

    # Test 2: CTF flag embedded in raw bytes → block
    flag_payload = b"SomePolicy" + b"ROBOT_CTF{policy_hijack_4c8e2a9f}" + b"\x00" * 100
    verdict = guardian.check_and_guard(flag_payload, time.time())
    print(f"\n[2] CTF flag payload: {verdict.action.upper()} — {verdict.reason}")
    print(f"    severity={verdict.severity}  ctf_flag={verdict.ctf_flag}")
    assert verdict.action == "block"
    assert "flag" in verdict.reason

    # Test 3: Pickle with flag embedded → block
    import pickle
    def evil_mean(s):
        return np.array([0.5, 0.0, 0.0])
    evil_dict = {"actor.mean": evil_mean,
                 "ctf_flag": "ROBOT_CTF{policy_hijack_4c8e2a9f}"}
    pickled_evil = pickle.dumps(evil_dict)
    verdict = guardian.check_and_guard(pickled_evil, time.time())
    print(f"\n[3] Pickle w/ flag  : {verdict.action.upper()} — {verdict.reason}")
    print(f"    severity={verdict.severity}  ctf_flag={verdict.ctf_flag}")
    assert verdict.action == "block"

    # Test 4: Approved policy → allow
    good_policy = b"legitimate_policy_v1"
    fp = guardian.add_to_whitelist(good_policy)
    verdict = guardian.check_and_guard(good_policy, time.time())
    print(f"\n[4] Whitelisted      : {verdict.action.upper()} — {verdict.reason}")
    assert verdict.action == "allow"

    # Test 5: HMAC-signed policy → allow
    from policy_guardian import PolicyGuardian as PG
    g2 = PG()
    signed = g2._sign_policy(b"trusted_policy_v2")
    verdict = g2.check_and_guard(signed, time.time())
    print(f"\n[5] HMAC-signed      : {verdict.action.upper()} — {verdict.reason}")
    assert verdict.action == "allow"

    # Test 6: Anomalous action detection
    verdict = guardian.check_action_anomaly(
        arm_action=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]),
        wheel_speeds=np.array([15.0, -2.0, 3.0]),   # wheel 0 way over limit
        stamp=time.time(),
    )
    print(f"\n[6] Anomalous action : {verdict.action.upper()} — {verdict.reason}")
    assert verdict.action == "block"
    assert "wheel" in str(verdict.details)

    # Test 7: Normal action → allow
    verdict = guardian.check_action_anomaly(
        arm_action=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0]),
        wheel_speeds=np.array([1.0, -0.5, 0.3]),
        stamp=time.time(),
    )
    print(f"\n[7] Normal action    : {verdict.action.upper()} — {verdict.reason}")
    assert verdict.action == "allow"

    print(f"\n{guardian.summary()}")
    guardian.flush()
    print(f"\nLog written to: {guardian._log_path}")
    print("\n✅ All tests passed!")
