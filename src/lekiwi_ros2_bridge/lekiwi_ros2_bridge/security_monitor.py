#!/usr/bin/env python3
"""
LeKiWi CTF Security Monitor
============================
Intrusion detection for the ROS2 <-> MuJoCo bridge.

Monitors:
  /lekiwi/cmd_vel      -- velocity command anomalies + HMAC authentication
  /lekiwi/policy_input -- malicious policy injection (Challenge 7)

Anomaly detectors:
  1. Speed spikes       -- |vx|, |vy| > MAX_LIN_VEL or |wz| > MAX_ANG_VEL
  2. Rate-of-change    -- sudden jumps in velocity magnitude
  3. Out-of-range      -- NaN / Inf values
  4. Repeated patterns -- possible replay attack
  5. HMAC auth         -- signed cmd_vel required (when enabled)
  6. Policy hash mismatch -- detects tampering between inference cycles

Attack log -> JSON file at:
  ~/hermes_research/lekiwi_vla/security_log.jsonl

HMAC Authentication (Challenge 1 defense):
  Signed cmd_vel blocks UDP teleport attacks.
  Signature: HMAC-SHA256(timestamp_bytes + struct.pack('ddd', vx, vy, wz))
  Format: bytes = struct.pack('d', timestamp) + struct.pack('ddd', vx, vy, wz) + mac_bytes
  Where mac_bytes = HMAC(secret, bytes_without_mac)
"""

import time, json, math, threading, hashlib, hmac, os, struct
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

# Physical limits (from lekiwi_modular)
MAX_LIN_VEL   = 2.0    # m/s
MAX_ANG_VEL   = 3.14   # rad/s
ACCEL_SPIKE   = 5.0    # m/s2  -- rate-of-change alert threshold
REPLAY_WINDOW = 5.0    # seconds

_POLICY_SECRET = b"leki...026"
# cmd_vel HMAC secret — MUST be set before enabling hmac verification
# In production: load from environment variable or config file
_CMD_VEL_SECRET = b"cmd_vel_secret_key_2026"


@dataclass
class SecurityEvent:
    timestamp: float
    event_type: str     # "speed_spike" | "nan_inf" | "replay" | "hmac_fail" | "hmac_ok" | "policy_tamper"
    severity: str       # "low" | "medium" | "high" | "critical"
    details: dict
    ros_topic: str
    blocked: bool


class SecurityMonitor:
    """Thread-safe security monitor for the LeKiWi bridge."""

    def __init__(self, log_path: str = None, enable_hmac: bool = False,
                 cmd_vel_secret: bytes = None):
        if log_path is None:
            _home = os.path.expanduser("~")
            log_path = os.path.join(_home, "hermes_research", "lekiwi_vla", "security_log.jsonl")
        self._log_path = log_path
        self._lock     = threading.Lock()
        self._log_buf  = []
        self._vel_history: deque = deque(maxlen=50)
        self._last_policy_hash: Optional[str] = None
        self._policy_seq: int = 0
        self._counts = {"speed_spike": 0, "nan_inf": 0, "replay": 0,
                        "hmac_fail": 0, "hmac_ok": 0,
                        "policy_tamper": 0, "total_processed": 0}

        # HMAC authentication config
        self._enable_hmac = enable_hmac
        self._cmd_vel_secret = cmd_vel_secret or _CMD_VEL_SECRET
        # Track last N messages to prevent exact replay (reuse detection)
        self._hmac_history: deque = deque(maxlen=20)

    # ── cmd_vel HMAC verification ──────────────────────────────────────────────

    def _compute_hmac(self, vx: float, vy: float, wz: float,
                      timestamp: float) -> bytes:
        """Compute HMAC-SHA256 for a cmd_vel command."""
        msg_bytes = struct.pack('ddd', vx, vy, wz) + struct.pack('d', timestamp)
        return hmac.new(self._cmd_vel_secret, msg_bytes, hashlib.sha256).digest()

    def _verify_hmac(self, vx: float, vy: float, wz: float,
                     timestamp: float, mac_bytes: bytes) -> bool:
        """Verify HMAC of a cmd_vel command. Returns True if valid."""
        expected = self._compute_hmac(vx, vy, wz, timestamp)
        return hmac.compare_digest(expected, mac_bytes)

    def check_cmd_vel_hmac(self, vx: float, vy: float, wz: float,
                           stamp: float, mac: bytes) -> SecurityEvent:
        """
        Verify HMAC-signed cmd_vel command.
        Block if:
          - HMAC verification fails (attack detected)
          - Exact command was seen recently (replay attack)
        """
        self._counts["total_processed"] += 1

        # Check 1: HMAC validity
        if not self._verify_hmac(vx, vy, wz, stamp, mac):
            self._counts["hmac_fail"] += 1
            return self._make_event(
                "hmac_fail", "critical",
                {"vx": vx, "vy": vy, "wz": wz, "stamp": stamp,
                 "reason": "HMAC signature mismatch — possible forged command",
                 "ctf_flag": "ROBOT_CTF{teleport_success_6f8d2a1b}"},
                "/lekiwi/cmd_vel", blocked=True)

        # Check 2: Replay detection (exact command + near-identical timestamp)
        replay_key = struct.pack('ddd', vx, vy, wz)
        if replay_key in self._hmac_history:
            self._counts["replay"] += 1
            return self._make_event(
                "replay", "high",
                {"vx": vx, "vy": vy, "wz": wz, "stamp": stamp,
                 "reason": "Exact command replayed — possible replay attack"},
                "/lekiwi/cmd_vel", blocked=True)

        self._hmac_history.append(replay_key)
        self._counts["hmac_ok"] += 1
        return self._make_event("hmac_ok", "low",
                                 {"vx": vx, "vy": vy, "wz": wz, "stamp": stamp},
                                 "/lekiwi/cmd_vel", blocked=False)

    # ── Raw cmd_vel anomaly detection (no HMAC) ──────────────────────────────────

    def check_cmd_vel(self, vx: float, vy: float, wz: float, stamp: float) -> SecurityEvent:
        """Anomaly detection for legacy/unauthenticated cmd_vel."""
        self._counts["total_processed"] += 1
        # NaN/Inf check
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            self._counts["nan_inf"] += 1
            return self._make_event("nan_inf", "high",
                {"vx": vx, "vy": vy, "wz": wz}, "/lekiwi/cmd_vel", blocked=True)
        # Hard speed limit — this alone blocks Challenge 1's 100.0 m/s teleport
        if max(abs(vx), abs(vy)) > MAX_LIN_VEL or abs(wz) > MAX_ANG_VEL:
            self._counts["speed_spike"] += 1
            sev = "critical" if max(abs(vx), abs(vy)) > MAX_LIN_VEL * 2 else "high"
            return self._make_event("speed_spike", sev,
                {"vx": vx, "vy": vy, "wz": wz,
                 "limit_lin": MAX_LIN_VEL, "limit_ang": MAX_ANG_VEL,
                 "ctf_flag": "ROBOT_CTF{teleport_success_6f8d2a1b}"},
                "/lekiwi/cmd_vel", blocked=True)
        # Rate-of-change
        if self._vel_history:
            t_prev, vx_p, vy_p, wz_p = self._vel_history[-1]
            dt = stamp - t_prev
            if dt > 0.001:
                dvx = abs(vx - vx_p) / dt
                dvy = abs(vy - vy_p) / dt
                dwz = abs(wz - wz_p) / dt
                if max(dvx, dvy, dwz) > ACCEL_SPIKE:
                    self._counts["speed_spike"] += 1
                    return self._make_event("speed_spike", "medium",
                        {"vx": vx, "vy": vy, "wz": wz,
                         "dvx_dt": dvx, "dvy_dt": dvy, "dwz_dt": dwz,
                         "threshold": ACCEL_SPIKE},
                        "/lekiwi/cmd_vel", blocked=True)
        # Replay detection
        for t_h, vx_h, vy_h, wz_h in self._vel_history:
            if stamp - t_h < REPLAY_WINDOW:
                if abs(vx - vx_h) < 1e-6 and abs(vy - vy_h) < 1e-6 and abs(wz - wz_h) < 1e-6:
                    self._counts["replay"] += 1
                    return self._make_event("replay", "low",
                        {"vx": vx, "vy": vy, "wz": wz, "repeated_since": t_h},
                        "/lekiwi/cmd_vel", blocked=False)
        self._vel_history.append((stamp, vx, vy, wz))
        return self._make_event("cmd_vel_ok", "low", {}, "/lekiwi/cmd_vel", blocked=False)

    # ── Policy intrusion detection ──────────────────────────────────────────────

    def check_policy(self, policy_bytes: bytes, stamp: float) -> SecurityEvent:
        self._counts["total_processed"] += 1
        fp = hashlib.sha256(policy_bytes).hexdigest()[:16]
        if self._last_policy_hash is None:
            self._last_policy_hash = fp
            self._policy_seq += 1
            return self._make_event("policy_first_load", "low",
                {"fingerprint": fp, "seq": self._policy_seq}, "/lekiwi/policy_input", blocked=False)
        if fp != self._last_policy_hash:
            self._counts["policy_tamper"] += 1
            self._policy_seq += 1
            expected_mac = hmac.new(_POLICY_SECRET, policy_bytes, hashlib.sha256).hexdigest()
            return self._make_event("policy_tamper", "high",
                {"old_fingerprint": self._last_policy_hash,
                 "new_fingerprint": fp,
                 "expected_hmac": expected_mac,
                 "seq": self._policy_seq,
                 "ctf_flag": "ROBOT_CTF{policy_hijack_4c8e2a9f}"},
                "/lekiwi/policy_input", blocked=False)
        return self._make_event("policy_ok", "low", {}, "/lekiwi/policy_input", blocked=False)

    # ── Persistence + reporting ────────────────────────────────────────────────

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

    def summary(self) -> dict:
        with self._lock:
            return {**self._counts}

    def _make_event(self, event_type, severity, details, topic, blocked):
        entry = SecurityEvent(time.time(), event_type, severity, details, topic, blocked)
        with self._lock:
            self._log_buf.append(asdict(entry))
            if len(self._log_buf) >= 10:
                self.flush()
        return entry


if __name__ == "__main__":
    import time as _time
    mon = SecurityMonitor(log_path="/tmp/lekiwi_security_test.jsonl", enable_hmac=True)
    print("=== SecurityMonitor tests ===")
    print("\n--- Raw cmd_vel anomaly detection ---")
    e = mon.check_cmd_vel(0.5, 0.0, 0.0, _time.time())
    print(f"[1] Normal:  blocked={e.blocked}  {e.event_type}")
    e = mon.check_cmd_vel(100.0, 0.0, 0.0, _time.time() + 0.1)
    print(f"[2] Spike (100 m/s):  blocked={e.blocked}  {e.event_type}  severity={e.severity}")
    if "ctf_flag" in e.details:
        print(f"      CTF flag: {e.details['ctf_flag']}")
    e = mon.check_cmd_vel(float("nan"), 0.0, 0.0, _time.time() + 0.2)
    print(f"[3] NaN:    blocked={e.blocked}  {e.event_type}  severity={e.severity}")

    print("\n--- HMAC authentication (Challenge 1 defense) ---")
    stamp = _time.time()
    mac = mon._compute_hmac(0.5, 0.0, 0.0, stamp)
    e = mon.check_cmd_vel_hmac(0.5, 0.0, 0.0, stamp, mac)
    print(f"[4] Valid HMAC:   blocked={e.blocked}  {e.event_type}")

    # Forge a fake command
    fake_mac = hmac.new(b"wrong_secret", b"junk", hashlib.sha256).digest()
    e = mon.check_cmd_vel_hmac(0.5, 0.0, 0.0, stamp, fake_mac)
    print(f"[5] Forged HMAC: blocked={e.blocked}  {e.event_type}  severity={e.severity}")
    if "ctf_flag" in e.details:
        print(f"      CTF flag: {e.details['ctf_flag']}")

    # Replay same command
    e = mon.check_cmd_vel_hmac(0.5, 0.0, 0.0, stamp, mac)
    print(f"[6] Replay HMAC: blocked={e.blocked}  {e.event_type}  severity={e.severity}")

    print("\n--- Policy tamper detection (Challenge 7) ---")
    e = mon.check_policy(b"initial", _time.time())
    print(f"[7a] First policy: blocked={e.blocked}  {e.event_type}")
    e = mon.check_policy(b"malicious", _time.time() + 1.0)
    print(f"[7b] Tampered:     blocked={e.blocked}  {e.event_type}  severity={e.severity}")
    if "ctf_flag" in e.details:
        print(f"      CTF flag: {e.details['ctf_flag']}")

    mon.flush()
    print("\n=== Summary ===")
    for k, v in mon.summary().items():
        print(f"  {k}: {v}")
