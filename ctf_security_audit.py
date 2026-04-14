#!/usr/bin/env python3
"""
LeKiWi CTF Security Audit Tool
==============================
Monitors all ROS2 input channels for CTF-relevant anomalies.

Channels monitored:
  /lekiwi/cmd_vel           — raw twist (rate, magnitude, replay, hmac)
  /lekiwi/cmd_vel_hmac      — HMAC-signed twist
  /lekiwi/joint_states       — sensor state (joint_states injection)
  /lekiwi/vla_action         — policy action (policy injection)
  /lekiwi/policy             — policy selection (policy hijacking)
  /lekiwi/security_alert     — internal alerts for correlation

CTF Challenges detected:
  C1: Forged cmd_vel (no HMAC / bad signature)
  C2: DoS via rate flooding
  C3: Command injection (magnitude violation)
  C4: Physics DoS (acceleration spike)
  C5: Replay attack (identical sequence)
  C6: Sensor spoofing (joint_states injection)
  C7: Policy injection (vla_action override)
  C8: Policy hijacking (unauthorized policy switch)

Usage:
  # Run standalone audit (simulates monitoring)
  python3 ctf_security_audit.py

  # Import as module:
  from ctf_security_audit import CTFSecurityAuditor
  auditor = CTFSecurityAuditor(alert_callback=print)
"""

import time
import json
import numpy as np
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import hashlib
import hmac as _hmac


# ── Constants ────────────────────────────────────────────────────────────────

# Physical limits (from lekiwi_modular URDF)
WHEEL_MAX_SPEED = 30.0     # rad/s (realistic motor limit)
ARM_JOINT_MAX   = 3.14     # rad/s (arm joint velocity limit)
MAX_LINEAR_VEL  = 1.5     # m/s
MAX_ANGULAR_VEL  = 3.0     # rad/s
MAX_ACCEL       = 5.0      # m/s²

# Rate limits
CMD_VEL_RATE_LIMIT = 50.0  # Hz (warn if exceeded)
STATE_RATE_LIMIT   = 100.0 # Hz

# CTF flag prefixes
CTF_FLAGS = {
    "C1": "ROBOT_CTF{cmdvel_hmac_missing_a1b2c3d4}",
    "C2": "ROBOT_CTF{cmdvel_dos_rate_flood_e5f6g7h8}",
    "C3": "ROBOT_CTF{cmdvel_injection_i9j0k1l2}",
    "C4": "ROBOT_CTF{physics_dos_accel_m3n4o5p6}",
    "C5": "ROBOT_CTF{replay_attack_q7r8s9t0}",
    "C6": "ROBOT_CTF{sensor_spoof_u1v2w3x4}",
    "C7": "ROBOT_CTF{policy_inject_y5z6a7b8}",
    "C8": "ROBOT_CTF{policy_hijack_c9d0e1f2}",
}


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class SecurityAlert:
    """A detected security event."""
    challenge_id: str       # e.g. "C1", "C2"
    timestamp: float
    channel: str           # e.g. "/lekiwi/cmd_vel"
    description: str
    severity: str          # "low", "medium", "high", "critical"
    raw_data: Optional[dict] = None
    flag: Optional[str] = None    # CTF flag if challenge solved
    source_ip: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CmdVelSample:
    """A single cmd_vel sample for analysis."""
    vx: float
    vy: float
    wz: float
    timestamp: float
    seq: int


# ── Core Auditor ─────────────────────────────────────────────────────────────

class CTFSecurityAuditor:
    """
    Monitors all LeKiWi ROS2 channels for CTF-relevant attacks.

    Usage:
        auditor = CTFSecurityAuditor(alert_callback=my_callback)
        auditor.on_cmd_vel(vx=0.1, vy=0.0, wz=0.0)
        auditor.on_joint_states(position=[...] , velocity=[...])
        auditor.on_vla_action(action=[...])
        report = auditor.get_report()
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[SecurityAlert], None]] = None,
        enable_flags: bool = True,
        log_path: Optional[str] = None,
    ):
        self.alert_callback = alert_callback
        self.enable_flags = enable_flags
        self.log_path = log_path

        # Per-channel state
        self._cmd_vel_history: deque[CmdVelSample] = deque(maxlen=200)
        self._joint_state_history: deque = deque(maxlen=100)
        self._vla_action_history: deque = deque(maxlen=50)
        self._policy_switches: list = []
        self._seq_counter = 0

        # Rate tracking
        self._cmd_vel_times: deque = deque(maxlen=1000)
        self._state_times: deque = deque(maxlen=1000)

        # Alert log
        self.alerts: list[SecurityAlert] = []
        self.stats = {
            "cmd_vel_received": 0,
            "cmd_vel_blocked": 0,
            "state_injection_attempts": 0,
            "policy_switches": 0,
            "vla_override_attempts": 0,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def on_cmd_vel(
        self,
        vx: float,
        vy: float,
        wz: float,
        timestamp: Optional[float] = None,
        hmac_verified: bool = False,
        source_ip: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """
        Process a cmd_vel command. Returns SecurityAlert if blocked, None if allowed.
        """
        if timestamp is None:
            timestamp = time.time()

        self.stats["cmd_vel_received"] += 1
        self._seq_counter += 1
        seq = self._seq_counter

        # Rate check
        self._cmd_vel_times.append(timestamp)
        rate = self._get_rate(self._cmd_vel_times)
        if rate > CMD_VEL_RATE_LIMIT * 2:
            alert = self._make_alert(
                challenge_id="C2",
                channel="/lekiwi/cmd_vel",
                description=f"DoS attack: cmd_vel rate {rate:.1f} Hz (limit={CMD_VEL_RATE_LIMIT})",
                severity="critical",
                source_ip=source_ip,
            )
            self._record(alert)
            self.stats["cmd_vel_blocked"] += 1
            return alert

        # Magnitude check
        if (abs(vx) > MAX_LINEAR_VEL or abs(vy) > MAX_LINEAR_VEL or abs(wz) > MAX_ANGULAR_VEL):
            alert = self._make_alert(
                challenge_id="C3",
                channel="/lekiki/cmd_vel",
                description=f"Command injection: vx={vx:.3f} vy={vy:.3f} wz={wz:.3f} (limits: lin={MAX_LINEAR_VEL}, ang={MAX_ANGULAR_VEL})",
                severity="high",
                source_ip=source_ip,
            )
            self._record(alert)
            self.stats["cmd_vel_blocked"] += 1
            return alert

        # Acceleration check
        if self._cmd_vel_history:
            last = self._cmd_vel_history[-1]
            dt = timestamp - last.timestamp
            if dt > 0:
                dvx = abs(vx - last.vx) / dt
                dvy = abs(vy - last.vy) / dt
                dwz = abs(wz - last.wz) / dt
                if (dvx > MAX_ACCEL or dvy > MAX_ACCEL or dwz > MAX_ACCEL * 2):
                    alert = self._make_alert(
                        challenge_id="C4",
                        channel="/lekiwi/cmd_vel",
                        description=f"Physics DoS: dvx={dvx:.2f} m/s² dvy={dvy:.2f} dwz={dwz:.2f} rad/s²",
                        severity="medium",
                        source_ip=source_ip,
                    )
                    self._record(alert)
                    self.stats["cmd_vel_blocked"] += 1
                    return alert

        # Replay check
        if self._is_replay_cmd_vel(vx, vy, wz):
            alert = self._make_alert(
                challenge_id="C5",
                channel="/lekiwi/cmd_vel",
                description=f"Replay attack detected: identical cmd_vel sequence",
                severity="high",
                source_ip=source_ip,
            )
            self._record(alert)
            self.stats["cmd_vel_blocked"] += 1
            return alert

        # HMAC check (if HMAC not verified, it's a C1 event)
        if not hmac_verified:
            alert = self._make_alert(
                challenge_id="C1",
                channel="/lekiwi/cmd_vel",
                description="Forged cmd_vel detected: no HMAC signature (unauthenticated command)",
                severity="high",
                source_ip=source_ip,
            )
            self._record(alert)
            self.stats["cmd_vel_blocked"] += 1
            return alert

        # Record valid sample
        self._cmd_vel_history.append(CmdVelSample(vx=vx, vy=vy, wz=wz, timestamp=timestamp, seq=seq))
        return None

    def on_joint_states(
        self,
        position: list[float],
        velocity: list[float],
        effort: Optional[list[float]] = None,
        timestamp: Optional[float] = None,
        source_ip: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """
        Process joint_states data. Detects sensor spoofing attacks.
        LeKiWi has 9 joints: 6 arm + 3 wheel.
        """
        if timestamp is None:
            timestamp = time.time()

        self._state_times.append(timestamp)
        rate = self._get_rate(self._state_times)

        # Rate check on joint_states
        if rate > STATE_RATE_LIMIT * 2:
            alert = self._make_alert(
                challenge_id="C6",
                channel="/lekiwi/joint_states",
                description=f"Sensor spoofing DoS: joint_states rate {rate:.1f} Hz",
                severity="medium",
                source_ip=source_ip,
            )
            self._record(alert)
            self.stats["state_injection_attempts"] += 1
            return alert

        # Position/velocity limit check
        if len(position) >= 6:
            # Arm joint position limits (from URDF)
            arm_pos = position[:6]
            if any(abs(p) > 3.15 for p in arm_pos):
                alert = self._make_alert(
                    challenge_id="C6",
                    channel="/lekiwi/joint_states",
                    description=f"Sensor spoofing: arm position exceeds limits {[f'{p:.2f}' for p in arm_pos]}",
                    severity="high",
                    source_ip=source_ip,
                )
                self._record(alert)
                self.stats["state_injection_attempts"] += 1
                return alert

        if len(velocity) >= 9:
            wheel_vel = velocity[6:9]
            if any(abs(v) > WHEEL_MAX_SPEED for v in wheel_vel):
                alert = self._make_alert(
                    challenge_id="C6",
                    channel="/lekiwi/joint_states",
                    description=f"Sensor spoofing: wheel velocity {[f'{v:.2f}' for v in wheel_vel]} exceeds {WHEEL_MAX_SPEED} rad/s",
                    severity="high",
                    source_ip=source_ip,
                )
                self._record(alert)
                self.stats["state_injection_attempts"] += 1
                return alert

        # Physical plausibility: sudden jumps
        if self._joint_state_history:
            last_pos = self._joint_state_history[-1]["position"]
            if len(position) == len(last_pos):
                dt = timestamp - self._joint_state_history[-1]["timestamp"]
                if dt > 0:
                    jumps = [abs(position[i] - last_pos[i]) / dt for i in range(len(position))]
                    max_jump = max(jumps)
                    if max_jump > 100:  # > 100 rad/s jump = physically impossible
                        alert = self._make_alert(
                            challenge_id="C6",
                            channel="/lekiwi/joint_states",
                            description=f"Sensor spoofing: impossible state jump {max_jump:.1f} rad/s",
                            severity="critical",
                            source_ip=source_ip,
                        )
                        self._record(alert)
                        self.stats["state_injection_attempts"] += 1
                        return alert

        self._joint_state_history.append({"position": list(position), "velocity": list(velocity), "timestamp": timestamp})
        return None

    def on_vla_action(
        self,
        action: list[float],
        policy_name: str = "unknown",
        timestamp: Optional[float] = None,
        source_ip: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """
        Process VLA action output. Detects policy injection.
        Action: [arm*6, wheel*3] = 9 DOF
        """
        if timestamp is None:
            timestamp = time.time()

        # Check action magnitude
        if len(action) >= 6:
            arm_actions = action[:6]
            if any(abs(a) > ARM_JOINT_MAX * 2 for a in arm_actions):
                alert = self._make_alert(
                    challenge_id="C7",
                    channel="/lekiwi/vla_action",
                    description=f"Policy injection: arm action {[f'{a:.2f}' for a in arm_actions]} exceeds limit",
                    severity="high",
                    source_ip=source_ip,
                )
                self._record(alert)
                self.stats["vla_override_attempts"] += 1
                return alert

        if len(action) >= 9:
            wheel_actions = action[6:9]
            if any(abs(a) > WHEEL_MAX_SPEED * 2 for a in wheel_actions):
                alert = self._make_alert(
                    challenge_id="C7",
                    channel="/lekiwi/vla_action",
                    description=f"Policy injection: wheel action {[f'{a:.2f}' for a in wheel_actions]} exceeds limit",
                    severity="high",
                    source_ip=source_ip,
                )
                self._record(alert)
                self.stats["vla_override_attempts"] += 1
                return alert

        # Track for anomaly detection
        self._vla_action_history.append({"action": list(action), "policy": policy_name, "timestamp": timestamp})
        return None

    def on_policy_switch(
        self,
        old_policy: str,
        new_policy: str,
        timestamp: Optional[float] = None,
        authorized: bool = False,
        source_ip: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """Detect unauthorized policy hijacking."""
        if timestamp is None:
            timestamp = time.time()

        self.stats["policy_switches"] += 1
        self._policy_switches.append({
            "old": old_policy,
            "new": new_policy,
            "timestamp": timestamp,
            "authorized": authorized,
        })

        if not authorized:
            alert = self._make_alert(
                challenge_id="C8",
                channel="/lekiwi/policy",
                description=f"Policy hijacking: switch from '{old_policy}' → '{new_policy}' (unauthorized)",
                severity="critical",
                source_ip=source_ip,
            )
            self._record(alert)
            return alert

        return None

    def get_report(self) -> dict:
        """Generate a security audit report."""
        return {
            "timestamp": time.time(),
            "stats": self.stats,
            "alert_count": len(self.alerts),
            "alerts": [a.to_dict() for a in self.alerts[-50:]],  # last 50
            "cmd_vel_rate_hz": self._get_rate(self._cmd_vel_times),
            "state_rate_hz": self._get_rate(self._state_times),
            "challenge_flags": {k: v for k, v in CTF_FLAGS.items()} if self.enable_flags else {},
        }

    def print_report(self):
        """Print human-readable report."""
        r = self.get_report()
        print("=" * 60)
        print("LeKiWi CTF Security Audit Report")
        print("=" * 60)
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['timestamp']))}")
        print(f"cmd_vel received: {r['stats']['cmd_vel_received']}")
        print(f"cmd_vel blocked:  {r['stats']['cmd_vel_blocked']}")
        print(f"State injection:  {r['stats']['state_injection_attempts']}")
        print(f"Policy switches:  {r['stats']['policy_switches']}")
        print(f"VLA overrides:     {r['stats']['vla_override_attempts']}")
        print(f"Current cmd_vel rate: {r['cmd_vel_rate_hz']:.1f} Hz")
        print(f"Current state rate:   {r['state_rate_hz']:.1f} Hz")
        print(f"Total alerts: {r['alert_count']}")
        if r['alerts']:
            print("\nRecent alerts:")
            for a in r['alerts'][-5:]:
                print(f"  [{a['challenge_id']}] {a['channel']}: {a['description']} ({a['severity']})")
        print("=" * 60)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_alert(
        self,
        challenge_id: str,
        channel: str,
        description: str,
        severity: str,
        source_ip: Optional[str] = None,
    ) -> SecurityAlert:
        flag = CTF_FLAGS.get(challenge_id) if self.enable_flags else None
        return SecurityAlert(
            challenge_id=challenge_id,
            timestamp=time.time(),
            channel=channel,
            description=description,
            severity=severity,
            raw_data=None,
            flag=flag,
            source_ip=source_ip,
        )

    def _record(self, alert: SecurityAlert):
        self.alerts.append(alert)
        if self.alert_callback:
            self.alert_callback(alert)

    def _get_rate(self, times_deque: deque, window: float = 1.0) -> float:
        """Calculate rate (events/second) from a times deque over a sliding window."""
        if len(times_deque) < 2:
            return 0.0
        now = times_deque[-1]
        # Count events within the last `window` seconds
        recent = [t for t in times_deque if now - t <= window]
        if len(recent) < 2:
            return 0.0
        dt = recent[-1] - recent[0]
        if dt <= 1e-6:
            return 0.0
        return (len(recent) - 1) / dt

    def _is_replay_cmd_vel(self, vx: float, vy: float, wz: float, window: int = 5, threshold: float = 1e-6) -> bool:
        """Check if cmd_vel matches recent history (replay attack detection)."""
        count = 0
        for i in range(len(self._cmd_vel_history) - 1, max(0, len(self._cmd_vel_history) - window - 1), -1):
            sample = self._cmd_vel_history[i]
            if abs(vx - sample.vx) < threshold and abs(vy - sample.vy) < threshold and abs(wz - sample.wz) < threshold:
                count += 1
        return count >= 3  # 3+ identical commands in a row = replay


# ── Standalone Demo ──────────────────────────────────────────────────────────

def demo():
    """Run a demonstration of the CTF security auditor."""
    print("LeKiWi CTF Security Auditor — Demo Mode")
    print("-" * 50)

    auditor = CTFSecurityAuditor(enable_flags=True)

    # Normal operations (spread over time to avoid rate limit)
    print("\n[1] Normal cmd_vel operations:")
    import time
    for i in range(5):
        t = time.time() + i * 0.1  # 100ms apart = 10 Hz
        result = auditor.on_cmd_vel(vx=0.1, vy=0.0, wz=0.0, timestamp=t, hmac_verified=True)
        print(f"  cmd_vel #{i+1}: {'BLOCKED' if result else 'ALLOWED'}")

    # C1: forged cmd_vel (no HMAC)
    print("\n[2] C1 Attack — Forged cmd_vel (no HMAC):")
    t = time.time() + 10.0
    result = auditor.on_cmd_vel(vx=0.5, vy=0.0, wz=0.0, hmac_verified=False, timestamp=t)
    if result:
        print(f"  BLOCKED [{result.challenge_id}]: {result.description}")
        print(f"  FLAG: {result.flag}")

    # C3: command injection (magnitude)
    print("\n[3] C3 Attack — Command injection (magnitude violation):")
    t = time.time() + 11.0
    result = auditor.on_cmd_vel(vx=5.0, vy=0.0, wz=0.0, hmac_verified=True, timestamp=t)
    if result:
        print(f"  BLOCKED [{result.challenge_id}]: {result.description}")
        print(f"  FLAG: {result.flag}")

    # C5: replay attack
    print("\n[4] C5 Attack — Replay attack (3x identical cmd_vel):")
    for i in range(3):
        t = time.time() + 12.0 + i * 0.1
        auditor.on_cmd_vel(vx=0.1, vy=0.0, wz=0.0, hmac_verified=True, timestamp=t)
    t = time.time() + 13.0
    result = auditor.on_cmd_vel(vx=0.1, vy=0.0, wz=0.0, hmac_verified=True, timestamp=t)
    if result:
        print(f"  BLOCKED [{result.challenge_id}]: {result.description}")
        print(f"  FLAG: {result.flag}")

    # C6: sensor spoofing
    print("\n[5] C6 Attack — Sensor spoofing (impossible velocity):")
    t1 = time.time() + 14.0
    t2 = time.time() + 14.1
    auditor.on_joint_states(position=[0.0]*9, velocity=[0.0]*9, timestamp=t1)
    result = auditor.on_joint_states(position=[0.0]*9, velocity=[1000.0]*9, timestamp=t2)
    if result:
        print(f"  BLOCKED [{result.challenge_id}]: {result.description}")
        print(f"  FLAG: {result.flag}")

    # C8: policy hijacking
    print("\n[6] C8 Attack — Policy hijacking (unauthorized switch):")
    result = auditor.on_policy_switch(old_policy="task_oriented", new_policy="attacker_policy", authorized=False)
    if result:
        print(f"  BLOCKED [{result.challenge_id}]: {result.description}")
        print(f"  FLAG: {result.flag}")

    # Print full report
    print()
    auditor.print_report()


if __name__ == "__main__":
    demo()
