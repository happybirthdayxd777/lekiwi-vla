"""
SecurityMonitor — LeKiWi ROS2 Bridge
CTF Challenges 1-8: Raw cmd_vel anomaly detection, HMAC verification, replay attack
                     prevention, and goal spoofing detection.

Challenges covered:
  C1: HMAC-verified cmd_vel   (forged /lekiwi/cmd_vel)
  C2: DoS rate flooding       (/lekiwi/cmd_vel)
  C3: Command injection      (magnitude violations)
  C4: Physics DoS            (acceleration spikes)
  C5: Replay attack          (identical cmd_vel sequences)
  C6: Sensor spoofing        (/lekiwi/joint_states fake feedback)
  C7: Policy hijack          (policy injection via /lekiwi/vla_action)
  C8: Goal spoofing          (/lekiwi/goal unexpected updates) ⭐ NEW
"""
import hashlib
import hmac
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


CTF_FLAGS = {
    "C1": "ROBOT_CTF{cmdvel_hmac_missing_a1b2c3d4}",
    "C2": "ROBOT_CTF{cmdvel_dos_rate_flood_e5f6g7h8}",
    "C3": "ROBOT_CTF{cmdvel_injection_i9j0k1l2}",
    "C4": "ROBOT_CTF{physics_dos_accel_m3n4o5p6}",
    "C5": "ROBOT_CTF{replay_attack_q7r8s9t0}",
    "C6": "ROBOT_CTF{sensor_spoof_u1v2w3x4}",
    "C7": "ROBOT_CTF{policy_inject_y5z6a7b8}",
    "C8": "ROBOT_CTF{goal_spoof_z9a0b1c2}",   # ⭐ NEW
}


@dataclass
class GoalEvent:
    """Record of a /lekiwi/goal update for spoofing detection."""
    goal_x: float
    goal_y: float
    timestamp: float
    source: str = "unknown"   # "cmd_vel", "vla", "external", "unknown"


class SecurityMonitor:
    """
    Monitors /lekiwi/cmd_vel, /lekiwi/goal, and /lekiwi/vla_action for anomalies.

    Layers:
      Layer 1 (enable_hmac=False): Rate + magnitude anomaly, replay buffer
      Layer 2 (enable_hmac=True):  HMAC-SHA256 cmd_vel signature verification
      Layer 3: Goal spoofing detection on /lekiwi/goal
      Layer 4: Joint_states sensor spoofing detection
      Layer 5: VLA action injection detection
    """

    def __init__(
        self,
        enable_hmac: bool = False,
        cmd_vel_secret: str = "",
        max_linear: float = 1.5,      # m/s
        max_angular: float = 3.0,     # rad/s
        max_accel: float = 5.0,       # m/s²  (max change per step)
        rate_limit: float = 50.0,     # Hz
        # Goal spoofing params
        goal_max_change_per_sec: float = 2.0,   # max radial speed of goal (m/s)
        goal_max_rate: float = 5.0,              # max goal updates per second
        # Sensor spoofing params
        joint_max_delta: float = 1.5,   # max joint change per 20ms step (rad)
        vla_max_delta: float = 2.0,     # max VLA action change per step
        logger=None,
    ):
        self.enable_hmac = enable_hmac
        self.secret = cmd_vel_secret.encode()
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.max_accel = max_accel
        self.rate_limit = rate_limit
        self._logger = logger

        # cmd_vel state
        self._last_vx = self._last_vy = self._last_wz = 0.0
        self._last_time = time.monotonic()
        self._cmd_times: deque = deque(maxlen=100)
        self._cmd_history: deque = deque(maxlen=50)

        # Goal state (for C8: goal spoofing)
        self._goal_history: deque = deque(maxlen=200)
        self._goal_times: deque = deque(maxlen=100)
        self._last_goal: Optional[GoalEvent] = None
        self._goal_max_change_per_sec = goal_max_change_per_sec
        self._goal_max_rate = goal_max_rate

        # Joint_states state (for C6: sensor spoofing)
        self._last_joints: Optional[np.ndarray] = None
        self._last_joint_time = time.monotonic()
        self.joint_max_delta = joint_max_delta

        # VLA action state (for C7: policy injection)
        self._last_vla_action: Optional[np.ndarray] = None
        self._last_vla_time = time.monotonic()
        self.vla_max_delta = vla_max_delta

        # Alert log
        self.alerts: list = []
        self.total_processed = 0

    # ── Public API: cmd_vel ─────────────────────────────────────────────────

    def verify(self, twist) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        allowed=True  → cmd_vel passed all checks.
        allowed=False → cmd_vel blocked; reason explains why.
        """
        self.total_processed += 1
        now = time.monotonic()
        dt = now - self._last_time

        # C1: HMAC layer
        if self.enable_hmac:
            if not self._verify_hmac(twist):
                self._alert(f"C1 HMAC verification FAILED — possible forged cmd_vel")
                return False, "hmac_failed"

        # C2: Rate limit
        self._cmd_times.append(now)
        recent = [t for t in self._cmd_times if now - t < 1.0]
        if len(recent) > self.rate_limit * 2:
            self._alert(f"C2 Rate limit exceeded: {len(recent)} cmd_vels/s (limit={self.rate_limit})")
            return False, "rate_limit"

        # C3: Magnitude check
        vx, vy, wz = twist.linear.x, twist.linear.y, twist.linear.z
        if abs(vx) > self.max_linear or abs(vy) > self.max_linear or abs(wz) > self.max_angular:
            self._alert(f"C3 Magnitude violation: vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")
            return False, "magnitude_violation"

        # C4: Acceleration check
        if dt > 0:
            dvx = abs(vx - self._last_vx) / dt
            dvy = abs(vy - self._last_vy) / dt
            dwz = abs(wz - self._last_wz) / dt
            if dvx > self.max_accel or dvy > self.max_accel or dwz > self.max_accel * 2:
                self._alert(f"C4 Acceleration spike: dvx={dvx:.2f} m/s² (limit={self.max_accel})")
                return False, "accel_spike"

        # C5: Replay attack detection
        if self._cmd_history and self._is_replay(vx, vy, wz):
            self._alert(f"C5 Replay attack detected — identical cmd_vel sequence")
            return False, "replay_detected"

        self._cmd_history.append((vx, vy, wz, now))
        self._last_vx, self._last_vy, self._last_wz = vx, vy, wz
        self._last_time = now
        return True, "ok"

    # ── Public API: goal (C8) ────────────────────────────────────────────────

    def check_goal_spoofing(self, goal_x: float, goal_y: float,
                            source: str = "unknown") -> tuple[bool, str, Optional[str]]:
        """
        Check a /lekiwi/goal update for spoofing attacks.

        Returns (allowed, reason, ctf_flag):
          allowed=True  → goal update passed checks.
          allowed=False → goal update flagged as suspicious; reason explains why.
          ctf_flag      → flag string if a CTF challenge was triggered (C8 or None).

        C8 detection layers:
          1. Rate limit: >5 goal updates/second is suspicious
          2. Radial speed: goal moving >2 m/s is teleportation attack
          3. Out-of-bounds: goal >5m from origin (outside workspace)
        """
        now = time.monotonic()

        # Rate check
        self._goal_times.append(now)
        recent_goals = [t for t in self._goal_times if now - t < 1.0]
        if len(recent_goals) > self._goal_max_rate:
            flag = CTF_FLAGS["C8"]
            self._alert(
                f"C8 [GOAL SPOOF] Rate limit: {len(recent_goals)} goals/s "
                f"(limit={self._goal_max_rate}/s) — flag={flag}"
            )
            self._record_goal(goal_x, goal_y, now, source)
            return False, "goal_rate_limit", flag

        # Radial speed check
        if self._last_goal is not None:
            dt = now - self._last_goal.timestamp
            if dt > 0:
                dx = goal_x - self._last_goal.goal_x
                dy = goal_y - self._last_goal.goal_y
                dist = np.sqrt(dx**2 + dy**2)
                radial_speed = dist / dt
                if radial_speed > self._goal_max_change_per_sec:
                    flag = CTF_FLAGS["C8"]
                    self._alert(
                        f"C8 [GOAL SPOOF] Radial speed={radial_speed:.2f} m/s "
                        f"(limit={self._goal_max_change_per_sec} m/s) — "
                        f"flag={flag}"
                    )
                    self._record_goal(goal_x, goal_y, now, source)
                    return False, "goal_teleport", flag

        # Out-of-bounds check
        total_dist = np.sqrt(goal_x**2 + goal_y**2)
        if total_dist > 5.0:
            flag = CTF_FLAGS["C8"]
            self._alert(
                f"C8 [GOAL SPOOF] Goal out of bounds: ({goal_x:.2f}, {goal_y:.2f}) "
                f"dist={total_dist:.2f}m > 5.0m — flag={flag}"
            )
            self._record_goal(goal_x, goal_y, now, source)
            return False, "goal_out_of_bounds", flag

        self._record_goal(goal_x, goal_y, now, source)
        return True, "ok", None

    def _record_goal(self, goal_x: float, goal_y: float, timestamp: float, source: str):
        """Internal: record goal event and update last_goal."""
        event = GoalEvent(goal_x=goal_x, goal_y=goal_y, timestamp=timestamp, source=source)
        self._goal_history.append(event)
        self._last_goal = event

    # ── Public API: joint_states (C6) ──────────────────────────────────────

    def check_joint_spoofing(self, positions: np.ndarray,
                              velocities: np.ndarray = None) -> tuple[bool, str, Optional[str]]:
        """
        Check joint_states for sensor spoofing (C6).

        Returns (allowed, reason, ctf_flag).
        Flags anomalous jumps in joint positions (indicates fake sensor data).
        """
        now = time.monotonic()

        if self._last_joints is not None and self._last_joint_time > 0:
            dt = now - self._last_joint_time
            if dt > 0:
                delta = np.abs(positions - self._last_joints)
                max_delta = delta.max()
                if max_delta > self.joint_max_delta:
                    flag = CTF_FLAGS["C6"]
                    self._alert(
                        f"C6 [SENSOR SPOOF] Joint delta={max_delta:.3f} rad "
                        f"(limit={self.joint_max_delta} rad) — flag={flag}"
                    )
                    self._last_joints = positions.copy()
                    self._last_joint_time = now
                    return False, "joint_spoof", flag

        self._last_joints = positions.copy()
        self._last_joint_time = now
        return True, "ok", None

    # ── Public API: VLA action (C7) ─────────────────────────────────────────

    def check_vla_action(self, action: np.ndarray) -> tuple[bool, str, Optional[str]]:
        """
        Check VLA action for policy injection (C7).

        Returns (allowed, reason, ctf_flag).
        Flags anomalous action jumps (indicates injected policy).
        """
        now = time.monotonic()

        if self._last_vla_action is not None and self._last_vla_time > 0:
            dt = now - self._last_vla_time
            if dt > 0:
                delta = np.abs(action - self._last_vla_action)
                max_delta = delta.max()
                if max_delta > self.vla_max_delta:
                    flag = CTF_FLAGS["C7"]
                    self._alert(
                        f"C7 [POLICY INJECT] VLA action delta={max_delta:.3f} "
                        f"(limit={self.vla_max_delta}) — flag={flag}"
                    )
                    self._last_vla_action = action.copy()
                    self._last_vla_time = now
                    return False, "vla_action_inject", flag

        self._last_vla_action = action.copy()
        self._last_vla_time = now
        return True, "ok", None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _verify_hmac(self, twist) -> bool:
        """HMAC-SHA256 verification (always passes for now — needs custom msg type)."""
        payload = (
            f"{twist.linear.x}|{twist.linear.y}|{twist.linear.z}|"
            f"{twist.angular.x}|{twist.angular.y}|{twist.angular.z}"
        )
        expected = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return True   # NOTE: integrate with signed Twist message type later

    def _is_replay(self, vx, vy, wz, threshold: float = 1e-6) -> bool:
        """Detect if cmd_vel is identical to last 5 commands."""
        for i in range(len(self._cmd_history) - 1, max(0, len(self._cmd_history) - 6), -1):
            ox, oy, ow, _ = self._cmd_history[i]
            if (abs(vx - ox) < threshold and abs(vy - oy) < threshold
                    and abs(wz - ow) < threshold):
                return True
        return False

    def _alert(self, msg: str):
        """Internal: record alert and log."""
        self.alerts.append({"time": time.time(), "msg": msg})
        if self._logger:
            self._logger.warning(f"[SecurityAlert] {msg}")

    # ── CTF helpers ──────────────────────────────────────────────────────────

    def get_alerts(self) -> list:
        return list(self.alerts)

    def reset_alerts(self):
        self.alerts.clear()

    def is_secure(self) -> bool:
        return self.enable_hmac

    def get_goal_history(self) -> list:
        """Return recent goal events (for CTF forensics)."""
        return [
            {"gx": e.goal_x, "gy": e.goal_y, "ts": e.timestamp, "src": e.source}
            for e in self._goal_history
        ]
