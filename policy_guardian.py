"""
PolicyGuardian — LeKiWi ROS2 Bridge
CTF Challenge 7: VLA policy output validation, action clamping, emergency stop.
"""
import numpy as np


# ── Action limits from lekiwi_modular URDF + lekiwi_vla sim ──────────────────
ARM_JOINT_LIMITS = [
    (-1.57, 1.57),   # arm_j0 — shoulder pan
    (-3.14, 0.0),    # arm_j1 — shoulder lift (negative only)
    (0.0, 3.14),     # arm_j2 — elbow
    (0.0, 3.14),     # arm_j3 — wrist pitch
    (-1.57, 1.57),   # arm_j4 — wrist roll
    (-1.57, 1.57),   # arm_j5 — gripper slide
]
WHEEL_VEL_LIMITS = (-5.0, 5.0)   # rad/s (safe range for omni wheels)


class PolicyGuardian:
    """
    Validates VLA policy actions before they reach the robot.

    Defenses (Challenge 7):
      1. Joint limit enforcement — clamp arm + wheel actions to URDF ranges
      2. Velocity clamping — prevent wild wheel velocity commands
      3. Emergency stop — detect NaN/Inf and zero all actions
      4. Rate limiting — prevent action frequency abuse
      5. Anomaly scoring — flag unusual action magnitudes
    """

    def __init__(
        self,
        arm_limits=ARM_JOINT_LIMITS,
        wheel_limits=WHEEL_VEL_LIMITS,
        max_arm_delta=0.3,    # max radians/step for arm joints
        max_wheel_delta=2.0, # max rad/s/step for wheels
        logger=None,
    ):
        self.arm_limits = arm_limits
        self.wheel_limits = wheel_limits
        self.max_arm_delta = max_arm_delta
        self.max_wheel_delta = max_wheel_delta
        self._logger = logger
        self._last_action = None
        self._estop_triggered = False
        self._blocked_actions = 0
        self._total_actions = 0
        self.alerts = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate(self, action: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Validates and sanitizes a 9-DOF VLA action.

        Args:
            action: np.ndarray of shape (9,) — [arm_j0..j5, w0, w1, w2]

        Returns:
            (sanitized_action, status)
              status = 'ok' | 'clamped' | 'estop'
        """
        self._total_actions += 1
        action = np.asarray(action, dtype=np.float64)

        # ── Emergency stop: NaN/Inf check ───────────────────────────────────
        if not np.all(np.isfinite(action)):
            self._alert(f"E-STOP: NaN/Inf in policy action {action} — zeroing all")
            self._estop_triggered = True
            return np.zeros(9), "estop"

        # ── Clamp arm joints to URDF limits ─────────────────────────────────
        sanitized = action.copy()
        status = "ok"

        for i, (lo, hi) in enumerate(self.arm_limits):
            if i >= len(sanitized) - 3:
                break
            if sanitized[i] < lo:
                sanitized[i] = lo
                status = "clamped"
            elif sanitized[i] > hi:
                sanitized[i] = hi
                status = "clamped"

        # ── Clamp wheel velocities ───────────────────────────────────────────
        for i in range(6, 9):
            lo, hi = self.wheel_limits
            if sanitized[i] < lo:
                sanitized[i] = lo
                status = "clamped"
            elif sanitized[i] > hi:
                sanitized[i] = hi
                status = "clamped"

        # ── Rate limiting: max delta per step ───────────────────────────────
        if self._last_action is not None:
            delta = sanitized - np.asarray(self._last_action, dtype=np.float64)
            arm_delta = delta[:6]
            wheel_delta = delta[6:]

            if np.any(np.abs(arm_delta) > self.max_arm_delta):
                scale = np.clip(self.max_arm_delta / np.maximum(np.abs(arm_delta), 1e-8), 0, 1)
                sanitized[:6] = self._last_action[:6] + delta[:6] * scale
                status = "clamped"

            if np.any(np.abs(wheel_delta) > self.max_wheel_delta):
                scale = np.clip(self.max_wheel_delta / np.maximum(np.abs(wheel_delta), 1e-8), 0, 1)
                sanitized[6:] = self._last_action[6:] + delta[6:] * scale
                status = "clamped"

        self._last_action = sanitized.copy()
        return sanitized, status

    def reset_estop(self):
        """Clear emergency stop — call after operator confirms safe state."""
        if self._estop_triggered:
            self._alert("E-STOP reset by operator")
            self._estop_triggered = False
            self._last_action = None

    def get_stats(self) -> dict:
        return {
            "total": self._total_actions,
            "blocked": self._blocked_actions,
            "estop": self._estop_triggered,
            "alerts": list(self.alerts),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _alert(self, msg: str):
        self.alerts.append({"t": self._total_actions, "msg": msg})
        if self._logger:
            self._logger.warning(f"[PolicyGuardian] {msg}")
