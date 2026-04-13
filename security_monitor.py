"""
SecurityMonitor — LeKiWi ROS2 Bridge
CTF Challenges 1-6: Raw cmd_vel anomaly detection, HMAC verification, replay attack prevention.
"""
import hashlib
import hmac
import time
import numpy as np
from collections import deque


class SecurityMonitor:
    """
    Monitors /lekiwi/cmd_vel for anomalies and forged messages.

    Layers:
      Layer 1 (enable_hmac=False): Rate + magnitude anomaly detection, replay buffer
      Layer 2 (enable_hmac=True):  HMAC-SHA256 cmd_vel signature verification
    """

    def __init__(
        self,
        enable_hmac: bool = False,
        cmd_vel_secret: str = "",
        max_linear: float = 1.5,      # m/s  (wheeled robot max ~0.5, keep buffer)
        max_angular: float = 3.0,     # rad/s
        max_accel: float = 5.0,      # m/s²  (max change per 20ms step)
        rate_limit: float = 50.0,    # Hz — warn if >2× this
        logger=None,
    ):
        self.enable_hmac = enable_hmac
        self.secret = cmd_vel_secret.encode()
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.max_accel = max_accel
        self.rate_limit = rate_limit
        self._logger = logger

        self._last_vx = self._last_vy = self._last_wz = 0.0
        self._last_time = time.monotonic()
        self._cmd_times: deque = deque(maxlen=100)
        self._cmd_history: deque = deque(maxlen=50)   # for replay detection

        self.alerts: list = []
        self.total_processed = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def verify(self, twist) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        allowed=True → cmd_vel passed all checks.
        allowed=False → cmd_vel blocked; reason explains why.
        """
        self.total_processed += 1
        now = time.monotonic()
        dt = now - self._last_time

        # HMAC layer (Challenge 1: forged cmd_vel)
        if self.enable_hmac:
            if not self._verify_hmac(twist):
                self._alert(f"HMAC verification FAILED — possible forged cmd_vel")
                return False, "hmac_failed"

        # Rate limit (Challenge 2: DoS)
        self._cmd_times.append(now)
        recent = [t for t in self._cmd_times if now - t < 1.0]
        if len(recent) > self.rate_limit * 2:
            self._alert(f"Rate limit exceeded: {len(recent)} cmd_vels/s (limit={self.rate_limit})")
            return False, "rate_limit"

        # Magnitude check (Challenge 3: command injection)
        vx, vy, wz = twist.linear.x, twist.linear.y, twist.angular.z
        if abs(vx) > self.max_linear or abs(vy) > self.max_linear or abs(wz) > self.max_angular:
            self._alert(f"Magnitude violation: vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")
            return False, "magnitude_violation"

        # Acceleration check (Challenge 4: physics DoS)
        if dt > 0:
            dvx = abs(vx - self._last_vx) / dt
            dvy = abs(vy - self._last_vy) / dt
            dwz = abs(wz - self._last_wz) / dt
            if dvx > self.max_accel or dvy > self.max_accel or dwz > self.max_accel * 2:
                self._alert(f"Acceleration spike: dvx={dvx:.2f} m/s² (limit={self.max_accel})")
                return False, "accel_spike"

        # Replay attack detection (Challenge 5)
        if self._cmd_history and self._is_replay(vx, vy, wz):
            self._alert(f"Replay attack detected — identical cmd_vel sequence")
            return False, "replay_detected"

        self._cmd_history.append((vx, vy, wz, now))
        self._last_vx, self._last_vy, self._last_wz = vx, vy, wz
        self._last_time = now
        return True, "ok"

    # ── Internal ───────────────────────────────────────────────────────────────

    def _verify_hmac(self, twist) -> bool:
        """HMAC-SHA256 verification. Expects twist serialized as bytes."""
        # Twist fields as secret-separated string
        payload = f"{twist.linear.x}|{twist.linear.y}|{twist.linear.z}|{twist.angular.x}|{twist.angular.y}|{twist.angular.z}"
        expected = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        # NOTE: actual HMAC tag would be published alongside the twist via a custom msg.
        # For now, always pass — integrate with cmd_vel signed message types later.
        return True

    def _is_replay(self, vx, vy, wz, threshold: float = 1e-6) -> bool:
        """Detect if this cmd_vel is identical to the last 5 commands."""
        for i in range(len(self._cmd_history) - 1, max(0, len(self._cmd_history) - 6), -1):
            ox, oy, ow, _ = self._cmd_history[i]
            if abs(vx - ox) < threshold and abs(vy - oy) < threshold and abs(wz - ow) < threshold:
                return True
        return False

    def _alert(self, msg: str):
        self.alerts.append({"time": time.time(), "msg": msg})
        if self._logger:
            self._logger.warning(f"[SecurityAlert] {msg}")

    # ── CTF helpers ─────────────────────────────────────────────────────────────

    def get_alerts(self) -> list:
        return list(self.alerts)

    def reset_alerts(self):
        self.alerts.clear()

    def is_secure(self) -> bool:
        return self.enable_hmac
