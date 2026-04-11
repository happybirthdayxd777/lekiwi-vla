"""
Real Hardware Adapter for LeKiWi Bridge
========================================
Handles serial communication with ST3215 servo controllers in real hardware mode.
Bypasses MuJoCo simulation and talks directly to physical servos.

ST3215 Protocol (half-duplex serial, 115200 baud):
  - Position write:  0xFF 0xFE 0x0D <ID> 0x03 0x0A <PosH> <PosL> <SpeedH> <SpeedL> <Checksum>
  - Position read:   0xFF 0xFE 0x0D <ID> 0x02 0x0A <Checksum>
  - Reply:          0xFF 0xFE 0x0D <ID> 0x03 0x0E <PosH> <PosL> <SpeedH> <SpeedL> <Checksum>

For arm joints:  0x0A (SERVO_WRITE_POS) / 0x0E (SERVO_READ_POS)
For wheel joints: same protocol, servo IDs [10, 11, 12]

All positions are 0-4095 (12-bit) mapped to 0-360°.
Speed is 0-1000 (rpm * 10).
"""

import struct
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ST3215 protocol constants
STX1, STX2 = 0xFF, 0xFE
CMD_SERVO_WRITE_POS = 0x0D
CMD_SERVO_READ_POS  = 0x0E
SERVO_WRITE_POS     = 0x03
SERVO_READ_POS      = 0x02


@dataclass
class ServoState:
    position: float   # radians (converted from 0-4095 raw)
    velocity: float  # rad/s (estimated from position delta)


class ST3215Protocol:
    """ST3215 half-duplex serial protocol handler."""

    @staticmethod
    def position_to_raw(rad: float) -> int:
        """Convert radians (0-2π) to ST3215 12-bit position (0-4095)."""
        # Normalize to [0, 2π]
        rad = rad % (2 * np.pi)
        raw = int(rad / (2 * np.pi) * 4095)
        return max(0, min(4095, raw))

    @staticmethod
    def raw_to_position(raw: int) -> float:
        """Convert ST3215 12-bit raw (0-4095) to radians (0-2π)."""
        return (raw / 4095.0) * 2 * np.pi

    @staticmethod
    def speed_to_raw(rpm: float) -> int:
        """Convert rpm to ST3215 speed (0-1000)."""
        raw = int(abs(rpm) * 10)
        return max(1, min(1000, raw))

    @staticmethod
    def checksum(data: bytes) -> int:
        """Calculate ST3215 checksum: ~(ID + CMD + params) & 0xFF."""
        return (~sum(data)) & 0xFF

    def build_write_packet(self, servo_id: int, position_raw: int, speed_raw: int) -> bytes:
        """Build a position-write packet."""
        params = bytes([servo_id, SERVO_WRITE_POS, CMD_SERVO_WRITE_POS,
                         (position_raw >> 8) & 0xFF, position_raw & 0xFF,
                         (speed_raw >> 8) & 0xFF, speed_raw & 0xFF])
        cs = self.checksum(params)
        return bytes([STX1, STX2]) + params + bytes([cs])

    def build_read_packet(self, servo_id: int) -> bytes:
        """Build a position-read packet."""
        params = bytes([servo_id, SERVO_READ_POS, CMD_SERVO_READ_POS])
        cs = self.checksum(params)
        return bytes([STX1, STX2]) + params + bytes([cs])

    def parse_reply(self, packet: bytes) -> Optional[tuple]:
        """Parse a position-reply packet. Returns (position_raw, speed_raw) or None."""
        if len(packet) < 9 or packet[0] != STX1 or packet[1] != STX2:
            return None
        try:
            pos_raw = (packet[5] << 8) | packet[6]
            spd_raw = (packet[7] << 8) | packet[8]
            return pos_raw, spd_raw
        except IndexError:
            return None


class RealHardwareAdapter:
    """
    Serial adapter for real LeKiWi hardware.
    
    Manages arm servos (IDs 1-5) and wheel servos (IDs 10-12).
    Uses a worker thread to continuously poll servo positions
    and a lock-protected command queue for sending.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        arm_servo_ids: List[int] = None,
        wheel_servo_ids: List[int] = None,
        arm_num_joints: int = 5,
        wheel_num_joints: int = 3,
    ):
        self.port = port
        self.baudrate = baudrate
        self.arm_servo_ids  = arm_servo_ids  or [1, 2, 3, 4, 5]
        self.wheel_servo_ids = wheel_servo_ids or [10, 11, 12]
        self.arm_num_joints  = arm_num_joints
        self.wheel_num_joints = wheel_num_joints

        self._protocol = ST3215Protocol()
        self._serial = None       # initialized in connect()
        self._lock   = threading.Lock()
        self._running = False
        self._reader_thread: threading.Thread = None

        # Servo state (updated by reader thread)
        self._arm_states: List[ServoState]  = [
            ServoState(position=0.0, velocity=0.0) for _ in range(arm_num_joints)
        ]
        self._wheel_states: List[ServoState] = [
            ServoState(position=0.0, velocity=0.0) for _ in range(wheel_num_joints)
        ]
        self._last_arm_positions  = [0.0] * arm_num_joints
        self._last_wheel_positions = [0.0] * wheel_num_joints
        self._last_poll_time = time.time()

        # Command queue for sending (processed in reader thread)
        self._cmd_queue: List[bytes] = []
        self._last_cmd_time = 0.0

    # ── Connection ─────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open serial connection. Returns True on success."""
        try:
            import serial
        except ImportError:
            print("[RealHardwareAdapter] pyserial not installed: pip install pyserial")
            return False

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.1,
            )
            self._running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            print(f"[RealHardwareAdapter] Connected to {self.port} @ {self.baudrate}")
            return True
        except Exception as e:
            print(f"[RealHardwareAdapter] Failed to connect to {self.port}: {e}")
            return False

    def disconnect(self):
        """Stop reader thread and close serial port."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[RealHardwareAdapter] Disconnected.")

    # ── Reader loop ──────────────────────────────────────────────────────────────

    def _reader_loop(self):
        """Continuously poll all servos and update state."""
        while self._running:
            try:
                self._poll_all_servos()
                self._send_queued_commands()
                time.sleep(0.02)   # ~50 Hz poll rate
            except Exception as e:
                print(f"[RealHardwareAdapter] Reader error: {e}")
                time.sleep(0.1)

    def _poll_all_servos(self):
        """Poll arm and wheel servo positions (read-query round-robin)."""
        # Round-robin: poll one servo per call to avoid flooding the bus
        now = time.time()

        # Poll one arm servo
        for i, sid in enumerate(self.arm_servo_ids):
            pos_raw = self._read_position(sid)
            if pos_raw is not None:
                pos_rad = self._protocol.raw_to_position(pos_raw)
                dt = max(now - self._last_poll_time, 0.001)
                vel = (pos_rad - self._last_arm_positions[i]) / dt
                self._arm_states[i] = ServoState(position=pos_rad, velocity=vel)
                self._last_arm_positions[i] = pos_rad

        # Poll one wheel servo
        for i, sid in enumerate(self.wheel_servo_ids):
            pos_raw = self._read_position(sid)
            if pos_raw is not None:
                pos_rad = self._protocol.raw_to_position(pos_raw)
                dt = max(now - self._last_poll_time, 0.001)
                vel = (pos_rad - self._last_wheel_positions[i]) / dt
                self._wheel_states[i] = ServoState(position=pos_rad, velocity=vel)
                self._last_wheel_positions[i] = pos_rad

        self._last_poll_time = now

    def _read_position(self, servo_id: int) -> Optional[int]:
        """Send read query and parse reply for one servo."""
        with self._lock:
            try:
                pkt = self._protocol.build_read_packet(servo_id)
                self._serial.write(pkt)
                # ST3215 reply is 10 bytes, wait up to 50ms
                resp = self._serial.read(10)
                if len(resp) >= 9:
                    result = self._protocol.parse_reply(resp)
                    return result[0] if result else None
            except Exception:
                pass
        return None

    def _send_queued_commands(self):
        """Send pending commands from queue (one per cycle to avoid bus collision)."""
        with self._lock:
            if self._cmd_queue:
                pkt = self._cmd_queue.pop(0)
                try:
                    self._serial.write(pkt)
                except Exception as e:
                    print(f"[RealHardwareAdapter] Write error: {e}")

    # ── Public API ───────────────────────────────────────────────────────────────

    def queue_arm_positions(self, positions_rad: List[float], speed_rpm: float = 30.0):
        """
        Queue arm position commands for all 5 joints.
        Commands are sent round-robin style (one per poll cycle).
        """
        speed_raw = self._protocol.speed_to_raw(speed_rpm)
        packets = []
        for servo_id, pos_rad in zip(self.arm_servo_ids, positions_rad):
            pos_raw = self._protocol.position_to_raw(pos_rad)
            packets.append(self._protocol.build_write_packet(servo_id, pos_raw, speed_raw))
        with self._lock:
            self._cmd_queue.extend(packets)

    def queue_wheel_velocities(self, velocities_rad_s: List[float]):
        """
        Queue wheel velocity commands.
        ST3215 doesn't support pure velocity control — we use position-stepping
        by computing the next target position based on velocity.
        
        For simplicity, convert each wheel velocity to a target position
        increment and command the next position.
        """
        speed_raw = self._protocol.speed_to_raw(50.0)   # fixed medium speed
        packets = []
        for servo_id, vel_rad_s in zip(self.wheel_servo_ids, velocities_rad_s):
            # Compute incremental position command from velocity
            dt = 0.02   # 50 Hz cycle
            delta_pos = vel_rad_s * dt
            # Get current position and add delta
            # Since we don't have instant access to current pos from reader thread,
            # we use the last known position stored in _last_wheel_positions
            idx = self.wheel_servo_ids.index(servo_id)
            last_pos = self._last_wheel_positions[idx]
            target_pos = last_pos + delta_pos
            pos_raw = self._protocol.position_to_raw(target_pos)
            packets.append(self._protocol.build_write_packet(servo_id, pos_raw, speed_raw))
        with self._lock:
            self._cmd_queue.extend(packets)

    def get_state(self) -> dict:
        """
        Return current servo state as a dict matching the MuJoCo observation format.
        Used by bridge_node._publish_joint_states() in real mode.
        """
        arm_pos  = [s.position for s in self._arm_states]
        arm_vel  = [s.velocity for s in self._arm_states]
        wheel_pos = [s.position for s in self._wheel_states]
        wheel_vel = [s.velocity for s in self._wheel_states]
        return {
            "arm_positions":  np.array(arm_pos,  dtype=np.float64),
            "arm_velocities": np.array(arm_vel,  dtype=np.float64),
            "wheel_positions": np.array(wheel_pos, dtype=np.float64),
            "wheel_velocities": np.array(wheel_vel, dtype=np.float64),
        }

    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open


# ── Mock adapter for testing without real hardware ──────────────────────────────

class MockHardwareAdapter:
    """
    Mock adapter — simulates servo feedback without real serial hardware.
    Useful for testing real-mode code paths on a development machine.
    """

    def __init__(
        self,
        arm_num_joints: int = 5,
        wheel_num_joints: int = 3,
    ):
        self.arm_num_joints  = arm_num_joints
        self.wheel_num_joints = wheel_num_joints
        self._arm_positions  = np.zeros(arm_num_joints)
        self._wheel_positions = np.zeros(wheel_num_joints)
        self._last_time = time.time()

    def connect(self) -> bool:
        print("[MockHardwareAdapter] Connected (mock mode)")
        return True

    def disconnect(self):
        print("[MockHardwareAdapter] Disconnected")

    def queue_arm_positions(self, positions_rad: List[float], speed_rpm: float = 30.0):
        self._arm_positions = np.array(positions_rad, dtype=np.float64)

    def queue_wheel_velocities(self, velocities_rad_s: List[float]):
        now = time.time()
        dt = now - self._last_time
        self._wheel_positions += np.array(velocities_rad_s) * dt
        self._last_time = now

    def get_state(self) -> dict:
        return {
            "arm_positions":   self._arm_positions,
            "arm_velocities":   np.zeros(self.arm_num_joints),
            "wheel_positions":  self._wheel_positions,
            "wheel_velocities": np.zeros(self.wheel_num_joints),
        }

    def is_connected(self) -> bool:
        return True
