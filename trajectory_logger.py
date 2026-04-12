#!/usr/bin/env python3
"""
LeKiWi Trajectory Logger + Recorder
===================================
Two interfaces:
  1. TrajectoryRecorder — non-ROS utility class (embedded in bridge_node)
  2. TrajectoryLogger    — standalone ROS2 node (for external recording)

Records (cmd_vel, joint_states, camera) triples during teleop or VLA runs.
Saves to HDF5 for later replay through replay_node.

Recording format mirrors lekiwi_urdf_5k.h5 so replay_data.py can
replay trajectories through the MuJoCo sim for offline policy evaluation.

Output: ~/hermes_research/lekiwi_vla/trajectories/<run_id>.h5

Usage (standalone node):
    python3 trajectory_logger.py --mode record --output /tmp/my_run.h5

Usage (bridge integration — no camera recording):
    ros2 launch lekiwi_ros2_bridge full.launch.py record:=true record_file:=/tmp/my_run.h5

    # Control recording:
    ros2 topic pub /lekiwi/record_control std_msgs/String "start"
    ros2 topic pub /lekiwi/record_control std_msgs/String "stop"
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String
from cv_bridge import CvBridge


# ─── Configuration ───────────────────────────────────────────────────────────

MAX_FRAMES = 50_000          # per file
IMAGE_H, IMAGE_W = 224, 224  # match VLA training resolution

ARM_JOINT_NAMES  = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINT_NAMES = ["w1", "w2", "w3"]


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryRecorder — non-ROS utility class for use inside bridge_node
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryRecorder:
    """
    In-process trajectory recorder for embedding in bridge_node.

    Takes cmd_vel + joint_states data directly from bridge callbacks
    and writes to HDF5 on flush().

    Recording format (HDF5):
        /cmd_vel       (N, 3)   float64   [vx, vy, wz]
        /joint_states  (N, 18) float64   [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
        /timestamps    (N,)    float64   seconds
        /metadata      {run_id, duration_s, num_frames}

    Usage in bridge_node:
        self.recorder = TrajectoryRecorder("/tmp/my_traj.h5")
        # In _on_cmd_vel:  self.recorder.record_cmd_vel(vx, vy, wz)
        # In _on_timer:    self.recorder.record_joint_state(...)
        # On shutdown:     self.recorder.flush()
    """

    def __init__(self, output_path: str, max_frames: int = MAX_FRAMES):
        self.output_path = os.path.expanduser(output_path)
        self.max_frames  = max_frames
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        self.cmd_vel_buf  = []
        self.js_buf       = []
        self.ts_buf       = []
        self._start_time  = None
        self._recording   = False

    def start(self):
        self._recording  = True
        self._start_time  = datetime.now()
        self.cmd_vel_buf.clear()
        self.js_buf.clear()
        self.ts_buf.clear()

    def stop(self):
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def num_frames(self) -> int:
        return len(self.js_buf)

    def record_cmd_vel(self, vx: float, vy: float, wz: float):
        if not self._recording or len(self.js_buf) >= self.max_frames:
            return
        self.cmd_vel_buf.append([vx, vy, wz])

    def record_joint_state(
        self,
        arm_positions: list,
        arm_velocities: list,
        wheel_positions: list,
        wheel_velocities: list,
        timestamp: float = None,
    ):
        if not self._recording or len(self.js_buf) >= self.max_frames:
            return
        row = list(arm_positions) + list(arm_velocities) + list(wheel_positions) + list(wheel_velocities)
        self.js_buf.append(row)
        self.ts_buf.append(timestamp or (datetime.now().timestamp()))

    def flush(self):
        """Write buffered data to HDF5. Call on shutdown or episode end."""
        if not self._recording and not self.js_buf:
            return

        n = len(self.js_buf)
        if n == 0:
            return

        # Pad cmd_vel to same length as joint_states
        cv_len = len(self.cmd_vel_buf)
        if cv_len < n:
            self.cmd_vel_buf.extend([[0.0, 0.0, 0.0]] * (n - cv_len))
        elif cv_len > n:
            self.cmd_vel_buf = self.cmd_vel_buf[:n]

        duration = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0

        with h5py.File(self.output_path, "w") as f:
            f.create_dataset("cmd_vel",      data=np.array(self.cmd_vel_buf, dtype=np.float64))
            f.create_dataset("joint_states", data=np.array(self.js_buf,      dtype=np.float64))
            f.create_dataset("timestamps",   data=np.array(self.ts_buf,     dtype=np.float64))
            f.attrs["run_id"]      = Path(self.output_path).stem
            f.attrs["duration_s"]   = float(duration)
            f.attrs["num_frames"]   = n
            f.attrs["arm_names"]    = ARM_JOINT_NAMES
            f.attrs["wheel_names"]  = WHEEL_JOINT_NAMES

        size_mb = os.path.getsize(self.output_path) / 1024 / 1024
        print(f"[TrajectoryRecorder] Saved {n} frames, {duration:.1f}s → "
              f"{self.output_path} ({size_mb:.1f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryLogger(Node):
    """
    Records cmd_vel + joint_states + camera frames to HDF5.

    Recording format (one HDF5 file per run):
        /cmd_vel       (N, 3)   float64   [vx, vy, wz]
        /joint_states  (N, 18) float64   [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
        /images        (N, H, W, 3) uint8
        /timestamps    (N,)    float64   ROS time seconds
        /metadata      {
            "run_id": str,
            "duration_s": float,
            "num_frames": int,
        }
    """

    def __init__(self, output_path: str, record_camera: bool = True):
        super().__init__("trajectory_logger")

        self.output_path = os.path.expanduser(output_path)
        self.record_camera = record_camera

        # Ensure output dir exists
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # ── HDF5 buffers ────────────────────────────────────────────────
        self.cmd_vel_buf   = []
        self.js_buf        = []   # [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
        self.img_buf       = []
        self.ts_buf        = []

        # ── ROS2 QoS ─────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.js_sub = self.create_subscription(
            JointState, "/lekiwi/joint_states", self._on_js, qos
        )
        self.cv_sub = self.create_subscription(
            Twist, "/lekiwi/cmd_vel", self._on_cmd_vel, qos
        )

        if record_camera:
            self.cam_sub = self.create_subscription(
                Image, "/lekiwi/camera/image_raw", self._on_cam, qos
            )
        else:
            self.cam_sub = None

        self.bridge = CvBridge()
        self.get_logger().info(f"TrajectoryLogger recording to: {self.output_path}")
        self.get_logger().info(
            f"Recording: joint_states={'/lekiwi/joint_states'}, "
            f"cmd_vel={'/lekiwi/cmd_vel'}, "
            f"camera={'/lekiwi/camera/image_raw' if record_camera else 'DISABLED'}"
        )

        self._js_received = False
        self._start_time = self.get_clock().now()

    # ── Subscriber callbacks ───────────────────────────────────────────────────

    def _on_js(self, msg: JointState):
        """Buffer joint_states: [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]."""
        if len(self.js_buf) >= MAX_FRAMES:
            return

        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity))

        # Bridge canonical names: j0..j5 + w1..w3
        arm_pos  = [pos.get(n, 0.0) for n in ARM_JOINT_NAMES]
        arm_vel  = [vel.get(n, 0.0) for n in ARM_JOINT_NAMES]
        wheel_pos = [pos.get(n, 0.0) for n in WHEEL_JOINT_NAMES]
        wheel_vel = [vel.get(n, 0.0) for n in WHEEL_JOINT_NAMES]

        row = arm_pos + arm_vel + wheel_pos + wheel_vel
        self.js_buf.append(row)
        self.ts_buf.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        if not self._js_received:
            self._js_received = True
            self.get_logger().info("First joint_states received — recording started.")

    def _on_cmd_vel(self, msg: Twist):
        """Buffer cmd_vel: [vx, vy, wz]."""
        if len(self.js_buf) >= MAX_FRAMES:
            return
        self.cmd_vel_buf.append([msg.linear.x, msg.linear.y, msg.angular.z])

    def _on_cam(self, msg: Image):
        """Buffer camera frame (resize to VLA resolution)."""
        if len(self.img_buf) >= MAX_FRAMES:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            img = self._resize(img, (IMAGE_W, IMAGE_H))
            self.img_buf.append(img)
        except Exception as e:
            self.get_logger().warn(f"Camera frame dropped: {e}")

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resize(img: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize with PIL (nearest) to avoid OpenCV interpolation artifacts."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img)
        pil = pil.resize(target_size, PILImage.NEAREST)
        return np.array(pil)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def flush(self):
        """Write buffered data to HDF5 file."""
        n = len(self.js_buf)
        if n == 0:
            self.get_logger().warn("No frames recorded — nothing to save.")
            return

        self.get_logger().info(f"Flushing {n} frames to {self.output_path}…")

        # Ensure cmd_vel and image buffers are the same length as joint_states
        # by padding/truncating if needed
        cv_len = len(self.cmd_vel_buf)
        if cv_len < n:
            self.cmd_vel_buf.extend([[0.0, 0.0, 0.0]] * (n - cv_len))
        elif cv_len > n:
            self.cmd_vel_buf = self.cmd_vel_buf[:n]

        if self.record_camera:
            img_len = len(self.img_buf)
            if img_len < n:
                # Pad with black frames
                black = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
                self.img_buf.extend([black] * (n - img_len))
            elif img_len > n:
                self.img_buf = self.img_buf[:n]
        else:
            self.img_buf = [np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)] * n

        duration = self.get_clock().now().seconds_nanoseconds()[0] - self._start_time.seconds_nanoseconds()[0]

        with h5py.File(self.output_path, "w") as f:
            f.create_dataset("cmd_vel",      data=np.array(self.cmd_vel_buf, dtype=np.float64))
            f.create_dataset("joint_states", data=np.array(self.js_buf,      dtype=np.float64))
            f.create_dataset("images",       data=np.stack(self.img_buf),            dtype=np.uint8)
            f.create_dataset("timestamps",   data=np.array(self.ts_buf, dtype=np.float64))
            f.attrs["run_id"]       = Path(self.output_path).stem
            f.attrs["duration_s"]  = float(duration)
            f.attrs["num_frames"]   = n
            f.attrs["arm_names"]    = ARM_JOINT_NAMES
            f.attrs["wheel_names"]  = WHEEL_JOINT_NAMES

        self.get_logger().info(
            f"Saved {n} frames, {duration:.1f}s → {self.output_path}  "
            f"({os.path.getsize(self.output_path) / 1024 / 1024:.1f} MB)"
        )

    def get_stats(self) -> dict:
        n = len(self.js_buf)
        return {
            "num_frames": n,
            "cmd_vel_frames": len(self.cmd_vel_buf),
            "image_frames": len(self.img_buf),
            "buffer_full": n >= MAX_FRAMES,
        }


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeKiWi Trajectory Logger")
    parser.add_argument("--mode",   default="record", choices=["record", "stats"],
                        help="record: save HDF5 on shutdown; stats: print current buffer")
    parser.add_argument("--output", default="~/hermes_research/lekiwi_vla/trajectories/run_default.h5",
                        help="Output HDF5 path")
    parser.add_argument("--no-camera", action="store_true",
                        help="Disable camera recording")
    args = parser.parse_args()

    rclpy.init()
    node = TrajectoryLogger(args.output, record_camera=not args.no_camera)

    if args.mode == "stats":
        import time
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            stats = node.get_stats()
            print(f"Frames: {stats['num_frames']}  |  cmd_vel: {stats['cmd_vel_frames']}  |  "
                  f"images: {stats['image_frames']}  |  buffer_full: {stats['buffer_full']}", flush=True)
            time.sleep(2)
    else:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.flush()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
