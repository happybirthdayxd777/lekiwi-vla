#!/usr/bin/env python3
"""
LeKiWi Trajectory Replay Node
==============================
Plays back recorded HDF5 trajectories through the LeKiWi bridge.
Used for:
  1. Offline policy evaluation (replay observed trajectories)
  2. Sim-to-real comparison (record real → replay in sim)
  3. Bug reproduction from recorded sessions

Playback topics:
  Input  : /lekiwi/replay_control  (String: "play", "pause", "stop", "step")
  Output : /lekiwi/joint_states    — replayed joint positions @ replay_hz
  Output : /lekiwi/cmd_vel         — replayed velocity commands (if in data)
  Output : /lekiwi/replay_status   (String: current frame / total frames)

Usage:
  # Replay a trajectory:
  ros2 run lekiwi_ros2_bridge replay_node --ros-args \
    -p replay_file:=~/hermes_research/lekiwi_vla/trajectories/run_default.h5 \
    -p replay_hz:=20.0

  # Control playback:
  ros2 topic pub /lekiwi/replay_control std_msgs/String "stop"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge


# ─── Configuration ───────────────────────────────────────────────────────────

ARM_JOINT_NAMES   = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINT_NAMES = ["w1", "w2", "w3"]

DEFAULT_REPLAY_HZ = 20.0   # Hz — match camera frame rate

# Package-relative path to lekiwi_vla root (for loading HDF5 paths)
_LEKIWI_VLA_ROOT = os.path.expanduser("~/hermes_research/lekiwi_vla")


# ─────────────────────────────────────────────────────────────────────────────

class ReplayNode(Node):
    """
    Replays HDF5 trajectory files through ROS2 topics.

    Reads from HDF5 (trajectory_logger.py format):
        /cmd_vel       (N, 3)   [vx, vy, wz]
        /joint_states  (N, 18)  [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
        /images        (N, H, W, 3) uint8
        /timestamps    (N,)    float64

    Publishes at replay_hz to:
        /lekiwi/joint_states  — JointState message
        /lekiwi/cmd_vel       — Twist message (if cmd_vel data exists)
        /lekiwi/replay_status — frame counter

    Subscribes to:
        /lekiwi/replay_control — "play", "pause", "stop", "step"
    """

    def __init__(
        self,
        replay_file: str,
        replay_hz: float = DEFAULT_REPLAY_HZ,
        loop: bool = True,
        start_frame: int = 0,
    ):
        super().__init__("replay_node")

        self.replay_file = os.path.expanduser(replay_file)
        self.replay_hz   = float(replay_hz)
        self.loop        = loop
        self.start_frame = start_frame

        self._load_trajectory()

        # ── State machine ────────────────────────────────────────────────
        self._playing    = True   # start immediately
        self._paused     = False
        self._stopped    = False
        self._current_idx = self.start_frame

        # ── Publishers ───────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.js_pub    = self.create_publisher(JointState, "/lekiwi/joint_states", qos)
        self.cv_pub    = self.create_publisher(Twist,       "/lekiwi/cmd_vel",        qos)
        self.stat_pub  = self.create_publisher(String,    "/lekiwi/replay_status",  qos)
        self.img_pub   = self.create_publisher(Image,      "/lekiwi/replay/image_raw", qos)
        self.bridge    = CvBridge()

        # ── Control subscriber ────────────────────────────────────────────
        self.ctrl_sub = self.create_subscription(
            String, "/lekiwi/replay_control", self._on_control, qos
        )

        # ── Timer for playback loop ───────────────────────────────────────
        period = 1.0 / self.replay_hz
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"ReplayNode: {self.replay_file}\n"
            f"  total_frames={self.n_frames}, replay_hz={self.replay_hz}Hz\n"
            f"  loop={self.loop}, start={self.start_frame}\n"
            f"  has_cmd_vel={self._has_cmd_vel}, has_images={self._has_images}"
        )

    # ── HDF5 loading ─────────────────────────────────────────────────────────

    def _load_trajectory(self):
        if not os.path.exists(self.replay_file):
            raise FileNotFoundError(f"Trajectory file not found: {self.replay_file}")

        self.get_logger().info(f"Loading trajectory: {self.replay_file}")
        with h5py.File(self.replay_file, "r") as f:
            self._js_data    = f["joint_states"][:]   # (N, 18)
            self._has_cmd_vel = "cmd_vel" in f
            self._cmd_vel_data = f["cmd_vel"][:] if self._has_cmd_vel else None
            self._has_images   = "images" in f
            self._img_data     = f["images"][:] if self._has_images else None
            self._timestamps   = f["timestamps"][:] if "timestamps" in f else None
            self.n_frames = len(self._js_data)

        self.get_logger().info(
            f"  joint_states: {self._js_data.shape}, range=[{self._js_data.min():.3f}, {self._js_data.max():.3f}]"
        )
        if self._has_cmd_vel:
            self.get_logger().info(f"  cmd_vel: {self._cmd_vel_data.shape}")
        if self._has_images:
            self.get_logger().info(f"  images: {self._img_data.shape}")

    # ── Control callbacks ─────────────────────────────────────────────────────

    def _on_control(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "play":
            self._playing = True
            self._paused  = False
            self._stopped = False
            self.get_logger().info("Replay: PLAY")
        elif cmd == "pause":
            self._paused  = True
            self._playing = False
            self.get_logger().info("Replay: PAUSE")
        elif cmd == "stop":
            self._stopped = True
            self._playing = False
            self._current_idx = self.start_frame
            self.get_logger().info("Replay: STOP (reset to frame 0)")
        elif cmd == "step":
            if self._paused:
                self._advance_frame()
            else:
                self.get_logger().warn("Step only works while paused — use 'pause' first")
        else:
            self.get_logger().warn(f"Unknown control command: {cmd} (use: play, pause, stop, step)")

    # ── Playback tick ─────────────────────────────────────────────────────────

    def _tick(self):
        """Called at replay_hz. Publishes next frame if playing."""
        if not self._playing or self._paused or self._stopped:
            return

        self._advance_frame()

    def _advance_frame(self):
        """Publish frame at self._current_idx, then advance."""
        if self._current_idx >= self.n_frames:
            if self.loop:
                self._current_idx = self.start_frame
                self.get_logger().info("Replay: loop restart")
            else:
                self._playing = False
                self.get_logger().info("Replay: END OF TRAJECTORY")
                return

        idx = self._current_idx

        # ── JointState ───────────────────────────────────────────────────
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name  = ARM_JOINT_NAMES + WHEEL_JOINT_NAMES
        # joint_states format: [arm_pos*6, arm_vel*6, wheel_pos*3, wheel_vel*3]
        arm_pos   = self._js_data[idx, 0:6]
        arm_vel   = self._js_data[idx, 6:12]
        wheel_pos = self._js_data[idx, 12:15]
        wheel_vel = self._js_data[idx, 15:18]
        js.position = list(arm_pos)  + list(wheel_pos)
        js.velocity = list(arm_vel) + list(wheel_vel)
        js.effort   = [0.0] * 9
        self.js_pub.publish(js)

        # ── cmd_vel (optional) ────────────────────────────────────────────
        if self._has_cmd_vel and self._cmd_vel_data is not None:
            cv = Twist()
            cv.linear.x  = self._cmd_vel_data[idx, 0]
            cv.linear.y  = self._cmd_vel_data[idx, 1]
            cv.angular.z = self._cmd_vel_data[idx, 2]
            self.cv_pub.publish(cv)

        # ── Camera image (optional) ───────────────────────────────────────
        if self._has_images and self._img_data is not None:
            img_hwc = self._img_data[idx]   # (H, W, 3) uint8
            img_msg = self.bridge.cv2_to_imgmsg(img_hwc, encoding="rgb8")
            img_msg.header.stamp = js.header.stamp
            img_msg.header.frame_id = "camera_link"
            self.img_pub.publish(img_msg)

        # ── Status ────────────────────────────────────────────────────────
        status = String()
        status.data = f"{self._current_idx + 1} / {self.n_frames}"
        self.stat_pub.publish(status)

        self._current_idx += 1

    # ── Status ───────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "current_frame": self._current_idx,
            "total_frames":  self.n_frames,
            "playing":       self._playing,
            "paused":        self._paused,
            "stopped":       self._stopped,
            "has_cmd_vel":   self._has_cmd_vel,
            "has_images":    self._has_images,
        }


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeKiWi Trajectory Replay Node")
    parser.add_argument("--replay-file", default="", help="Path to HDF5 trajectory file")
    parser.add_argument("--replay-hz",    type=float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument("--no-loop",     action="store_true", help="Stop at end (no loop)")
    parser.add_argument("--start-frame", type=int,   default=0)
    args = rclpy.utilities.parse_arguments(sys.argv[1:])

    # Extract remap args from full argv
    rclpy.init(args=args)

    params = {}
    for raw in args:
        if "replay-file:=" in raw:
            params["replay_file"] = raw.split(":=", 1)[1]
        elif "replay-hz:=" in raw:
            params["replay_hz"] = float(raw.split(":=", 1)[1])
        elif "no-loop" in raw:
            params["loop"] = False

    replay_file = params.get("replay_file", os.path.expanduser(
        "~/hermes_research/lekiwi_vla/trajectories/run_default.h5"))
    replay_hz   = params.get("replay_hz",  DEFAULT_REPLAY_HZ)
    loop        = params.get("loop", True)
    start_frame = int(params.get("start_frame", 0))

    try:
        node = ReplayNode(replay_file, replay_hz, loop, start_frame)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    except FileNotFoundError as e:
        print(f"[ReplayNode] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
