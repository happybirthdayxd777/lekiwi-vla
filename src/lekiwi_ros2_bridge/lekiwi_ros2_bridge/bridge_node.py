#!/usr/bin/env python3
"""
LeKiWi ROS2 ↔ MuJoCo Bridge Node
================================
Bridges ROS2 cmd_vel → MuJoCo sim, and MuJoCo sensor data → ROS2 joint_states.

Topics:
  Input  : /lekiwi/cmd_vel        (geometry_msgs/Twist)
  Output : /lekiwi/joint_states   (sensor_msgs/JointState)

The bridge maps the 3-wheel omni kinematics from omni_controller.py onto
LeKiWiSim (sim_lekiwi.py) ctrl[6:9], and converts the MuJoCo observation
back into a ROS2 JointState message.

Architecture:
  ROS2 /lekiwi/cmd_vel
        ↓ (Twist → [vx, vy, wz])
  BridgeNode.cmd_vel_callback()
        ↓
  LeKiWiSim.step(action=[arm*6, wheel_speeds*3])
        ↓
  BridgeNode.publish_joint_states()
        ↓ (JointState ← MuJoCo obs)
  ROS2 /lekiwi/joint_states
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import numpy as np
import sys
import os

# ── LeKiWiSim import ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi import LeKiWiSim


# ── Kinematics constants (from omni_controller.py) ──────────────────────────────
WHEEL_RADIUS   = 0.05    # m
WHEEL_POSITIONS = np.array([
    [ 0.1732,  0.0,    0.0 ],   # wheel 0 — front
    [-0.0866,  0.15,   0.0 ],   # wheel 1 — back-left
    [-0.0866, -0.15,   0.0 ],   # wheel 2 — back-right
], dtype=np.float64)

# Normalised joint axes (omni rollers)
_JOINT_AXES = np.array([
    [0.866025, 0.0, 0.5],
    [0.866025, 0.0, 0.5],
    [0.866025, 0.0, 0.5],
], dtype=np.float64)
_JOINT_AXES /= np.linalg.norm(_JOINT_AXES, axis=1, keepdims=True)


def twist_to_wheel_speeds(vx: float, vy: float, wz: float) -> np.ndarray:
    """
    Convert robot-level Twist (vx, vy, wz) into 3 wheel angular velocities.
    Mirrors the exact kinematics from lekiwi_modular/omni_controller.py.
    Returns shape (3,).
    """
    wheel_speeds = np.zeros(3, dtype=np.float64)
    for i in range(3):
        wheel_vel = np.array([
            vx - wz * WHEEL_POSITIONS[i, 1],
            vy + wz * WHEEL_POSITIONS[i, 0],
            0.0,
        ])
        angular_speed = np.dot(wheel_vel, _JOINT_AXES[i]) / WHEEL_RADIUS
        wheel_speeds[i] = angular_speed
    return wheel_speeds


class LeKiWiBridge(Node):
    """
    ROS2 ↔ MuJoCo bridge for LeKiWi robot.

    Publishers:
      /lekiwi/joint_states   — arm (6) + wheel (3) joint positions & velocities

    Subscribers:
      /lekiwi/cmd_vel        — Twist (linear x/y, angular z)
    """

    # MuJoCo ctrl index layout (from sim_lekiwi.py):
    #   ctrl[0:6]  = arm joints   (j0..j5, range ±3.14)
    #   ctrl[6:9]  = wheel joints (w1..w3, range ±5.0)
    ARM_CTRL_MIN  = -3.14
    ARM_CTRL_MAX  =  3.14
    WHEEL_CTRL_MIN = -5.0
    WHEEL_CTRL_MAX =  5.0

    ARM_NAMES  = ["j0", "j1", "j2", "j3", "j4", "j5"]
    WHEEL_NAMES = ["w1", "w2", "w3"]

    def __init__(self):
        super().__init__("lekiwi_ros2_bridge")

        # ── Initialise MuJoCo simulation ────────────────────────────────────
        self.get_logger().info("Starting LeKiWi MuJoCo simulation…")
        self.sim = LeKiWiSim()
        self.get_logger().info("MuJoCo simulation initialised.")

        # ── ROS2 publishers ────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.joint_state_pub = self.create_publisher(JointState, "/lekiwi/joint_states", qos)

        # Camera bridge: MuJoCo → ROS2 image
        self.camera_pub = self.create_publisher(Image, "/lekiwi/camera/image_raw", qos)
        self.bridge = CvBridge()

        # Separate publishers for each wheel (mirrors omni_controller output)
        self.wheel_pubs = [
            self.create_publisher(Float64, f"/lekiwi/wheel_{i}/cmd_vel", qos)
            for i in range(3)
        ]

        # ── ROS2 subscribers ───────────────────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/lekiwi/cmd_vel", self._on_cmd_vel, qos
        )

        # ── Timer: step MuJoCo & publish at 20 Hz (camera is expensive) ────────
        self.timer = self.create_timer(0.05, self._on_timer)   # 20 Hz

        # ── State ───────────────────────────────────────────────────────────────
        self._last_action = np.zeros(9, dtype=np.float64)   # [arm*6, wheel*3]
        self._frame_count = 0
        self.get_logger().info(
            "LeKiWi ROS2 bridge ready.  Topics:\n"
            "  /lekiwi/cmd_vel       ← subscribe\n"
            "  /lekiwi/joint_states  → publish\n"
            "  /lekiwi/camera/image_raw → publish (20 Hz)\n"
            "  /lekiwi/wheel_N/cmd_vel → publish"
        )

    # ── cmd_vel callback ────────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """Convert Twist → wheel speeds, combine with current arm action, step sim."""
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)

        # Compute wheel angular velocities from kinematics
        wheel_speeds = twist_to_wheel_speeds(vx, vy, wz)

        # Keep arm portion of last action (held at 0 for now; future: arm topics)
        arm_action = self._last_action[0:6]

        # Build full action vector for MuJoCo
        action = np.concatenate([arm_action, wheel_speeds]).astype(np.float64)

        # Execute one simulation step
        self.sim.step(action)
        self._last_action = action

        # Also republish individual wheel velocities (mirrors real robot)
        for i, speed in enumerate(wheel_speeds):
            wm = Float64()
            wm.data = float(speed)
            self.wheel_pubs[i].publish(wm)

    # ── Timer callback ─────────────────────────────────────────────────────────

    def _on_timer(self):
        """Publish MuJoCo state as JointState + camera Image."""
        obs = self.sim._obs()

        # ── JointState ──────────────────────────────────────────────────────
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.ARM_NAMES + self.WHEEL_NAMES

        # Arm positions (6) + integrate wheel velocities → positions (3)
        arm_pos = list(obs.get("arm_positions", np.zeros(6)))
        # Accumulate wheel positions from velocity integration
        wheel_vel = obs.get("wheel_velocities", np.zeros(3))
        dt = 0.05   # matches timer period
        self._wheel_posAccum = getattr(self, '_wheel_posAccum', np.zeros(3)) + wheel_vel * dt
        wheel_pos = list(self._wheel_posAccum)

        msg.position = arm_pos + wheel_pos
        msg.velocity = list(obs.get("arm_velocities", np.zeros(6))) + list(wheel_vel)

        self.joint_state_pub.publish(msg)

        # ── Camera Image ────────────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % 1 == 0:   # publish every frame (20 Hz)
            try:
                img_pil = self.sim.render(640, 480)
                img_np  = np.asarray(img_pil)
                ros_img = self.bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
                ros_img.header.stamp = msg.header.stamp
                ros_img.header.frame_id = "lekiwi_camera"
                self.camera_pub.publish(ros_img)
            except Exception as e:
                self.get_logger().warn(f"Camera render failed: {e}", once=True)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def apply_arm_action(self, arm_pos: np.ndarray):
        """
        Directly set arm joint targets.
        arm_pos: shape (6,) in radians.
        Used by external VLA policy nodes.
        """
        arm_pos = np.clip(arm_pos, self.ARM_CTRL_MIN, self.ARM_CTRL_MAX)
        self._last_action[0:6] = arm_pos


def main(args=None):
    rclpy.init(args=args)
    node = LeKiWiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("LeKiWi bridge interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
