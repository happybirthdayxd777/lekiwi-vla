#!/usr/bin/env python3
"""
LeKiWi ROS2 Bridge Node
=======================
Bridges ROS2 cmd_vel → MuJoCo simulation and MuJoCo state → ROS2 joint_states.

Topics:
  Subscribe:  /lekiwi/cmd_vel      (geometry_msgs/Twist)   → robot motion
  Publish:    /lekiwi/joint_states  (sensor_msgs/JointState) ← MuJoCo state
  Publish:    /lekiwi/odom          (nav_msgs/Odometry)     ← base pose

The bridge maps /lekiwi/cmd_vel (vx, vy, wz) to the same 3-wheel omni-kinematics
used by lekiwi_modular/omni_controller.py, then applies torques to the MuJoCo
LeKiWiSim instance.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import numpy as np
import sys
import os

# Add lekiwi_vla to path for LeKiWiSim
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sim_lekiwi import LeKiWiSim, WHEEL_JOINTS, ARM_JOINTS


class LeKiWiBridge(Node):
    """
    ROS2 ↔ MuJoCo Bridge

    Receives /lekiwi/cmd_vel (Twist) and converts to wheel torques using the same
    kinematics as lekiwi_modular's omni_controller.py.  Feeds the torques into
    the LeKiWiSim MuJoCo instance and publishes the resulting state back as
    ROS2 joint_states and odometry.
    """

    def __init__(self, sim_mode: bool = True):
        super().__init__("lekiwi_ros2_bridge")

        # ── Robot kinematics (mirrors omni_controller.py) ──────────────────────
        self.wheel_radius = 0.05
        self.wheel_base   = 0.1732
        self.joint_axes = [
            np.array([0.866025, 0.0, 0.5]) / np.linalg.norm([0.866025, 0.0, 0.5]),
            np.array([0.866025, 0.0, 0.5]) / np.linalg.norm([0.866025, 0.0, 0.5]),
            np.array([0.866025, 0.0, 0.5]) / np.linalg.norm([0.866025, 0.0, 0.5]),
        ]
        self.wheel_positions = [
            np.array([0.1732,  0.0,   0.0]),
            np.array([-0.0866, 0.15,  0.0]),
            np.array([-0.0866, -0.15, 0.0]),
        ]

        # ── MuJoCo simulation ───────────────────────────────────────────────────
        self.sim = LeKiWiSim()
        self.get_logger().info(f"LeKiWi Bridge initialised (sim_mode={sim_mode})")

        # ── ROS2 interface ─────────────────────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/lekiwi/cmd_vel", self._on_cmd_vel, 10
        )

        self.joint_state_pub = self.create_publisher(JointState, "/lekiwi/joint_states", 10)
        self.odom_pub        = self.create_publisher(Odometry,    "/lekiwi/odom",        10)

        # Timer: step sim + publish at 50 Hz
        self.timer = self.create_timer(0.02, self._on_timer)

        # Current cmd_vel command (held between callbacks)
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0

        # Odometry state
        self._x     = 0.0
        self._y     = 0.0
        self._theta = 0.0
        self._last_time = self.get_clock().now()

        self.get_logger().info("LeKiWi Bridge ready — subscribed to /lekiwi/cmd_vel")

    # ── cmd_vel → MuJoCo ───────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """Store latest cmd_vel command."""
        self._vx = float(msg.linear.x)
        self._vy = float(msg.linear.y)
        self._wz = float(msg.angular.z)

    # ── Timer callback ───────────────────────────────────────────────────────────

    def _on_timer(self):
        """Called at 50 Hz: apply cmd_vel to MuJoCo, publish state to ROS2."""
        now = self.get_clock().now()
        dt  = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now

        # ── Kinematics: cmd_vel → wheel angular velocities ──────────────────
        # Same formula as omni_controller.py
        wheel_speeds = []
        for i in range(3):
            wheel_vel = np.array([
                self._vx - self._wz * self.wheel_positions[i][1],
                self._vy + self._wz * self.wheel_positions[i][0],
                0.0,
            ])
            angular_speed = np.dot(wheel_vel, self.joint_axes[i]) / self.wheel_radius
            wheel_speeds.append(angular_speed)

        # wheel_speeds are in rad/s.  MuJoCo step() takes action[6:9] which is
        # motor torque in Nm with gear=10.  We approximate: torque ≈ wheel_speed
        # (with wheel_radius=0.05, gear=10 → torque ≈ angular_speed * 0.5).
        # Scale to [-1, 1] action space (action[6:9] * 10.0 = ctrl, gear=10 → torque).
        # The MuJoCo sim expects action units of motor_torque_normalised where
        # ctrl = action * 10.0  →  max torque = 10 Nm.
        action = np.zeros(9, dtype=np.float64)
        action[6:9] = np.clip(np.array(wheel_speeds) / 10.0, -1.0, 1.0)

        # ── Step MuJoCo ───────────────────────────────────────────────────────
        obs = self.sim.step(action)

        # ── Publish joint_states ──────────────────────────────────────────────
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = ARM_JOINTS + WHEEL_JOINTS
        js.position = list(obs["arm_positions"]) + [0.0, 0.0, 0.0]  # wheels are continuous
        js.velocity = [0.0] * 6 + list(obs["wheel_velocities"])
        js.effort   = [0.0] * 9
        self.joint_state_pub.publish(js)

        # ── Publish odometry ──────────────────────────────────────────────────
        # Integrate base pose from sim
        base_pos = obs["base_position"]
        base_lin = obs["base_linear_velocity"]
        base_ang = obs["base_angular_velocity"]

        # Update pose
        self._theta += base_ang[2] * dt
        self._x     += (base_lin[0] * np.cos(self._theta) - base_lin[1] * np.sin(self._theta)) * dt
        self._y     += (base_lin[0] * np.sin(self._theta) + base_lin[1] * np.cos(self._theta)) * dt

        odom = Odometry()
        odom.header.stamp        = now.to_msg()
        odom.header.frame_id     = "odom"
        odom.child_frame_id      = "base_link"
        odom.pose.pose.position.x    = self._x
        odom.pose.pose.position.y    = self._y
        odom.pose.pose.orientation.z = np.sin(self._theta / 2.0)
        odom.pose.pose.orientation.w = np.cos(self._theta / 2.0)
        odom.twist.twist.linear.x  = base_lin[0]
        odom.twist.twist.linear.y  = base_lin[1]
        odom.twist.twist.angular.z = base_ang[2]
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = LeKiWiBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
