#!/usr/bin/env python3
"""
ROS2 ↔ LeKiWi-MuJoCo Bridge Node

Bridges commands between ROS2 omni_controller and the LeKiWi MuJoCo simulator.
Handles bidirectional conversion between ROS2 topics and MuJoCo physics state.

ROS2 side:
  - Subscribe: /lekiwi/cmd_vel (Twist)              → base + arm
  - Publish:   /lekiwi/wheel_{i}/cmd_vel (Float64)  → wheel angular velocities
  - Publish:   /lekiwi/odom (Odometry)               → simulated odometry
  - Publish:   /lekiwi/joint_states (JointState)     → full joint state

MuJoCo side (LeKiWiSim):
  - action[0..5] = arm joint position targets (rad, clamped -3.14..3.14)
  - action[6..9] = wheel angular velocities (rad/s, clamped -5..5)
  - qpos[0:3]    = base position (free joint)
  - qpos[3:7]    = base quaternion
  - qvel[0:3]    = base linear velocity
  - qvel[3:6]    = base angular velocity
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import transforms3d
from threading import Thread, Event


# MuJoCo joint name → ROS2 wheel index mapping
MUJOCO_WHEEL_JOINTS = ["w1", "w2", "w3"]
# MuJoCo arm joints (6 DOF)
ARM_JOINTS = ["j0", "j1", "j2", "j3", "j4", "j5"]


class LeKiWiBridge(Node):
    def __init__(self, sim, rate=50.0):
        super().__init__('lekiwi_ros2_bridge')
        self.sim = sim
        self.dt = 1.0 / rate
        self.rate = rate

        # MuJoCo joint index lookups (from sim)
        self._jpos_idx = sim._jpos_idx
        self._jvel_idx = sim._jvel_idx

        # ROS2 parameters (match lekiwi_modular/omni_controller.py)
        self.wheel_radius = 0.05
        self.wheel_positions = [
            np.array([0.1732, 0.0, 0.0]),
            np.array([-0.0866, 0.15, 0.0]),
            np.array([-0.0866, -0.15, 0.0]),
        ]
        # Omni-wheel joint axes — from lekiwi.urdf (verified empirically)
        # wheel_0 (front-right, Revolute-64): [-0.866, 0, 0.5]
        # wheel_1 (back-left,   Revolute-62): [ 0.866, 0, 0.5]
        # wheel_2 (back,        Revolute-60): [ 0,     0, -1  ]
        self.joint_axes = [
            np.array([-0.866025, 0.0, 0.5]),
            np.array([ 0.866025, 0.0, 0.5]),
            np.array([ 0.0,      0.0, -1.0]),
        ]

        # --- ROS2 Publishers ---
        # Wheel velocity publishers (to feed omni_odometry)
        self.wheel_pubs = [
            self.create_publisher(Float64, f'/lekiwi/wheel_{i}/cmd_vel', 10)
            for i in range(3)
        ]
        # Odometry publisher
        self.odom_pub = self.create_publisher(Odometry, '/lekiwi/odom', 10)
        # Full joint state publisher
        self.joint_state_pub = self.create_publisher(
            JointState, '/lekiwi/joint_states', 10
        )

        # --- ROS2 Subscribers ---
        # Base velocity command (from teleop or Nav2)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/lekiwi/cmd_vel', self._on_cmd_vel, 10
        )

        # Arm joint position commands (radians, one topic per joint)
        self.arm_subs = []
        for i, joint_name in enumerate(ARM_JOINTS):
            sub = self.create_subscription(
                Float64,
                f'/lekiwi/arm_joint_{i}/cmd_pos',
                lambda msg, idx=i: self._on_arm_cmd(idx, msg),
                10
            )
            self.arm_subs.append(sub)

        # --- State ---
        self._cmd_vel = np.zeros(3)    # wheel angular speeds [w1, w2, w3] (rad/s)
        self._arm_targets = np.zeros(6) # arm joint positions (rad)
        self._running = Event()
        self._spin_thread: Thread = None

        self.get_logger().info(
            f"LeKiWi Bridge initialized (rate={rate}Hz, "
            f"wheels={MUJOCO_WHEEL_JOINTS}, arm={ARM_JOINTS})"
        )

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """Convert Twist (vx, vy, wz) → wheel angular velocities.
        
        Matches the kinematics from lekiwi_modular/omni_controller.py.
        """
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        wheel_speeds = []
        for i in range(3):
            wheel_vel = np.array([
                vx - wz * self.wheel_positions[i][1],
                vy + wz * self.wheel_positions[i][0],
                0.0
            ])
            angular_speed = np.dot(wheel_vel, self.joint_axes[i]) / self.wheel_radius
            wheel_speeds.append(float(angular_speed))

        self._cmd_vel = np.array(wheel_speeds)
        self.get_logger().debug(
            f"cmd_vel → wheel speeds: {wheel_speeds}", throttle_duration_sec=1.0
        )

    def _on_arm_cmd(self, idx: int, msg: Float64):
        """Receive arm joint position command (radians)."""
        self._arm_targets[idx] = float(msg.data)

    # ─── Apply commands to MuJoCo ────────────────────────────────────────────

    def _apply_to_sim(self):
        """Send current commands to MuJoCo simulator.
        
        LeKiWiSim action layout:
          action[0..5] = arm joint positions (rad, clamped ±3.14 internally)
          action[6..8] = wheel angular velocities (rad/s, clamped ±5 internally)
        """
        # Build 9-D action: [arm(6), wheel(3)]
        action = np.concatenate([self._arm_targets, self._cmd_vel])
        # step() clamps internally; pass raw values
        self.sim.step(action)

    # ─── Read state from MuJoCo ───────────────────────────────────────────────

    def _read_state(self) -> dict:
        """Read current state from MuJoCo simulator."""
        return {
            "base_pos":    self.sim.data.qpos[:3].copy(),
            "base_quat":   self.sim.data.qpos[3:7].copy(),
            "base_linvel": self.sim.data.qvel[:3].copy(),
            "base_angvel": self.sim.data.qvel[3:6].copy(),
            "arm_pos":     np.array([
                self.sim.data.qpos[self._jpos_idx[n]] for n in ARM_JOINTS
            ]),
            "arm_vel":     np.array([
                self.sim.data.qvel[self._jvel_idx[n]] for n in ARM_JOINTS
            ]),
            "wheel_pos":   np.array([
                self.sim.data.qpos[self._jpos_idx[n]] for n in MUJOCO_WHEEL_JOINTS
            ]),
            "wheel_vel":   np.array([
                self.sim.data.qvel[self._jvel_idx[n]] for n in MUJOCO_WHEEL_JOINTS
            ]),
        }

    # ─── Publish ROS2 topics ──────────────────────────────────────────────────

    def _publish_wheel_cmdvel(self, wheel_vels: np.ndarray):
        """Publish wheel angular velocities (for omni_odometry)."""
        for i, vel in enumerate(wheel_vels):
            msg = Float64()
            msg.data = float(vel)
            self.wheel_pubs[i].publish(msg)

    def _publish_odom(self, state: dict):
        """Publish odometry (Odometry message)."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(state['base_pos'][0])
        msg.pose.pose.position.y = float(state['base_pos'][1])
        msg.pose.pose.position.z = float(state['base_pos'][2])
        msg.pose.pose.orientation.x = float(state['base_quat'][0])
        msg.pose.pose.orientation.y = float(state['base_quat'][1])
        msg.pose.pose.orientation.z = float(state['base_quat'][2])
        msg.pose.pose.orientation.w = float(state['base_quat'][3])
        msg.twist.twist.linear.x = float(state['base_linvel'][0])
        msg.twist.twist.linear.y = float(state['base_linvel'][1])
        msg.twist.twist.linear.z = float(state['base_linvel'][2])
        msg.twist.twist.angular.x = float(state['base_angvel'][0])
        msg.twist.twist.angular.y = float(state['base_angvel'][1])
        msg.twist.twist.angular.z = float(state['base_angvel'][2])
        self.odom_pub.publish(msg)

    def _publish_joint_states(self, state: dict):
        """Publish full joint state (sensor_msgs/JointState)."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS) + list(MUJOCO_WHEEL_JOINTS)
        msg.position = list(state['arm_pos']) + list(state['wheel_pos'])
        msg.velocity = list(state['arm_vel']) + list(state['wheel_vel'])
        self.joint_state_pub.publish(msg)

    # ─── Control loop ─────────────────────────────────────────────────────────

    def start(self):
        """Start the bridge spin loop in a background thread."""
        self._running.set()
        self._spin_thread = Thread(target=self._run, daemon=True)
        self._spin_thread.start()
        self.get_logger().info("Bridge spin loop started")

    def stop(self):
        """Stop the bridge spin loop."""
        self._running.clear()
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)
        self.get_logger().info("Bridge spin loop stopped")

    def _run(self):
        """Main loop: apply commands → step sim → publish state."""
        publish_counter = 0
        publish_every = max(1, int(self.rate / 20))  # publish at ~20Hz

        while rclpy.ok() and self._running.is_set():
            loop_start = self.get_clock().now()

            # 1. Apply commands to MuJoCo
            self._apply_to_sim()

            # 2. Read state (state is available after step())
            state = self._read_state()

            # 3. Publish ROS2 topics (throttled to ~20Hz)
            publish_counter += 1
            if publish_counter >= publish_every:
                publish_counter = 0
                self._publish_wheel_cmdvel(state['wheel_vel'])
                self._publish_odom(state)
                self._publish_joint_states(state)

            # 4. Sleep to maintain rate
            elapsed = (self.get_clock().now() - loop_start).nanoseconds * 1e-9
            sleep_time = max(0.001, self.dt - elapsed)
            rclpy.sleep(sleep_time)
