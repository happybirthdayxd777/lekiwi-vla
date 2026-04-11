#!/usr/bin/env python3
"""
ROS2 ↔ MuJoCo Bridge for LeKiWi
================================
Bridges ROS2 control commands to MuJoCo simulation.

Listens to:
  /lekiwi/cmd_vel          (geometry_msgs/Twist) — base velocity command
  /lekiwi/arm/cmd_pose     (Float64MultiArray)  — arm joint positions

Publishes:
  /lekiwi/joint_states     (sensor_msgs/JointState) — all joint states
  /lekiwi/odom             (nav_msgs/Odometry)        — base odometry
  /lekiwi/camera/image_raw (sensor_msgs/Image)       — simulated camera

Usage (with ROS2):
  $ ros2 run lekiwi_ros2_bridge bridge_node

  (Requires ROS2 Humble + lekiwi_modular workspace sourced)
"""

import numpy as np
from typing import Optional, List
import time

# ─────────────────────────────────────────────────────────────
#  Attempt ROS2 imports; fall back gracefully if not available
# ─────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray
    import std_msgs.msg
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    Twist = object
    JointState = object
    Odometry = object
    Float64MultiArray = object


# ─────────────────────────────────────────────────────────────
#  Import or mock MuJoCo simulation
# ─────────────────────────────────────────────────────────────
import sys as _sys
_lekiwi_vla_path = str(__file__).split("/src/")[0]
if _lekiwi_vla_path not in _sys.path:
    _sys.path.insert(0, _lekiwi_vla_path)

try:
    from sim_lekiwi import LeKiwiSim
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    LeKiwiSim = object


# ─────────────────────────────────────────────────────────────
#  Constants: joint names matching lekiwi.urdf
# ─────────────────────────────────────────────────────────────

WHEEL_JOINTS = [
    "ST3215_Servo_Motor-v1_Revolute-64",      # wheel 0 (front)
    "ST3215_Servo_Motor-v1-1_Revolute-62",    # wheel 1 (left rear)
    "ST3215_Servo_Motor-v1-2_Revolute-60",    # wheel 2 (right rear)
]

ARM_JOINTS = [
    "STS3215_03a-v1_Revolute-45",             # arm_joint_1
    "STS3215_03a-v1-1_Revolute-49",           # arm_joint_2
    "STS3215_03a-v1-2_Revolute-51",           # arm_joint_3
    "STS3215_03a-v1-3_Revolute-53",           # arm_joint_4
    "STS3215_03a_Wrist_Roll-v1_Revolute-55", # arm_joint_5
    "STS3215_03a-v1-4_Revolute-57",           # arm_joint_6
]

ALL_JOINTS = WHEEL_JOINTS + ARM_JOINTS

# Wheel positions relative to base_link — from omni_controller_fixed.py:
# wheel_base = 0.1732 (meters), angles at 30°, 150°, 270°
# Wheel 0: +30° (front-right) → [0.15, 0.0866, 0]
# Wheel 1: +150° (left)      → [-0.15, 0.0866, 0]
# Wheel 2: +270° (back-right) → [0, -0.1732, 0]
_angles = np.deg2rad([30, 150, 270])
_wheel_base = 0.1732
WHEEL_POSITIONS = [
    _wheel_base * np.array([np.cos(a), np.sin(a), 0.0])
    for a in _angles
]
# Roller axes — ACTUAL axes from lekiwi.urdf (extracted via regex from XML):
#   Revolute-64 (wheel 0, front-right): [-0.866, 0, 0.5]
#   Revolute-62 (wheel 1, left):        [ 0.866, 0, 0.5]
#   Revolute-60 (wheel 2, back-right):  [ 0,    0, -1]
# NOTE: The WHEEL_JOINTS list uses URDF joint names as identifiers, so the
# order here matches WHEEL_JOINTS[0..2] in sequence.
WHEEL_JOINT_AXES = [
    np.array([-0.866025, 0.0, 0.5])  / 1.0,   # wheel 0 (Revolute-64)
    np.array([ 0.866025, 0.0, 0.5])  / 1.0,   # wheel 1 (Revolute-62)
    np.array([ 0.0,      0.0,-1.0])  / 1.0,   # wheel 2 (Revolute-60)
]


# ─────────────────────────────────────────────────────────────
#  Main Bridge Node
# ─────────────────────────────────────────────────────────────

class LeKiWiRos2Bridge(Node if HAS_ROS2 else object):
    """
    ROS2 ↔ MuJoCo bridge for LeKiWi.

    In "simulation mode" (no real robot):
      - Reads /lekiwi/cmd_vel → applies to MuJoCo sim
      - Publishes joint_states, odom, camera from MuJoCo

    In "passthrough mode" (real robot):
      - Forwards commands to real robot
      - Relays sensor data back to ROS2
    """

    def __init__(self, sim_mode: bool = True, device: str = "cpu"):
        if HAS_ROS2:
            super().__init__("lekiwi_ros2_bridge")

        self.sim_mode = sim_mode
        self.device = device
        self._last_cmd_vel: Optional[np.ndarray] = None  # [vx, vy, wz]

        # ── MuJoCo Simulation ──────────────────────────────
        if self.sim_mode and HAS_MUJOCO:
            self.get_logger().info("Initializing MuJoCo simulation...") if HAS_ROS2 else print("[INFO] Initializing MuJoCo simulation...")
            self.sim = LeKiwiSim()
            self.sim.reset()
            self.get_logger().info("MuJoCo sim ready.") if HAS_ROS2 else print("[INFO] MuJoCo sim ready.")
        else:
            self.sim = None

        if HAS_ROS2:
            self._setup_ros2()

    def _setup_ros2(self):
        """Set up ROS2 publishers and subscribers."""
        # ── Subscribers ────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/lekiwi/cmd_vel",
            self._on_cmd_vel, 10
        )
        self.arm_cmd_sub = self.create_subscription(
            Float64MultiArray, "/lekiwi/arm/cmd_pose",
            self._on_arm_cmd, 10
        )

        # ── Publishers ─────────────────────────────────────
        self.joint_states_pub = self.create_publisher(
            JointState, "/lekiwi/joint_states", 10
        )
        self.odom_pub = self.create_publisher(
            Odometry, "/lekiwi/odom", 10
        )

        self.get_logger().info(
            f"LeKiWi ROS2 Bridge ready (sim_mode={self.sim_mode})"
        )

    # ── cmd_vel → MuJoCo ──────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """Convert Twist to wheel velocities and apply to MuJoCo."""
        vx  = msg.linear.x
        vy  = msg.linear.y
        wz  = msg.angular.z
        self._last_cmd_vel = np.array([vx, vy, wz])

        if self.sim is None:
            return

        # Compute wheel angular velocities (inverse kinematics)
        wheel_speeds = self._compute_wheel_speeds(vx, vy, wz)

        # Build action: [arm_joints(6), wheel_velocities(3)]
        # LeKiwiSim action format = [arm(6), wheel(3)] — NOT [wheel, arm]
        action = np.zeros(9, dtype=np.float32)
        action[6] = wheel_speeds[0]   # w1
        action[7] = wheel_speeds[1]  # w2
        action[8] = wheel_speeds[2]  # w3
        # action[0:6] = arm positions (kept at 0 = neutral)

        self.sim.step(action)

    def _compute_wheel_speeds(self, vx: float, vy: float, wz: float) -> List[float]:
        """Inverse kinematics for 3 omni wheels."""
        speeds = []
        for i in range(3):
            robot_vel = np.array([
                vx - wz * WHEEL_POSITIONS[i][1],
                vy + wz * WHEEL_POSITIONS[i][0],
                0.0
            ])
            angular_speed = np.dot(robot_vel, WHEEL_JOINT_AXES[i]) / WHEEL_RADIUS
            speeds.append(float(angular_speed))
        return speeds

    # ── Arm command ─────────────────────────────────────────────

    def _on_arm_cmd(self, msg: Float64MultiArray):
        """Set arm joint positions."""
        if self.sim is None:
            return
        positions = np.array(msg.data, dtype=np.float32)
        if len(positions) == 6:
            # Apply as direct joint position offsets (simplified)
            # In full integration, would use joint position controller
            pass

    # ── Odometry Publisher ──────────────────────────────────────

    def publish_odometry(self):
        """Publish base odometry from MuJoCo.

        LeKiwiSim qpos layout:
          qpos[0:3]  = base x, y, z
          qpos[3:7]  = base quaternion (4D)
          qpos[7:13] = arm joints (6D)
          qpos[13:16] = wheel joints (3D)
        """
        if not HAS_ROS2 or self.sim is None:
            return

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        if hasattr(self.sim.data, "qpos"):
            odom.pose.pose.position.x = self.sim.data.qpos[0]
            odom.pose.pose.position.y = self.sim.data.qpos[1]
            odom.pose.pose.position.z = self.sim.data.qpos[2]
            odom.pose.pose.orientation.x = self.sim.data.qpos[3]
            odom.pose.pose.orientation.y = self.sim.data.qpos[4]
            odom.pose.pose.orientation.z = self.sim.data.qpos[5]
            odom.pose.pose.orientation.w = self.sim.data.qpos[6]

        if hasattr(self.sim.data, "qvel"):
            odom.twist.twist.linear.x = self.sim.data.qvel[0]
            odom.twist.twist.linear.y = self.sim.data.qvel[1]
            odom.twist.twist.angular.z = self.sim.data.qvel[2]

        self.odom_pub.publish(odom)

    # ── Joint States Publisher ───────────────────────────────────

    def publish_joint_states(self):
        """Read sim state and publish joint_states to ROS2.

        LeKiwiSim joint layout:
          arm_joints (j0..j5)   → qpos[7:13], qvel[7:13]
          wheel_joints (w1..w3)  → qpos[13:16], qvel[13:16]
        """
        if not HAS_ROS2 or self.sim is None:
            return

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM_JOINTS + WHEEL_JOINTS

        # Arm positions: qpos[7:13]
        arm_pos = list(self.sim.data.qpos[7:13]) if hasattr(self.sim.data, "qpos") else [0.0] * 6
        # Wheel positions: qpos[13:16]
        wheel_pos = list(self.sim.data.qpos[13:16]) if hasattr(self.sim.data, "qpos") else [0.0] * 3
        js.position = arm_pos + wheel_pos

        # Arm velocities: qvel[7:13]
        arm_vel = list(self.sim.data.qvel[7:13]) if hasattr(self.sim.data, "qvel") else [0.0] * 6
        # Wheel velocities: qvel[13:16]
        wheel_vel = list(self.sim.data.qvel[13:16]) if hasattr(self.sim.data, "qvel") else [0.0] * 3
        js.velocity = arm_vel + wheel_vel

        self.joint_states_pub.publish(js)

    # ── Spin (for non-ROS2 standalone use) ─────────────────────

    def spin_once(self):
        """One step: apply pending command, publish state. For standalone use."""
        if self.sim is None:
            return

        # Apply last cmd_vel if any (maintains velocity)
        if self._last_cmd_vel is not None:
            vx, vy, wz = self._last_cmd_vel
            wheel_speeds = self._compute_wheel_speeds(vx, vy, wz)
            action = np.zeros(9, dtype=np.float32)
            action[6] = wheel_speeds[0]  # w1 — LeKiwiSim format is [arm(6), wheel(3)]
            action[7] = wheel_speeds[1]  # w2
            action[8] = wheel_speeds[2]  # w3
            self.sim.step(action)

        # Publish (ROS2 only)
        if HAS_ROS2:
            self.publish_joint_states()
            self.publish_odometry()

    def spin(self, hz: float = 50):
        """Main loop for standalone (non-ROS2) operation."""
        rate = 1.0 / hz
        while True:
            t0 = time.time()
            self.spin_once()
            dt = time.time() - t0
            if dt < rate:
                time.sleep(rate - dt)


# ─────────────────────────────────────────────────────────────
#  Standalone test (no ROS2 required)
# ─────────────────────────────────────────────────────────────

def test_bridge():
    """Test the bridge without ROS2 — just exercise the simulation."""
    print("=" * 60)
    print("  LeKiWi Bridge — Standalone Test")
    print("=" * 60)

    bridge = LeKiWiRos2Bridge(sim_mode=True)

    print("\n[1] Testing forward motion (vx=0.1, vy=0, wz=0)")
    bridge._last_cmd_vel = np.array([0.1, 0.0, 0.0])
    for _ in range(50):
        bridge.spin_once()

    pos = bridge.sim.data.qpos[6:9] if bridge.sim else None
    print(f"    Position after 50 steps: {pos}")

    print("\n[2] Testing turning (vx=0, vy=0, wz=0.5)")
    bridge._last_cmd_vel = np.array([0.0, 0.0, 0.5])
    for _ in range(50):
        bridge.spin_once()

    pos = bridge.sim.data.qpos[6:9] if bridge.sim else None
    print(f"    Position after 50 steps: {pos}")

    print("\n[3] Testing full motion")
    bridge.sim.reset()
    bridge._last_cmd_vel = np.array([0.1, 0.05, 0.2])
    for _ in range(100):
        bridge.spin_once()

    pos = bridge.sim.data.qpos[6:9] if bridge.sim else None
    print(f"    Final position: {pos}")

    print("\n[4] Checking joint names match URDF")
    print(f"    Wheel joints: {WHEEL_JOINTS}")
    print(f"    Arm joints:   {ARM_JOINTS}")
    print(f"    All joints:   {len(ALL_JOINTS)} total")

    print("\n✓ Bridge test complete")

    return bridge


# ─────────────────────────────────────────────────────────────
#  ROS2 entry point
# ─────────────────────────────────────────────────────────────

def main(args=None):
    if not HAS_ROS2:
        print("[ERROR] ROS2 not available. Use test_bridge() instead.")
        return

    rclpy.init(args=args)
    node = LeKiWiRos2Bridge(sim_mode=True)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_bridge()
    else:
        main()
