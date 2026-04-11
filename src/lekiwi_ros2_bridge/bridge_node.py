#!/usr/bin/env python3
"""
LeKiWi ROS2 ↔ MuJoCo Bridge Node
================================
Bridges ROS2 cmd_vel → MuJoCo sim, and MuJoCo sensor data → ROS2 joint_states.

Supports two simulation backends:
  - LeKiWiSim      (sim_lekiwi.py)   — cylinder primitives, fast & stable
  - LeKiWiSimURDF  (sim_lekiwi_urdf.py) — real STL meshes from lekiwi_modular

Topics:
  Input  : /lekiwi/cmd_vel        (geometry_msgs/Twist)
  Input  : /lekiwi/vla_action     (Float64MultiArray, arm*6 + wheel*3, native units)
  Output : /lekiwi/joint_states   (sensor_msgs/JointState)
  Output : /lekiwi/camera/image_raw (Image, 20 Hz, URDF model only)
  Output : /lekiwi/wheel_N/cmd_vel (Float64, mirrors real robot)

Architecture:
  ROS2 /lekiwi/cmd_vel
        ↓ (Twist → [vx, vy, wz])
  BridgeNode._on_cmd_vel()
        ↓
  BridgeNode._on_vla_action()     ← NEW: arm override from VLA policy
        ↓
  LeKiWiSim.step(action=[arm*6, wheel_speeds*3])
        ↓
  BridgeNode._publish_joint_states()
        ↓ (JointState ← MuJoCo obs)
  ROS2 /lekiwi/joint_states  →  LeKiWiVLAPolicyNode → /lekiwi/vla_action  (closed loop)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import sys
import os

# ── Simulation backend imports ─────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi import LeKiWiSim
from sim_lekiwi_urdf import LeKiWiSimURDF
from security_monitor import SecurityMonitor
from policy_guardian import PolicyGuardian

# ── Joint name mapping: URDF Gazebo names → bridge canonical names ─────────────
# From lekiwi.urdf Gazebo plugin joint list:
#   wheel_0 → ST3215_Servo_Motor-v1_Revolute-64
#   wheel_1 → ST3215_Servo_Motor-v1-1_Revolute-62
#   wheel_2 → ST3215_Servo_Motor-v1-2_Revolute-60
# Arm joints (from URDF Revolute joints):
#   arm_j0 → ST3215_Servo_Motor-v1-1_Revolute-49  (first arm joint)
#   arm_j1 → ST3215_Servo_Motor-v1-2_Revolute-51
#   arm_j2 → ST3215_Servo_Motor-v1-3_Revolute-53
#   arm_j3 → STS3215_03a_Wrist_Roll-v1_Revolute-55
#   arm_j4 → STS3215_03a-v1-4_Revolute-57
#   arm_j5 → (gripper slide — not in URDF Gazebo plugin)
URDF_WHEEL_JOINT_NAMES = [
    "ST3215_Servo_Motor-v1_Revolute-64",       # wheel_0 → w1 in bridge
    "ST3215_Servo_Motor-v1-1_Revolute-62",     # wheel_1 → w2 in bridge
    "ST3215_Servo_Motor-v1-2_Revolute-60",     # wheel_2 → w3 in bridge
]
URDF_ARM_JOINT_NAMES = [
    "STS3215_03a-v1_Revolute-45",             # arm_j0 — shoulder pan (axis≈Z, range ±1.57)
    "STS3215_03a-v1-1_Revolute-49",           # arm_j1 — shoulder lift (axis=[1,0,0], range -3.14..0)
    "STS3215_03a-v1-2_Revolute-51",           # arm_j2 — elbow (axis=[1,0,0], range 0..3.14)
    "STS3215_03a-v1-3_Revolute-53",           # arm_j3 — wrist pitch (axis=[1,0,0], range 0..3.14)
    "STS3215_03a_Wrist_Roll-v1_Revolute-55", # arm_j4 — wrist roll (axis=[0,0.423,-0.906])
    "STS3215_03a-v1-4_Revolute-57",          # arm_j5 — gripper slide (axis=[0,-0.906,-0.423], range ±1.57)
]


# ── Kinematics constants (from omni_controller.py) ──────────────────────────────
WHEEL_RADIUS   = 0.05    # m
WHEEL_POSITIONS = np.array([
    [ 0.1732,  0.0,    0.0 ],   # wheel 0 — front
    [-0.0866,  0.15,   0.0 ],   # wheel 1 — back-left
    [-0.0866, -0.15,   0.0 ],   # wheel 2 — back-right
], dtype=np.float64)

# Roller axes from URDF (how each wheel contacts the ground)
# Wheel 0: pure rotation around Z; Wheel 1/2: 30° offset + 26.6° roller tilt
# (Corrected from omni_controller_fixed.py analysis)
_JOINT_AXES = np.array([
    [0.0,        0.0, -1.0      ],   # wheel 0 — front
    [0.866025,   0.0,  0.5      ],   # wheel 1 — back-left
    [-0.866025,  0.0,  0.5      ],   # wheel 2 — back-right
], dtype=np.float64)


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

    def __init__(self, sim_type: str = "primitive"):
        """
        Parameters
        ----------
        sim_type : str
            "primitive" → LeKiWiSim (fast cylinders, stable)
            "urdf"      → LeKiWiSimURDF (STL mesh geometry)
        """
        super().__init__("lekiwi_ros2_bridge")

        # Declare and retrieve sim_type parameter (set via launch file)
        self.declare_parameter("sim_type", "primitive")
        p = self.get_parameter("sim_type")
        sim_type = str(p.value) if p.value else "primitive"

        # ── Initialise MuJoCo simulation ────────────────────────────────────
        if sim_type == "urdf":
            self.get_logger().info("Starting LeKiWiSimURDF (STL mesh geometry)…")
            self.sim: LeKiWiSim | LeKiWiSimURDF = LeKiWiSimURDF()
            self.get_logger().info("URDF simulation initialised.")
        else:
            self.get_logger().info("Starting LeKiWiSim (cylinder primitives)…")
            self.sim = LeKiWiSim()
            self.get_logger().info("Primitive simulation initialised.")

        # CTF security monitor
        self.security_monitor = SecurityMonitor()
        self.get_logger().info("SecurityMonitor active (CTF mode).")

        # Active policy guardian — blocks unknown policy fingerprints
        self.policy_guardian = PolicyGuardian()
        self.get_logger().info("PolicyGuardian active (Challenge 7 defense).")

        # ── ROS2 QoS profile ──────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Security alert publisher
        from std_msgs.msg import String
        self.alert_pub = self.create_publisher(String, "/lekiwi/security_alert", qos)
        self.joint_state_pub = self.create_publisher(JointState, "/lekiwi/joint_states", qos)

        # Odometry publisher (mirrors lekiwi_modular omni_odometry.py)
        self.odom_pub = self.create_publisher(Odometry, "/lekiwi/odom", qos)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # URDF-compatible joint states publisher (real URDF joint names)
        self.joint_state_urdf_pub = self.create_publisher(
            JointState, "/lekiwi/joint_states_urdf", qos
        )

        # Camera bridge: MuJoCo → ROS2 image
        self.camera_pub  = self.create_publisher(Image, "/lekiwi/camera/image_raw",    qos)
        self.wrist_cam_pub = self.create_publisher(Image, "/lekiwi/wrist_camera/image_raw", qos)
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

        # VLA action subscriber: arm (6) + wheel (3) in native units from policy node
        from std_msgs.msg import Float64MultiArray
        self.vla_action_sub = self.create_subscription(
            Float64MultiArray, "/lekiwi/vla_action", self._on_vla_action, qos
        )

        # CTF Challenge 7: policy injection topic
        from std_msgs.msg import ByteMultiArray
        self.policy_sub = self.create_subscription(
            ByteMultiArray, "/lekiwi/policy_input", self._on_policy_input, qos
        )
        self._blocked_count = 0

        # ── Timer: step MuJoCo & publish at 20 Hz (camera is expensive) ────────
        self.timer = self.create_timer(0.05, self._on_timer)   # 20 Hz

        # ── State ───────────────────────────────────────────────────────────────
        self._last_action = np.zeros(9, dtype=np.float64)   # [arm*6, wheel*3]
        self._vla_action_fresh = False                       # set True when VLA writes action
        self._frame_count = 0
        # Odometry state (mirrors omni_odometry.py)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._last_odom_time = self.get_clock().now()
        self.get_logger().info(
            "LeKiWi ROS2 bridge ready.  Topics:\n"
            "  /lekiwi/cmd_vel       ← subscribe\n"
            "  /lekiwi/vla_action    ← subscribe (VLA arm override)\n"
            "  /lekiwi/joint_states  → publish\n"
            "  /lekiwi/odom          → publish (20 Hz)\n"
            "  /lekiwi/camera/image_raw → publish (20 Hz)\n"
            "  /lekiwi/wheel_N/cmd_vel → publish"
        )

    # ── cmd_vel callback ────────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """
        Convert Twist → wheel speeds, combine with current arm action, step sim.

        When VLA is active (_vla_action_fresh=True), arm portion from _last_action
        is preserved and only the wheel portion is overridden by cmd_vel.
        When VLA is not active, arms stay at their last commanded position.

        Note: _vla_action_fresh is set True by _on_vla_action; the timer
        callback clears it at the end of each tick so stale VLA actions
        (e.g., if the VLA node dies) are automatically ignored.
        """
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        wz = float(msg.angular.z)

        # CTF security check -- drop anomalous commands
        stamp = self.get_clock().now().nanoseconds / 1e9
        verdict = self.security_monitor.check_cmd_vel(vx, vy, wz, stamp)
        if verdict.blocked:
            self._blocked_count += 1
            self.get_logger().warn(
                "Blocked cmd_vel #{} {} severity={} vx={:.3f} vy={:.3f} wz={:.3f}".format(
                    self._blocked_count, verdict.event_type, verdict.severity, vx, vy, wz),
                throttle_duration_sec=2.0)
            return

        # Compute wheel angular velocities from kinematics
        wheel_speeds = twist_to_wheel_speeds(vx, vy, wz)

        # If VLA has set an arm action, keep arms; otherwise keep last arm pos.
        # _vla_action_fresh is cleared after each timer step so we know whether
        # to trust the arm portion of _last_action.
        if self._vla_action_fresh:
            arm_action = self._last_action[0:6]
        else:
            # No VLA action yet: hold arms at last commanded position
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

    # ── VLA action callback ────────────────────────────────────────────────────

    def _on_vla_action(self, msg):
        """
        Receive VLA policy action (arm*6 + wheel*3 in native units).
        Arms override whatever _on_cmd_vel set for the arm portion.
        VLA action always wins for arm joints to enable closed-loop control.
        """
        try:
            action = np.array(msg.data, dtype=np.float64)
            if action.shape != (9,):
                self.get_logger().warn(f"VLA action shape {action.shape} != (9,), ignoring", once=True)
                return
            # VLA action = native units; clamp to safe limits
            action[:6] = np.clip(action[:6], self.ARM_CTRL_MIN,  self.ARM_CTRL_MAX)
            action[6:] = np.clip(action[6:], self.WHEEL_CTRL_MIN, self.WHEEL_CTRL_MAX)
            self._last_action = action
            self._vla_action_fresh = True
        except Exception as e:
            self.get_logger().warn(f"VLA action parse error: {e}", once=True)

    # ── Timer callback ─────────────────────────────────────────────────────────

    def _on_timer(self):
        """Publish MuJoCo state as JointState + Odometry + camera Image."""
        now = self.get_clock().now()
        obs = self.sim._obs()
        dt = 0.05   # matches timer period

        # ── Odometry ─────────────────────────────────────────────────────────
        wheel_vel = obs.get("wheel_velocities", np.zeros(3))
        # Compute vx, vy, wz from wheel velocities using same kinematics as omni_odometry.py
        vx_total = 0.0
        vy_total = 0.0
        wz_total = 0.0
        wheel_base = 0.1732
        for i in range(3):
            angular_speed = wheel_vel[i]
            wheel_vel_world = angular_speed * WHEEL_RADIUS * _JOINT_AXES[i]
            vx_total += wheel_vel_world[0] / 3.0
            vy_total += wheel_vel_world[1] / 3.0
            wz_total += np.cross(WHEEL_POSITIONS[i], wheel_vel_world)[2] / (3.0 * wheel_base)

        # Integrate position
        self._odom_theta += wz_total * dt
        self._odom_x += (vx_total * np.cos(self._odom_theta) - vy_total * np.sin(self._odom_theta)) * dt
        self._odom_y += (vx_total * np.sin(self._odom_theta) + vy_total * np.cos(self._odom_theta)) * dt

        # Publish /lekiwi/odom
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = self._odom_x
        odom_msg.pose.pose.position.y = self._odom_y
        odom_msg.pose.pose.orientation.z = np.sin(self._odom_theta / 2.0)
        odom_msg.pose.pose.orientation.w = np.cos(self._odom_theta / 2.0)
        odom_msg.twist.twist.linear.x = vx_total
        odom_msg.twist.twist.linear.y = vy_total
        odom_msg.twist.twist.angular.z = wz_total
        self.odom_pub.publish(odom_msg)

        # Publish TF odom → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now.to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = self._odom_x
        tf_msg.transform.translation.y = self._odom_y
        tf_msg.transform.rotation.z = np.sin(self._odom_theta / 2.0)
        tf_msg.transform.rotation.w = np.cos(self._odom_theta / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)

        # ── JointState ──────────────────────────────────────────────────────
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = self.ARM_NAMES + self.WHEEL_NAMES

        # Arm positions (6) + integrate wheel velocities → positions (3)
        arm_pos = list(obs.get("arm_positions", np.zeros(6)))
        # Accumulate wheel positions from velocity integration
        self._wheel_posAccum = getattr(self, '_wheel_posAccum', np.zeros(3)) + wheel_vel * dt
        wheel_pos = list(self._wheel_posAccum)

        msg.position = arm_pos + wheel_pos
        msg.velocity = list(obs.get("arm_velocities", np.zeros(6))) + list(wheel_vel)

        self.joint_state_pub.publish(msg)

        # ── URDF-compatible JointState (real joint names from lekiwi.urdf) ────
        # Maps bridge canonical names → URDF Gazebo plugin joint names
        urdf_msg = JointState()
        urdf_msg.header.stamp = now.to_msg()
        # Arm: bridge arm_names → URDF arm joint names (all 6, including gripper j5)
        urdf_arm_names = URDF_ARM_JOINT_NAMES   # 6 joints
        urdf_wheel_names = URDF_WHEEL_JOINT_NAMES
        urdf_msg.name = urdf_arm_names + urdf_wheel_names
        # arm_positions[0:6] includes j5/gripper (slide joint)
        urdf_msg.position = list(obs.get("arm_positions", np.zeros(6))) + wheel_pos
        urdf_msg.velocity = list(obs.get("arm_velocities", np.zeros(6))) + list(wheel_vel)
        self.joint_state_urdf_pub.publish(urdf_msg)

        # ── Camera Images ────────────────────────────────────────────────────
        self._frame_count += 1
        if self._frame_count % 1 == 0:   # publish every frame (20 Hz)
            try:
                # Front camera
                img_pil = self.sim.render(640, 480)
                img_np  = np.asarray(img_pil)
                ros_img = self.bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
                ros_img.header.stamp = msg.header.stamp
                ros_img.header.frame_id = "lekiwi_camera"
                self.camera_pub.publish(ros_img)

                # Wrist camera (URDF model only)
                if hasattr(self.sim, 'render_wrist'):
                    wrist_pil = self.sim.render_wrist()
                    wrist_np  = np.asarray(wrist_pil)
                    wrist_ros = self.bridge.cv2_to_imgmsg(wrist_np, encoding="rgb8")
                    wrist_ros.header.stamp = msg.header.stamp
                    wrist_ros.header.frame_id = "wrist_camera"
                    self.wrist_cam_pub.publish(wrist_ros)
            except Exception as e:
                self.get_logger().warn(f"Camera render failed: {e}", once=True)

        # Clear VLA freshness flag at end of each tick so stale VLA actions
        # (e.g., if the VLA node crashes) are automatically ignored.
        self._vla_action_fresh = False

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def apply_arm_action(self, arm_pos: np.ndarray):
        """
        Directly set arm joint targets.
        arm_pos: shape (6,) in radians.
        Used by external VLA policy nodes.
        """
        arm_pos = np.clip(arm_pos, self.ARM_CTRL_MIN, self.ARM_CTRL_MAX)
        self._last_action[0:6] = arm_pos


    # ── Policy input callback (CTF Challenge 7) ───────────────────────────
    def _on_policy_input(self, msg):
        from std_msgs.msg import ByteMultiArray
        policy_bytes = bytes(msg.data)
        stamp = self.get_clock().now().nanoseconds / 1e9

        # ── Layer 1: Passive SecurityMonitor (log-only, backward compat) ────
        sec_verdict = self.security_monitor.check_policy(policy_bytes, stamp)

        # ── Layer 2: Active PolicyGuardian (blocks, alerts, rolls back) ────
        guardian_verdict = self.policy_guardian.check_and_guard(policy_bytes, stamp)

        # Publish security alert to /lekiwi/security_alert
        alert_msg = String()
        if guardian_verdict.action in ("block", "rollback"):
            import json
            alert_payload = {
                "type": guardian_verdict.reason,
                "severity": guardian_verdict.severity,
                "fingerprint": guardian_verdict.details.get("fingerprint", "?"),
                "ctf_flag": guardian_verdict.ctf_flag,
                "action": guardian_verdict.action,
            }
            alert_msg.data = json.dumps(alert_payload)
            self.alert_pub.publish(alert_msg)
            self.get_logger().error(
                "⚔️ POLICY BLOCKED [%s] %s  fingerprint=%s  flag=%s".format(
                    guardian_verdict.severity.upper(),
                    guardian_verdict.reason,
                    guardian_verdict.details.get("fingerprint", "?"),
                    guardian_verdict.ctf_flag or ""))
        else:
            # Silent allow — publish a heartbeat so monitoring can see it
            alert_payload = {
                "type": "policy_allowed",
                "severity": "low",
                "fingerprint": guardian_verdict.details.get("fingerprint", "?"),
            }
            alert_msg.data = json.dumps(alert_payload)
            self.alert_pub.publish(alert_msg)

        # Flush both monitors
        self.security_monitor.flush()
        self.policy_guardian.flush()


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
