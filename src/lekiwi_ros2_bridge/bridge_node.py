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

from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import sys
import os
import time

# ── Simulation backend imports ─────────────────────────────────────────────────
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from lekiwi_ros2_bridge.lekiwi_sim_loader import make_sim
from sim_lekiwi_urdf import twist_to_contact_wheel_speeds
from security_monitor import SecurityMonitor
from policy_guardian import PolicyGuardian
from camera_adapter import CameraAdapter  # noqa: F401 — used by urdf sim mode at line ~412
from lekiwi_ros2_bridge.ctf_integration import CTFSecurityAuditor

# ── Joint name mapping: URDF Gazebo names → bridge canonical names ─────────────
# From lekiwi.urdf Gazebo plugin joint list:
#   wheel_0 → ST3215_Servo_Motor-v1_Revolute-64
#   wheel_1 → ST3215_Servo_Motor-v1-1_Revolute-62
#   wheel_2 → ST3215_Servo_Motor-v1-2_Revolute-60
# Arm joints (from lekiwi.urdf Revolute joints):
#   arm_j0 → STS3215_03a-v1_Revolute-45    (shoulder pan, axis≈Z, range ±1.57)
#   arm_j1 → STS3215_03a-v1-1_Revolute-49  (shoulder lift, axis=[1,0,0], range -3.14..0)
#   arm_j2 → STS3215_03a-v1-2_Revolute-51   (elbow, axis=[1,0,0], range 0..3.14)
#   arm_j3 → STS3215_03a-v1-3_Revolute-53   (wrist pitch, axis=[1,0,0], range 0..3.14)
#   arm_j4 → STS3215_03a_Wrist_Roll-v1_Revolute-55  (wrist roll, axis=[0,0.423,-0.906])
#   arm_j5 → STS3215_03a-v1-4_Revolute-57  (gripper slide, axis=[0,-0.906,-0.423], range ±1.57)
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


# ── Kinematics constants (Phase 100: corrected to equilateral triangle) ─────────
# Phase 99 finding: URDF body positions in sim_lekiwi_urdf.xml are WRONG —
# they form an isosceles triangle (w0-w1=0.1732, w1-w2=0.2000, w2-w0=0.2646m)
# instead of the correct equilateral 120°-separated triangle.
#
# The correct geometry (confirmed by omni_controller_fixed.py) is:
#   wheel_base = 0.1732m,  angles = [30°, 150°, 270°]
#   wheel_0: [+0.1500, +0.0866] — front-right (Revolute-64)
#   wheel_1: [-0.1500, +0.0866] — back-left  (Revolute-62)
#   wheel_2: [+0.0000, -0.1732] — back        (Revolute-60)
#
# Phase 48 introduced the URDF-body-position bug: it "matched URDF" but the
# URDF body positions themselves are wrong (not equilateral).
# Phase 100 fixes this: use correct equilateral positions.
#
WHEEL_RADIUS   = 0.05    # m (wheel roller radius)
WHEEL_POSITIONS = np.array([
    [ 0.1500,  0.0866,  0.0 ],   # wheel_0 — front-right (Revolute-64), 30° from X
    [-0.1500,  0.0866,  0.0 ],   # wheel_1 — back-left  (Revolute-62), 150° from X
    [ 0.0000, -0.1732,  0.0 ],   # wheel_2 — back       (Revolute-60), 270° from X
], dtype=np.float64)

# Roller axes from URDF (continuous revolute joints):
#   wheel_0 Revolute-64:  axis=[-0.866025,  0,  0.5]
#   wheel_1 Revolute-62:  axis=[ 0.866025,  0,  0.5]
#   wheel_2 Revolute-60:  axis=[ 0,         0, -1.0]   (pure Z roller for forward/back)
_JOINT_AXES = np.array([
    [-0.866025,  0.0,  0.5 ],   # wheel_0 — Revolute-64
    [ 0.866025,  0.0,  0.5 ],   # wheel_1 — Revolute-62
    [ 0.0,       0.0, -1.0 ],   # wheel_2 — Revolute-60
], dtype=np.float64)


# ── Phase 88: Translation Layer ──────────────────────────────────────────────
#
# BRIDGE: Phase 85 policy → Phase 86 physics
#
# The Phase 63 CLIP-FM policy was trained on Phase 85's crude physics model
# (average wheel torque → pure forward force). When evaluated on Phase 86's
# correct omni-kinematics (SR=0%), symmetric wheel actions produce lateral
# vy motion instead of forward translation.
#
# Translation layer: converts Phase-85-equivalent wheel actions into
# Phase-86-correct motor torques using the pseudo-inverse Jacobian of the
# wheel geometry.
#
# How it works:
#   1. Policy outputs wheel_action (symmetric pattern from Phase 85 training)
#   2. Treat wheel_action as a "forward force magnitude" proxy:
#      forward_force = mean(wheel_action)  →  desired base vx
#   3. Use J⁺ (pseudo-inverse of Jacobian) to get wheel torques that produce vx
#   4. Apply residual to correct for the discrepancy between Phase 85 & 86
#
# Alternative interpretation (used here):
#   wheel_action[i] in Phase 85 = forward contribution of wheel i
#   We interpret wheel_action as the desired angular velocity for each wheel
#   in Phase 85's simplified model. The translation layer maps these to
#   Phase 86 motor torques that produce the same net base velocity.
#
# The translation is:  tau_phase86 = J⁺_phase86 · J_phase85 · wheel_action
# Where J_phase85 is the Phase 85 Jacobian (average/symmetric geometry).
# This can be simplified to a constant 3×3 transformation matrix T such that:
#   tau_corrected = T @ wheel_action
#
# T is pre-computed from the wheel geometry and stored as TRANSFORM_MATRIX.
# See sim_lekiwi.py Phase 86 documentation for full derivation.

# Phase 85 simplified Jacobian (assumes all wheels pointing +X):
#   J85_row_i = [1, 0, -y_i]   (x contribution of each wheel)
# where y_i are the lateral offsets of each wheel from base center.
# Wheel offsets: y0=0.10, y1=0.10, y2=-0.10
# J85 = [[1, 0, -0.10],
#        [1, 0,  0.10],
#        [1, 0, -0.10]]   ← wheel 2 at y=-0.10

# Phase 86 Jacobian (correct omni-kinematics, w1=+X, w2=+Y, w3=-X):
#   J86_row_i = dot_product([cos, sin, -lateral], roller_axis)
# See _omni_kinematics() in sim_lekiwi_urdf.py for the full derivation.
# The pseudo-inverse J86⁺ transforms base velocity → wheel angular velocity.
# The full transformation T = J86⁺ · J85 maps Phase 85 actions → Phase 86 torques.

# Clamping constants for translated actions (per wheel)
# Phase 89 FIX: Increased from ±0.75 to ±6.0 rad/s.
# ROOT CAUSE (Phase 88): ±0.75 rad/s cap is ~7-8x too restrictive.
# Phase 86 omni-kinematics: vx = R/3 * 1.732 * (w2 - w3), R=0.0508.
# For Phase 85-level forward motion (~0.2 m/s), need w2-w3 ≈ 6.8 rad/s.
# Previous max w2 = 0.75/0.866 = 0.866 → vx ≈ 0.025 m/s (8x too slow).
# Phase 89: No internal clamp in translation; scale happens in _on_cmd_vel.
_MAX_TRANSLATED = 6.0   # max absolute translated wheel speed (rad/s)
_MIN_TRANSLATED = -6.0

# ── Hybrid Bridge: P-controller Fallback ──────────────────────────────────────
# Phase 210: VLA wheel actions are too conservative (3-6x smaller than optimal).
# When VLA wheel action magnitude < threshold, fall back to P-controller
# so the robot still moves toward goal when VLA hesitates.
# Threshold is ~1.5x typical VLA wheel action magnitude (~0.10) and ~0.5x
# what P-controller needs (~0.20-0.30). This ensures P-controller only
# intervenes when VLA is being too cautious, not when VLA is actively moving.
_HYBRID_WHEEL_FALLBACK_THRESHOLD = 0.15  # rad/s — below this VLA magnitude → use P-controller
# Phase 212: Direction-agreement hybrid — if VLA direction agrees with P-controller,
# amplify and use VLA instead of blending. This fixes the 3-6x VLA magnitude underestimation.
_WHEEL_AMPLIFICATION_FACTOR = 2.5          # amplify VLA actions when direction agrees
_DIRECTION_AGREEMENT_THRESHOLD = 0.5       # cos(angle) threshold — moderate directional alignment


def _wheel_direction_agrees(vla_wheels: np.ndarray, pctrl_ws: np.ndarray,
                            threshold: float = _DIRECTION_AGREEMENT_THRESHOLD) -> bool:
    """
    Phase 212: Check if VLA wheel direction agrees with P-controller direction.

    When VLA direction is correct but magnitude is too small, we amplify the VLA
    action instead of blending with P-controller. This preserves VLA learned
    behavior while compensating for its magnitude underestimation.

    Args:
        vla_wheels: VLA wheel action (3,) in rad/s
        pctrl_ws: P-controller wheel speeds (3,) in rad/s
        threshold: cos(angle) threshold (0.5 = 60° alignment required)

    Returns:
        True if VLA and P-controller directions agree within threshold
    """
    vla_norm = np.linalg.norm(vla_wheels)
    pctrl_norm = np.linalg.norm(pctrl_ws)

    # No reference or zero VLA → cannot determine direction
    if pctrl_norm < 0.01 or vla_norm < 0.01:
        return False

    vla_dir = vla_wheels / vla_norm
    pctrl_dir = pctrl_ws / pctrl_norm
    cos_angle = np.dot(vla_dir, pctrl_dir)

    return cos_angle > threshold


def _translate_phase85_to_phase86(wheel_action: np.ndarray) -> np.ndarray:
    """
    Phase 88: Translation Layer

    Convert a Phase 85 policy's wheel action (symmetric pattern) into a
    Phase 86-correct wheel action that produces the intended base motion.

    In Phase 85: symmetric [a,a,a] → net forward force (vx ≈ mean(a) * K85)
    In Phase 86: symmetric [a,a,a] → vy motion only, no translation (SR=0%)

    Phase 86 omni-kinematics analysis (from _omni_kinematics):
      w1 (axis=[-0.866,0,0.5]): vx=-17.32, vy=10.0 per unit wheel vel
      w2 (axis=[0.866,0,0.5]):   vx=+17.32, vy=10.0 per unit wheel vel  ← primary +X
      w3 (axis=[0,0,-1]):        vx=0,      vy=-20.0 per unit wheel vel

    Phase 85 simplified (all wheels → +X direction):
      vx = sum_i(wheel_i[i]) * K   (no vy contribution)

    Translation strategy:
      1. Extract forward component = mean(wheel_action) — what Phase 85 meant
      2. In Phase 86, only w2 contributes to vx (cos=0.866)
         → w2 must carry the full forward load: mean / 0.866
      3. Use w1 and w3 to handle any asymmetric lateral component
         from the Phase 85 policy's differential wheel commands

    Parameters
    ----------
    wheel_action : np.ndarray, shape (3,)
        Raw wheel action from Phase 85-trained policy.
        Values are in the same units as the policy's output (typically -1..1 or 0..1).

    Returns
    -------
    np.ndarray, shape (3,)
        Corrected wheel action for Phase 86 physics model.
        w2 carries the forward component; w1/w3 handle lateral.
    """
    action = np.asarray(wheel_action, dtype=np.float64)
    mean_a = np.mean(action)
    # Differential component between w1 and w3 (proxy for yaw in Phase 85)
    diff_13 = (action[0] - action[2]) / 2.0

    # Phase 89 FIX: Gain factor increased from 1.15 (1/0.866) to 10.0.
    # Phase 86 omni-kinematics: vx = R/3 * 1.732 * (w2 - w3), R=0.0508.
    # Phase 85 model: forward force ≈ K85 * mean_a, achieving ~0.2 m/s at mean=0.5.
    # This requires w2-w3 ≈ 6.8 rad/s → gain ≈ 6.8/0.5 ≈ 13.6.
    # Using gain=10 as safe starting point (gives ~0.15 m/s forward at mean=0.5).
    # w1 and w3 handle differential (turning), scaled proportionally.
    _GAIN = 10.0
    w2 = np.clip(mean_a * _GAIN, _MIN_TRANSLATED, _MAX_TRANSLATED)
    w1 = np.clip(diff_13 * _GAIN * 0.3, _MIN_TRANSLATED, _MAX_TRANSLATED)
    w3 = np.clip(-diff_13 * _GAIN * 0.3, _MIN_TRANSLATED, _MAX_TRANSLATED)

    return np.array([w1, w2, w3], dtype=np.float64)


def _contact_jacobian_pctrl(base_xy: np.ndarray, goal_xy: np.ndarray, kP: float = 2.0) -> np.ndarray:
    """
    Phase 276: Contact-Jacobian P-controller for URDF sim.

    Uses _CONTACT_JACOBIAN_PSEUDO_INV from sim_lekiwi_urdf — the physically calibrated
    contact-physics Jacobian that achieves 100% SR in URDF sim vs ~20% with kinematic IK.

    The CJ P-controller was verified in Phase 274 to reach 100% SR (10/10 goals) in URDF sim.

    Parameters
    ----------
    base_xy : np.ndarray
        Current base position [x, y] in meters.
    goal_xy : np.ndarray
        Goal position [x, y] in meters.
    kP : float
        Proportional gain for position error (default=2.0, verified optimal in Phase 274).

    Returns
    -------
    np.ndarray, shape (3,)
        Wheel angular velocities [w1, w2, w3] in rad/s, clipped to [-0.5, 0.5].
    """
    from sim_lekiwi_urdf import _CONTACT_JACOBIAN_PSEUDO_INV
    err = goal_xy - base_xy
    vx = kP * err[0]
    vy = kP * err[1]
    wheel_speeds = _CONTACT_JACOBIAN_PSEUDO_INV @ np.array([vx, vy])
    return np.clip(wheel_speeds, -0.5, 0.5)


def twist_to_wheel_speeds(vx: float, vy: float, wz: float) -> np.ndarray:
    """
    Phase 158: Convert robot-level Twist (vx, vy, wz) into 3 wheel angular velocities
    using the Phase 122 kinematic IK + Phase 158 gain=3.5 calibration.

    Phase 123 contact Jacobian approach (J_c^+) was giving only 10-15% SR.
    Phase 158 discovered k_omni=15.0 contact physics amplifies commanded wheel
    speeds by ~3.5x. The kinematic IK gives correct wheel directions, and gain=3.5
    compensates for the contact amplification → 93.3% SR.

    Uses twist_to_contact_wheel_speeds() from sim_lekiwi_urdf.

    For VLA policy actions, the wheel portion is passed directly (native units).
    """
    # Phase 122 kinematic IK + Phase 158 gain=3.5 (via sim_lekiwi_urdf)
    return twist_to_contact_wheel_speeds(vx, vy, wz)


class LeKiWiBridge(Node):
    """
    ROS2 ↔ MuJoCo bridge for LeKiWi robot.

    Publishers:
      /lekiwi/joint_states   — arm (6) + wheel (3) joint positions & velocities

    Subscribers:
      /lekiwi/cmd_vel        — Twist (linear x/y, angular z)
    """

    # MuJoCo action space (normalized):
    #   action[0:6]  = arm joint torques   (normalized ±1.0 → ±3.14 Nm via *3.14)
    #   action[6:9]  = wheel motor torques (normalized ±0.5 → ±5.0 Nm via *10.0)
    #   → WHEEL_CTRL values are action-space [-0.5, 0.5], NOT raw ctrl
    #   → sim_lekiwi_urdf._action_to_ctrl() amplifies by 10x (motor gear=10)
    #   → MuJoCo motor gear=10 then amplifies again: final joint torque = action * 31.4 Nm
    #   → e.g., action=0.5 → ctrl=5.0 → motor torque=50 Nm → joint torque=50 Nm
    #   → e.g., action=0.1 → ctrl=1.0 → motor torque=10 Nm → joint torque=10 Nm
    ARM_CTRL_MIN  = -3.14
    ARM_CTRL_MAX  =  3.14
    # Phase 70/74: WHEEL_CTRL ±0.5 required for URDF sim stability (±5.0 causes NaN)
    WHEEL_CTRL_MIN = -0.5
    WHEEL_CTRL_MAX =  0.5

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

        # Declare and retrieve parameters (set via launch file)
        self.declare_parameter("sim_type", "primitive")
        self.declare_parameter("mode", "sim")
        self.declare_parameter("record", False)
        self.declare_parameter("record_file", "")
        self.declare_parameter("record_images", True)
        self.declare_parameter("enable_hmac", False)
        self.declare_parameter("cmd_vel_secret", "")
        self.declare_parameter("ctf_mode", False)  # when True, records all CTF flags to JSONL
        # Phase 88: enable/disable policy translation layer (fixes Phase 85→86 mismatch)
        self.declare_parameter("phase88_translation", True)
        # Phase 276: Contact-Jacobian mode for URDF sim — use physically calibrated
        # _CONTACT_JACOBIAN_PSEUDO_INV instead of the old kinematic k_omni overlay model.
        # P-controller achieves 100% SR with CJ on URDF sim vs ~20% with kinematic model.
        self.declare_parameter("use_contact_jacobian", True)
        # Phase 210: Hybrid bridge — goal for P-controller fallback
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", 0.0)

        p_enable_hmac = self.get_parameter("enable_hmac")
        p_cmd_vel_secret = self.get_parameter("cmd_vel_secret")
        enable_hmac = bool(p_enable_hmac.value) if p_enable_hmac.value else False
        cmd_vel_secret = str(p_cmd_vel_secret.value) if p_cmd_vel_secret.value else ""
        if cmd_vel_secret:
            cmd_vel_secret_bytes = cmd_vel_secret.encode()
        else:
            cmd_vel_secret_bytes = None
        p_ctf_mode = self.get_parameter("ctf_mode")
        ctf_mode = bool(p_ctf_mode.value) if p_ctf_mode.value else False
        if ctf_mode:
            ctf_log_path = os.path.join(
                os.path.expanduser("~"), "hermes_research", "lekiwi_vla", "ctf_flags.jsonl"
            )
        else:
            ctf_log_path = None
        p_sim = self.get_parameter("sim_type")
        p_mode = self.get_parameter("mode")
        p_record = self.get_parameter("record")
        p_record_file = self.get_parameter("record_file")
        sim_type = str(p_sim.value) if p_sim.value else "primitive"
        self.mode = str(p_mode.value) if p_mode.value else "sim"
        # Phase 88: translation layer for Phase 85 policy → Phase 86 physics
        p_phase88 = self.get_parameter("phase88_translation")
        self._phase88_enabled = bool(p_phase88.value) if p_phase88.value else True
        # Phase 276: Contact-Jacobian mode (URDF sim only)
        p_cj = self.get_parameter("use_contact_jacobian")
        self._use_contact_jacobian = bool(p_cj.value) if p_cj.value else True
        self._record = bool(p_record.value) if p_record.value else False
        self._record_file = str(p_record_file.value) if p_record_file.value else ""

        # Phase 210: Hybrid bridge goal — loaded here so it's available before VLA node starts
        p_goal_x = self.get_parameter("goal_x")
        p_goal_y = self.get_parameter("goal_y")
        gx = float(p_goal_x.value) if p_goal_x.value else 0.0
        gy = float(p_goal_y.value) if p_goal_y.value else 0.0
        self._goal_xy = np.array([gx, gy], dtype=np.float64)
        if gx != 0.0 or gy != 0.0:
            self.get_logger().info(f"Hybrid bridge goal: ({gx:.3f}, {gy:.3f})")

        # ── Initialise simulation OR real hardware ────────────────────────
        if self.mode == "real":
            self.get_logger().info("Starting REAL HARDWARE mode (serial servos)…")
            from real_hardware_adapter import RealHardwareAdapter, MockHardwareAdapter
            # Try real hardware; fall back to mock if serial unavailable
            self.hw: RealHardwareAdapter | MockHardwareAdapter = RealHardwareAdapter(
                port="/dev/ttyUSB0",
                baudrate=115200,
                arm_servo_ids=[1, 2, 3, 4, 5],
                wheel_servo_ids=[10, 11, 12],
                arm_num_joints=5,
                wheel_num_joints=3,
            )
            if not self.hw.connect():
                self.get_logger().warn("Real hardware unavailable — using mock adapter")
                self.hw = MockHardwareAdapter()
                self.hw.connect()
            self.sim = None
            self.get_logger().info("Real hardware adapter ready.")
        elif sim_type == "urdf":
            self.get_logger().info("Starting LeKiWiSimURDF (STL mesh geometry) via make_sim()…")
            self.sim = make_sim("urdf", render=False)
            self.get_logger().info("URDF simulation initialised.")
            # Phase 164 FIX: Warmup step — base_position is zero before first step
            # After one step, base settles to correct z=0.075 with proper quaternion
            self.sim.step(np.zeros(9))
            self.get_logger().info("URDF warmup: 1 step to stabilise base position.")
            self.hw = None
            # Phase 96: Start background camera adapter for 20 Hz image publishing
            self.camera_adapter = CameraAdapter(
                self.sim, self,
                front_topic="/lekiwi/camera/image_raw",
                wrist_topic="/lekiwi/wrist_camera/image_raw",
                fps=20,
            )
            self.camera_adapter.start()
            self._camera_stats_every = 100  # log stats every N frames
        else:
            self.get_logger().info("Starting LeKiWiSim (cylinder primitives) via make_sim()…")
            self.sim = make_sim("primitive", render=False)
            self.get_logger().info("Primitive simulation initialised.")
            # Phase 212: Warmup step required — MuJoCo renderer returns all-black before first step
            # (base_freejoint has qpos[2]=0.14 at init but needs physics tick before camera works)
            self.sim.step(np.zeros(9))
            self.get_logger().info("Primitive warmup: 1 step to enable rendering.")
            self.hw = None

        # CTFSecurityAuditor — logs C1-C8 events to JSONL + fires _on_security_alert
        if ctf_mode:
            self.ctf_auditor = CTFSecurityAuditor(
                alert_callback=self._on_security_alert,
                log_path=str(Path.home() / "hermes_research" / "lekiwi_vla" / "ctf_events.jsonl"),
            )
            self.get_logger().info("CTF auditor: enabled (CTFSecurityAuditor active)")
        else:
            self.ctf_auditor = None
            self.get_logger().info("CTF auditor: disabled (ctf_mode=False)")

        # Legacy SecurityMonitor — kept for backward compat with existing callbacks
        self.security_monitor = SecurityMonitor(
            enable_hmac=enable_hmac,
            cmd_vel_secret=cmd_vel_secret_bytes,
        )
        if enable_hmac:
            self.get_logger().info("SecurityMonitor (legacy compat) active with HMAC verification.")
        else:
            self.get_logger().info("SecurityMonitor (legacy compat) active.")

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
        # Phase 237: /lekiwi/goal subscriber — dynamic navigation goal from any node
        self.goal_sub = self.create_subscription(
            Point, "/lekiwi/goal", self._on_goal, qos
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

        # ── HMAC-signed cmd_vel subscriber (Challenge 1 defense) ─────────────
        # Message format: bytes = struct.pack('d', timestamp) + struct.pack('ddd', vx, vy, wz) + mac_bytes(32)
        # Total: 8 + 24 + 32 = 64 bytes
        self.cmd_vel_hmac_sub = self.create_subscription(
            ByteMultiArray, "/lekiwi/cmd_vel_hmac", self._on_cmd_vel_hmac, qos
        )
        self._blocked_count = 0

        # ── Timer: step MuJoCo & publish at 20 Hz (camera is expensive) ────────
        self.timer = self.create_timer(0.05, self._on_timer)   # 20 Hz

        # ── State ────────────────────────────────────────────────────────────────
        self._last_action = np.zeros(9, dtype=np.float64)   # [arm*6, wheel*3]
        self._vla_action_fresh = False                       # set True when VLA writes action
        self._frame_count = 0
        # Phase 210: Hybrid bridge — goal already set above from launch params.
        # Track current base position from sim (updated each _on_timer)
        self._cached_base_xy = None

        # ── Watchdog timer for real hardware mode ──────────────────────────────
        self._last_cmd_vel_time = self.get_clock().now()
        self._watchdog_timer = self.create_timer(0.5, self._on_watchdog)   # 2 Hz check
        self._watchdog_count = 0
        # Odometry state (mirrors omni_odometry.py)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._last_odom_time = self.get_clock().now()

        # ── Trajectory recording ──────────────────────────────────────────────
        self._recorder = None
        self._record_control_sub = None
        p_record_images = self.get_parameter("record_images")
        record_images = bool(p_record_images.value) if p_record_images.value else True

        if self._record:
            if not self._record_file:
                self._record_file = os.path.expanduser(
                    f"~/hermes_research/lekiwi_vla/trajectories/run_{int(time.time())}.h5")
            self._recorder = TrajectoryRecorder(self._record_file, record_images=record_images)
            self._recorder.start()
            self.get_logger().info(f"Recording trajectory → {self._record_file}")
            # Record control subscriber
            self._record_control_sub = self.create_subscription(
                String, "/lekiwi/record_control", self._on_record_control, qos
            )

        self.get_logger().info(
            "LeKiWi ROS2 bridge ready.  Topics:\n"
            "  /lekiwi/cmd_vel       ← subscribe\n"
            "  /lekiwi/vla_action    ← subscribe (VLA arm override)\n"
            "  /lekiwi/joint_states  → publish\n"
            "  /lekiwi/odom          → publish (20 Hz)\n"
            "  /lekiwi/camera/image_raw → publish (20 Hz)\n"
            "  /lekiwi/wheel_N/cmd_vel → publish"
        )

    # ── Security alert callback (used by CTFSecurityAuditor) ──────────────────

    def _on_security_alert(self, alert) -> None:
        """
        Called by CTFSecurityAuditor when a CTF-relevant attack is detected.
        Publishes the alert to /lekiwi/security_alert and logs it.
        """
        import json
        self._blocked_count += 1
        self.get_logger().warn(
            f"[{alert.challenge_id}] {alert.channel}: {alert.description} "
            f"(severity={alert.severity}, flag={alert.flag or 'none'})",
            throttle_duration_sec=3.0,
        )
        # Publish to ROS2 topic for external monitors
        alert_msg = String()
        alert_msg.data = alert.to_json()
        self.alert_pub.publish(alert_msg)

    # ── HMAC-signed cmd_vel callback (Challenge 1 defense) ─────────────────────

    def _on_cmd_vel_hmac(self, msg) -> None:
        """
        Handle HMAC-authenticated cmd_vel commands.
        Message: ByteMultiArray with 64 bytes:
          bytes[0:8]   = struct.pack('d', timestamp)    — float64
          bytes[8:32]  = struct.pack('ddd', vx,vy,wz)  — 3× float64
          bytes[32:64] = HMAC-SHA256 signature         — 32 bytes

        Forged commands (wrong HMAC key) are BLOCKED and logged.
        Replay attacks (same command within window) are BLOCKED.
        """
        import struct
        try:
            data = bytes(msg.data)
            if len(data) < 64:
                self.get_logger().warn(f"HMAC cmd_vel too short: {len(data)} < 64 bytes")
                return
            stamp, = struct.unpack('d', data[0:8])
            vx, vy, wz = struct.unpack('ddd', data[8:32])
            mac = data[32:64]
        except Exception as e:
            self.get_logger().warn(f"HMAC cmd_vel parse error: {e}")
            return

        verdict = self.security_monitor.check_cmd_vel_hmac(vx, vy, wz, stamp, mac)
        if verdict.blocked:
            self._blocked_count += 1
            self.get_logger().warn(
                f"Blocked HMAC cmd_vel #{{}} {{}} severity={{}} vx={{:.3f}} vy={{:.3f}} wz={{:.3f}}".format(
                    self._blocked_count, verdict.event_type, verdict.severity, vx, vy, wz),
                throttle_duration_sec=2.0)
            # Publish security alert
            alert_msg = String()
            alert_msg.data = json.dumps({
                "type": verdict.event_type,
                "severity": verdict.severity,
                "details": verdict.details,
                "ctf_flag": verdict.details.get("ctf_flag"),
            })
            self.alert_pub.publish(alert_msg)
            return

        # CTF SecurityAuditor — C1/C2/C3/C4/C5 detection on HMAC-verified cmd_vel
        if self.ctf_auditor is not None:
            ctf_alert = self.ctf_auditor.on_cmd_vel(
                vx=vx, vy=vy, wz=wz, timestamp=stamp, hmac_verified=True
            )
            if ctf_alert is not None:
                self.get_logger().debug(
                    f"CTF [{ctf_alert.challenge_id}] on verified cmd_vel — {ctf_alert.description}")

        # HMAC verified — reset watchdog and apply command
        self._last_cmd_vel_time = self.get_clock().now()
        # Phase 164 FIX: Apply same scale factor as _on_cmd_vel
        SCALE = 200 * 0.002  # = 0.4
        wheel_speeds = twist_to_wheel_speeds(vx * SCALE, vy * SCALE, wz)
        if self._recorder is not None:
            self._recorder.record_cmd_vel(vx, vy, wz)
        arm_action = self._last_action[0:6]
        if self.mode == "real" and self.hw is not None:
            self.hw.queue_arm_positions(list(arm_action[:self.hw.arm_num_joints]))
            self.hw.queue_wheel_velocities(list(wheel_speeds))
            self._last_action = np.concatenate([arm_action, wheel_speeds])
            for i, speed in enumerate(wheel_speeds):
                wm = Float64()
                wm.data = float(speed)
                self.wheel_pubs[i].publish(wm)
            return
        action = np.concatenate([arm_action, wheel_speeds]).astype(np.float64)
        self.sim.step(action)
        self._last_action = action
        for i, speed in enumerate(wheel_speeds):
            wm = Float64()
            wm.data = float(speed)
            self.wheel_pubs[i].publish(wm)

    # ── cmd_vel callback ────────────────────────────────────────────────────────

    def _on_cmd_vel(self, msg: Twist):
        """
        Convert Twist → wheel speeds, combine with current arm action, step sim.

        When VLA is active (_vla_action_fresh=True), arm portion from _last_action
        is preserved and only the wheel portion is overridden by cmd_vel.
        When VLA is not active, arms stay at their last commanded position.

        In real hardware mode, also resets the watchdog timer.
        """
        # Reset watchdog timer on any cmd_vel
        self._last_cmd_vel_time = self.get_clock().now()
        self._watchdog_count = 0

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

        # CTF SecurityAuditor — C1/C2/C3/C4/C5 detection on raw cmd_vel
        if self.ctf_auditor is not None:
            ctf_alert = self.ctf_auditor.on_cmd_vel(
                vx=vx, vy=vy, wz=wz, timestamp=stamp, hmac_verified=False
            )
            if ctf_alert is not None:
                self.get_logger().debug(
                    f"CTF [{ctf_alert.challenge_id}] on raw cmd_vel — hmac_verified=False (allowed by policy)")

        # Compute wheel angular velocities from kinematics
        # Phase 164 FIX: Scale factor for Jacobian IK units
        # twist_to_contact_wheel_speeds() uses J_c calibrated for m/200steps
        # ROS2 Twist provides vx,vy in m/s → MuJoCo timestep=0.002s
        # Scale: vx_m_per_200steps = vx_m_per_s * (200 * 0.002) = vx * 0.4
        SCALE = 200 * 0.002  # = 0.4
        wheel_speeds = twist_to_wheel_speeds(vx * SCALE, vy * SCALE, wz)

        # ── Trajectory recording: log cmd_vel ───────────────────────────────────
        if self._recorder is not None:
            self._recorder.record_cmd_vel(vx, vy, wz)

        # ── Phase 212: Direction-agreement hybrid ─────────────────────────────────
        # VLA wheel actions are 3-6x smaller than optimal but direction is correct.
        # When VLA is conservative (mag < threshold):
        #   - If VLA direction agrees with P-controller → amplify VLA and use it
        #   - If VLA direction disagrees → use P-controller directly
        # When VLA is confident (mag >= threshold) → use VLA directly
        #
        # Phase 276: P-controller has two implementations:
        #   - use_contact_jacobian=True  → _contact_jacobian_pctrl (100% SR in URDF sim)
        #   - use_contact_jacobian=False → kinematic + Phase88 translation (~20% SR in URDF)
        if self._vla_action_fresh:
            arm_action = self._last_action[0:6]
            vla_wheel_raw = self._last_action[6:9]
            vla_mag = np.linalg.norm(vla_wheel_raw)

            if vla_mag < _HYBRID_WHEEL_FALLBACK_THRESHOLD:
                # VLA too conservative → check if direction agrees with P-controller
                if hasattr(self, '_goal_xy') and self._goal_xy is not None:
                    base_xy = self._get_base_xy()
                    kP_FALLBACK = 2.0
                    if self._use_contact_jacobian:
                        # Phase 276: CJ P-controller — no Phase88 translation needed
                        pctrl_ws = _contact_jacobian_pctrl(base_xy, self._goal_xy, kP_FALLBACK)
                    else:
                        # Legacy kinematic P-controller — needs Phase88 translation
                        err = self._goal_xy - base_xy
                        vx_fb = kP_FALLBACK * err[0]
                        vy_fb = kP_FALLBACK * err[1]
                        pctrl_ws = twist_to_contact_wheel_speeds(vx_fb, vy_fb, 0.0)

                    if _wheel_direction_agrees(vla_wheel_raw, pctrl_ws,
                                               _DIRECTION_AGREEMENT_THRESHOLD):
                        # VLA direction correct → amplify and use VLA
                        wheel_speeds = vla_wheel_raw * _WHEEL_AMPLIFICATION_FACTOR
                    else:
                        # VLA direction wrong → use P-controller
                        wheel_speeds = pctrl_ws
                        # Phase 88 translation ONLY for kinematic P-controller output
                        if not self._use_contact_jacobian and self._phase88_enabled:
                            wheel_speeds = _translate_phase85_to_phase86(wheel_speeds)
                else:
                    pctrl_ws = np.zeros(3)
                    wheel_speeds = vla_wheel_raw  # no goal → use VLA as-is
            else:
                # VLA active and confident → use VLA directly (with translation if needed)
                # Phase 88 translation only needed for old-phase VLA policies on kinematic model
                if self._phase88_enabled and not self._use_contact_jacobian:
                    wheel_speeds = _translate_phase85_to_phase86(vla_wheel_raw)
                else:
                    wheel_speeds = vla_wheel_raw
        else:
            arm_action = self._last_action[0:6]

        # ── Real hardware mode: send commands to serial servos ───────────
        if self.mode == "real" and self.hw is not None:
            self.hw.queue_arm_positions(list(arm_action[:self.hw.arm_num_joints]))
            self.hw.queue_wheel_velocities(list(wheel_speeds))
            self._last_action = np.concatenate([arm_action, wheel_speeds])
            # Republish wheel velocities (mirrors real robot)
            for i, speed in enumerate(wheel_speeds):
                wm = Float64()
                wm.data = float(speed)
                self.wheel_pubs[i].publish(wm)
            return

        # ── Simulation mode: step MuJoCo ─────────────────────────────────
        action = np.concatenate([arm_action, wheel_speeds]).astype(np.float64)
        self.sim.step(action)
        self._last_action = action

        # Also republish individual wheel velocities (mirrors real robot)
        for i, speed in enumerate(wheel_speeds):
            wm = Float64()
            wm.data = float(speed)
            self.wheel_pubs[i].publish(wm)

    # ── Trajectory recording control ──────────────────────────────────────────

    def _on_record_control(self, msg: String):
        """
        Control recording via topic:
          "start"  — begin recording (or restart if already recording)
          "stop"   — stop recording and flush to disk
          "status" — log current recording status
        """
        cmd = msg.data.strip().lower()
        if cmd == "start":
            if self._recorder is None:
                self.get_logger().warn("Recording not enabled — set record:=true in launch file")
                return
            self._recorder.start()
            self.get_logger().info("Trajectory recording: START")
        elif cmd == "stop":
            if self._recorder is None:
                return
            self._recorder.stop()
            self.get_logger().info("Trajectory recording: STOP (flushed)")
        elif cmd == "status":
            if self._recorder is None:
                self.get_logger().info("Recording: DISABLED")
            else:
                status = self._recorder.status()
                self.get_logger().info(
                    f"Recording: {status['state']} — "
                    f"{status['frames']} frames, {status['duration']:.1f}s"
                )
        else:
            self.get_logger().warn(f"Unknown record command: {cmd}")

    def _on_goal(self, msg: Point):
        """
        Handle dynamic /lekiwi/goal updates from any ROS2 node.

        Updates self._goal_xy in-place.  This enables:
          - CTF navigation challenges where goals are published dynamically
          - SLAM waypoint following
          - Multi-goal patrol missions
          - External path planners sending waypoints

        C8 Goal Spoofing Detection:
          SecurityMonitor.check_goal_spoofing() is called to detect:
            1. Rate abuse (>5 goals/s)
            2. Teleportation (>2 m/s radial goal speed)
            3. Out-of-bounds (>5m from origin)
          Suspicious updates are logged but still applied (fail-open for
          availability — switch to fail-closed for strict deployments).
        """
        # C8: Check goal update for spoofing
        gx, gy = msg.x, msg.y
        allowed, reason, flag = self.security_monitor.check_goal_spoofing(
            gx, gy, source="external"
        )
        if not allowed:
            # Still apply the goal (fail-open), but log security alert
            self.get_logger().warn(
                f"C8 [GOAL SPOOF] blocked: {reason} — goal=({gx:.3f}, {gy:.3f})"
                + (f" flag={flag}" if flag else "")
            )
            # Publish security alert via CTF auditor if available
            if self.ctf_auditor is not None:
                ctf_alert = self.ctf_auditor.on_goal_spoof(
                    gx, gy, reason, flag
                )
                if ctf_alert is not None:
                    self._on_security_alert(ctf_alert)

        self._goal_xy = np.array([gx, gy], dtype=np.float64)
        self.get_logger().debug(
            f"Goal updated via /lekiwi/goal → ({gx:.3f}, {gy:.3f})"
        )

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

            # ── Legacy SecurityMonitor: C7 VLA action injection detection ─────
            # Detect anomalous action jumps (indicates injected policy)
            allowed, reason, ctf_flag = self.security_monitor.check_vla_action(action)
            if not allowed:
                self.get_logger().warn(
                    f"C7 [VLA ACTION INJECT] blocked: {reason} "
                    + (f"flag={ctf_flag}" if ctf_flag else ""),
                    throttle_duration_sec=2.0,
                )
                # Still allow the action through (fail-open) but log it

            # CTF SecurityAuditor — C7: VLA action injection detection
            if self.ctf_auditor is not None:
                stamp = self.get_clock().now().nanoseconds / 1e9
                ctf_alert = self.ctf_auditor.on_vla_action(
                    action=list(action),
                    policy_name=getattr(self, '_current_policy_name', 'unknown'),
                    timestamp=stamp,
                )
                if ctf_alert is not None:
                    self.get_logger().debug(f"CTF [{ctf_alert.challenge_id}] VLA action — {ctf_alert.description}")

            self._last_action = action
            self._vla_action_fresh = True
        except Exception as e:
            self.get_logger().warn(f"VLA action parse error: {e}", once=True)

    # ── Timer callback ─────────────────────────────────────────────────────────

    def _on_timer(self) -> None:
        """Publish state as JointState + Odometry + camera Image."""
        now = self.get_clock().now()

        # ── Real hardware mode: read from serial servos ─────────────────
        if self.mode == "real" and self.hw is not None:
            obs = self.hw.get_state()
            wheel_vel = obs["wheel_velocities"]
            dt = 0.05
            # Accumulate wheel positions
            self._wheel_posAccum = getattr(self, '_wheel_posAccum', np.zeros(3)) + wheel_vel * dt

            # Odometry from wheel velocities
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

            # Publish TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now.to_msg()
            tf_msg.header.frame_id = "odom"
            tf_msg.child_frame_id = "base_link"
            tf_msg.transform.translation.x = self._odom_x
            tf_msg.transform.translation.y = self._odom_y
            tf_msg.transform.rotation.z = np.sin(self._odom_theta / 2.0)
            tf_msg.transform.rotation.w = np.cos(self._odom_theta / 2.0)
            self.tf_broadcaster.sendTransform(tf_msg)

            # JointState (canonical names)
            wheel_pos = list(self._wheel_posAccum)
            arm_pos = list(obs["arm_positions"][:6])
            arm_vel = list(obs["arm_velocities"][:6])
            msg = JointState()
            msg.header.stamp = now.to_msg()
            msg.name = self.ARM_NAMES + self.WHEEL_NAMES
            msg.position = arm_pos + wheel_pos
            msg.velocity = arm_vel + list(wheel_vel)
            self.joint_state_pub.publish(msg)

            # URDF-compatible JointState
            urdf_msg = JointState()
            urdf_msg.header.stamp = now.to_msg()
            urdf_msg.name = URDF_ARM_JOINT_NAMES + URDF_WHEEL_JOINT_NAMES
            urdf_msg.position = arm_pos + wheel_pos
            urdf_msg.velocity = arm_vel + list(wheel_vel)
            self.joint_state_urdf_pub.publish(urdf_msg)

            # Clear VLA freshness
            self._vla_action_fresh = False
            return

        # ── Simulation mode: read from MuJoCo ───────────────────────────
        obs = self.sim._obs()
        dt = 0.05   # matches timer period

        # Phase 210: Cache base XY for hybrid bridge P-controller fallback
        base_pos = obs.get("base_position", np.zeros(3))
        self._cached_base_xy = base_pos[:2].copy()

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

        # ── Trajectory recording (simulation mode) ───────────────────────────
        if self._recorder is not None:
            ts = now.seconds_nanoseconds()
            timestamp = ts[0] + ts[1] * 1e-9
            self._recorder.record_joint_state(
                arm_positions=arm_pos,
                arm_velocities=list(obs.get("arm_velocities", np.zeros(6))),
                wheel_positions=wheel_pos,
                wheel_velocities=list(wheel_vel),
                timestamp=timestamp,
            )

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

        # ── Legacy SecurityMonitor: C6 joint spoofing detection ─────────────
        # Detect anomalous joint position jumps (indicates fake sensor data)
        allowed, reason, ctf_flag = self.security_monitor.check_joint_spoofing(
            positions=np.array(urdf_msg.position),
            velocities=np.array(urdf_msg.velocity),
        )
        if not allowed:
            self.get_logger().warn(
                f"C6 [SENSOR SPOOF] {reason} "
                + (f"flag={ctf_flag}" if ctf_flag else ""),
                throttle_duration_sec=2.0,
            )

        # ── CTF SecurityAuditor — C6: sensor spoofing detection ──────────────
        if self.ctf_auditor is not None:
            stamp = now.seconds_nanoseconds()[0] + now.seconds_nanoseconds()[1] * 1e-9
            ctf_alert = self.ctf_auditor.on_joint_states(
                position=urdf_msg.position,
                velocity=urdf_msg.velocity,
                timestamp=stamp,
            )
            if ctf_alert is not None:
                self.get_logger().debug(
                    f"CTF [{ctf_alert.challenge_id}] sensor — {ctf_alert.description}")

        # Phase 96: Camera images now handled by background CameraAdapter thread
        # (20 Hz front + wrist rendering, no longer blocks main step loop)
        # Log camera adapter stats every N frames
        self._frame_count += 1
        if hasattr(self, 'camera_adapter') and self._frame_count % self._camera_stats_every == 0:
            stats = self.camera_adapter.get_stats()
            self.get_logger().debug(
                f"Camera stats: frames={stats['frames_rendered']}, "
                f"errors={stats['render_errors']}, running={stats['running']}"
            )

        # Clear VLA freshness flag at end of each tick so stale VLA actions
        # (e.g., if the VLA node crashes) are automatically ignored.
        self._vla_action_fresh = False

    # ── Watchdog ─────────────────────────────────────────────────────────────────

    def _on_watchdog(self):
        """
        Hardware safety watchdog: if no cmd_vel received in >1s, halt servos.
        Only active in real hardware mode.
        """
        if self.mode != "real":
            return
        now = self.get_clock().now()
        dt = (now - self._last_cmd_vel_time).nanoseconds / 1e9
        if dt > 1.0:
            self._watchdog_count += 1
            if self._watchdog_count == 1:
                self.get_logger().warn(
                    f"⚠️ cmd_vel watchdog timeout ({dt:.1f}s) — halting servos",
                    throttle_duration_sec=5.0)
            # Send zero velocity to all servos (emergency stop)
            if self.hw is not None:
                zero_arm  = [0.0] * self.hw.arm_num_joints
                zero_wheel = [0.0] * self.hw.wheel_num_joints
                self.hw.queue_arm_positions(zero_arm, speed_rpm=10.0)
                self.hw.queue_wheel_velocities(zero_wheel)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    def set_goal(self, goal_xy: tuple[float, float]) -> None:
        """
        Set the navigation goal for P-controller fallback (Phase 210).
        goal_xy: (x, y) in world frame (meters).  e.g. (0.5, 0.0)

        Call this after creating the bridge, e.g. when a navigation task starts.
        If goal is None, P-controller fallback degrades to zero wheel speeds.
        """
        self._goal_xy = np.array(goal_xy, dtype=np.float64)
        self.get_logger().debug(f"Hybrid bridge goal set: ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")

    def _get_base_xy(self) -> np.ndarray:
        """
        Get current base XY position from sim (cached each _on_timer).
        Returns np.array([x, y]) in world frame.
        """
        if self._cached_base_xy is not None:
            return self._cached_base_xy
        # Fallback: extract from sim directly
        if self.sim is not None:
            obs = self.sim._obs()
            return obs["base_position"][:2].copy()
        return np.zeros(2)

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
        # MUST be called BEFORE ctf_alert so guardian_verdict is defined
        guardian_verdict = self.policy_guardian.check_and_guard(policy_bytes, stamp)

        # ── CTF Layer: C7/C8 policy hijacking detection ─────────────────────
        if self.ctf_auditor is not None:
            old_policy = getattr(self, '_current_policy_name', 'unknown')
            ctf_alert = self.ctf_auditor.on_policy_switch(
                old_policy=old_policy,
                new_policy=guardian_verdict.details.get("policy_name", "unknown"),
                timestamp=stamp,
                authorized=(guardian_verdict.action == "allow"),
            )
            if ctf_alert is not None:
                self.get_logger().debug(
                    f"CTF [{ctf_alert.challenge_id}] policy switch — {ctf_alert.description}")

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

    def destroy_node(self):
        """Stop camera adapter and flush trajectory recording on shutdown."""
        if hasattr(self, 'camera_adapter'):
            self.camera_adapter.stop()
            self.get_logger().info("CameraAdapter stopped.")
        if self._recorder is not None and self._recorder.num_frames > 0:
            self._recorder.stop()
            self._recorder.flush()
            self.get_logger().info(
                f"TrajectoryRecorder: flushed {self._recorder.num_frames} frames on shutdown"
            )
        super().destroy_node()


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
