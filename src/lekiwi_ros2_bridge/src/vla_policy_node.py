#!/usr/bin/env python3
"""
LeKiWi VLA Policy Node
=======================
ROS2 node that runs a LeRobot VLA policy on the LeKiWi robot.

Listens to:
  /lekiwi/joint_states    — arm (6) + wheel (3) positions & velocities
  /lekiwi/camera/image_raw — camera image @ 20 Hz

Runs policy inference each time a joint_states message arrives,
then publishes the action to /lekiwi/vla_action.

Usage:
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=mock
  ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=pi0 pretrained:=/path/to/model
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import numpy as np
import sys
import os
import time
from typing import Optional

# ── LeKiWi action limits (mirrors lerobot_policy_inference.py) ────────────────
LEKIWI_ARM_LIMITS = np.array([
    [-3.14,  3.14],   # j0 shoulder pan
    [-1.57,  1.57],   # j1 shoulder lift
    [-1.57,  1.57],   # j2 elbow
    [-1.57,  1.57],   # j3 wrist flex
    [-3.14,  3.14],   # j4 wrist roll
    [ 0.00,  0.04],   # j5 gripper slide
], dtype=np.float32)

LEKIWI_WHEEL_LIMITS = np.array([
    [-5.0,  5.0],   # w1
    [-5.0,  5.0],   # w2
    [-5.0,  5.0],   # w3
], dtype=np.float32)

ARM_JOINT_NAMES  = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINT_NAMES = ["w1", "w2", "w3"]


def normalize_action(raw_action: np.ndarray) -> np.ndarray:
    """Policy (-1..1) → LeKiWi native units."""
    arm    = raw_action[:6]
    wheel  = raw_action[6:9]
    arm_n  = LEKIWI_ARM_LIMITS[:, 0] + (arm + 1) / 2 * (
        LEKIWI_ARM_LIMITS[:, 1] - LEKIWI_ARM_LIMITS[:, 0])
    wheel_n = LEKIWI_WHEEL_LIMITS[:, 0] + (wheel + 1) / 2 * (
        LEKIWI_WHEEL_LIMITS[:, 1] - LEKIWI_WHEEL_LIMITS[:, 0])
    return np.concatenate([arm_n, wheel_n]).astype(np.float32)


# ── Mock policy (no GPU needed) ───────────────────────────────────────────────

class MockPolicyRunner:
    """Always-works mock: sinusoidal arm + random base. No GPU needed."""

    def predict(self, obs: dict) -> np.ndarray:
        t = time.time()
        action = np.zeros(9, dtype=np.float32)
        action[0] = 0.5 * np.sin(t * 2 * np.pi)          # shoulder pan
        action[1] = 0.3 * np.sin(t * 4 * np.pi)          # shoulder lift
        action[2] = -0.3 * np.sin(t * 4 * np.pi)         # elbow
        action[3] = 0.1                                  # wrist flex
        action[4] = 0.0                                  # wrist roll
        action[5] = 0.02                                 # gripper slide
        action[6] = 0.1 * np.sin(t * np.pi)              # wheel 0
        action[7] = 0.1 * np.sin(t * np.pi)              # wheel 1
        action[8] = 0.1 * np.sin(t * np.pi)              # wheel 2
        return action

    def reset(self):
        pass


# ── Real LeRobot policy loader ────────────────────────────────────────────────

def _make_mock_policy():
    return MockPolicyRunner()


def _make_lerobot_policy(policy_name: str, pretrained: Optional[str], device: str):
    """Dynamically import and configure LeRobot policy."""
    sys.path.insert(0, os.path.expanduser("~/lerobot/src"))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    import torch

    policy_name = policy_name.lower()
    config: PreTrainedConfig

    if policy_name == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        config = PI0Config(max_action_dim=9, max_state_dim=32, num_inference_steps=10)
    elif policy_name == "pi0_fast":
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
        config = PI0FastConfig(max_action_dim=9, max_state_dim=32, num_inference_steps=5)
    elif policy_name == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        config = ACTConfig(name="act", device=device, output_dir="~/act_lekiwi")
    elif policy_name == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        config = DiffusionConfig(name="diffusion", device=device, output_dir="~/diffusion_lekiwi")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if pretrained:
        config.from_pretrained = pretrained

    policy = make_policy(config)
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy


_POLICY_LOADERS = {
    "mock": _make_mock_policy,
}


def _load_policy(policy_name: str, pretrained: Optional[str], device: str):
    """Load policy by name, falling back to LeRobot if unknown."""
    loader = _POLICY_LOADERS.get(policy_name.lower())
    if loader:
        return loader()
    return _make_lerobot_policy(policy_name, pretrained, device)


# ─────────────────────────────────────────────────────────────────────────────

class LeKiWiVLAPolicyNode(Node):
    """
    VLA Policy Node for LeKiWi.

    Subscribes:
      /lekiwi/joint_states     — joint positions + velocities (27 Hz)
      /lekiwi/camera/image_raw — camera image (20 Hz)

    Publishes:
      /lekiwi/vla_action       — 9-DOF action [arm*6, wheel*3] in native units
                                 (published whenever joint_states arrives)
    """

    def __init__(self, policy_name: str = "mock", pretrained: Optional[str] = None):
        super().__init__("lekiwi_vla_policy_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("policy",     "mock")
        self.declare_parameter("pretrained", "")
        self.declare_parameter("device",    "cpu")

        policy_name  = str(self.get_parameter("policy").value)
        pretrained_v = self.get_parameter("pretrained").value
        device       = str(self.get_parameter("device").value)

        pretrained = str(pretrained_v) if pretrained_v else None

        self.get_logger().info(f"Loading VLA policy: '{policy_name}' on {device}")
        self.policy = _load_policy(policy_name, pretrained, device)
        self.get_logger().info(f"Policy '{policy_name}' loaded.")

        # ── ROS2 publishers ──────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.action_pub = self.create_publisher(
            Float64MultiArray, "/lekiwi/vla_action", qos
        )

        # ── ROS2 subscribers ──────────────────────────────────────────────────
        self.joint_state_sub = self.create_subscription(
            JointState, "/lekiwi/joint_states", self._on_joint_states, qos
        )
        self.image_sub = self.create_subscription(
            Image, "/lekiwi/camera/image_raw", self._on_image, qos
        )

        # ── State ────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self._last_image: Optional[np.ndarray] = None
        self._last_joints: Optional[np.ndarray] = None
        self._inference_count = 0
        self._last_inference_time = 0.0

        self.get_logger().info(
            "LeKiWi VLA Policy Node ready.\n"
            "  /lekiwi/joint_states     ← subscribe\n"
            "  /lekiwi/camera/image_raw ← subscribe\n"
            "  /lekiwi/vla_action       → publish"
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        """Store latest joint state; trigger policy inference."""
        # Map names → ordered arrays
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        arm_pos = np.array([
            msg.position[name_to_idx[n]] for n in ARM_JOINT_NAMES
        ], dtype=np.float32)
        wheel_pos = np.array([
            msg.position[name_to_idx[n]] for n in WHEEL_JOINT_NAMES
        ], dtype=np.float32)
        arm_vel = np.array([
            msg.velocity[name_to_idx[n]] for n in ARM_JOINT_NAMES
        ], dtype=np.float32)
        wheel_vel = np.array([
            msg.velocity[name_to_idx[n]] for n in WHEEL_JOINT_NAMES
        ], dtype=np.float32)

        self._last_joints = {
            "arm_positions":    arm_pos,
            "wheel_positions": wheel_pos,
            "arm_velocities":   arm_vel,
            "wheel_velocities": wheel_vel,
        }

        self._run_inference()

    def _on_image(self, msg: Image):
        """Store latest camera image."""
        try:
            self._last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")

    def _run_inference(self):
        """Run policy inference and publish action."""
        if self._last_image is None:
            return   # wait for first image

        # Build observation dict
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(self._last_image)
        img_resized = img_pil.resize((224, 224), PILImage.BILINEAR)
        img_arr = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0

        joints = self._last_joints
        state = np.concatenate([
            joints["arm_positions"],
            joints["wheel_positions"],
        ]).astype(np.float32)

        obs = {
            "observation.images.primary": img_arr[np.newaxis, ...],   # (1,3,224,224)
            "observation.state":          state[np.newaxis, ...],     # (1,9)
        }

        # Policy inference
        raw_action = self.policy.predict(obs)         # (9,) in [-1, 1]
        native_action = normalize_action(raw_action)   # (9,) in native units

        # Publish
        msg = Float64MultiArray()
        msg.data = native_action.tolist()
        self.action_pub.publish(msg)

        self._inference_count += 1
        now = time.time()
        dt = now - self._last_inference_time
        if dt > 5.0:
            self.get_logger().info(
                f"Policy inference running at {1/max(dt, 0.001):.1f} Hz "
                f"({self._inference_count} total inferences)"
            )
            self._last_inference_time = now


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LeKiWiVLAPolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
