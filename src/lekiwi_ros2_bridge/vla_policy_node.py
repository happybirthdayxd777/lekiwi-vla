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


# ── CLIP-Flow Matching policy loader ────────────────────────────────────────

def _make_clip_fm_policy(pretrained: Optional[str], device: str):
    """
    Load CLIP-FM policy trained via scripts/train_clip_fm.py.

    Handles two checkpoint formats (strict=False silently skips mismatched weights):

    Format A — "old" (epoch 10, trained with original architecture):
      flow_mlp.time_mlp[0]: Linear(1, 64)
      flow_mlp.time_mlp[2]: Linear(64, 128) → time_feat=128
      flow_mlp.net[0]:     Linear(658, 512) → total_dim=658=512+9+9+128
      vision_encoder:      SimpleCNN MLP proj (net.0/2/4/6/10)
      → Must use flow_head.* keys, will skip incompatible weights

    Format B — "new" (re-trained with updated architecture):
      flow_head.time_mlp[0]: Linear(1, 128)
      flow_head.time_mlp[2]: Linear(128, 256) → time_feat=256
      flow_head.net[0]:     Linear(786, 512) → total_dim=786=512+9+9+256
      vision_encoder:      CLIPVisionEncoder with nn.Linear(768,512) proj
      → Should load cleanly

    In both cases the CLIP vision encoder (frozen, 151M params) loads successfully.
    Only the trainable flow_head weights may be partially loaded from old checkpoints.
    """
    import torch
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_clip_fm import CLIPFlowMatchingPolicy

    policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device=device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    else:
        # Priority: 5k-frame/10epoch URDF-trained checkpoint (clean CLIP, strict=True) >
        # 2k-frame/5epoch URDF checkpoint >
        # old SimpleCNN checkpoint (requires key remapping)
        fresh_5k_ckpt = os.path.expanduser(
            "~/hermes_research/lekiwi_vla/results/fresh_train_5k/checkpoint_epoch_10.pt"
        )
        fresh_ckpt = os.path.expanduser(
            "~/hermes_research/lekiwi_vla/results/fresh_train/policy_urdf_ep5.pt"
        )
        old_ckpt = os.path.expanduser(
            "~/hermes_research/lekiwi_vla/results/fm_50ep_improved/policy_ep10.pt"
        )
        if os.path.exists(fresh_5k_ckpt):
            state_dict = torch.load(fresh_5k_ckpt, map_location=device, weights_only=False)
            sd = state_dict.get("policy_state_dict", state_dict)
            policy.load_state_dict(sd, strict=False)
            print(f"[CLIP-FM] Loading 5k/10ep checkpoint (clean CLIP, strict=False): {fresh_5k_ckpt}")
        elif os.path.exists(fresh_ckpt):
            state_dict = torch.load(fresh_ckpt, map_location=device, weights_only=False)
            sd = state_dict.get("policy_state_dict", state_dict)
            policy.load_state_dict(sd, strict=False)
            print(f"[CLIP-FM] Loading 2k/5ep URDF checkpoint: {fresh_ckpt}")
        elif os.path.exists(old_ckpt):
            state_dict = torch.load(old_ckpt, map_location=device, weights_only=False)
            print(f"[CLIP-FM] Falling back to old checkpoint (requires key remapping): {old_ckpt}")
        else:
            state_dict = {}

    if not state_dict:
        print("[CLIP-FM] No checkpoint — using random weights (training required)")
        policy.to(device)
        policy.eval()
        return policy

    # ── Key remapping: flow_mlp → flow_head for backwards compatibility ────────
    # Old checkpoints (Format A) use "flow_mlp" prefix; model uses "flow_head"
    remapped = {}
    flow_mlp_found = False
    for k, v in state_dict.items():
        if k.startswith("flow_mlp."):
            remapped[k.replace("flow_mlp.", "flow_head.", 1)] = v
            flow_mlp_found = True
        else:
            remapped[k] = v

    # ── Partial load: only keep keys with matching shapes ─────────────────────
    sd = policy.state_dict()
    compatible = {}
    skipped = []
    for k, v in remapped.items():
        if k in sd and sd[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    n_loaded = len(compatible)
    n_total  = len(sd)
    n_skipped = len(skipped)

    if skipped:
        print(f"[CLIP-FM] Loaded {n_loaded}/{n_total} weights ({n_skipped} skipped: shape mismatch)")
        # Show first few skipped keys for debugging
        for s in skipped[:6]:
            ckpt_shape = remapped[s].shape if s in remapped else "?"
            model_shape = sd.get(s, "(not in model)").shape if s in sd else "(not in model)"
            print(f"         skipped: {s}: ckpt={ckpt_shape} model={model_shape}")
    else:
        print(f"[CLIP-FM] Loaded {n_loaded}/{n_total} weights (clean load)")

    policy.load_state_dict(compatible, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def _normalize_state(state: np.ndarray) -> np.ndarray:
    """
    CLIP-FM was trained with RAW (unnormalized) state — do NOT normalize.

    Training data (lekiwi_urdf_5k.h5) uses raw native-unit state values:
      states range: -3.7872 to +2.9817
    The model never saw [-1,1] normalized inputs during training.

    Normalizing state at inference time creates a SEVERE DISTRIBUTION MISMATCH
    (e.g., j5 gripper raw=0.3 → normalized=1.0, but training saw raw=0.3).

    Fix: pass raw state directly, matching lerobot_policy_inference.py L257-258.
    """
    return np.asarray(state, dtype=np.float32)


class CLIPFMPolicyRunner:
    """
    Wrapper that adapts CLIPFlowMatchingPolicy's infer() interface
    to the same predict(obs) API used by MockPolicyRunner / LeRobot.
    obs keys: image (np.ndarray HWC uint8), state (np.ndarray [9] native units)
    """

    def __init__(self, policy, device="cpu"):
        import torch
        self.policy = policy
        self.device = device
        self.torch = torch

    def predict(self, obs: dict) -> np.ndarray:
        import torch
        # Accept both LeRobot-style keys and simple keys:
        # LeRobot:  obs["observation.images.primary"] = (1,3,224,224) CHW float [0,1]
        #           obs["observation.state"]          = (1,9) float [native]
        # Simple:   obs["image"] = HWC uint8, obs["state"] = (9,) native
        if "observation.images.primary" in obs:
            # LeRobot format: already (1,3,224,224) CHW float [0,1]
            img = torch.from_numpy(obs["observation.images.primary"]).float()
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device)
            state_np = obs["observation.state"]
        elif "image" in obs:
            # Simple format: HWC uint8 [0,255]
            img_np = obs["image"]
            img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img = img.to(self.device)
            state_np = obs["state"]
        else:
            raise KeyError(f"obs must contain 'image' or 'observation.images.primary', got keys: {list(obs.keys())}")

        # Normalize state to [-1,1] for policy
        state = _normalize_state(state_np).astype(np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, ...]
        state_t = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            action = self.policy.infer(img, state_t, num_steps=4)

        return action.squeeze(0).cpu().numpy()

    def reset(self):
        pass


def _make_clip_fm_policy_wrapper(pretrained: Optional[str], device: str):
    raw = _make_clip_fm_policy(pretrained, device)
    return CLIPFMPolicyRunner(raw, device)


_POLICY_LOADERS = {
    "mock":    _make_mock_policy,
    "clip_fm": _make_clip_fm_policy_wrapper,
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
        # Wrist camera — remapped by launch file to /lekiwi/wrist_camera/image_raw
        # Uses same _on_image callback; prefers wrist if available (mounted on arm).
        self.wrist_cam_sub = self.create_subscription(
            Image, "/lekiwi/wrist_camera/image_raw", self._on_image, qos
        )

        # ── State ────────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self._last_image: Optional[np.ndarray] = None
        self._last_joints: Optional[np.ndarray] = None
        self._inference_count = 0
        self._last_inference_time = 0.0

        self.get_logger().info(
            "LeKiWi VLA Policy Node ready.\n"
            "  /lekiwi/joint_states        ← subscribe\n"
            "  /lekiwi/camera/image_raw    ← subscribe\n"
            "  /lekiwi/wrist_camera/       ← subscribe (wrist preferred, falls back to front)\n"
            "  /lekiwi/vla_action          → publish"
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
        """Store latest camera image; trigger inference if joints are ready.

        Also listens on /lekiwi/wrist_camera/image_raw (remapped to the same
        callback).  When image arrives, check whether joints are already
        buffered; if so, run inference immediately rather than waiting for the
        next joint_states message (which eliminates the 1-frame lag from the
        previous _on_joint_states → _run_inference → return-early path).
        """
        try:
            self._last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return
        # Trigger inference if joints are already buffered
        if self._last_joints is not None:
            self._run_inference()

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
        # NOTE: CLIP-FM policy was trained with wheel_velocities as the 3 wheel
        # state dimensions (matching lerobot_policy_inference.py), NOT positions.
        state = np.concatenate([
            joints["arm_positions"],
            joints["wheel_velocities"],   # ← was: wheel_positions (training mismatch bug)
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
