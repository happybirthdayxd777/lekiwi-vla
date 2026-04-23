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
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import numpy as np
import sys
import os
import time
from typing import Optional, Union

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


# ── Action Smoother (Phase 61) ─────────────────────────────────────────────────
class ActionSmoother:
    """
    Exponential moving average smoother for VLA policy actions.
    
    Reduces jerky wheel commands that cause URDF sim instability.
    Uses separate smoothing factors for arms (stiffer) vs wheels (softer).
    
    Phase 61: Addresses SR=33% root cause — raw policy output has high-frequency
    wheel velocity oscillations that destabilize the URDF simulation after step 50.
    """

    def __init__(
        self,
        wheel_alpha: float = 0.25,   # EMA coefficient for wheels (0=wet, 1=no smooth)
        arm_alpha: float  = 0.70,   # EMA coefficient for arms (less smoothing needed)
        wheel_max_delta: float = 0.8,  # Max wheel action change per step (rad/s)
        arm_max_delta: float  = 0.5,  # Max arm action change per step (rad)
        warmup_steps: int = 10,       # No smoothing for first N steps (let policy settle)
    ):
        self.wheel_alpha    = wheel_alpha
        self.arm_alpha      = arm_alpha
        self.wheel_max_delta = wheel_max_delta
        self.arm_max_delta   = arm_max_delta
        self.warmup_steps    = warmup_steps
        self._wheel_smoothed: Optional[np.ndarray] = None
        self._arm_smoothed:  Optional[np.ndarray]  = None
        self._step = 0

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """
        Apply EMA smoothing + delta clipping to action.
        
        Parameters
        ----------
        action : np.ndarray (9,)
            Native-unit action [arm*6, wheel*3]
            
        Returns
        -------
        np.ndarray (9,)
            Smoothed action with clamped deltas
        """
        arm_action   = action[:6]
        wheel_action = action[6:9]
        
        # Warmup: no smoothing, just store
        if self._step < self.warmup_steps:
            if self._wheel_smoothed is None:
                self._wheel_smoothed = wheel_action.copy()
                self._arm_smoothed   = arm_action.copy()
            else:
                self._wheel_smoothed = wheel_action.copy()
                self._arm_smoothed    = arm_action.copy()
            self._step += 1
            return action
        
        # ── Wheel smoothing (lower alpha = more smoothing) ─────────────────
        delta_wheel = wheel_action - self._wheel_smoothed
        # Clamp large changes to prevent jerky starts
        delta_wheel_clamped = np.clip(
            delta_wheel,
            -self.wheel_max_delta,
            self.wheel_max_delta
        )
        self._wheel_smoothed = self._wheel_smoothed + self.wheel_alpha * delta_wheel_clamped
        
        # ── Arm smoothing (higher alpha = less smoothing — arms need responsiveness) ─
        delta_arm = arm_action - self._arm_smoothed
        delta_arm_clamped = np.clip(
            delta_arm,
            -self.arm_max_delta,
            self.arm_max_delta
        )
        self._arm_smoothed = self._arm_smoothed + self.arm_alpha * delta_arm_clamped
        
        result = np.concatenate([self._arm_smoothed, self._wheel_smoothed])
        self._step += 1
        return result

    def reset(self):
        """Reset smoothing state (called on episode boundary)."""
        self._wheel_smoothed = None
        self._arm_smoothed   = None
        self._step = 0


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


def _make_task_oriented_policy(pretrained: Optional[str], device: str):
    """
    Load task-oriented policy trained via scripts/train_task_oriented.py.

    Same CLIP-FM architecture but trained with reward-weighted sampling.
    Checkpoint format: {'epoch': int, 'policy_state_dict': state_dict, ...}

    Default checkpoint: results/task_oriented_50ep/checkpoint_epoch_30.pt

    Phase 16: Uses state_dim=11 (goal-aware: arm_pos6 + wheel_vel3 + goal_xy2)
    """
    import torch
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_task_oriented import CLIPFlowMatchingPolicy as TOClipFlowMatchingPolicy

    policy = TOClipFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device=device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get("policy_state_dict", ckpt)
            missing, unexpected = policy.load_state_dict(sd, strict=False)
            print(f"[task_oriented] Loaded checkpoint epoch={ckpt.get('epoch','?')} | "
                  f"missing={missing} unexpected={len(unexpected)}")
        else:
            print(f"[task_oriented] WARNING: checkpoint not found: {ckpt_path}")
    return policy


def _make_task_oriented_wrapper(pretrained: Optional[str], device: str):
    raw = _make_task_oriented_policy(pretrained, device)
    return CLIPFMPolicyRunner(raw, device)


def _make_clip_fm_wrapper(pretrained: Optional[str], device: str):
    """Wrapper for clip_fm: loads CLIPFlowMatchingPolicy + CLIPFMPolicyRunner."""
    raw = _make_clip_fm_policy(pretrained, device)
    return CLIPFMPolicyRunner(raw, device)


def _make_phase196_policy(pretrained: Optional[str], device: str):
    """
    Load Phase 196 Contact-Jacobian VLA policy.

    Architecture: GoalConditionedPolicy (same as train_phase196.py)
      - CLIP ViT-B/32 vision encoder (frozen)
      - Goal MLP: 2 → 256 → 128
      - State net: 11D → 256 → 128
      - Cross-attention: goal(Q) attends to CLIP(K,V)
      - Flow matching head: 4-step Euler inference
      - 155M total params, 0 NaN/Inf tensors

    Checkpoint: results/phase196_contact_jacobian_train/epoch_14.pt
      - loss=0.3267, 14 epochs on CORRECT Contact-Jacobian data
      - 90% SR in standalone eval (10/10 random goals, 200 steps)

    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    Action: arm_torque(6) + wheel_speed(3) = 9D
    """
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_phase196 import GoalConditionedPolicy as GCPolicy

    policy = GCPolicy(state_dim=11, action_dim=9).to(device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
    else:
        ckpt_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/phase196_contact_jacobian_train/epoch_14.pt")

    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("policy_state_dict", ckpt)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[phase196] Loaded checkpoint epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?')} | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[phase196] WARNING: checkpoint not found: {ckpt_path}")

    return policy


def _make_phase196_wrapper(pretrained: Optional[str], device: str):
    """Wrapper for phase196: GoalConditionedPolicy with Phase196Replay-style inference."""
    raw = _make_phase196_policy(pretrained, device)
    return Phase196PolicyRunner(raw, device)


def _make_dagger_policy(pretrained: Optional[str], device: str):
    """
    Load Phase 252 DAgger policy (trained on P-controller expert corrections).

    Architecture: GoalConditionedPolicy (same as train_phase227/train_phase196.py)
      - CLIP ViT-B/32 vision encoder (frozen)
      - Goal MLP: 2 → 256 → 128
      - State net: 11D → 256 → 128
      - Cross-attention: goal(Q) attends to CLIP(K,V)
      - Flow matching head: 4-step Euler inference
      - 155M total params
      - Train: 30 epochs on 50ep DAgger data + 50ep base data
      - Base: results/phase227_contact_jacobian_train/best_policy.pt (loss=0.195)
      - Best checkpoint: results/dagger_phase254_train/best_policy.pt (loss=0.0018, epoch 20)

    Checkpoint order: best_policy.pt > final_policy.pt (best is epoch with lowest loss)
    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    Action: arm_torque(6) + wheel_speed(3) = 9D
    """
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_phase227 import GoalConditionedPolicy as GCPolicy

    policy = GCPolicy(state_dim=11, action_dim=9, hidden=512, device=device).to(device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
    else:
        # Prefer best_policy.pt (lowest loss epoch) over final_policy.pt (last epoch)
        best_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/dagger_phase254_train/best_policy.pt")
        final_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/dagger_phase254_train/final_policy.pt")
        ckpt_path = best_path if os.path.exists(best_path) else final_path

    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("policy_state_dict", ckpt)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[dagger] Loaded checkpoint epoch={ckpt.get('epoch','?')} "
              f"loss={ckpt.get('loss','?')} best_loss={ckpt.get('best_loss','?')} | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[dagger] WARNING: checkpoint not found: {ckpt_path}")

    return policy


def _make_dagger_wrapper(pretrained: Optional[str], device: str):
    """Wrapper for dagger: GoalConditionedPolicy DAgger-trained policy."""
    raw = _make_dagger_policy(pretrained, device)
    return Phase196PolicyRunner(raw, device)


class Phase196PolicyRunner:
    """
    Inference runner for Phase 196 GoalConditionedPolicy.

    Uses 4-step Euler flow matching, same as train_phase196.py infer().
    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    Action: raw policy output (9D), no normalization applied.
    """
    def __init__(self, policy, device: str = "cpu"):
        self.policy = policy
        self.device = device
        self.policy.eval()

    def __call__(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Args:
            image: [3, 224, 224] preprocessed image (normalized)
            state: [11] — arm_pos(6) + wheel_vel(3) + goal_norm(2)
        Returns:
            [9] raw action (policy output, no denormalization)
        """
        import torch
        img_t = torch.from_numpy(image[None]).to(self.device)
        st_t = torch.from_numpy(state[None]).to(self.device)
        with torch.no_grad():
            action = self.policy.infer(img_t, st_t, num_steps=4)[0].cpu().numpy()
        return action

    def reset(self):
        """No internal state to reset."""
        pass


class Stage2PolicyRunner:
    """
    Inference runner for Stage 2 Curriculum policy (GoalConditionedPolicy).

    Architecture (from scripts/train_curriculum.py):
      - CLIP ViT-B/32 vision encoder (frozen)
      - Goal MLP: 2 → 256 → 128 → 768 (goal_q_proj)
      - State net: 11D → 256 → 128
      - Cross-attention: goal_query attends to CLIP image features
      - Flow matching head: 4-step Euler inference
      - State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
      - Action: arm_torque(6) + wheel_speed(3) = 9D

    Checkpoint: results/phase260_curriculum_train/stage2_r045.pt
      - 72% SR on |r|<0.45m goals (50-goal eval, 200 steps)
      - This is the best practical VLA policy we have

    NOTE: Stage 2 is constrained to |r|<0.45m goals.
    At inference, only goals within this radius should be sent.
    """
    def __init__(self, policy, device: str = "cpu"):
        self.policy = policy
        self.device = device
        self.policy.eval()

    def __call__(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Args:
            image: [3, 224, 224] preprocessed image (normalized)
            state: [11] — arm_pos(6) + wheel_vel(3) + goal_norm(2)
        Returns:
            [9] raw action (policy output)
        """
        # Phase 268: Stage2 goal-radius filtering
        # Stage2 was trained on |r|<0.45m goals. If goal is outside this radius,
        # return zeros so the bridge falls back to P-controller (safe fallback).
        goal_norm = state[9:11]   # [-1, 1] normalized goal
        goal_xy_m = goal_norm * 0.4   # un-normalize to meters (same scale as training)
        goal_radius = np.linalg.norm(goal_xy_m)
        if goal_radius > 0.45:
            return np.zeros(9, dtype=np.float32)

        import torch
        img_t = torch.from_numpy(image[None]).to(self.device)
        st_t = torch.from_numpy(state[None]).to(self.device)
        with torch.no_grad():
            action = self.policy.infer(img_t, st_t, num_steps=4)[0].cpu().numpy()
        return action

    def reset(self):
        """No internal state to reset."""
        pass


def _make_stage2_policy(pretrained: Optional[str], device: str):
    """
    Load Stage 2 Curriculum policy from scripts/train_curriculum.py.

    Architecture: GoalConditionedPolicy (same as train_curriculum.py)
      - CLIP ViT-B/32 vision encoder (frozen)
      - Goal MLP: 2 → 256 → 128 → 768
      - State net: 11D → 256 → 128
      - Cross-attention: goal_query attends to CLIP(K,V)
      - Flow matching head: 4-step Euler inference
      - 155M total params

    Checkpoint: results/phase260_curriculum_train/stage2_r045.pt
      - 72% SR on |r|<0.45m goals (50-goal eval, 200 steps)
      - Trained on phase227_extended_65ep.h5 with max_goal_radius=0.45m

    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    Action: arm_torque(6) + wheel_speed(3) = 9D
    """
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_curriculum import GoalConditionedPolicy as GCPolicy

    policy = GCPolicy(state_dim=11, action_dim=9, hidden=512, device=device).to(device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
    else:
        ckpt_path = os.path.expanduser(
            "~/hermes_research/lekiwi_vla/results/phase260_curriculum_train/stage2_r045.pt"
        )

    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("policy_state_dict", ckpt)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[stage2] Loaded checkpoint epoch={ckpt.get('epoch','?')} "
              f"loss={ckpt.get('loss','?')} | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[stage2] WARNING: checkpoint not found: {ckpt_path}")

    return policy


def _make_stage2_wrapper(pretrained: Optional[str], device: str):
    """Wrapper for stage2: GoalConditionedPolicy Stage-2 curriculum policy."""
    raw = _make_stage2_policy(pretrained, device)
    return Stage2PolicyRunner(raw, device)


def _make_stage3_policy(pretrained: Optional[str], device: str):
    """
    Load Stage 3 Curriculum policy from scripts/train_curriculum_stage3.py.

    Architecture: GoalConditionedPolicy (same as train_curriculum.py)
      - CLIP ViT-B/32 vision encoder (frozen)
      - Goal MLP: 2 → 256 → 128 → 768 (goal_q_proj)
      - State net: 11D → 256 → 128
      - Cross-attention: goal_query attends to CLIP(K,V)
      - Flow matching head: 4-step Euler inference
      - 155M total params

    Checkpoints (results/phase264_curriculum_train/):
      - s3_epoch3.pt  — loss=0.2761
      - s3_epoch6.pt  — loss=0.2558
      - s3_epoch9.pt  — loss=0.2324 ← BEST (lowest loss, epoch 9/15)
      - s3_epoch12.pt — loss=0.2372 ← OVERFITTING (loss increased after epoch 9)

    NOTE: Stage 3 is trained on ALL goals (|r|=any distance) with 7589 frames.
    Best checkpoint is s3_epoch9.pt (72% SR on |r|<0.45m was Stage2; Stage3
    with all goals is still failing — 0-15% SR in evals).
    Default is s3_epoch9.pt (best loss).

    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    Action: arm_torque(6) + wheel_speed(3) = 9D
    """
    sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
    from scripts.train_curriculum_stage3 import GoalConditionedPolicy as GCPolicy

    policy = GCPolicy(state_dim=11, action_dim=9, hidden=512, device=device).to(device)

    if pretrained:
        ckpt_path = os.path.expanduser(pretrained)
    else:
        # Use s3_epoch9.pt as default (lowest loss = best model)
        ckpt_path = os.path.expanduser(
            "~/hermes_research/lekiwi_vla/results/phase264_curriculum_train/s3_epoch9.pt"
        )

    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("policy_state_dict", ckpt)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        print(f"[stage3] Loaded checkpoint epoch={ckpt.get('epoch','?')} "
              f"loss={ckpt.get('loss','?')} | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[stage3] WARNING: checkpoint not found: {ckpt_path}")

    return policy


def _make_stage3_wrapper(pretrained: Optional[str], device: str):
    """Wrapper for stage3: GoalConditionedPolicy Stage-3 curriculum policy."""
    raw = _make_stage3_policy(pretrained, device)
    return Stage2PolicyRunner(raw, device)


_POLICY_LOADERS = {
    "mock":          _make_mock_policy,
    "clip_fm":       _make_clip_fm_wrapper,
    "task_oriented": _make_task_oriented_wrapper,
    "phase196":      _make_phase196_wrapper,
    "dagger":        _make_dagger_wrapper,
    "stage2":        _make_stage2_wrapper,
    "stage3":        _make_stage3_wrapper,
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

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("policy",     "mock")
        self.declare_parameter("pretrained", "")
        self.declare_parameter("device",    "cpu")
        # Phase 16: goal_xy for goal-aware 11D state
        self.declare_parameter("goal_x",    0.5)
        self.declare_parameter("goal_y",    0.0)
        # Phase 61: action smoothing parameters
        self.declare_parameter("wheel_alpha",   0.25)
        self.declare_parameter("arm_alpha",     0.70)
        self.declare_parameter("wheel_max_delta", 0.8)
        self.declare_parameter("arm_max_delta",  0.5)
        self.declare_parameter("smooth_warmup",  10)

        policy_name  = str(self.get_parameter("policy").value)
        pretrained_v = self.get_parameter("pretrained").value
        device       = str(self.get_parameter("device").value)
        goal_x       = float(self.get_parameter("goal_x").value)
        goal_y       = float(self.get_parameter("goal_y").value)

        # Phase 16: normalize goal to [-1, 1] for 11D state embedding
        self._goal_xy = np.array([goal_x, goal_y], dtype=np.float32)
        self._goal_xy_norm = np.clip(self._goal_xy / 1.0, -1.0, 1.0)

        pretrained = str(pretrained_v) if pretrained_v else None

        self.get_logger().info(f"Loading VLA policy: '{policy_name}' on {device}")
        self.policy = _load_policy(policy_name, pretrained, device)
        # Phase 278: Store policy name to avoid double-normalization for native-unit policies
        self._policy_name = policy_name
        self.get_logger().info(f"Policy '{policy_name}' loaded.")

        # ── Phase 61: Action Smoother ────────────────────────────────────────
        wheel_alpha      = float(self.get_parameter("wheel_alpha").value)
        arm_alpha        = float(self.get_parameter("arm_alpha").value)
        wheel_max_delta  = float(self.get_parameter("wheel_max_delta").value)
        arm_max_delta    = float(self.get_parameter("arm_max_delta").value)
        smooth_warmup    = int(self.get_parameter("smooth_warmup").value)
        self._smoother = ActionSmoother(
            wheel_alpha=wheel_alpha,
            arm_alpha=arm_alpha,
            wheel_max_delta=wheel_max_delta,
            arm_max_delta=arm_max_delta,
            warmup_steps=smooth_warmup,
        )
        self.get_logger().info(
            f"ActionSmoother active: wheel_alpha={wheel_alpha}, arm_alpha={arm_alpha}, "
            f"wheel_max_delta={wheel_max_delta}, arm_max_delta={arm_max_delta}, "
            f"warmup={smooth_warmup}"
        )

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
        # Phase 237: /lekiwi/goal subscriber — allows dynamic goal updates at runtime
        # without restarting the VLA node.  message type: geometry_msgs/Point
        # (x, y, z).  Only x and y are used for navigation; z is ignored.
        self.goal_sub = self.create_subscription(
            Point, "/lekiwi/goal", self._on_goal, qos
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
            "  /lekiwi/goal                ← subscribe (dynamic goal update)\n"
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
        # Phase 16: Build state with goal_xy (11D) or without (9D legacy)
        # NOTE: CLIP-FM policy was trained with wheel_velocities as the 3 wheel
        # state dimensions (matching lerobot_policy_inference.py), NOT positions.
        base_state = np.concatenate([
            joints["arm_positions"],
            joints["wheel_velocities"],
        ]).astype(np.float32)
        # Append normalized goal_xy for goal-aware policy
        state = np.concatenate([base_state, self._goal_xy_norm]).astype(np.float32)

        obs = {
            "observation.images.primary": img_arr[np.newaxis, ...],   # (1,3,224,224)
            "observation.state":          state[np.newaxis, ...],     # (1,11) goal-aware
        }

        # Policy inference
        raw_action = self.policy.predict(obs)         # (9,)

        # Phase 278 FIX: Skip normalize_action for policies that output native units.
        # - Stage2/DAgger: policy.infer() outputs native units (arm_torque in Nm, wheel_speed in rad/s)
        # - normalize_action() is only for policies that output [-1,1] (ACT, diffusion, pi0, etc.)
        # - Bridge's _action_to_ctrl() handles normalization for ALL policies uniformly.
        # Policies outputting native units: stage2, stage3, dagger
        # Policies outputting [-1,1]: act, diffusion, pi0, pi0_fast, task_oriented, clip_fm, phase196, mock
        _NATIVE_UNIT_POLICIES = frozenset(["stage2", "stage3", "dagger"])
        if hasattr(self, '_policy_name') and self._policy_name in _NATIVE_UNIT_POLICIES:
            native_action = raw_action  # Skip normalize_action — bridge handles it
        else:
            native_action = normalize_action(raw_action)   # (9,) in native units

        # ── Phase 61: Action Smoother ─────────────────────────────────────────
        # Apply EMA smoothing + delta clipping to reduce jerky wheel commands
        # that destabilize URDF simulation after step 50.
        # Arms: less smoothing (alpha=0.70) to preserve manipulation responsiveness
        # Wheels: more smoothing (alpha=0.25) to reduce high-frequency oscillations
        smoothed_action = self._smoother.smooth(native_action)

        # Publish
        msg = Float64MultiArray()
        msg.data = smoothed_action.tolist()
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

    def _on_goal(self, msg: Point):
        """Handle dynamic /lekiwi/goal updates (geometry_msgs/Point).

        Updates self._goal_xy and self._goal_xy_norm in-place so that the next
        policy inference call uses the new goal immediately.  This enables
        closed-loop navigation where the goal can be updated at runtime from
        any ROS2 node (e.g. a higher-level planner, SLAM, or CTF attack).
        """
        self._goal_xy = np.array([msg.x, msg.y], dtype=np.float32)
        self._goal_xy_norm = np.clip(self._goal_xy / 1.0, -1.0, 1.0)
        self.get_logger().debug(
            f"Goal updated → ({msg.x:.3f}, {msg.y:.3f}) norm={self._goal_xy_norm}"
        )


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
