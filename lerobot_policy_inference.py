#!/usr/bin/env python3
"""
LeRobot Policy Inference for LeKiwi
===================================
Loads a LeRobot VLA policy and runs it on the LeKiwi MuJoCo simulation.

Supported policies: pi0, pi0_fast, act, diffusion, smolvla, tdmpc, mock
Usage:
    python3 lerobot_policy_inference.py --policy mock --steps 50
    python3 lerobot_policy_inference.py --policy pi0 --pretrained <hf_repo>
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy
from lerobot.types import PolicyAction

# ─── Policy Config Factories (import only what we need, avoiding __init__.py)

# ─── LeKiwi Action Normalization ────────────────────────────────────────────

LEKIWI_ARM_LIMITS = np.array([
    [-3.14, 3.14],   # j0 shoulder pan
    [-1.57, 1.57],   # j1 shoulder lift
    [-1.57, 1.57],   # j2 elbow
    [-1.57, 1.57],   # j3 wrist flex
    [-3.14, 3.14],   # j4 wrist roll
    [0.00,  0.04],   # j5 gripper slide
], dtype=np.float32)

LEKIWI_WHEEL_LIMITS = np.array([
    [-5.0, 5.0],   # w1
    [-5.0, 5.0],   # w2
    [-5.0, 5.0],   # w3
], dtype=np.float32)


def normalize_action(raw_action: np.ndarray) -> np.ndarray:
    """Policy (-1..1) → LeKiwi native units."""
    arm    = raw_action[:6]
    wheel  = raw_action[6:9]
    arm_n  = LEKIWI_ARM_LIMITS[:, 0] + (arm + 1) / 2 * (
        LEKIWI_ARM_LIMITS[:, 1] - LEKIWI_ARM_LIMITS[:, 0])
    wheel_n = LEKIWI_WHEEL_LIMITS[:, 0] + (wheel + 1) / 2 * (
        LEKIWI_WHEEL_LIMITS[:, 1] - LEKIWI_WHEEL_LIMITS[:, 0])
    return np.concatenate([arm_n, wheel_n]).astype(np.float32)


# ─── Policy Config Factory ────────────────────────────────────────────────────

def _make_pi0_config(device):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    return PI0Config(
        max_action_dim=9,
        max_state_dim=32,
        num_inference_steps=10,
    )

def _make_pi0_fast_config(device):
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    return PI0FastConfig(max_action_dim=9, max_state_dim=32, num_inference_steps=5)

def _make_act_config(device):
    from lerobot.policies.act.configuration_act import ACTConfig
    return ACTConfig(name="act", device=device, output_dir="~/act_lekiwi")

def _make_diffusion_config(device):
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    return DiffusionConfig(name="diffusion", device=device, output_dir="~/diffusion_lekiwi")

def _make_smolvla_config(device):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    return SmolVLAConfig(action_dim=9, state_dim=32)

def _make_tdmpc_config(device):
    from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
    return TDMPCConfig(device=device, action_dim=9, model_dir="~/tdmpc_lekiwi")


_POLICY_CONFIG_FACTORIES = {
    "pi0":        _make_pi0_config,
    "pi0_fast":   _make_pi0_fast_config,
    "act":        _make_act_config,
    "diffusion":  _make_diffusion_config,
    "smolvla":    _make_smolvla_config,
    "tdmpc":      _make_tdmpc_config,
}


# ─── Observation Adapter ──────────────────────────────────────────────────────

def make_lekiwi_observation(
    image: Image.Image,
    arm_positions: np.ndarray,
    wheel_velocities: np.ndarray,
    goal_xy: Optional[np.ndarray] = None,
) -> dict:
    """Convert LeKiwi state → LeRobot observation dict.

    Phase 16 Goal-Aware:
      If goal_xy is provided, state = [arm_pos(6), wheel_vel(3), goal_xy(2)] = 11D
      Otherwise state = [arm_pos(6), wheel_vel(3)] = 9D (legacy)
    """
    img_resized = image.resize((224, 224), Image.BILINEAR)
    img_tensor = torch.from_numpy(
        np.array(img_resized).transpose(2, 0, 1)
    ).float() / 255.0

    base_state = np.concatenate([arm_positions, wheel_velocities]).astype(np.float32)
    if goal_xy is not None:
        # Normalize goal to [-1, 1] based on 1m arena radius
        goal_norm = np.clip(goal_xy / 1.0, -1.0, 1.0).astype(np.float32)
        state = np.concatenate([base_state, goal_norm]).astype(np.float32)
    else:
        state = base_state

    return {
        "observation.images.primary": img_tensor.unsqueeze(0),   # (1,3,224,224)
        "observation.state":          torch.from_numpy(state).unsqueeze(0),  # (1,9) or (1,11)
    }


# ─── Policy Runner ────────────────────────────────────────────────────────────

class LeRobotPolicyRunner:
    """LeRobot VLA policy inference on LeKiwi."""

    def __init__(self, policy_name: str, device: Optional[str] = None):
        self.policy_name = policy_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        self.config = None

    def load(self, pretrained_path: Optional[str] = None):
        factory = _POLICY_CONFIG_FACTORIES.get(self.policy_name)
        if factory is None:
            raise ValueError(f"Unknown policy: {self.policy_name}")
        self.config = factory(self.device)

        if pretrained_path:
            self.config.from_pretrained = pretrained_path

        self.policy = make_policy(self.config)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        print(f"[LeRobotPolicy] Loaded '{self.policy_name}' on {self.device}")

    def predict(self, obs: dict) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("Call load() first")
        with torch.no_grad():
            obs_dev = {k: v.to(self.device) for k, v in obs.items()}
            output: PolicyAction = self.policy.predict(obs_dev)
            # PolicyAction is a tensor (chunk_size, action_dim)
            action_01 = output.cpu().numpy()[0]
        return normalize_action(action_01)

    def reset(self):
        if self.policy:
            self.policy.reset()


# ─── Mock Policy ──────────────────────────────────────────────────────────────

class MockPolicyRunner:
    """Always-works mock: sinusoidal arm + random base. No GPU needed."""

    def predict(self, obs: dict) -> np.ndarray:
        t = time.time()
        action = np.zeros(9, dtype=np.float32)
        action[0] = 0.5 * np.sin(t * 2 * np.pi)   # shoulder pan
        action[1] = 0.3 * np.sin(t * 4 * np.pi)   # shoulder lift
        action[2] = -0.3 * np.sin(t * 4 * np.pi)  # elbow
        action[3] = 0.1
        action[4] = 0.0
        action[5] = 0.02
        action[6] = 0.1 * np.sin(t * np.pi)  # wheels
        action[7] = 0.1 * np.sin(t * np.pi)
        action[8] = 0.1 * np.sin(t * np.pi)
        return action

    def reset(self):
        pass


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_with_sim(
    policy_name: str,
    pretrained_path: Optional[str] = None,
    device: Optional[str] = None,
    n_steps: int = 100,
):
    from sim_lekiwi import LeKiwiSim

    print("=" * 60)
    print(f"  LeRobot Policy Demo — {policy_name} × LeKiwi Sim")
    print("=" * 60)

    use_mock = (policy_name == "mock")
    runner: Optional[LeRobotPolicyRunner] = None

    if use_mock:
        print("[INFO] Using mock policy")
        runner = MockPolicyRunner()
    else:
        try:
            runner = LeRobotPolicyRunner(policy_name, device=device)
            runner.load(pretrained_path=pretrained_path)
        except Exception as e:
            print(f"\n[WARNING] Could not load '{policy_name}': {e}")
            print("[INFO] Falling back to mock policy.")
            runner = MockPolicyRunner()
            use_mock = True

    sim = LeKiwiSim()
    sim.reset()

    print(f"\nRunning {n_steps} steps...\n")
    total_reward = 0.0

    for step in range(n_steps):
        img = sim.render()
        obs = sim._obs()

        if use_mock:
            action = runner.predict(obs)
        else:
            policy_obs = make_lekiwi_observation(
                image=img,
                arm_positions=obs["arm_positions"],
                wheel_velocities=obs["wheel_velocities"],
            )
            action = runner.predict(policy_obs)

        obs = sim.step(action)
        reward = sim.get_reward()
        total_reward += reward

        if step % 25 == 0:
            print(f"  step {step:4d} | reward={reward:+.3f} | "
                  f"arm[0]={action[0]:+.3f} | "
                  f"base=({obs['base_position'][0]:+.3f}, {obs['base_position'][1]:+.3f})")

    print(f"\nTotal reward: {total_reward:.3f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LeRobot VLA on LeKiwi")
    parser.add_argument("--policy", required=True,
                        choices=["act","diffusion","pi0","pi0_fast","smolvla","tdmpc","mock"])
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--device", choices=["cuda","cpu","mps"], default=None)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    demo_with_sim(args.policy, args.pretrained, args.device, args.steps)


if __name__ == "__main__":
    main()