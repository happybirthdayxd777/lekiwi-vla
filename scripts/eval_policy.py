#!/usr/bin/env python3
"""
LeKiwi Policy Evaluation Script
===============================
Evaluates a trained LeRobot policy on the LeKiwi simulation or real robot.

Supports: ACT, Diffusion, Multi-Task DiT (Flow Matching), GR00T, SmolVLA, pi0

Usage:
  # Sim: mock policy
  python3 scripts/eval_policy.py --policy mock --sim

  # Sim: LeRobot Multi-Task DiT (Flow Matching)
  python3 scripts/eval_policy.py \
    --policy multi_task_dit \
    --checkpoint results/lekiwi_fm/checkpoints/latest \
    --dataset <hf_repo> \
    --sim

  # Real robot: GR00T-N1.5
  python3 scripts/eval_policy.py \
    --policy groot \
    --groot-model nvidia/GR00T-N1.5-3B \
    --robot-ip 172.18.134.136 \
    --task "move forward"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import get_policy_class, make_policy
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.factory import make_pre_post_processors

from sim_lekiwi import LeKiwiSim


def make_policy_for_lekiwi(policy_name, dataset_stats, device, checkpoint_path=None):
    """Instantiate a LeRobot policy for LeKiwi (9-DOF)."""

    # Standard LeKiwi feature shapes
    input_features = {
        "observation.images.primary": FeatureType.create("image", shape=(3, 224, 224)),
        "observation.state":          FeatureType.create("state",  shape=(9,)),
    }
    output_features = {
        "action": FeatureType.create("action", shape=(9,)),
    }

    if policy_name == "act":
        cfg = ACTConfig(input_features=input_features, output_features=output_features)
        cfg.chunk_size = 8
        cfg.n_obs_steps = 2

    elif policy_name == "diffusion":
        cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
        cfg.n_obs_steps = 2
        cfg.horizon = 16
        cfg.n_action_steps = 8

    elif policy_name == "multi_task_dit":
        cfg = MultiTaskDiTConfig(
            input_features=input_features,
            output_features=output_features,
            objective="flow_matching",
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            hidden_dim=512,
            num_layers=6,
            num_heads=8,
            vision_encoder_name="openai/clip-vit-base-patch16",
            image_crop_shape=[224, 224],
        )
        cfg.drop_n_last_frames = 7

    elif policy_name == "groot":
        cfg = GrootConfig(
            input_features=input_features,
            output_features=output_features,
            base_model_path="nvidia/GR00T-N1.5-3B",
            tune_llm=False,
            tune_visual=False,
            tune_projector=True,
            tune_diffusion_model=True,
            n_obs_steps=1,
            chunk_size=50,
            n_action_steps=50,
        )

    elif policy_name == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            action_dim=9,
            state_dim=9,
        )

    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    policy = make_policy(cfg)

    if checkpoint_path:
        print(f"  Loading checkpoint: {checkpoint_path}")
        # LeRobot's from_pretrained pattern
        if hasattr(policy, "from_pretrained"):
            policy = policy.__class__.from_pretrained(checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu")
            policy.load_state_dict(state)

    policy.to(device)
    policy.eval()

    # Attach dataset stats for normalization
    _, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

    return policy, postprocessor


def normalize_01_to_lekiwi(action_01):
    """Convert policy (-1..1) to LeKiwi native units."""
    ARM_LIMITS = np.array([
        [-3.14, 3.14], [-1.57, 1.57], [-1.57, 1.57],
        [-1.57, 1.57], [-3.14, 3.14], [0.00, 0.04],
    ], dtype=np.float32)
    WHEEL_LIMITS = np.array([[-5.0, 5.0]] * 3, dtype=np.float32)

    arm   = action_01[:6]
    wheel = action_01[6:9]

    arm_n   = ARM_LIMITS[:,0] + (arm   + 1) / 2 * (ARM_LIMITS[:,1] - ARM_LIMITS[:,0])
    wheel_n = WHEEL_LIMITS[:,0] + (wheel + 1) / 2 * (WHEEL_LIMITS[:,1] - WHEEL_LIMITS[:,0])

    return np.concatenate([arm_n, wheel_n]).astype(np.float32)


def run_sim_eval(policy, postprocessor, policy_name, num_steps=200):
    """Run evaluation on LeKiwi simulation."""
    from sim_lekiwi import LeKiwiSim

    sim = LeKiwiSim()
    sim.reset()

    print(f"\n{'='*60}")
    print(f"  Sim Evaluation — {policy_name}")
    print(f"{'='*60}")

    total_reward = 0.0
    device = next(policy.parameters()).device

    for step in range(num_steps):
        img = sim.render()
        obs = sim._obs()

        # Prepare observation dict
        img_resized = img.resize((224, 224), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img_resized).transpose(2,0,1)).float() / 255.0
        state_t = torch.from_numpy(np.concatenate([
            obs["arm_positions"], obs["wheel_velocities"]])).float()

        policy_obs = {
            "observation.images.primary": img_t.unsqueeze(0).to(device),
            "observation.state":          state_t.unsqueeze(0).to(device),
        }

        # Run policy
        with torch.no_grad():
            if policy_name in ("act", "diffusion", "multi_task_dit", "smolvla"):
                output = policy(policy_obs)
                if isinstance(output, dict):
                    action_01 = output["action"].cpu().numpy()[0]
                else:
                    action_01 = output.cpu().numpy()[0]
            elif policy_name == "groot":
                output = policy(policy_obs)
                action_01 = output.cpu().numpy()[0]
            elif policy_name == "mock":
                t = step / 50.0
                action_01 = np.zeros(9, dtype=np.float32)
                action_01[0] = 0.5 * np.sin(t * 2 * np.pi)
                action_01[1] = 0.3 * np.sin(t * 4 * np.pi)

        # Denormalize
        action_lekiwi = normalize_01_to_lekiwi(action_01)

        # Step sim
        obs_out = sim.step(action_lekiwi)
        reward = sim.get_reward()
        total_reward += reward

        if step % 40 == 0:
            print(f"  step {step:4d} | reward: {reward:+.3f} | "
                  f"arm[0]: {action_lekiwi[0]:+.3f} | "
                  f"base: ({obs_out['base_position'][0]:+.2f}, {obs_out['base_position'][1]:+.2f})")

    print(f"\n  Total reward: {total_reward:.3f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeKiwi policy")
    parser.add_argument("--policy", required=True,
                        choices=["act","diffusion","multi_task_dit","groot","smolvla","mock"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--groot-model", type=str, default="nvidia/GR00T-N1.5-3B",
                        help="GR00T model ID (only for groot policy)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sim", action="store_true", help="Run in simulation")
    parser.add_argument("--robot-ip", type=str, default="172.18.134.136",
                        help="LeKiwi robot IP (for real robot)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Policy: {args.policy} | Device: {device}")

    # Load dataset stats
    print("Loading dataset...")
    dataset_meta = LeRobotDatasetMetadata(args.dataset)
    stats = dataset_meta.stats

    # Make policy
    print("Loading policy...")
    policy, postprocessor = make_policy_for_lekiwi(
        args.policy, stats, device, args.checkpoint)

    if args.sim:
        run_sim_eval(policy, postprocessor, args.policy, args.steps)
    else:
        print("[Real robot mode] Connect to robot first, then run...")
        print(f"  python3 -m lerobot.robots.lekiwi.lekiwi_host --robot.id=lekiwi")
        print("  Then press ENTER to start evaluation...")
        input()


if __name__ == "__main__":
    main()