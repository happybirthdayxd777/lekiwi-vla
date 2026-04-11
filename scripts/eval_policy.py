#!/usr/bin/env python3
"""
Evaluate a trained Flow Matching policy on LeKiwi simulation.
Compares trained policy vs random policy baseline.

Usage:
  python3 scripts/eval_policy.py --policy flow_matching --checkpoint /tmp/fm_real/final_policy.pt --episodes 10
  python3 scripts/eval_policy.py --policy random --episodes 10
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from sim_lekiwi import LeKiwiSim


# ─── Policy Network (must match training architecture) ──────────────────────

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, embed_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class FlowMatchingMLP(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 128))
        self.net = nn.Sequential(
            nn.Linear(vision_dim + state_dim + action_dim + 128, hidden),
            nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, image_embed, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([image_embed, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class FlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=hidden)
        self.flow_mlp = FlowMatchingMLP(hidden, state_dim, action_dim, hidden)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_mlp(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=image.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=image.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_mlp(vis, state, action, t)
            action = action - dt * velocity
        return action


class RandomPolicy:
    """Baseline: random action uniformly in [-1, 1]."""
    def __init__(self, action_dim=9):
        self.action_dim = action_dim
    def infer(self, image, state, num_steps=4):
        return torch.rand(1, self.action_dim) * 2 - 1


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(policy, device, episodes=10, max_steps=200):
    """Run episodes and collect metrics."""
    sim = LeKiwiSim()
    all_rewards = []
    all_distances = []

    for ep in range(episodes):
        sim.reset()
        total_reward = 0.0
        start_pos = sim.data.qpos[6:9].copy()  # base position [x, y, theta]

        for step in range(max_steps):
            # Get observation
            img_pil = sim.render()
            img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t  = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            arm_pos  = sim.data.qpos[0:6]
            wheel_v  = sim.data.qvel[0:3]
            state_t  = torch.from_numpy(np.concatenate([arm_pos, wheel_v])).float().unsqueeze(0).to(device)

            # Get action
            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]

            # Denormalize to [-1, 1] range (already in that range)
            action_clipped = np.clip(action_np, -1, 1)

            # Step — standalone sim returns dict with reward via get_reward()
            sim.step(action_clipped)
            reward = sim.get_reward()  # standalone: -dist - 0.01*effort
            total_reward += reward

        # Final distance
        end_pos = sim.data.qpos[6:9]
        dist = np.linalg.norm(end_pos[:2] - start_pos[:2])

        all_rewards.append(total_reward)
        all_distances.append(dist)
        print(f"  Episode {ep+1:2d}: reward={total_reward:+.3f}, distance={dist:.3f}m")

    return {
        "mean_reward":  np.mean(all_rewards),
        "std_reward":   np.std(all_rewards),
        "mean_distance": np.mean(all_distances),
        "std_distance":  np.std(all_distances),
        "all_rewards":   all_rewards,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",    type=str,   default="flow_matching")
    parser.add_argument("--checkpoint",type=str,   default="/tmp/fm_real/final_policy.pt")
    parser.add_argument("--episodes",  type=int,   default=10)
    parser.add_argument("--device",    type=str,   default="mps")
    parser.add_argument("--hidden",    type=int,   default=512)
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"  Evaluating: {args.policy}")
    print(f"  Episodes: {args.episodes} | Device: {args.device}")
    print(f"=" * 60)

    # Load policy
    if args.policy == "flow_matching":
        print(f"\n[1] Loading trained policy from {args.checkpoint}...")
        policy = FlowMatchingPolicy(state_dim=9, action_dim=9, hidden=args.hidden)
        state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        policy.load_state_dict(state_dict)
        policy.to(args.device)
        policy.eval()
        print(f"  ✓ Loaded (device={args.device})")
    elif args.policy == "random":
        print(f"\n[1] Using random baseline policy")
        policy = RandomPolicy(action_dim=9)
        device = "cpu"
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Evaluate
    print(f"\n[2] Running {args.episodes} episodes...")
    device = args.device if args.policy == "flow_matching" else "cpu"
    metrics = evaluate(policy, device, episodes=args.episodes)

    print(f"\n[3] Results:")
    print(f"  Mean reward:    {metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Mean distance:  {metrics['mean_distance']:.3f} ± {metrics['std_distance']:.3f}m")
    print(f"  Best reward:    {max(metrics['all_rewards']):+.3f}")
    print(f"  Worst reward:   {min(metrics['all_rewards']):+.3f}")

    print("\n✓ Evaluation complete")

if __name__ == "__main__":
    main()