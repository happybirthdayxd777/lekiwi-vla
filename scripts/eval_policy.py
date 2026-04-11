#!/usr/bin/env python3
"""
Evaluate trained policies on LeKiwi simulation.
Supports both SimpleCNN-Flow Matching and CLIP-Flow Matching architectures.

Usage:
  # SimpleCNN-Flow Matching
  python3 scripts/eval_policy.py --arch simple_cnn_fm --checkpoint /tmp/fm_real/final_policy.pt --episodes 10

  # CLIP-Flow Matching
  python3 scripts/eval_policy.py --arch clip_fm --checkpoint /tmp/clip_fm_test/final_policy.pt --episodes 10

  # Random baseline
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


# ─── SimpleCNN Vision Encoder (from train_flow_matching_real.py) ──────────────

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
        total_dim = vision_dim + state_dim + action_dim + 128  # 658
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class SimpleCNNFlowMatchingPolicy(nn.Module):
    """SimpleCNN + Flow Matching (8M params)."""
    def __init__(self, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=hidden)
        self.flow_head = FlowMatchingMLP(hidden, state_dim, action_dim, hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=next(self.parameters()).device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=image.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── CLIP Vision Encoder ─────────────────────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """CLIP ViT-B/32 frozen encoder → 768 → 512."""
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        for p in self.clip.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(768, 512).to(device)

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            out = self.clip.vision_model(pixel_values=pixel_values)
            pooled = out.pooler_output
        return self.proj(pooled)


class CLIPFlowMatchingHead(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        total_dim = vision_dim + state_dim + action_dim + 256  # 786
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class CLIPFlowMatchingPolicy(nn.Module):
    """CLIP ViT-B/32 + Flow Matching (151M frozen + 970K trainable)."""
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = CLIPFlowMatchingHead(vision_dim=hidden, state_dim=state_dim,
                                              action_dim=action_dim, hidden=hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device = device

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── Random Baseline ─────────────────────────────────────────────────────────

class RandomPolicy:
    def __init__(self, action_dim=9):
        self.action_dim = action_dim
    def infer(self, image, state, num_steps=4):
        return torch.rand(1, self.action_dim) * 2 - 1


# ─── Evaluation ──────────────────────────────────────────────────────────────

def make_policy(arch, checkpoint, device):
    """Load a policy from checkpoint based on architecture."""
    if arch == "simple_cnn_fm":
        policy = SimpleCNNFlowMatchingPolicy(state_dim=9, action_dim=9)
    elif arch == "clip_fm":
        policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, device=device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("policy_state_dict", ckpt)   # handle both raw sd and wrapped ckpt
    policy.load_state_dict(sd)
    policy.to(device)
    policy.eval()
    return policy


def evaluate(policy, device, episodes=10, max_steps=200):
    """Run episodes and collect metrics."""
    from sim_lekiwi import LeKiwiSim
    sim = LeKiwiSim()
    all_rewards = []
    all_distances = []

    for ep in range(episodes):
        sim.reset()
        total_reward = 0.0
        start_pos = sim.data.qpos[6:9].copy()

        for step in range(max_steps):
            img_pil = sim.render()
            img_np  = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t   = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            arm_pos = sim.data.qpos[0:6]
            wheel_v = sim.data.qvel[0:3]
            state_t = torch.from_numpy(np.concatenate([arm_pos, wheel_v])).float().unsqueeze(0).to(device)

            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = np.clip(action.cpu().numpy()[0], -1, 1)

            sim.step(action_np)
            reward = sim.get_reward()
            total_reward += reward

        end_pos = sim.data.qpos[6:9]
        dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
        all_rewards.append(total_reward)
        all_distances.append(dist)
        print(f"  Episode {ep+1:2d}: reward={total_reward:+.3f}, distance={dist:.3f}m")

    return {
        "mean_reward":    np.mean(all_rewards),
        "std_reward":     np.std(all_rewards),
        "mean_distance":  np.mean(all_distances),
        "std_distance":   np.std(all_distances),
        "all_rewards":    all_rewards,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",     type=str,   default=None, help="'random' for baseline")
    parser.add_argument("--arch",       type=str,   default="simple_cnn_fm",
                        choices=["simple_cnn_fm", "clip_fm"])
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--episodes",   type=int,   default=10)
    parser.add_argument("--device",     type=str,   default="mps")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    if args.policy == "random":
        print(f"  Policy: RANDOM BASELINE")
        policy = RandomPolicy(action_dim=9)
        device = "cpu"
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for trained policies")
        print(f"  Architecture: {args.arch}")
        print(f"  Checkpoint:  {args.checkpoint}")
        policy = make_policy(args.arch, args.checkpoint, args.device)
        device = args.device
    print(f"  Episodes: {args.episodes} | Device: {device}")
    print(f"{'='*60}\n")

    print(f"[Running {args.episodes} episodes...]")
    metrics = evaluate(policy, device, episodes=args.episodes)

    print(f"\n{'='*40}")
    print(f"  Mean reward:   {metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Mean distance: {metrics['mean_distance']:.3f} ± {metrics['std_distance']:.3f}m")
    print(f"  Best reward:   {max(metrics['all_rewards']):+.3f}")
    print(f"  Worst reward:  {min(metrics['all_rewards']):+.3f}")
    print(f"{'='*40}")
    print("✓ Evaluation complete")

if __name__ == "__main__":
    main()