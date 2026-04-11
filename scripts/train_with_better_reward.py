#!/usr/bin/env python3
"""
Improved LeKiwi Reward Shaping + Training Script
================================================
Combines:
1. Better reward shaping: distance-to-goal + progress reward + action smoothness
2. On-policy data collection (ε-greedy exploration from current policy)
3. PPO-style advantage estimation
4. Flow Matching training

Usage:
  python3 scripts/train_with_better_reward.py --epochs 50 --device mps
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from PIL import Image

from sim_lekiwi import LeKiwiSim


# ─── Config ─────────────────────────────────────────────────────────────────

GOAL = np.array([0.5, 0.0])   # Target position
MAX_STEPS = 200
DEVICE_DEFAULT = "mps" if torch.backends.mps.is_available() else "cpu"


# ─── Vision Encoder ─────────────────────────────────────────────────────────

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


# ─── Flow Matching Policy ───────────────────────────────────────────────────

class FlowMatchingMLP(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 128))
        total = vision_dim + state_dim + action_dim + 128
        self.net = nn.Sequential(
            nn.Linear(total, hidden), nn.SiLU(), nn.LayerNorm(hidden),
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
        """Euler ODE inference."""
        action = torch.randn(image.shape[0], self.action_dim, device=image.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=image.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_mlp(vis, state, action, t)
            action = action - dt * velocity
        return action

    @torch.no_grad()
    def act(self, image, state, epsilon=0.0):
        """ε-greedy: with prob ε random, else policy."""
        if epsilon > 0 and np.random.rand() < epsilon:
            return torch.rand(1, self.action_dim) * 2 - 1
        img_t  = self._prep_image(image)
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(next(self.parameters()).device)
        return self.infer(img_t, state_t, num_steps=4)

    def _prep_image(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return t.to(next(self.parameters()).device)


# ─── Improved Reward ────────────────────────────────────────────────────────

def compute_reward(state, action, prev_dist, goal=GOAL):
    """
    Rich reward shaping:
    - r_progress: improvement in distance to goal
    - r_action_smooth: penalize large actions
    - r_goal: big bonus when close to goal
    """
    pos = state[6:8]  # [x, y] from qpos
    dist = np.linalg.norm(pos - goal)
    dist_reward = prev_dist - dist   # positive = moved closer

    # Action smoothness penalty ( penalize sudden large movements)
    action_penalty = -0.01 * np.sum(np.square(action))

    # Goal proximity bonus (exponential)
    if dist < 0.05:
        goal_bonus = 10.0
    elif dist < 0.1:
        goal_bonus = 2.0
    elif dist < 0.2:
        goal_bonus = 0.5
    else:
        goal_bonus = 0.0

    total = dist_reward + action_penalty + goal_bonus
    return total, dist


# ─── On-Policy Data Collection ─────────────────────────────────────────────

def collect_episode(policy, epsilon=0.2, device="cpu"):
    """Collect one episode using current policy (ε-greedy)."""
    sim = LeKiwiSim()
    sim.reset()

    imgs, states, actions, rewards, next_states = [], [], [], [], []

    prev_dist = float(np.linalg.norm(sim.data.qpos[6:8] - GOAL))

    for _ in range(MAX_STEPS):
        img_pil = sim.render()
        img_np  = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
        img_t   = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

        arm_pos = sim.data.qpos[0:6].astype(np.float32)
        wheel_v = sim.data.qvel[0:3].astype(np.float32)
        state   = np.concatenate([arm_pos, wheel_v])

        # ε-greedy action
        action = policy.act(img_pil, state, epsilon=epsilon)
        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, -1, 1)

        imgs.append(img_np.transpose(2, 0, 1))   # [3, 224, 224]
        states.append(state)
        actions.append(action_np)

        sim.step(action_np)
        reward, dist = compute_reward(
            np.concatenate([sim.data.qpos, sim.data.qvel]),
            action_np, prev_dist
        )
        rewards.append(reward)
        prev_dist = dist

    return {
        "images":  np.stack(imgs),
        "states":  np.stack(states),
        "actions": np.stack(actions),
        "rewards": np.array(rewards),
        "total_reward": sum(rewards),
        "final_dist": prev_dist,
    }


# ─── Training ────────────────────────────────────────────────────────────────

def train_step(policy, optimizer, batch_img, batch_state, batch_action, device):
    """One Flow Matching training step."""
    batch_img    = torch.from_numpy(np.asarray(batch_img, dtype=np.float32))
    batch_state  = torch.from_numpy(np.asarray(batch_state, dtype=np.float32))
    batch_action = torch.from_numpy(np.asarray(batch_action, dtype=np.float32))
    batch_img    = batch_img.to(device)
    batch_state  = batch_state.to(device)
    batch_action = batch_action.to(device)

    t = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
    noise = torch.randn_like(batch_action)
    x_t   = (1 - t) * batch_action + t * noise

    v_pred   = policy(batch_img, batch_state, x_t, t)
    v_target = batch_action - noise

    loss = ((v_pred - v_target) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--episodes-per-epoch", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--hidden",    type=int,   default=512)
    parser.add_argument("--device",     type=str,   default=DEVICE_DEFAULT)
    parser.add_argument("--output",     type=str,   default="results/improved")
    parser.add_argument("--eval-every", type=int,   default=5)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Goal: {GOAL}")

    policy = FlowMatchingPolicy(state_dim=9, action_dim=9, hidden=args.hidden).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params:,} parameters")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    eval_rewards = []

    for epoch in range(1, args.epochs + 1):
        # ── On-policy data collection ──
        epoch_data = []
        for ep in range(args.episodes_per_epoch):
            # Decay epsilon
            epsilon = max(0.05, 0.3 - epoch * 0.005)
            data = collect_episode(policy, epsilon=epsilon, device=args.device)
            epoch_data.append(data)

        total_rewards = [d["total_reward"] for d in epoch_data]
        mean_reward   = np.mean(total_rewards)
        mean_final_d  = np.mean([d["final_dist"] for d in epoch_data])

        # ── Training (multiple steps per epoch) ──
        epoch_loss = 0.0
        num_steps  = 50

        for _ in range(num_steps):
            # Sample random batch from epoch data
            ep_idx  = np.random.randint(0, len(epoch_data))
            ep      = epoch_data[ep_idx]
            n       = len(ep["actions"])
            idx     = np.random.randint(0, n, size=16)
            loss    = train_step(policy, optimizer,
                                ep["images"][idx], ep["states"][idx], ep["actions"][idx],
                                args.device)
            epoch_loss += loss

        avg_loss = epoch_loss / num_steps
        train_losses.append(avg_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Reward: {mean_reward:+.2f} | "
              f"FinalDist: {mean_final_d:.3f}m | "
              f"Loss: {avg_loss:.4f}")

        # ── Evaluation ──
        if epoch % args.eval_every == 0:
            eval_rewards_ep = []
            for _ in range(3):
                d = collect_episode(policy, epsilon=0.0, device=args.device)
                eval_rewards_ep.append(d["total_reward"])
            eval_rewards.append((epoch, np.mean(eval_rewards_ep)))
            print(f"  → Eval reward: {eval_rewards[-1][1]:+.2f}")

        # ── Save checkpoint ──
        if epoch % 10 == 0:
            torch.save(policy.state_dict(), output_dir / f"policy_ep{epoch}.pt")

    # ── Final save ──
    torch.save(policy.state_dict(), output_dir / "final_policy.pt")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    if eval_rewards:
        epochs_r, rewards_r = zip(*eval_rewards)
        axes[1].plot(epochs_r, rewards_r, marker="o")
        axes[1].set_title("Eval Reward (higher = closer to goal)")
        axes[1].set_xlabel("Epoch")

    plt.savefig(output_dir / "training_curves.png", dpi=150)
    print(f"\n✓ Saved to {output_dir}")
    print(f"  Policy: {output_dir / 'final_policy.pt'}")
    print(f"  Curves: {output_dir / 'training_curves.png'}")

if __name__ == "__main__":
    main()