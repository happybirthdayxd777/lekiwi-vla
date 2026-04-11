#!/usr/bin/env python3
"""
LeKiwi Flow Matching Training — Real Data Version
====================================================
Trains on real data collected from the LeKiwi MuJoCo simulation.
Uses HDF5 format from collect_data.py.

Usage:
  python3 train_flow_matching_real.py --data /tmp/lekiwi_demo.h5 --epochs 50
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sim_lekiwi import LeKiwiSim


# ─── Vision Encoder ─────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  nn.ReLU(),   # 112
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 56
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),  # 28
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),# 14
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, embed_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


# ─── Flow Matching Policy ────────────────────────────────────────────────────

class FlowMatchingMLP(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 128))

        total_dim = vision_dim + state_dim + action_dim + 128  # 658
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
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


# ─── Replay Buffer from HDF5 ─────────────────────────────────────────────────

class ReplayBuffer:
    """Loads HDF5 data and samples batches."""
    def __init__(self, h5_path, batch_size=32):
        with h5py.File(h5_path, "r") as f:
            self.images  = f["images"][:]
            self.states  = f["states"][:]
            self.actions = f["actions"][:]
        print(f"  Loaded {len(self.images)} frames from {h5_path}")
        self.N = len(self.actions)
        self.bs = batch_size

    def sample(self):
        idx = np.random.randint(0, self.N, size=self.bs)
        # images: [N, H, W, C] uint8 → normalize to [0,1], convert to [C, H, W]
        imgs  = self.images[idx].astype(np.float32) / 255.0
        imgs  = torch.from_numpy(imgs.transpose(0, 3, 1, 2))  # [B, C, H, W]
        states  = torch.from_numpy(self.states[idx].astype(np.float32))
        actions = torch.from_numpy(self.actions[idx].astype(np.float32))
        return imgs, states, actions


# ─── Training ────────────────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=50, device="cpu", output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy.to(device)
    policy.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx in range(100):
            batch_img, batch_state, batch_action = replay.sample()
            batch_img    = batch_img.to(device)
            batch_state  = batch_state.to(device)
            batch_action = batch_action.to(device)

            # Timestep t
            t = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999

            # Linear interpolation (Flow Matching)
            noise = torch.randn_like(batch_action)
            x_t   = (1 - t) * batch_action + t * noise

            v_pred   = policy(batch_img, batch_state, x_t, t)
            v_target = batch_action - noise

            loss = ((v_pred - v_target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / 100
        losses.append(avg)

        if epoch % 10 == 9:
            torch.save({"epoch": epoch, "policy_state_dict": policy.state_dict()},
                       output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

    torch.save(policy.state_dict(), output_dir / "final_policy.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Flow Matching — Real Data Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Velocity Loss")
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    print(f"\n✓ Policy: {output_dir / 'final_policy.pt'}")
    return losses


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=str, default="/tmp/lekiwi_demo.h5")
    parser.add_argument("--epochs",    type=int,  default=50)
    parser.add_argument("--batch-size",type=int,  default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--hidden",    type=int,   default=512)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--output",    type=str,   default="results")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    print("\n[1] Loading replay buffer...")
    replay = ReplayBuffer(args.data, batch_size=args.batch_size)

    print("[2] Building Flow Matching policy...")
    policy = FlowMatchingPolicy(state_dim=9, action_dim=9, hidden=args.hidden)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  ✓ {n_params:,} parameters")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print("\n[3] Training on real data...")
    train(policy, optimizer, replay, epochs=args.epochs,
          device=args.device, output_dir=args.output)

    print("\n[4] Inference test...")
    policy.eval()
    with torch.no_grad():
        sim = LeKiwiSim()
        sim.reset()
        img = sim.render()
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        # Resize to 224x224 for CNN
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img.astype(np.uint8)).resize((224, 224))
        img_np  = np.array(img_pil, dtype=np.float32) / 255.0
        img_t   = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
        arm_pos = sim.data.qpos[0:6]
        wheel_vel = sim.data.qvel[0:3]
        state_t = torch.from_numpy(np.concatenate([arm_pos, wheel_vel])).float().unsqueeze(0).to(args.device)

        action = policy.infer(img_t, state_t, num_steps=4)
        print(f"  ✓ Action: {action.shape}, range=[{action.min().item():.3f}, {action.max().item():.3f}]")

    print("\n✓ All done!")

if __name__ == "__main__":
    main()