#!/usr/bin/env python3
"""
LeKiwi Flow Matching Training — Working Version
=================================================
Trains on synthetic random data. Uses LeKiwiSim just to verify the sim loads.

Usage:
  python3 train_flow_matching_lekiwi.py --epochs 50 --device cpu
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sim_lekiwi import LeKiwiSim


# ─── Vision Encoder (CNN) ───────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """Tiny conv encoder → 512-dim embedding. No pretrained weights."""
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
        return self.net(x)   # [B, 512]


# ─── Flow Matching Policy ────────────────────────────────────────────────────

class FlowMatchingMLP(nn.Module):
    """
    Simple MLP that predicts velocity v = x_0 - x_noise.
    Takes: image_embed (512) + state (9) + noisy_action (9) + timestep (1)
    Outputs: velocity (9)
    """
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim

        total_dim = vision_dim + state_dim + action_dim + 1  # 531

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )

        # Core network: state+action+timestep → hidden
        self.net = nn.Sequential(
            nn.Linear(vision_dim + state_dim + action_dim + 128, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )

        # Skip connection: noisy_action → output
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, image_embed, state, noisy_action, timestep):
        """
        image_embed:  [B, 512]
        state:        [B, 9]
        noisy_action: [B, 9]
        timestep:     [B, 1]
        Returns:      [B, 9] velocity
        """
        t_feat = self.time_mlp(timestep)                   # [B, 128]

        x = torch.cat([image_embed, state, noisy_action, t_feat], dim=-1)  # [B, 531+128=659?]
        # Wait - vision (512) + state (9) + action (9) + t_feat (128) = 658
        x = self.net(x)                                     # [B, 9]
        x = x + self.skip(noisy_action)                     # skip connection
        return x


class FlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=hidden)
        self.flow_mlp = FlowMatchingMLP(vision_dim=hidden, state_dim=state_dim,
                                       action_dim=action_dim, hidden=hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)                          # [B, 512]
        return self.flow_mlp(vis, state, noisy_action, timestep) # [B, 9]

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        """Euler ODE: start from noise, denoise in num_steps."""
        action = torch.randn(image.shape[0], self.action_dim, device=image.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=image.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_mlp(vis, state, action, t)
            action = action - dt * velocity   # Euler step

        return action


# ─── Training ────────────────────────────────────────────────────────────────

def train(policy, optimizer, epochs=50, batch_size=32, device="cpu", output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy.to(device)
    policy.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(50):
            # Synthetic batch (replace with real replay buffer)
            batch_img    = torch.randn(batch_size, 3, 224, 224, device=device)
            batch_state  = torch.randn(batch_size, 9, device=device)
            batch_action = torch.randn(batch_size, 9, device=device)

            # Timestep t ~ Beta(1.5, 1) → more weight near 0 and 1
            t = (torch.rand(batch_size, 1, device=device) ** 1.5) * 0.999

            # Linear interpolation: x_t = (1-t)*x_0 + t*noise
            noise = torch.randn_like(batch_action)
            x_t   = (1 - t) * batch_action + t * noise

            # Flow Matching target: velocity = x_0 - noise
            v_pred   = policy(batch_img, batch_state, x_t, t)
            v_target = batch_action - noise

            loss = ((v_pred - v_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg = epoch_loss / 50
        losses.append(avg)

        if epoch % 10 == 9:
            torch.save({
                "epoch": epoch,
                "policy_state_dict": policy.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f}")

    torch.save(policy.state_dict(), output_dir / "final_policy.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Flow Matching Training — LeKiwi")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Velocity Loss")
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    print(f"\n✓ Final policy: {output_dir / 'final_policy.pt'}")
    print(f"✓ Loss curve:   {output_dir / 'loss_curve.png'}")
    return losses


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--hidden",     type=int,   default=512)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--output",      type=str,   default="results")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Verify sim loads
    print("\n[1] Loading LeKiwi simulation...")
    sim = LeKiwiSim()
    sim.reset()
    img = sim.render()
    print(f"  ✓ Camera OK: {img.size} | State dim: {sim.action_dim}")

    # Build policy
    print("[2] Building Flow Matching policy...")
    policy = FlowMatchingPolicy(state_dim=9, action_dim=9, hidden=args.hidden)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  ✓ Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Train
    print("\n[3] Training Flow Matching (synthetic data)...")
    train(policy, optimizer, epochs=args.epochs, batch_size=args.batch_size,
          device=args.device, output_dir=args.output)

    # Quick inference test
    print("\n[4] 4-step Euler inference test...")
    policy.eval()
    with torch.no_grad():
        test_img   = torch.randn(1, 3, 224, 224, device=args.device)
        test_state = torch.randn(1, 9, device=args.device)
        action = policy.infer(test_img, test_state, num_steps=4)
        print(f"  ✓ Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    print("\n✓ All done!")

if __name__ == "__main__":
    main()