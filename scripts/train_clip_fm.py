#!/usr/bin/env python3
"""
LeKiwi Flow Matching + CLIP Vision Encoder
==========================================
CLIP ViT-B/32 (151M params, frozen) → 768-dim visual features → Flow Matching.

Improves over SimpleCNN (5M) by leveraging pretrained visual representations.
CLIP is frozen (no gradient) so training is fast.

Usage:
  python3 train_clip_fm.py --data /tmp/lekiwi_demo_224.h5 --epochs 50 --device mps
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
from PIL import Image
import time

from sim_lekiwi import LeKiwiSim

# ─── CLIP Vision Encoder (pretrained, frozen) ───────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """CLIP ViT-B/32 frozen encoder → 768-dim visual tokens [B, 50, 768]."""
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        print("[INFO] Loading CLIP ViT-B/32 (pretrained, frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

        # Freeze CLIP entirely
        for p in self.clip.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in self.clip.parameters())
        print(f"[INFO] CLIP loaded: {n_params:,} params (frozen)")

        # Projection: 768 → hidden_dim
        self.proj = nn.Linear(768, 512).to(device)

    def forward(self, images):
        """
        images: [B, 3, 224, 224] in [0, 1] range
        Returns: [B, 512] pooled visual features
        """
        # images are [0,1] float, CLIP expects [0,255] uint8
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            # pooled: [B, 768]
            pooled = outputs.pooler_output
        return self.proj(pooled)   # [B, 512]


# ─── Flow Matching Policy ────────────────────────────────────────────────────

class FlowMatchingHead(nn.Module):
    """Flow Matching MLP: predicts velocity = x_0 - x_noise."""
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )
        total_dim = vision_dim + state_dim + action_dim + 256  # 786
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)   # [B, 256]
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)  # [B, 786]
        return self.net(x) + self.skip(noisy_action)


class CLIPFlowMatchingPolicy(nn.Module):
    """
    Full VLA policy: CLIP vision encoder + Flow Matching action head.
    Vision: frozen CLIP ViT-B/32 (151M)
    Action: Flow Matching MLP (8M trainable)
    """
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead(vision_dim=hidden, state_dim=state_dim,
                                          action_dim=action_dim, hidden=hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device = device

        n_vision   = sum(p.numel() for p in self.vision_encoder.parameters())
        n_flow     = sum(p.numel() for p in self.flow_head.parameters())
        n_trainable = sum(p.numel() for p in self.flow_head.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {n_vision + n_flow:,} (vision frozen, {n_trainable:,} trainable)")

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        """4-step Euler ODE inference."""
        action = torch.randn(image.shape[0], self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, h5_path, batch_size=16):
        with h5py.File(h5_path, "r") as f:
            self.images  = f["images"][:]    # [N, 224, 224, 3] uint8
            self.states  = f["states"][:]    # [N, 9] float
            self.actions = f["actions"][:]   # [N, 9] float
        print(f"  Loaded {len(self.images)} frames")
        self.N = len(self.actions)
        self.bs = batch_size

    def sample(self):
        idx = np.random.randint(0, self.N, size=self.bs)
        # Images: uint8 [0,255] → [0,1] float, then [B,3,224,224]
        imgs = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)
        imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
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

    print(f"\n[3] Training CLIP-Flow Matching on {len(replay.images)} frames...")
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx in range(100):
            batch_img, batch_state, batch_action = replay.sample()
            batch_img    = batch_img.to(device)
            batch_state  = batch_state.to(device)
            batch_action = batch_action.to(device)

            t = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            x_t    = (1 - t) * batch_action + t * noise

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

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch, "policy_state_dict": policy.state_dict()},
                       output_dir / f"checkpoint_epoch_{epoch+1}.pt")

        elapsed = time.time() - t_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f} | ETA: {eta:.0f}s")

    torch.save(policy.state_dict(), output_dir / "final_policy.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("CLIP-Flow Matching Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Velocity Loss")
    plt.savefig(output_dir / "loss_curve.png", dpi=150)

    total_time = time.time() - t_start
    print(f"\n✓ Training done in {total_time:.0f}s")
    print(f"✓ Policy: {output_dir / 'final_policy.pt'}")
    return losses


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,   default="/tmp/lekiwi_demo_224.h5")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--hidden",      type=int,   default=512)
    parser.add_argument("--device",      type=str,   default="mps")
    parser.add_argument("--output",      type=str,   default="results")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    print("\n[1] Loading replay buffer...")
    replay = ReplayBuffer(args.data, batch_size=args.batch_size)

    print("[2] Building CLIP-Flow Matching policy...")
    policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9,
                                    hidden=args.hidden, device=args.device)

    optimizer = torch.optim.Adam(policy.flow_head.parameters(), lr=args.lr)

    train(policy, optimizer, replay, epochs=args.epochs,
          device=args.device, output_dir=args.output)

    # Inference test
    print("\n[4] 4-step inference test...")
    policy.eval()
    with torch.no_grad():
        sim = LeKiwiSim()
        sim.reset()
        img = sim.render()
        img_np = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
        img_t  = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(args.device)
        arm  = sim.data.qpos[0:6]
        whl  = sim.data.qvel[0:3]
        state_t = torch.from_numpy(np.concatenate([arm, whl])).float().unsqueeze(0).to(args.device)
        action  = policy.infer(img_t, state_t, num_steps=4)
        print(f"  ✓ Action: {action.shape}, range=[{action.min().item():.3f}, {action.max().item():.3f}]")

    print("\n✓ All done!")

if __name__ == "__main__":
    main()