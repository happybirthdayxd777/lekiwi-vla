#!/usr/bin/env python3
"""
Phase 190: Train GoalConditioned VLA on phase189 CLEAN vision data
==================================================================
Train on CORRECT 11D state + per-step images (10000 images, 1:1 ratio).

KEY FIX vs Phase 187:
  - Phase 187 data: images=(50,) — only 1 image per episode (BUG)
  - Phase 189 data: images=(10000,) — 1 image per step (FIXED)
  → Policy now trains with REAL vision (CLIP sees different images per frame)

Architecture (from train_phase187.py):
  - CLIP ViT-B/32 spatial tokens [B, 50, 768] — NOW WITH REAL IMAGES
  - Goal MLP: 2 → 256 → 128
  - State net: 11D → 256 → 128
  - Cross-attention: goal(Q) attends to CLIP(K,V) → [B, 1, 768]
  - Fusion: [cls, state_feat, cross_out] → [B, 2048] → [B, 9]
  - 4-step Euler flow matching

Data (phase189_clean_50ep.h5):
  - states: (10000, 11) — arm_pos(6) + wheel_vel(3) + goal_norm(2)
  - actions: (10000, 9) — arm_torque(6) + wheel_speed(3)
  - images: (10000, 640, 480, 3) — one per step (FIXED in Phase 189)
  - goals: (10000, 2)
  - episode_starts: (51,) — 50 episodes × ~200 steps
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Phase 190: Use CPU — MPS has device mismatch issues with scalar tensors in nn.Linear
DEVICE = "cpu"
print(f"[Phase 190] Device: {DEVICE} (CPU only — MPS scalar tensor issues)")


# ─── CLIP Vision Encoder ───────────────────────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        images: [B, 3, 224, 224] in [0,1].
        Returns: [B, 50, 768] spatial tokens (CLS + 49 patches).
        """
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            return outputs.last_hidden_state  # [B, 50, 768]


# ─── GoalConditioned Policy ─────────────────────────────────────────────────────

class GoalConditionedPolicy(nn.Module):
    """
    CLIP-FM with goal-conditioned state (11D) + REAL vision (per-step images).
    Architecture from train_task_oriented.py:
      - CLIP spatial tokens [B, 50, 768] — NOW with real per-step images
      - Goal MLP: 2 → 256 → 128 (stronger than Phase 131)
      - State net: 11D → 256 → 128
      - Cross-attention: goal(Q) attends to CLIP(K,V) → [B, 1, 768]
      - Fusion: [cls, state_feat, cross_out] → [B, 2048] → [B, 9]
      - 4-step Euler flow matching
    """
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.encoder = CLIPVisionEncoder(device=device)

        # Goal conditioning (2D goal position)
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )

        # State encoder (11D: arm(6) + wheel_vel(3) + goal_norm(2))
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )

        # Cross-attention: goal attends to CLIP visual tokens
        self.goal_q_proj = nn.Linear(128, 768)  # Project goal embedding to 768 for Q
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(768)

        # Action head: cls(768) + cross_out(768) + state_feat(128) + t_emb(256) + noisy_action(9) = 1929
        self.flow_head = nn.Sequential(
            nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        # Time embedding: sin/cos of timestep → 256-dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU()
        )

    def forward(self, images, state, noisy_action, timestep):
        """
        images: [B, 3, 224, 224]
        state: [B, 11] (arm_pos + wheel_vel + goal_norm)
        noisy_action: [B, 9]
        timestep: [B, 1]
        """
        # CLIP spatial tokens — NOW with real per-step images from phase189
        clip_tokens = self.encoder(images)  # [B, 50, 768]

        # Goal embedding
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]

        # Cross-attention: goal (Q) attends to visual tokens (K, V)
        # Project goal_emb [B, 128] → [B, 1, 768] for Q
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]

        # State features
        state_feat = self.state_net(state)  # [B, 128]

        # Time embedding
        t_emb = self.time_mlp(timestep)  # [B, 256]

        # Cat: cls_token(1,768) + cross_out(1,768) + state_feat(1,128) + time(256) + noisy_action(1,9)
        cls_token = clip_tokens[:, 0:1, :]  # [B, 1, 768]
        x = torch.cat([
            cls_token,               # [B, 1, 768]
            cross_out,               # [B, 1, 768]
            state_feat.unsqueeze(1),  # [B, 1, 128]
            t_emb.unsqueeze(1),      # [B, 1, 256]
            noisy_action.unsqueeze(1),  # [B, 1, 9]
        ], dim=-1)  # [B, 1, 768+768+128+256+9] = [B, 1, 2029]

        x = x.squeeze(1)  # [B, 2029]
        return self.flow_head(x)  # [B, 9]

    def infer(self, images, state, num_steps=4):
        """4-step Euler flow matching inference."""
        self.eval()
        with torch.no_grad():
            x = torch.zeros_like(state[:, :self.action_dim])
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((images.shape[0], 1), i * dt, device=state.device)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ─── Replay Buffer ──────────────────────────────────────────────────────────────

class GoalConditionedReplay:
    """
    Replay for phase189 data: 11D state + per-step images (10000 frames, 1:1 ratio).
    
    KEY DIFFERENCE from Phase 187:
      - Phase 187: images=(50,) — only 1 image per episode → zero images used
      - Phase 189: images=(10000,) — 1 image per step → REAL vision training
    
    state: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
    action: arm_torque(6) + wheel_speed(3) = 9D
    image: (640, 480, 3) → resized to (224, 224, 3) via PIL bicubic
    """
    def __init__(self, h5_path, batch_size=32):
        self.batch_size = batch_size
        with h5py.File(h5_path, 'r') as f:
            self.states  = f['states'][:]            # (10000, 11) float32
            self.actions = f['actions'][:]           # (10000, 9) float32
            self.rewards = f['rewards'][:]           # (10000,) float32
            self.goals    = f['goal_positions'][:]    # (10000, 2) float32
            self.images_raw = f['images'][:]          # (10000, 640, 480, 3) uint8

        N = len(self.actions)
        self.n = N
        # Phase 189 uses continuous rewards (not binary); threshold at 0.5 for "goal-near" frames
        is_goal_near = (self.rewards >= 0.5)
        self.weights = np.ones(N, dtype=np.float32)
        self.weights[is_goal_near] = 3.0
        print(f"[Replay] {N} frames, state=11D, REAL images={self.images_raw.shape} (1:1 ratio with states)")
        n_goal = is_goal_near.sum()
        print(f"[Replay] goal_near_frames(reward>=0.5)={n_goal}, weight_goal={self.weights[is_goal_near][0] if n_goal > 0 else 'N/A'}x")

        # Precompute image mean/std for normalization
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_image(self, raw_img: np.ndarray) -> torch.Tensor:
        """
        Convert raw (640, 480, 3) uint8 → (3, 224, 224) float32 normalized.
        Uses PIL bicubic resize for high-quality downsampling.
        """
        from PIL import Image
        # Raw is HWC uint8
        img = Image.fromarray(raw_img)
        img = img.resize((224, 224), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        # Normalize with ImageNet stats
        arr = (arr - self.img_mean) / self.img_std
        # HWC → CHW
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def sample(self):
        """
        Returns:
          imgs: [B, 3, 224, 224] — REAL per-step images (FIXED vs Phase 187)
          states: [B, 11]
          actions: [B, 9]
          weights: [B]
        """
        idx = np.random.choice(len(self.actions), self.batch_size,
                               p=(self.weights / self.weights.sum()).astype(np.float64))
        states  = torch.from_numpy(self.states[idx].astype(np.float32))
        actions = torch.from_numpy(self.actions[idx].astype(np.float32))
        weights = torch.from_numpy(self.weights[idx].astype(np.float32))
        # Load and preprocess REAL images (not zeros like Phase 187)
        imgs = torch.stack([self._preprocess_image(self.images_raw[i])
                           for i in idx], dim=0)
        return imgs, states, actions, weights


# ─── Training ───────────────────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=30, device=DEVICE,
         output_dir="results/phase190_vision_train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.train()
    losses = []

    print(f"\n[Training] {epochs} epochs on {replay.n} frames with REAL vision...")
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 200
        for _ in range(n_batches):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img = batch_img.to(device)
            batch_state = batch_state.to(device)
            batch_action = batch_action.to(device)
            batch_weights = batch_weights.to(device)

            # Flow matching
            t_batch = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            alpha = 1 - t_batch.squeeze(-1)
            x_t = alpha.unsqueeze(-1) * batch_action + t_batch.squeeze(-1).unsqueeze(-1) * noise

            v_pred = policy(batch_img, batch_state, x_t, t_batch)
            v_target = batch_action - noise

            # Weighted loss
            loss = ((v_pred - v_target) ** 2).mean(dim=-1)
            loss = (loss * batch_weights / batch_weights.mean()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        elapsed = time.time() - t_start

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={lr:.6f}, elapsed={elapsed:.0f}s")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'loss': avg_loss,
                'policy_config': {
                    'state_dim': 11, 'action_dim': 9, 'hidden': 512
                }
            }, output_dir / f'epoch_{epoch}.pt')

    # Save best (lowest loss)
    best_epoch = np.argmin(losses)
    torch.save({
        'epoch': best_epoch,
        'policy_state_dict': policy.state_dict(),
        'losses': losses,
        'policy_config': {
            'state_dim': 11, 'action_dim': 9, 'hidden': 512
        }
    }, output_dir / 'best_policy.pt')

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Phase 190 — GoalConditioned VLA Training Loss (REAL VISION)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(output_dir / 'training_loss.png', dpi=100)
    plt.close()

    import json
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump([{'epoch': i, 'loss': losses[i]} for i in range(len(losses))], f)

    print(f"\n✓ Best epoch: {best_epoch+1}, loss={losses[best_epoch]:.4f}")
    print(f"✓ Saved: {output_dir / 'best_policy.pt'}")
    return losses


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/phase189_clean_50ep.h5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='results/phase190_vision_train')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.data) or '.', exist_ok=True)

    # Verify data has real images before training
    with h5py.File(args.data, 'r') as f:
        images_shape = f['images'].shape
        states_shape = f['states'].shape
    print(f"[Data Check] images={images_shape}, states={states_shape}")
    if images_shape[0] != states_shape[0]:
        raise ValueError(f"CRITICAL: images={images_shape[0]} != states={states_shape[0]} — phase189 data not fixed!")
    if images_shape[0] < 5000:
        raise ValueError(f"CRITICAL: only {images_shape[0]} images — likely phase187 data (only 50 images)")

    replay = GoalConditionedReplay(args.data, batch_size=args.batch_size)
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train(policy, optimizer, replay, epochs=args.epochs, device=DEVICE, output_dir=args.output)
    scheduler.step()

    # Save final
    output_dir = Path(args.output)
    torch.save(policy.state_dict(), output_dir / 'final_policy.pt')
    print(f"✓ Final policy: {output_dir / 'final_policy.pt'}")


if __name__ == '__main__':
    main()
