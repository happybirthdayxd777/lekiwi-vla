#!/usr/bin/env python3
"""
Phase 187: Train GoalConditioned VLA on phase187 clean data
==========================================================
Train on CORRECT 11D state data with strong goal-wheel correlation.
Uses the train_task_oriented.py architecture (CLIP-FM with task-oriented reward weighting).
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

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 187] Device: {DEVICE}")


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
    CLIP-FM with goal-conditioned state (11D).
    Architecture from train_task_oriented.py:
      - CLIP spatial tokens [B, 50, 768]
      - Goal MLP: 2 → 256 → 128 (stronger than Phase 131)
      - State net: 11D → 256
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
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(768)

        # Action head
        self.flow_head = nn.Sequential(
            nn.Linear(768 + 128 + state_dim, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

    def forward(self, images, state, noisy_action, timestep):
        """
        images: [B, 3, 224, 224]
        state: [B, 11] (arm_pos + wheel_vel + goal_norm)
        noisy_action: [B, 9]
        timestep: [B, 1]
        """
        # CLIP spatial tokens
        clip_tokens = self.encoder(images)  # [B, 50, 768]

        # Goal embedding
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]

        # Cross-attention: goal (Q) attends to visual tokens (K, V)
        goal_q = goal_emb.unsqueeze(1)  # [B, 1, 128]
        # Project goal to 768 for attention
        goal_q_proj = nn.functional.linear(goal_q, torch.eye(768, 128, device=self.device).T).squeeze(1) if goal_emb.shape[-1] != 768 else nn.functional.linear(goal_emb, torch.eye(768, device=self.device)).unsqueeze(1)
        cross_out, _ = self.cross_attn(goal_q_proj, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q_proj)  # [B, 1, 768]

        # State features
        state_feat = self.state_net(state)  # [B, 128]

        # Time embedding
        t_feat = torch.sin(6.283 * timestep / 2.0).squeeze(-1)  # [B]
        t_emb = nn.functional.linear(t_feat.unsqueeze(-1),
                                     torch.zeros(256, 1, device=self.device)).squeeze(-1) if t_feat.shape[-1] != 1 else torch.zeros(state.shape[0], 256, device=self.device)

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
                t = torch.full((images.shape[0], 1), i * dt, device=DEVICE)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ─── Replay Buffer ──────────────────────────────────────────────────────────────

class GoalConditionedReplay:
    """Replay for phase187 data: 11D state (arm+wheel+goal_norm), 9D action.
    NOTE: phase187_clean_50ep.h5 has only 50 images (1 per episode), so images
    are not used for training. Policy trains on state+goal only (no vision).
    """
    def __init__(self, h5_path, batch_size=32):
        self.batch_size = batch_size
        with h5py.File(h5_path, 'r') as f:
            self.states  = f['states'][:]            # (10000, 11) float32
            self.actions = f['actions'][:]           # (10000, 9) float32
            self.rewards = f['rewards'][:]           # (10000,) float32
            self.goals    = f['goal_positions'][:]    # (10000, 2) float32

        N = len(self.actions)
        self.n = N
        is_goals = (self.rewards >= 1.0)
        self.weights = np.ones(N, dtype=np.float32)
        self.weights[is_goals] = 3.0
        print(f"[Replay] {N} frames, state=11D, goal_frames={is_goals.sum()} (NO images — state-only training)")

    def sample(self):
        # Use CPU for weighted sampling to avoid MPS/CPU tensor mismatch
        idx = np.random.choice(len(self.actions), self.batch_size,
                               p=(self.weights / self.weights.sum()).astype(np.float64))
        states  = torch.from_numpy(self.states[idx].astype(np.float32))
        actions = torch.from_numpy(self.actions[idx].astype(np.float32))
        weights = torch.from_numpy(self.weights[idx].astype(np.float32))
        # Zero images as placeholder (no per-frame images available)
        imgs = torch.zeros(self.batch_size, 3, 224, 224, dtype=torch.float32)
        return imgs, states, actions, weights


# ─── Training ───────────────────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=30, device=DEVICE,
         output_dir="results/phase187_goal_conditioned_train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.train()
    losses = []

    print(f"\n[Training] {epochs} epochs on {replay.n} frames...")
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
            # Alternative: standard MSE
            # loss = ((v_pred - v_target) ** 2).mean()

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
    plt.title('Phase 187 — GoalConditioned VLA Training Loss')
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
    parser.add_argument('--data', type=str, default='data/phase187_clean_50ep.h5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='results/phase187_goal_conditioned_train')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.data) or '.', exist_ok=True)

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
