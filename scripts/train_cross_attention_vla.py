#!/usr/bin/env python3
"""
Phase 131 — Cross-Attention VLA Policy
======================================
CLIP ViT-B/32 spatial tokens [B, 50, 768] + cross-attention with goal embedding.

Key insight: Pooled CLIP features [B, 768] lose SPATIAL information about WHERE
the goal is in the image. Cross-attention allows the policy to attend to specific
image regions based on goal position.

Architecture:
  CLIP tokens:     [B, 50, 768]        (frozen encoder, spatial info preserved)
  Goal embed:      [B, 2] → [B, 64]    (learnable goal MLP)
  Cross-attention: goal (Q) attends to CLIP tokens (K,V) → [B, 1, 768]
  Output:          [B, 9] action via CLS + state + cross_out concatenation

Flow Matching: noise → x_t → v_pred → v_target = action - noise
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
import json

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 131] Device: {DEVICE}")


# ─── Dataset ────────────────────────────────────────────────────────────────

class GoalDirectedReplay:
    """Replays goal-directed frames with 11D state (9D + goal_xy)."""
    def __init__(self, h5_path, batch_size=16):
        self.batch_size = batch_size
        with h5py.File(h5_path, 'r') as h:
            self.n = len(h['images'])
            self.states = h['states'][:].astype(np.float32)       # [N, 9]
            self.actions = h['actions'][:].astype(np.float32)     # [N, 9]
            self.goals = h['goal_positions'][:].astype(np.float32) # [N, 2]
            self.images = h['images'][:]                            # [N, 224, 224, 3]
        print(f"[Replay] {self.n} frames, state=11D (9+2)")

    def sample(self):
        idx = np.random.randint(0, self.n, self.batch_size)
        imgs = np.stack([self.images[i] for i in idx]).transpose(0, 3, 1, 2) / 255.0
        ext_state = np.concatenate([self.states[idx], self.goals[idx]], axis=1)
        return (torch.from_numpy(imgs.astype(np.float32)),
                torch.from_numpy(ext_state.astype(np.float32)),
                torch.from_numpy(self.actions[idx].astype(np.float32)))


# ─── CLIP Spatial Encoder ──────────────────────────────────────────────────

class CLIPSpatialEncoder(nn.Module):
    """CLIP ViT-B/32 preserving spatial tokens [B, 50, 768]."""
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        print("[INFO] Loading CLIP ViT-B/32 (spatial tokens mode)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        for p in self.clip.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in self.clip.parameters())
        print(f"[INFO] CLIP: {n_params:,} params (frozen)")

    def forward(self, images):
        """images: [B, 3, 224, 224] in [0,1]. Returns: [B, 50, 768] spatial tokens."""
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            # last_hidden_state: [B, 50, 768] — CLS token + 49 patches
            hidden = outputs.last_hidden_state
        return hidden


# ─── Cross-Attention Goal-Conditioned Policy ────────────────────────────────

class CrossAttentionPolicy(nn.Module):
    """
    Cross-attention VLA: CLIP spatial tokens + goal → cross-attention → action.
    Flow: clip_tokens[B,50,768] + goal_emb[B,64]
        → cross_attn(Q=goal, K=V=clip_tokens) → [B,1,768]
        → concat[cls, state_feat, cross_out, time_feat] → [B,2048] → [B,9]
    """
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device

        # CLIP encoder (frozen, spatial)
        self.clip_encoder = CLIPSpatialEncoder(device=device)

        # Goal embedding: 2D → 64D
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 128), nn.SiLU(), nn.Linear(128, 64)
        )

        # State processing: [B, 11] → [B, 256]
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )

        # Cross-attention: goal (Q) attends to CLIP tokens (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(768)

        # Time embedding: t ∈ [0,1] → [B, 256]
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )

        # Combined dim: cls(768) + state(256) + cross_out(768) + time(256) = 2048
        total_dim = 768 + 256 + 768 + 256
        self.action_head = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)
        self.to(device)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] CrossAttentionPolicy: {n_params:,} params")

    def forward(self, images, state, noisy_action, timestep):
        """Returns [B, 9] velocity prediction."""
        clip_tokens = self.clip_encoder(images)            # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])           # [B, 64]
        state_feat = self.state_net(state)               # [B, 256]

        # Cross-attention: goal attends to image regions
        goal_q = nn.Linear(64, 768, device=self.device)(goal_emb.unsqueeze(1))  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)   # [B, 1, 768]

        # CLS token (first token = global image representation)
        cls_token = clip_tokens[:, 0:1, :]                 # [B, 1, 768]

        # Combine: cls + state + cross_out + time
        combined = torch.cat([
            cls_token,              # [B, 1, 768]
            state_feat.unsqueeze(1), # [B, 1, 256]
            cross_out,              # [B, 1, 768]
        ], dim=-1)                  # [B, 1, 1792]

        t_feat = self.time_mlp(timestep)   # [B, 256]
        combined = torch.cat([combined, t_feat.unsqueeze(1)], dim=-1)  # [B, 1, 2048]

        v_pred = self.action_head(combined).squeeze(1)  # [B, 9]
        return v_pred + self.skip(noisy_action)

    def infer(self, images, state, num_steps=4):
        """4-step Euler flow matching inference."""
        self.eval()
        with torch.no_grad():
            x = torch.zeros_like(state[:, :9])
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((images.shape[0], 1), i * dt, device=DEVICE)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ─── Training ──────────────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=50, device=DEVICE, output_dir="results/phase131"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.train()
    losses = []
    t_start = time.time()

    print(f"\n[3] Training Cross-Attention VLA on {replay.n} frames...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(200):
            batch_img, batch_state, batch_action = replay.sample()
            batch_img = batch_img.to(device)
            batch_state = batch_state.to(device)
            batch_action = batch_action.to(device)

            # Flow matching: interpolate between action and noise
            t_batch = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            alpha = 1 - t_batch.squeeze(-1)  # [B]
            x_t = alpha.unsqueeze(-1) * batch_action + t_batch.squeeze(-1).unsqueeze(-1) * noise

            v_pred = policy(batch_img, batch_state, x_t, t_batch)
            v_target = batch_action - noise

            loss = ((v_pred - v_target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / 200)
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1}/{epochs}: loss={losses[-1]:.4f}, elapsed={elapsed:.0f}s")

    # Save
    torch.save({'epoch': epochs-1, 'policy_state_dict': policy.state_dict(), 'losses': losses},
               output_dir / f'checkpoint_epoch_{epochs}.pt')
    torch.save(policy.state_dict(), output_dir / 'final_policy.pt')

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Phase 131 — Cross-Attention VLA Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(output_dir / 'training_loss.png', dpi=100)
    plt.close()
    print(f"\n✓ Checkpoint: {output_dir / f'checkpoint_epoch_{epochs}.pt'}")
    return losses


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_cross_attn_vla(policy, n_episodes=10, max_steps=200):
    from sim_lekiwi import LeKiwiSim
    sim = LeKiwiSim()
    successes = 0
    dists = []
    policy.eval()
    for ep in range(n_episodes):
        sim.reset()
        goal = np.random.uniform(-1.5, 1.5, 2)
        for step in range(max_steps):
            img = np.array(sim.render().resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            arm = sim.data.qpos[0:6]
            whl = sim.data.qvel[0:3]
            state_11d = np.concatenate([arm, whl, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)
            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = np.clip(action.cpu().numpy()[0], -0.5, 0.5)
            sim.step(action_np)
            base_pos = sim.data.qpos[6:9]
            dist = np.sqrt((goal[0]-base_pos[0])**2 + (goal[1]-base_pos[1])**2)
            if dist < 0.3:
                successes += 1
                break
        dists.append(dist)
    sr = successes / n_episodes
    print(f"  CrossAttn VLA: {100*sr:.0f}% SR, mean_dist={np.mean(dists):.3f}m")
    return sr


def evaluate_pcontroller(n_episodes=10, max_steps=200):
    from sim_lekiwi import LeKiwiSim
    sim = LeKiwiSim()
    successes = 0
    for ep in range(n_episodes):
        sim.reset()
        goal = np.random.uniform(-1.5, 1.5, 2)
        for step in range(max_steps):
            base_pos = sim.data.qpos[6:9]
            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            action = np.zeros(9)
            action[0] = np.clip(2.0 * dx, -0.5, 0.5)
            action[1] = np.clip(2.0 * dy, -0.5, 0.5)
            sim.step(action)
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 0.3:
                successes += 1
                break
    sr = successes / n_episodes
    print(f"  P-controller: {100*sr:.0f}% SR")
    return sr


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       type=str,   default="data/phase63_reachable_10k_converted.h5")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--device",      type=str,   default=DEVICE)
    parser.add_argument("--output",      type=str,   default="results/phase131")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 131 — Cross-Attention VLA Policy")
    print("CLIP spatial tokens [B,50,768] + cross-attention goal conditioning")
    print("=" * 60)

    print(f"\n[1] Loading replay: {args.data}")
    replay = GoalDirectedReplay(args.data, batch_size=args.batch_size)

    print("\n[2] Building Cross-Attention policy...")
    policy = CrossAttentionPolicy(state_dim=11, goal_dim=2, action_dim=9,
                                   cross_heads=8, hidden=512, device=args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    print("\n[3] Training...")
    losses = train(policy, optimizer, replay, epochs=args.epochs,
                   device=args.device, output_dir=args.output)

    print("\n[4] Evaluating CrossAttn VLA...")
    sr_vla = evaluate_cross_attn_vla(policy, n_episodes=5, max_steps=200)

    print("\n[5] P-controller baseline...")
    sr_pctrl = evaluate_pcontroller(n_episodes=5, max_steps=200)

    print("\n" + "=" * 60)
    print("PHASE 131 RESULTS")
    print("=" * 60)
    print(f"  CrossAttn VLA (50 epochs): {100*sr_vla:.0f}% SR")
    print(f"  P-controller baseline:    {100*sr_pctrl:.0f}% SR")

    results = {"phase": 131, "architecture": "cross_attention", "epochs": args.epochs,
               "vla_sr": float(sr_vla), "pcontroller_sr": float(sr_pctrl)}
    with open(Path(args.output) / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()