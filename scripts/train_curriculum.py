#!/usr/bin/env python3
"""
Phase 260: Curriculum Training — VLA Learns Goals in Difficulty Stages
======================================================================
New direction after DAgger abandoned (Phase259: DAgger-254 = 50% SR < VLA Phase227 = 70% SR).

Key insight: DAgger failed because it added more data from the same (imperfect) P-controller
without addressing the visual goal encoding difficulty. Instead, curriculum training
teaches the VLA easier goals first, then gradually expands to harder goals.

Curriculum Stages:
  Stage 1 (5 epochs): |r| < 0.25m — easy, close goals
  Stage 2 (10 epochs): |r| < 0.45m — medium goals
  Stage 3 (15 epochs): ALL goals — full distribution

Base checkpoint: results/phase227_contact_jacobian_train/epoch_30.pt
Data: phase227_extended_65ep.h5 (7589 frames, 65 episodes)

Usage:
  python3 scripts/train_curriculum.py --output results/phase260_curriculum_train
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cpu"
print(f"[Phase 260 Curriculum Training] Device: {DEVICE}")


# ─── CLIP Vision Encoder ────────────────────────────────────────────────────────

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


# ─── Goal-Conditioned VLA Policy ──────────────────────────────────────────────

class GoalConditionedPolicy(nn.Module):
    """
    CLIP-FM Goal-Conditioned VLA with cross-attention goal injection.
    Architecture matches Phase 227 exactly.
    """
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.encoder = CLIPVisionEncoder(device)

        self.goal_mlp = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )
        self.goal_q_proj = nn.Linear(128, 768)

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )

        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(768)

        self.flow_head = nn.Sequential(
            nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim)
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU()
        )

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)  # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)
        state_feat = self.state_net(state)  # [B, 128]
        t_emb = self.time_mlp(timestep)  # [B, 256]

        cls_token = clip_tokens[:, 0:1, :]
        x = torch.cat([
            cls_token,
            cross_out,
            state_feat.unsqueeze(1),
            t_emb.unsqueeze(1),
            noisy_action.unsqueeze(1),
        ], dim=-1)
        x = x.squeeze(1)
        return self.flow_head(x)

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


# ─── Curriculum Replay Buffer ──────────────────────────────────────────────────

class CurriculumReplay:
    """
    Replay buffer with curriculum filtering by goal radius.
    Caches images on first access (same pattern as Phase196Replay).

    Data keys (phase227_extended_65ep.h5):
      - states: (N, 11)
      - actions: (N, 9)
      - images: (N, 640, 480, 3)
      - goals: (N, 2) — raw goal coordinates
      - rewards: (N,) — binary
    """
    def __init__(self, h5_path, max_goal_radius=None, batch_size=32):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.max_goal_radius = max_goal_radius
        self.cache = {}

        with h5py.File(h5_path, 'r') as f:
            self.actions   = f['actions'][:]
            self.states    = f['states'][:]
            self.goals_raw = f['goals'][:]
            self.rewards   = f['rewards'][:]
            self.img_mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.img_std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.n_total = len(self.actions)
        goal_r = np.sqrt(self.goals_raw[:,0]**2 + self.goals_raw[:,1]**2)

        if max_goal_radius is not None:
            self.mask = goal_r <= max_goal_radius
            self.n = int(self.mask.sum())
        else:
            self.mask = np.ones(self.n_total, dtype=bool)
            self.n = self.n_total

        print(f"[CurriculumReplay] {self.n}/{self.n_total} frames (|r|<={max_goal_radius})")

        # Higher weight for goal-near frames (apply mask to is_goal_near)
        is_goal_near = (self.rewards >= 0.5) & self.mask
        masked_is_goal_near = is_goal_near[self.mask]
        self.weights = np.ones(self.n, dtype=np.float32)
        self.weights[masked_is_goal_near] = 5.0

    def _masked_indices(self):
        """Return original indices for masked elements."""
        return np.where(self.mask)[0]

    def sample(self):
        """Sample a batch from curriculum-filtered data. Returns (batch_img, batch_state, batch_action, batch_weights)."""
        orig_idx = self._masked_indices()
        chosen = np.random.choice(len(orig_idx), size=self.batch_size, replace=True)
        idx = orig_idx[chosen]

        batch_img = np.zeros((self.batch_size, 224, 224, 3), dtype=np.float32)
        for i, ii in enumerate(idx):
            raw = self._load_image(ii)
            img = Image.fromarray(raw).resize((224, 224), Image.BICUBIC)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - self.img_mean) / self.img_std
            batch_img[i] = arr
        batch_img = batch_img.transpose(0, 3, 1, 2)  # NHWC -> NCHW

        weights = self.weights[chosen]
        return (
            torch.from_numpy(batch_img).float(),
            torch.from_numpy(self.states[idx]).float(),
            torch.from_numpy(self.actions[idx]).float(),
            torch.from_numpy(weights).float(),
        )

    def _load_image(self, idx):
        """Load and cache image from HDF5."""
        if idx in self.cache:
            return self.cache[idx]
        with h5py.File(self.h5_path, 'r') as f:
            img = f['images'][idx]
        self.cache[idx] = img
        return img


# ─── Training ──────────────────────────────────────────────────────────────────

def train_stage(policy, replay, epochs, lr, stage_name, batch_size):
    """Train for one curriculum stage. Freezes CLIP encoder, trains flow head + goal/state nets."""

    # Freeze CLIP
    for p in policy.clip.parameters():
        p.requires_grad = False

    # Train: goal_mlp, state_net, cross_attn, time_mlp, flow_head
    for p in policy.encoder.parameters():
        p.requires_grad = False
    for p in policy.goal_mlp.parameters():
        p.requires_grad = True
    for p in policy.goal_q_proj.parameters():
        p.requires_grad = True
    for p in policy.state_net.parameters():
        p.requires_grad = True
    for p in policy.cross_attn.parameters():
        p.requires_grad = True
    for p in policy.cross_norm.parameters():
        p.requires_grad = True
    for p in policy.time_mlp.parameters():
        p.requires_grad = True
    for p in policy.flow_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    policy.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    n_batches = max(1, replay.n // batch_size)
    n_batches = min(n_batches, 200)  # Cap at 200 batches per epoch

    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(n_batches):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img    = batch_img.to(DEVICE)
            batch_state  = batch_state.to(DEVICE)
            batch_action = batch_action.to(DEVICE)
            batch_weights = batch_weights.to(DEVICE)

            # Flow matching: noise → action in 1 step (t=1 → t=0)
            noise = torch.randn_like(batch_action)
            t = torch.rand(batch_size, 1, device=DEVICE)
            alpha = 1 - t.squeeze(-1)
            x_t = alpha.unsqueeze(-1) * batch_action + t.squeeze(-1).unsqueeze(-1) * noise
            v_target = batch_action - noise

            t_scaled = t * 0.01
            v_pred = policy(batch_img, batch_state, x_t, t_scaled)

            # Weighted MSE
            loss = ((v_pred - v_target) ** 2).mean(dim=-1)
            loss = (loss * batch_weights / batch_weights.mean()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"  Stage {stage_name} epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    return losses


def train_curriculum(base_checkpoint, data_path, output_dir,
                     epochs_1=5, epochs_2=10, epochs_3=15,
                     batch_size=32, lr=1e-4):
    """Multi-stage curriculum training from a base checkpoint."""

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Curriculum Training] Output: {output_dir}")
    print(f"  Base: {base_checkpoint}")
    print(f"  Data: {data_path}")

    # Load base policy
    print("\n[1] Loading base policy from Phase227 epoch_30...")
    policy = GoalConditionedPolicy().to(DEVICE)
    ckpt = torch.load(base_checkpoint, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
    print(f"  Loaded checkpoint (epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')})")

    all_losses = {}
    start_time = time.time()

    # ── Stage 1: Easy goals |r| < 0.25 ───────────────────────────────────────
    print(f"\n[Stage 1] |r| <= 0.25m ({epochs_1} epochs)")
    s1_replay = CurriculumReplay(data_path, max_goal_radius=0.25, batch_size=batch_size)
    s1_losses = train_stage(policy, s1_replay, epochs_1, lr, "s1_r025", batch_size)
    all_losses['stage1_r025'] = s1_losses
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': f's1_{epochs_1}',
        'stage': 'stage1_r025',
        'loss': s1_losses[-1],
    }, f"{output_dir}/stage1_r025.pt")
    print(f"  Stage 1 done: {len(s1_losses)} epochs, final loss={s1_losses[-1]:.4f}")

    # ── Stage 2: Medium goals |r| < 0.45 ───────────────────────────────────────
    print(f"\n[Stage 2] |r| <= 0.45m ({epochs_2} epochs)")
    s2_replay = CurriculumReplay(data_path, max_goal_radius=0.45, batch_size=batch_size)
    s2_losses = train_stage(policy, s2_replay, epochs_2, lr, "s2_r045", batch_size)
    all_losses['stage2_r045'] = s2_losses
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': f's2_{epochs_2}',
        'stage': 'stage2_r045',
        'loss': s2_losses[-1],
    }, f"{output_dir}/stage2_r045.pt")
    print(f"  Stage 2 done: {len(s2_losses)} epochs, final loss={s2_losses[-1]:.4f}")

    # ── Stage 3: All goals ─────────────────────────────────────────────────────
    print(f"\n[Stage 3] ALL goals ({epochs_3} epochs)")
    s3_replay = CurriculumReplay(data_path, max_goal_radius=None, batch_size=batch_size)
    s3_losses = train_stage(policy, s3_replay, epochs_3, lr, "s3_all", batch_size)
    all_losses['stage3_all'] = s3_losses

    # Save final and best
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': 'final',
        'stage': 'stage3_all',
        'loss': s3_losses[-1],
    }, f"{output_dir}/final_policy.pt")

    best_idx = int(np.argmin(s3_losses))
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': f's3_epoch{best_idx+1}',
        'stage': 'stage3_all_best',
        'loss': s3_losses[best_idx],
    }, f"{output_dir}/best_policy.pt")

    elapsed = (time.time() - start_time) / 60
    print(f"\n[Training Complete] Total time: {elapsed:.1f} min")
    print(f"  Best stage3 epoch: {best_idx+1} (loss={s3_losses[best_idx]:.4f})")

    # Plot
    plot_curriculum_loss(all_losses, output_dir)

    # Save training log
    with open(f"{output_dir}/training_log.txt", 'w') as f:
        for stage, losses in all_losses.items():
            for ep, loss in enumerate(losses):
                f.write(f"{stage} epoch {ep+1} loss={loss:.6f}\n")
        f.write(f"\nTotal time: {elapsed:.1f} min\n")
        f.write(f"Best: stage3 epoch {best_idx+1} loss={s3_losses[best_idx]:.6f}\n")

    return policy, all_losses


def plot_curriculum_loss(all_losses, output_dir):
    """Plot combined loss curve across all stages."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {'stage1_r025': '#2ecc71', 'stage2_r045': '#3498db', 'stage3_all': '#e74c3c'}
    offset = 0
    for stage_name, losses in all_losses.items():
        epochs = list(range(offset+1, offset+len(losses)+1))
        ax.plot(epochs, losses, label=stage_name, color=colors[stage_name], linewidth=2, marker='o', markersize=3)
        offset += len(losses)
    ax.axvline(x=5, color='#2ecc71', linestyle='--', alpha=0.5, label='Stage 1→2 boundary')
    ax.axvline(x=5+10, color='#3498db', linestyle='--', alpha=0.5, label='Stage 2→3 boundary')
    ax.set_xlabel('Global Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Phase 260: Curriculum Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/curriculum_loss.png", dpi=150)
    plt.close()
    print(f"[PLOT] {output_dir}/curriculum_loss.png")


# ─── Sanity Check ──────────────────────────────────────────────────────────────

def sanity_check(policy, data_path):
    """Quick inference sanity check."""
    print("\n[Sanity Check] Running 1-step policy inference...")
    with h5py.File(data_path, 'r') as f:
        img_raw = f['images'][0]
        state = f['states'][0]

    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = Image.fromarray(img_raw).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - img_mean) / img_std
    img_t = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
    state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)

    policy.eval()
    with torch.no_grad():
        action = policy.infer(img_t, state_t)
    print(f"  action[0:3] (arm): {action.cpu().numpy()[0, 0:3]}")
    print(f"  action[6:9] (wheels): {action.cpu().numpy()[0, 6:9]}")
    print("[PASS] Policy inference OK")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 260: Curriculum Training')
    parser.add_argument('--base', default='results/phase227_contact_jacobian_train/epoch_30.pt')
    parser.add_argument('--data', default='data/phase227_extended_65ep.h5')
    parser.add_argument('--output', default='results/phase260_curriculum_train')
    parser.add_argument('--epochs_1', type=int, default=5,
                        help='Stage 1 epochs (|r|<0.25)')
    parser.add_argument('--epochs_2', type=int, default=10,
                        help='Stage 2 epochs (|r|<0.45)')
    parser.add_argument('--epochs_3', type=int, default=15,
                        help='Stage 3 epochs (all)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sanity_only', action='store_true',
                        help='Load base, run sanity check, then exit (no training)')
    args = parser.parse_args()

    # Sanity check mode — load base and verify inference
    if args.sanity_only:
        policy = GoalConditionedPolicy().to(DEVICE)
        ckpt = torch.load(args.base, map_location=DEVICE, weights_only=False)
        policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
        sanity_check(policy, args.data)
        return

    start = time.time()
    policy, all_losses = train_curriculum(
        base_checkpoint=args.base,
        data_path=args.data,
        output_dir=args.output,
        epochs_1=args.epochs_1,
        epochs_2=args.epochs_2,
        epochs_3=args.epochs_3,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    sanity_check(policy, args.data)
    elapsed = (time.time() - start) / 60
    print(f"\n[DONE] Total elapsed: {elapsed:.1f} min")

if __name__ == '__main__':
    main()
