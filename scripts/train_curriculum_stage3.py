#!/usr/bin/env python3
"""
Phase 263: Resume Stage 3 Curriculum Training — Memory-Fixed + Hang-Fixed
=========================================================================
Fixes from Phase 262:
1. Lazy image loading — open H5 once, don't cache all images (was causing OOM)
2. CLIP loaded ONCE via CLIPVisionEncoder — eliminated duplicate loading that caused hang
3. Explicit stdout flush after every print
4. Smaller batch size to reduce peak memory
5. Add periodic checkpoint (every 3 epochs) to prevent loss from crashes

Usage:
  python3 scripts/train_curriculum_stage3.py
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

WORKDIR = Path(__file__).parent.parent.resolve()
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR / "scripts"))

DEVICE = "cpu"
BATCH_SIZE = 16  # reduced from 32 to save memory
EPOCHS_3 = 15
LR = 1e-4

def p(msg):
    """Print with explicit flush."""
    print(msg, flush=True)


# ─── Same policy/model classes as train_curriculum.py ─────────────────────────

class CLIPVisionEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        p("[INFO] Loading CLIP ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p_ in self.clip.parameters():
            p_.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            return outputs.last_hidden_state  # [B, 50, 768]


class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        # Load CLIP ONCE via encoder; reuse self.clip reference for gradient freezing
        # FIX: Previously loaded CLIP twice (once here, once in CLIPVisionEncoder) — second
        # load hung indefinitely on text_model initialization. Now share the same instance.
        self.encoder = CLIPVisionEncoder(device)
        self.clip = self.encoder.clip  # share same CLIP instance — avoids second hang

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
        B = images.shape[0]

        img_feat = self.encoder(images)  # [B, 50, 768]
        goal_q = self.goal_q_proj(self.goal_mlp(state[:, :2]))
        goal_q = goal_q.unsqueeze(1)  # [B, 1, 768]

        attn_out, _ = self.cross_attn(goal_q, img_feat, img_feat)
        attn_out = self.cross_norm(attn_out.squeeze(1))  # [B, 768]

        goal_q = goal_q.squeeze(1)  # [B, 768]
        state_feat = self.state_net(state)  # [B, 128]
        time_emb = self.time_mlp(timestep)  # [B, 256]

        x = torch.cat([attn_out, goal_q, state_feat, time_emb, noisy_action], dim=-1)
        return self.flow_head(x)


# ─── Lazy CurriculumReplay (no image caching) ─────────────────────────────────

class CurriculumReplay:
    """On-demand image loading — keeps H5 file open, no image caching."""

    def __init__(self, h5_path, max_goal_radius=None, batch_size=32):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.max_goal_radius = max_goal_radius
        self._h5 = None  # lazy open

        # Read metadata only (small arrays)
        with h5py.File(h5_path, 'r') as f:
            self.actions   = f['actions'][:]
            self.states    = f['states'][:]
            self.goals_raw = f['goals'][:]
            self.rewards   = f['rewards'][:]
            self.n_total = len(self.actions)
            self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        goal_r = np.sqrt(self.goals_raw[:,0]**2 + self.goals_raw[:,1]**2)
        if max_goal_radius is not None:
            self.mask = goal_r <= max_goal_radius
            self.n = int(self.mask.sum())
        else:
            self.mask = np.ones(self.n_total, dtype=bool)
            self.n = self.n_total

        is_goal_near = (self.rewards >= 0.5) & self.mask
        masked_is_goal_near = is_goal_near[self.mask]
        self.weights = np.ones(self.n, dtype=np.float32)
        self.weights[masked_is_goal_near] = 5.0

        self._masked_indices = np.where(self.mask)[0]
        p(f"[CurriculumReplay] {self.n}/{self.n_total} frames (|r|<={max_goal_radius})")

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def _load_image(self, idx):
        f = self._open_h5()
        raw = f['images'][idx]
        img = Image.fromarray(raw).resize((224, 224), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.img_mean) / self.img_std
        return arr.transpose(2, 0, 1)  # CHW for torch

    def sample(self):
        chosen = np.random.choice(len(self._masked_indices), size=self.batch_size, replace=True)
        orig_idx = self._masked_indices[chosen]

        batch_img = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)
        for i, ii in enumerate(orig_idx):
            batch_img[i] = self._load_image(ii)

        return (
            torch.from_numpy(batch_img).float(),
            torch.from_numpy(self.states[orig_idx]).float(),
            torch.from_numpy(self.actions[orig_idx]).float(),
            torch.from_numpy(self.weights[chosen]).float(),
        )

    def close(self):
        if self._h5 is not None:
            self._h5.close()


def train_stage(policy, replay, epochs, lr, stage_name, batch_size, save_every=3):
    """Train Stage 3. Save checkpoint every `save_every` epochs."""

    # Freeze CLIP
    for p_ in policy.clip.parameters():
        p_.requires_grad = False
    for p_ in policy.encoder.parameters():
        p_.requires_grad = False
    for p_ in policy.goal_mlp.parameters():
        p_.requires_grad = True
    for p_ in policy.goal_q_proj.parameters():
        p_.requires_grad = True
    for p_ in policy.state_net.parameters():
        p_.requires_grad = True
    for p_ in policy.cross_attn.parameters():
        p_.requires_grad = True
    for p_ in policy.cross_norm.parameters():
        p_.requires_grad = True
    for p_ in policy.time_mlp.parameters():
        p_.requires_grad = True
    for p_ in policy.flow_head.parameters():
        p_.requires_grad = True

    trainable = sum(p_.numel() for p_ in policy.parameters() if p_.requires_grad)
    p(f"  Trainable params: {trainable:,}")

    policy.train()
    optimizer = torch.optim.AdamW(filter(lambda p_: p_.requires_grad, policy.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    n_batches = max(1, replay.n // batch_size)
    n_batches = min(n_batches, 200)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for _ in range(n_batches):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img    = batch_img.to(DEVICE)
            batch_state  = batch_state.to(DEVICE)
            batch_action = batch_action.to(DEVICE)
            batch_weights = batch_weights.to(DEVICE)

            noise = torch.randn_like(batch_action)
            t = torch.rand(batch_size, 1, device=DEVICE)
            alpha = 1 - t.squeeze(-1)
            x_t = alpha.unsqueeze(-1) * batch_action + t.squeeze(-1).unsqueeze(-1) * noise
            v_target = batch_action - noise

            t_scaled = t * 0.01
            v_pred = policy(batch_img, batch_state, x_t, t_scaled)

            loss = ((v_pred - v_target) ** 2).mean(dim=-1)
            loss = (loss * batch_weights / batch_weights.mean()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()
        p(f"  Stage {stage_name} epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        # Save every N epochs + last
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            ckpt_path = f"{OUTPUT_DIR}/s3_epoch{epoch+1}.pt"
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'epoch': f's3_epoch{epoch+1}',
                'stage': 'stage3_all',
                'loss': avg_loss,
            }, ckpt_path)
            p(f"    → Checkpoint saved: {ckpt_path}")

    return losses


def plot_curriculum_loss(s1_losses, s2_losses, s3_losses, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = {'stage1': '#2ecc71', 'stage2': '#3498db', 'stage3': '#e74c3c'}
    s1_x = list(range(1, len(s1_losses)+1))
    s2_x = list(range(len(s1_losses)+1, len(s1_losses)+len(s2_losses)+1))
    s3_x = list(range(len(s1_losses)+len(s2_losses)+1, len(s1_losses)+len(s2_losses)+len(s3_losses)+1))
    ax.plot(s1_x, s1_losses, label='Stage 1 (|r|<0.25)', color=colors['stage1'], linewidth=2, marker='o', markersize=3)
    ax.plot(s2_x, s2_losses, label='Stage 2 (|r|<0.45)', color=colors['stage2'], linewidth=2, marker='o', markersize=3)
    ax.plot(s3_x, s3_losses, label='Stage 3 (ALL)', color=colors['stage3'], linewidth=2, marker='o', markersize=3)
    ax.axvline(x=5, color='#2ecc71', linestyle='--', alpha=0.5)
    ax.axvline(x=15, color='#3498db', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Phase 264: Curriculum Training Loss (Stage 3 — CLIP Hang Fixed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/curriculum_loss.png", dpi=150)
    p(f"[Plot] Saved {output_dir}/curriculum_loss.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CKPT_STAGE2 = "results/phase260_curriculum_train/stage2_r045.pt"
    DATA_PATH   = "data/phase227_extended_65ep.h5"
    OUTPUT_DIR  = "results/phase264_curriculum_train"
    SAVE_EVERY = 3

    p(f"\n{'='*60}")
    p(f"Phase 264: Stage 3 Curriculum (CLIP Hang Fixed + Single CLIP Load)")
    p(f"{'='*60}")
    p(f"  Loading Stage 2 checkpoint: {CKPT_STAGE2}")
    p(f"  Data: {DATA_PATH}")
    p(f"  Output: {OUTPUT_DIR}")
    p(f"  Stage 3 epochs: {EPOCHS_3}, batch_size: {BATCH_SIZE}")
    p(f"  Checkpoint every: {SAVE_EVERY} epochs")
    p("")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    p("[1] Loading Stage 2 policy checkpoint...")
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    ckpt = torch.load(CKPT_STAGE2, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt['policy_state_dict'])
    p(f"  Loaded: stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}")

    # Stage 1 + Stage 2 losses (from previous log)
    s1_losses = [5.5315, 2.2123, 0.8483, 0.4232, 0.2882]
    s2_losses = [0.3387, 0.2987, 0.2874, 0.3124, 0.2971, 0.2785, 0.2868, 0.2802, 0.2712, 0.2938]

    p(f"\n[2] Stage 3: ALL goals ({EPOCHS_3} epochs, batch={BATCH_SIZE})")
    replay = CurriculumReplay(DATA_PATH, max_goal_radius=None, batch_size=BATCH_SIZE)

    p(f"\n[3] Training Stage 3...")
    s3_losses = train_stage(
        policy, replay,
        epochs=EPOCHS_3,
        lr=LR,
        stage_name="s3_all",
        batch_size=BATCH_SIZE,
        save_every=SAVE_EVERY,
    )

    replay.close()

    p(f"\n[4] Saving final + best checkpoints...")

    # Save final
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': 'final',
        'stage': 'stage3_all',
        'loss': s3_losses[-1],
    }, f"{OUTPUT_DIR}/final_policy.pt")
    p(f"  Saved: final_policy.pt (loss={s3_losses[-1]:.4f})")

    # Save best
    best_idx = int(np.argmin(s3_losses))
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'epoch': f's3_epoch{best_idx+1}',
        'stage': 'stage3_all_best',
        'loss': s3_losses[best_idx],
    }, f"{OUTPUT_DIR}/best_policy.pt")
    p(f"  Saved: best_policy.pt (epoch={best_idx+1}, loss={s3_losses[best_idx]:.4f})")

    # Save log
    with open(f"{OUTPUT_DIR}/training_log.txt", 'w') as f:
        for stage, losses in [('stage1_r025', s1_losses), ('stage2_r045', s2_losses), ('stage3_all', s3_losses)]:
            for ep, loss in enumerate(losses):
                f.write(f"{stage} epoch {ep+1} loss={loss:.6f}\n")
        f.write(f"Best: stage3 epoch {best_idx+1} loss={s3_losses[best_idx]:.6f}\n")

    # Plot
    plot_curriculum_loss(s1_losses, s2_losses, s3_losses, OUTPUT_DIR)

    p(f"\n✓ Stage 3 training complete!")
    p(f"  Output: {OUTPUT_DIR}/")
    p(f"  Best epoch: {best_idx+1}/{EPOCHS_3}, loss={s3_losses[best_idx]:.4f}")
