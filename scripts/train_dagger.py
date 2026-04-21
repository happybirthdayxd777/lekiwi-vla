#!/usr/bin/env python3
"""
Phase 246: DAgger Training for LeKiWi VLA
==========================================

DAgger (Ross & Bagnell 2013) retraining:
- Load DAgger dataset (from collect_dagger.py): (obs, states, vla_actions, expert_actions, labels)
- For each frame: train VLA to predict expert action for label=1, VLA action for label=0
- Expert actions weighted more heavily in loss
- Fine-tune from Phase 227 checkpoint (epoch_30.pt)

Key insight: DAgger corrects the VLA's large-displacement failure mode
by providing expert (P-controller) corrections on VLA failure trajectories.

Usage:
  python3 scripts/train_dagger.py \
    --dagger_data data/dagger_pilot_5ep.h5 \
    --base_data data/phase196_clean_50ep.h5 \
    --checkpoint results/phase227_contact_jacobian_train/epoch_30.pt \
    --output results/dagger_phase246_train \
    --epochs 30 --batch_size 32

Expected outcome: VLA improved on large |goal| (|g| > 0.3m) from ~60% → 85%+
"""
import os, sys, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Reuse GoalConditionedPolicy from train_phase227.py
from scripts.train_phase227 import GoalConditionedPolicy, DEVICE

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Data Loading ──────────────────────────────────────────────────────────

def load_dagger_data(path):
    """Load DAgger dataset from collect_dagger.py output."""
    with h5py.File(path, 'r') as f:
        obs = f['obs'][:]           # (N, 3, 224, 224) — preprocessed tensors (float32)
        states = f['states'][:]      # (N, 11)
        vla_actions = f['vla_actions'][:]   # (N, 9)
        expert_actions = f['expert_actions'][:]  # (N, 9)
        labels = f['labels'][:]     # (N,) 0=VLA, 1=expert
        goals = f['goals'][:]        # (N, 2)
        rewards = f['rewards'][:]   # (N,)
        episode_starts = f['episode_starts'][:]

    n_expert = int(labels.sum())
    n_vla = len(labels) - n_expert
    print(f"  DAgger data: {len(labels)} frames, expert={n_expert} ({n_expert/len(labels)*100:.1f}%), "
          f"vla={n_vla} ({n_vla/len(labels)*100:.1f}%)")
    return obs, states, vla_actions, expert_actions, labels, goals, rewards, episode_starts


def load_base_data(path):
    """Load base dataset (from collect_phase196_clean.py or similar)."""
    with h5py.File(path, 'r') as f:
        images = f['images'][:]         # (N, 640, 480, 3)
        states = f['states'][:]         # (N, 11)
        actions = f['actions'][:]       # (N, 9)
        goals = f['goals'][:]           # (N, 2)
        rewards = f['rewards'][:]      # (N,)
        episode_starts = f['episode_starts'][:]

    print(f"  Base data: {len(actions)} frames, {len(episode_starts)-1} episodes")
    return images, states, actions, goals, rewards, episode_starts


def preprocess_images(raw_images, mean=IMG_MEAN, std=IMG_STD):
    """Convert raw (N, H, W, 3) uint8 to (N, 3, 224, 224) normalized tensors."""
    from PIL import Image
    tensors = []
    for img in raw_images:
        pil = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        tensors.append(arr.transpose(2, 0, 1))
    return np.array(tensors, dtype=np.float32)


# ── DAgger Training Loop ──────────────────────────────────────────────────

def train_dagger(dagger_obs, dagger_states, dagger_vla_actions, dagger_expert_actions,
                 dagger_labels, dagger_goals, dagger_rewards,
                 base_images, base_states, base_actions, base_goals, base_rewards,
                 policy, output_dir, epochs=30, batch_size=32, lr=1e-4,
                 dagger_weight=3.0, base_weight=1.0):
    """
    DAgger training with expert correction weighting.

    For each batch:
      - Sample from base data (P-controller demonstration)
      - Sample from DAgger data (VLA failures + expert corrections)
      - Loss = base_weight * MSE(vla_pred, base_action) + dagger_weight * label * MSE(vla_pred, expert_action)
      - label=1 → expert correction gets dagger_weight multiplier
      - label=0 → no gradient (stay with VLA action)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess base images
    print("[Preprocessing base images]")
    base_obs = preprocess_images(base_images)
    del base_images  # free memory

    # DAgger obs is already preprocessed (from collect_dagger.py)
    dagger_obs_tensor = torch.from_numpy(dagger_obs).float()
    dagger_states_tensor = torch.from_numpy(dagger_states).float()
    dagger_expert_tensor = torch.from_numpy(dagger_expert_actions).float()
    dagger_vla_tensor = torch.from_numpy(dagger_vla_actions).float()
    dagger_labels_tensor = torch.from_numpy(dagger_labels).float()
    dagger_goals_tensor = torch.from_numpy(dagger_goals).float()

    base_obs_tensor = torch.from_numpy(base_obs).float()
    base_states_tensor = torch.from_numpy(base_states).float()
    base_actions_tensor = torch.from_numpy(base_actions).float()
    base_goals_tensor = torch.from_numpy(base_goals).float()

    n_base = len(base_obs_tensor)
    n_dagger = len(dagger_obs_tensor)
    print(f"  Base: {n_base} frames, DAgger: {n_dagger} frames")
    print(f"  Training for {epochs} epochs, batch_size={batch_size}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []

    # Freeze CLIP encoder (only train policy head)
    for name, p in policy.named_parameters():
        if 'encoder' in name:
            p.requires_grad = False

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    for epoch in range(epochs):
        epoch_losses = []
        batches_per_epoch = max(n_base, n_dagger) // batch_size

        # Shuffle indices
        base_idx = torch.randperm(n_base)
        dagger_idx = torch.randperm(n_dagger)

        for b in range(batches_per_epoch):
            bi_start = b * batch_size
            if bi_start >= n_base:
                break
            # Sample base batch
            bi = base_idx[b * batch_size:(b + 1) * batch_size]
            base_img_batch = base_obs_tensor[bi].to(DEVICE)
            base_state_batch = base_states_tensor[bi].to(DEVICE)
            base_action_batch = base_actions_tensor[bi].to(DEVICE)

            # Sample DAgger batch
            di = dagger_idx[b * batch_size:(b + 1) * batch_size]
            if len(di) == 0:
                continue

            # Build DAgger batch first, then validate
            dagger_img_batch = dagger_obs_tensor[di].float().to(DEVICE)
            dagger_state_batch = dagger_states_tensor[di].to(DEVICE)
            dagger_expert_batch = dagger_expert_tensor[di].to(DEVICE)
            dagger_vla_batch = dagger_vla_tensor[di].to(DEVICE)
            dagger_labels_batch = dagger_labels_tensor[di].to(DEVICE)

            # Explicit shape check — catch CLIP reshape bug early
            assert base_img_batch.shape[0] > 0, f"Empty base batch at b={b}"
            assert dagger_img_batch.shape[0] > 0, f"Empty dagger batch at b={b}"
            assert base_img_batch.shape[1:] == (3, 224, 224), f"Bad base img shape: {base_img_batch.shape}"
            assert dagger_img_batch.shape[1:] == (3, 224, 224), f"Bad dagger img shape: {dagger_img_batch.shape}"

            # Pad if needed
            if base_img_batch.shape[0] < batch_size:
                pad = batch_size - base_img_batch.shape[0]
                base_img_batch = torch.cat([base_img_batch, base_img_batch[:pad]], dim=0)
                base_state_batch = torch.cat([base_state_batch, base_state_batch[:pad]], dim=0)
                base_action_batch = torch.cat([base_action_batch, base_action_batch[:pad]], dim=0)

            if dagger_img_batch.shape[0] < batch_size:
                pad = batch_size - dagger_img_batch.shape[0]
                dagger_img_batch = torch.cat([dagger_img_batch, dagger_img_batch[:pad]], dim=0)
                dagger_state_batch = torch.cat([dagger_state_batch, dagger_state_batch[:pad]], dim=0)
                dagger_expert_batch = torch.cat([dagger_expert_batch, dagger_expert_batch[:pad]], dim=0)
                dagger_vla_batch = torch.cat([dagger_vla_batch, dagger_vla_batch[:pad]], dim=0)
                dagger_labels_batch = torch.cat([dagger_labels_batch, dagger_labels_batch[:pad]], dim=0)

            optimizer.zero_grad()

            # ── Base data loss (P-controller demonstrations) ────────────────
            t_base = torch.zeros(base_img_batch.shape[0], 1, device=DEVICE)
            pred_base = policy(base_img_batch, base_state_batch,
                               torch.zeros_like(base_action_batch), t_base)
            loss_base = ((pred_base - base_action_batch) ** 2).mean(dim=-1).mean()

            # ── DAgger loss ───────────────────────────────────────────────────
            # VLA prediction on DAgger states
            t_dagger = torch.zeros(dagger_img_batch.shape[0], 1, device=DEVICE)
            pred_dagger = policy(dagger_img_batch, dagger_state_batch,
                                 torch.zeros_like(dagger_expert_batch), t_dagger)

            # Expert correction loss: weighted by label (1=expert correction)
            expert_loss = ((pred_dagger - dagger_expert_batch) ** 2).mean(dim=-1)
            vla_loss = ((pred_dagger - dagger_vla_batch) ** 2).mean(dim=-1)

            # DAgger loss = label * expert_loss + (1-label) * vla_loss
            # Weight expert corrections more heavily
            dagger_loss_per_frame = dagger_labels_batch * expert_loss * dagger_weight + \
                                    (1 - dagger_labels_batch) * vla_loss
            loss_dagger = dagger_loss_per_frame.mean()

            # Combined loss
            loss = base_weight * loss_base + loss_dagger

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        lr_now = scheduler.get_last_lr()[0]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={lr_now:.2e}")

    # Plot loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase 246: DAgger Training Loss')
    plt.savefig(f"{output_dir}/training_loss.png", dpi=100)
    plt.close()

    # Save final model
    torch.save({
        'epoch': epochs,
        'loss': losses[-1],
        'policy_state_dict': policy.state_dict(),
    }, f"{output_dir}/final_policy.pt")

    print(f"\n✅ Training complete: {output_dir}/final_policy.pt")
    return losses


def main():
    parser = argparse.ArgumentParser(description='Phase 246: DAgger Training for LeKiWi VLA')
    parser.add_argument('--dagger_data', type=str, required=True,
                        help='DAgger dataset from collect_dagger.py')
    parser.add_argument('--base_data', type=str, default='data/phase196_clean_50ep.h5',
                        help='Base P-controller dataset for training')
    parser.add_argument('--checkpoint', type=str,
                        default='results/phase227_contact_jacobian_train/epoch_30.pt',
                        help='Starting checkpoint (Phase 227 or Phase 196)')
    parser.add_argument('--output', type=str,
                        default='results/dagger_phase246_train',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dagger_weight', type=float, default=3.0,
                        help='Weight for expert correction loss (default 3.0)')
    parser.add_argument('--base_weight', type=float, default=1.0,
                        help='Weight for base data loss (default 1.0)')
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 246: DAgger Training")
    print("=" * 60)
    print(f"  DAgger data: {args.dagger_data}")
    print(f"  Base data: {args.base_data}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}, batch_size={args.batch_size}")
    print(f"  DAgger weight: {args.dagger_weight}, base_weight: {args.base_weight}")
    print()

    # ── Load policy from checkpoint ─────────────────────────────────────────
    print("[Loading policy from checkpoint]")
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    if 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
    else:
        policy.load_state_dict(ckpt, strict=False)
    print(f"  Loaded: {args.checkpoint}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("[Loading DAgger data]")
    (dagger_obs, dagger_states, dagger_vla_actions, dagger_expert_actions,
     dagger_labels, dagger_goals, dagger_rewards, dagger_ep_starts) = load_dagger_data(args.dagger_data)

    print("[Loading base data]")
    base_images, base_states, base_actions, base_goals, base_rewards, base_ep_starts = load_base_data(args.base_data)

    print()

    # ── Train ────────────────────────────────────────────────────────────────
    losses = train_dagger(
        dagger_obs, dagger_states, dagger_vla_actions, dagger_expert_actions,
        dagger_labels, dagger_goals, dagger_rewards,
        base_images, base_states, base_actions, base_goals, base_rewards,
        policy, args.output,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        dagger_weight=args.dagger_weight, base_weight=args.base_weight,
    )

    print(f"\n[Final losses]")
    print(f"  First: {losses[0]:.4f}")
    print(f"  Last:  {losses[-1]:.4f}")
    print(f"  Delta: {losses[0] - losses[-1]:.4f}")


if __name__ == "__main__":
    main()