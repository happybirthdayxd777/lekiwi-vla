#!/usr/bin/env python3
"""
Phase 29 — Train on high-quality URDF P-controller data
=======================================================
Training on lekiwi_goal_urdf_10k.h5 (4000 frames, P-controller, URDF physics)
Hypothesis: P-controller data is higher quality (rewards go to 1.0,
actions are more purposeful) vs GridSearchController (max reward=0.1).
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path
from datetime import datetime

from scripts.train_task_oriented import CLIPFlowMatchingPolicy


def load_data(h5_path):
    """Load URDF goal-directed dataset."""
    f = h5py.File(h5_path, 'r')
    images = f['images'][:]        # (N, 224, 224, 3) uint8
    states_9d = f['states'][:]    # (N, 9) = arm_pos(6) + wheel_vel(3)
    actions = f['actions'][:]     # (N, 9) = arm(6) + wheel(3)
    goals = f['goal_positions'][:] # (N, 2)
    rewards = f['rewards'][:]      # (N,)
    f.close()

    # Build 11D goal-aware state: arm_pos(6) + wheel_vel(3) + goal_xy(2)
    goal_norm = np.clip(goals / 1.0, -1.0, 1.0).astype(np.float32)
    states_11d = np.concatenate([states_9d, goal_norm], axis=1)  # (N, 11)

    # Normalize images to tensor format
    imgs = []
    for img in images:
        pil_resized = PILImage.fromarray(img).resize((224, 224))
        img_np = np.array(pil_resized).astype(np.float32) / 255.0
        img_chw = img_np.transpose(2, 0, 1)
        imgs.append(torch.from_numpy(img_chw))

    print(f"Loaded {len(states_11d)} frames from {h5_path}")
    print(f"  States: {states_9d.shape} → 11D goal-aware")
    print(f"  Rewards: mean={rewards.mean():.3f}, max={rewards.max():.3f}")
    print(f"  Goals: {goals.min(axis=0)} to {goals.max(axis=0)}")
    print(f"  Positive reward frames: {(rewards > 0).sum()} / {len(rewards)}")

    return torch.stack(imgs), torch.from_numpy(states_11d), torch.from_numpy(actions), torch.from_numpy(rewards)


def train(policy, imgs, states, actions, rewards, epochs=50, device='cpu',
          output_dir='results/phase29_urdf_train'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy.to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    n = len(imgs)
    indices = np.arange(n)

    epoch_losses = []
    grad_norms = []

    print(f"\nTraining on {n} frames for {epochs} epochs...")

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, 16):  # batch_size=16
            batch_idx = indices[i:i+16]
            img_batch = imgs[batch_idx].to(device)
            state_batch = states[batch_idx].to(device)
            action_batch = actions[batch_idx].to(device)

            # Flow matching: x_t = (1-t)*action + t*noise
            t = (torch.rand(batch_idx.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(action_batch)
            x_t = (1 - t) * action_batch + t * noise

            optimizer.zero_grad()

            # Forward: v_pred = policy(img, state, x_t, t)
            v_pred = policy(img_batch, state_batch, x_t, t)
            v_target = action_batch - noise

            loss = ((v_pred - v_target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        epoch_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            grad_norm = sum(p.grad.norm().item() for p in policy.parameters() if p.grad is not None)
            grad_norms.append(grad_norm)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.5f}, grad={grad_norm:.2f}")

        if (epoch + 1) % 20 == 0:
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  => Saved {ckpt_path.name}")

    # Save final
    final_path = output_dir / 'final_policy.pt'
    torch.save({
        'epoch': epochs,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Training Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(grad_norms)
    plt.title('Gradient Norm')
    plt.xlabel('Epoch × 5')
    plt.ylabel('||grad||')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=100)
    plt.close()

    print(f"\nTraining complete. Final loss: {epoch_losses[-1]:.5f}")
    print(f"Checkpoints: {output_dir}")

    with open(output_dir / 'training_log.txt', 'w') as log:
        log.write(f"Phase 29 — URDF P-controller data training\n")
        log.write(f"Started: {datetime.now()}\n")
        log.write(f"Epochs: {epochs}\n")
        log.write(f"Dataset: lekiwi_goal_urdf_10k.h5 (4000 frames)\n")
        log.write(f"Final loss: {epoch_losses[-1]:.5f}\n")
        for i, l in enumerate(epoch_losses):
            log.write(f"epoch {i}: loss={l:.5f}\n")

    return epoch_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=str, default="data/lekiwi_goal_urdf_10k.h5")
    parser.add_argument("--output", type=str, default="results/phase29_urdf_train")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 29 — Training on URDF P-controller data")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print()

    imgs, states, actions, rewards = load_data(args.data)

    print(f"\nInitializing policy (state_dim=11, action_dim=9, hidden=512)...")
    policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device=args.device)

    losses = train(policy, imgs, states, actions, rewards,
                   epochs=args.epochs, device=args.device, output_dir=args.output)

    print("\n✓ Phase 29 training complete")
