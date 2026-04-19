#!/usr/bin/env python3
"""
resume_phase196.py — Resume Phase 196 VLA training from epoch 4, continue to epoch 30.
Phase 197 found epoch_4 gets 67% SR (beats P-ctrl 30% in 3-goal quick test).
Continue training to get the full 30-epoch policy.
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
from pathlib import Path
import time

DEVICE = "cpu"
print(f"[Resume Phase 196] Device: {DEVICE}")

from train_phase196 import GoalConditionedPolicy, Phase196Replay, train

def resume_training(checkpoint_path, data_path, output_dir, resume_epoch=4, total_epochs=30, batch_size=32, lr=1e-4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"[Resume] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if 'policy_state_dict' in ckpt:
        epoch_start = ckpt.get('epoch', resume_epoch) + 1
        saved_losses = ckpt.get('losses', [float('nan')] * (resume_epoch + 1))
        config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    else:
        epoch_start = resume_epoch + 1
        saved_losses = [float('nan')] * (resume_epoch + 1)
        config = {'state_dim': 11, 'action_dim': 9, 'hidden': 512}

    print(f"[Resume] Starting from epoch {epoch_start}, total to train: {total_epochs - epoch_start}")

    # Create policy and load weights
    policy = GoalConditionedPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden=config['hidden'],
        device=DEVICE
    )
    if 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
    else:
        policy.load_state_dict(ckpt, strict=False)
    policy.to(DEVICE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    # Step scheduler to where it would be after resume_epoch
    for _ in range(resume_epoch):
        scheduler.step()

    replay = Phase196Replay(data_path, batch_size=batch_size)

    # Run training
    policy.train()
    losses = list(saved_losses)
    t_start = time.time()

    for epoch in range(epoch_start, total_epochs):
        epoch_loss = 0.0
        n_batches = max(1, replay.n // batch_size)
        n_batches = min(n_batches, 200)

        for _ in range(n_batches):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img = batch_img.to(DEVICE)
            batch_state = batch_state.to(DEVICE)
            batch_action = batch_action.to(DEVICE)
            batch_weights = batch_weights.to(DEVICE)

            t_batch = (torch.rand(batch_img.shape[0], 1, device=DEVICE) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            alpha = 1 - t_batch.squeeze(-1)
            x_t = alpha.unsqueeze(-1) * batch_action + t_batch.squeeze(-1).unsqueeze(-1) * noise

            v_pred = policy(batch_img, batch_state, x_t, t_batch)
            v_target = batch_action - noise

            loss = ((v_pred - v_target) ** 2).mean(dim=-1)
            loss = (loss * batch_weights / batch_weights.mean()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()
        elapsed = time.time() - t_start

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{total_epochs}: loss={avg_loss:.4f}, lr={lr_now:.6f}, elapsed={elapsed:.0f}s")

        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'loss': avg_loss,
                'losses': losses,
                'policy_config': config
            }, output_dir / f'epoch_{epoch}.pt')

    # Save best
    best_epoch = np.nanargmin(losses)
    torch.save({
        'epoch': best_epoch,
        'policy_state_dict': policy.state_dict(),
        'losses': losses,
        'policy_config': config
    }, output_dir / 'best_policy.pt')

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Phase 196 — Resume: GoalConditioned VLA (epoch 4→30)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(output_dir / 'training_loss.png', dpi=100)
    plt.close()

    import json
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump([{'epoch': i, 'loss': float(losses[i])} for i in range(len(losses))], f)

    print(f"\n✓ Best epoch: {best_epoch+1}, loss={losses[best_epoch]:.4f}")
    print(f"✓ Saved: {output_dir / 'best_policy.pt'}")
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='results/phase196_contact_jacobian_train/epoch_4.pt')
    parser.add_argument('--data', type=str, default='data/phase196_clean_50ep.h5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='results/phase196_contact_jacobian_train')
    args = parser.parse_args()

    losses = resume_training(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        resume_epoch=4,
        total_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    print(f"\n✅ Training complete. Final losses: {[f'{l:.4f}' for l in losses[-5:]]}")
