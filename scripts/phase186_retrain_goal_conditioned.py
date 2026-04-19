#!/usr/bin/env python3
"""
Phase 186: FIXED — Retrain VLA with state_dim=11 (goal conditioning)

Dataset: phase181_symmetrized_10k.h5
  - 10,000 frames with goal_positions (2D) stored SEPARATELY
  - States: (10000, 9) — arm_pos(6) + wheel_vel(3) — NO goal in state
  - Goals: (10000, 2) — goal x, y

FIX: Build 11D state at training time: concat(states[9D], goals[2D]) = 11D
  - state[:, 0:9] = arm_pos(6) + wheel_vel(3) 
  - state[:, 9:11] = goal_xy(2)
  - Normalize: arm_pos/2.0, wheel_vel/0.5, goal_xy/0.5 (all to [-1,1])

Architecture: CLIPFlowMatchingPolicy(state_dim=11)
  - CLIP ViT-B/32 frozen encoder
  - FlowMatchingHead(state_dim=11) — total_dim = 512 + 11 + 9 + 256 = 788
  - Only flow_head is trainable (~970K params)

Training: 5 epochs, batch=16, lr=1e-4
  - Goal: verify SR > 0% with goal conditioning
"""
import sys, os, h5py, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from scripts.train_task_oriented import CLIPFlowMatchingPolicy

DEVICE = 'cpu'
STATE_DIM = 11   # arm_pos(6) + wheel_vel(3) + goal_xy(2)
ACTION_DIM = 9
HIDDEN = 512
LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 16

DATA_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/data/phase181_symmetrized_10k.h5'
OUT_DIR = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase186_goal_conditioned_train'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


class GoalConditionedDataset(Dataset):
    """Dataset that builds 11D goal-conditioned state from h5.

    h5 stores:
      states: (N, 9) = arm_pos(6) + wheel_vel(3)
      goals:  (N, 2) = goal_x, goal_y
      images: (N, 224, 224, 3) uint8
      actions: (N, 9) normalized [-1, 1]
      rewards: (N,) float

    We build: state11 = concat(states, goals) at indices [0:9, 9:11]
    """
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.images  = f['images'][:]       # (N, 224, 224, 3) uint8
            self.states  = f['states'][:]         # (N, 9)
            self.actions = f['actions'][:]         # (N, 9)
            self.goals   = f['goal_positions'][:] # (N, 2)
            self.rewards = f['rewards'][:]

        self.n = len(self.actions)
        pos_y = (self.goals[:, 1] >= 0).sum()
        neg_y = (self.goals[:, 1] < 0).sum()
        w1_pos = self.actions[self.goals[:, 1] >= 0][:, 6].mean()
        w1_neg = self.actions[self.goals[:, 1] < 0][:, 6].mean()

        print(f'[Dataset] {self.n} frames')
        print(f'  +Y: {pos_y}, -Y: {neg_y}')
        print(f'  w1 mean +Y: {w1_pos:+.4f}, w1 mean -Y: {w1_neg:+.4f}')
        print(f'  States: {self.states.shape}, range [{self.states.min():.2f}, {self.states.max():.2f}]')
        print(f'  Goals: {self.goals.shape}, range [{self.goals.min():.3f}, {self.goals.max():.3f}]')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW

        # Build 11D state: [arm_pos(6), wheel_vel(3), goal_xy(2)]
        state9 = self.states[idx].astype(np.float32)  # arm_pos(6) + wheel_vel(3)
        goal = self.goals[idx].astype(np.float32)     # goal_x, goal_y

        # Normalize: arm_pos/2.0, wheel_vel/0.5, goal_xy/0.5
        arm_norm = np.clip(state9[:6] / 2.0, -1, 1)
        wheel_norm = np.clip(state9[6:9] / 0.5, -1, 1)
        goal_norm = np.clip(goal / 0.5, -1, 1)

        state11 = np.concatenate([arm_norm, wheel_norm, goal_norm]).astype(np.float32)
        action = self.actions[idx].astype(np.float32)
        reward = self.rewards[idx]

        return (torch.from_numpy(img),
                torch.from_numpy(state11),
                torch.from_numpy(action),
                torch.tensor(reward, dtype=torch.float32))


def main():
    print(f'Phase 186: Retrain VLA with state_dim=11 (goal conditioning)')
    print(f'Device: {DEVICE}')
    print(f'LR: {LR}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}')
    print(f'Data: {DATA_PATH}')
    print(f'State dim: {STATE_DIM} (goal-aware)')

    dataset = GoalConditionedDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Create NEW policy with state_dim=11
    print(f'\n[Policy] CLIPFlowMatchingPolicy (state_dim={STATE_DIM}, action_dim={ACTION_DIM})')
    policy = CLIPFlowMatchingPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden=HIDDEN,
        device=DEVICE
    )

    # Only train flow_head; vision encoder frozen
    optimizer = torch.optim.AdamW(policy.flow_head.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')
    history = []

    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        policy.flow_head.train()

        for batch_idx, (imgs, states11, actions, rewards) in enumerate(loader):
            imgs    = imgs.to(DEVICE)
            states11 = states11.to(DEVICE)
            actions = actions.to(DEVICE)

            # Random timestep in [0, 1)
            t = (torch.rand(len(imgs), 1, device=DEVICE) ** 1.5) * 0.999

            # Flow matching: create noisy action
            noise = torch.randn_like(actions)
            noisy = t.expand_as(actions) * actions + (1 - t.expand_as(actions)) * noise
            target = actions - noise  # flow velocity

            # Forward pass
            pred = policy(imgs, states11, noisy, t)

            # MSE loss on flow
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.flow_head.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'lr': scheduler.get_last_lr()[0]
        })
        elapsed = time.time() - t0

        # Quick eval: check action magnitude
        policy.flow_head.eval()
        with torch.no_grad():
            eval_imgs = torch.zeros(4, 3, 224, 224, device=DEVICE)
            eval_states = torch.randn(4, STATE_DIM, device=DEVICE)
            eval_t = torch.rand(4, 1, device=DEVICE)
            eval_pred = policy(eval_imgs, eval_states, torch.randn(4, 9, device=DEVICE), eval_t)
            pred_mean = eval_pred[:, 6:9].abs().mean().item()

        print(f'Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.6f}, '
              f'lr={scheduler.get_last_lr()[0]:.2e}, '
              f'wheel_pred_mean={pred_mean:.6f}, '
              f'elapsed={elapsed:.0f}s')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save({
                'flow_head_state_dict': policy.flow_head.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'history': history,
                'policy_config': {
                    'state_dim': STATE_DIM,
                    'action_dim': ACTION_DIM,
                    'hidden': HIDDEN
                }
            }, OUT_DIR + '/best_policy.pt')

        torch.save({
            'flow_head_state_dict': policy.flow_head.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'history': history,
            'policy_config': {
                'state_dim': STATE_DIM,
                'action_dim': ACTION_DIM,
                'hidden': HIDDEN
            }
        }, OUT_DIR + f'/epoch_{epoch}.pt')

    print(f'\nTraining complete. Best epoch: {best_epoch+1} (loss={best_loss:.6f})')
    print(f'Output: {OUT_DIR}')

    # Save training history
    with open(OUT_DIR + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()
