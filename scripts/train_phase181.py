#!/usr/bin/env python3
"""
Phase 181: Train VLA on symmetrized image dataset (10k frames WITH images).

Dataset: phase181_symmetrized_10k.h5
  - 10,000 frames (5k original + 5k Y-mirrored)
  - Images: (10000, 224, 224, 3) uint8 — REAL robot images
  - States: (10000, 9) normalized z-score
  - Actions: (10000, 9) normalized [-1, 1]
  - Goals: (10000, 2) — balanced +Y/-Y

Key architecture: CLIPFlowMatchingPolicy (state_dim=9)
  - CLIP ViT-B/32 vision encoder (frozen)
  - Flow matching action head
  - 9D wheel+arm action output

Goal: Train vision VLA with proper images, balanced +Y/-Y dataset.
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
STATE_DIM = 9
ACTION_DIM = 9
HIDDEN = 512
LR = 2e-5
EPOCHS = 10
BATCH_SIZE = 16  # Small batch for MPS

DATA_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/data/phase181_symmetrized_10k.h5'
OUT_DIR = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_vision_train'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

class SymVisionDataset(Dataset):
    """Dataset with real images and normalized states."""
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.images  = f['images'][:]       # (N, 224, 224, 3) uint8
            self.states  = f['states'][:]         # (N, 9) normalized
            self.actions = f['actions'][:]         # (N, 9) normalized [-1, 1]
            self.goals   = f['goal_positions'][:] # (N, 2)
            self.rewards = f['rewards'][:]
            if 'state_mean' in f:
                self.state_mean = f['state_mean'][:]
                self.state_std  = f['state_std'][:]
            else:
                self.state_mean = None
                self.state_std  = None

        self.n = len(self.actions)
        # Verify balance
        pos_y = (self.goals[:, 1] >= 0).sum()
        neg_y = (self.goals[:, 1] < 0).sum()
        w1_pos = self.actions[self.goals[:, 1] >= 0][:, 6].mean()
        w1_neg = self.actions[self.goals[:, 1] < 0][:, 6].mean()
        
        print(f'[Dataset] {self.n} frames')
        print(f'  +Y: {pos_y}, -Y: {neg_y}')
        print(f'  w1 mean +Y: {w1_pos:+.4f}, w1 mean -Y: {w1_neg:+.4f}')
        print(f'  Images: {self.images.shape}, range [{self.images.min()}, {self.images.max()}]')
        print(f'  States: {self.states.shape}, range [{self.states.min():.2f}, {self.states.max():.2f}]')
        print(f'  Actions: {self.actions.shape}, range [{self.actions.min():.4f}, {self.actions.max():.4f}]')
        print(f'  NaN states: {np.isnan(self.states).sum()}, NaN actions: {np.isnan(self.actions).sum()}')
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Image: uint8 → [0, 1] float → [B, C, H, W] (CLIP expects this)
        img = self.images[idx].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        
        state  = self.states[idx].astype(np.float32)
        action = self.actions[idx].astype(np.float32)
        goal   = self.goals[idx].astype(np.float32)
        reward = self.rewards[idx]
        
        return (torch.from_numpy(img),
                torch.from_numpy(state),
                torch.from_numpy(action),
                torch.tensor(reward, dtype=torch.float32))

def main():
    print(f'Phase 181: Vision VLA Training on Symmetrized Image Dataset')
    print(f'Device: {DEVICE}')
    print(f'LR: {LR}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}')
    print(f'Data: {DATA_PATH}')
    
    # Load dataset
    dataset = SymVisionDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Policy: CLIPFlowMatchingPolicy with 9D state
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
        
        for batch_idx, (imgs, states, actions, rewards) in enumerate(loader):
            imgs    = imgs.to(DEVICE)
            states  = states.to(DEVICE)
            actions = actions.to(DEVICE)
            
            # Random timestep in [0, 1)
            t = (torch.rand(len(imgs), 1, device=DEVICE) ** 1.5) * 0.999
            
            # Flow matching: create noisy action
            noise = torch.randn_like(actions)
            noisy = t.expand_as(actions) * actions + (1 - t.expand_as(actions)) * noise
            target = actions - noise  # flow velocity
            
            # Forward pass through CLIP + flow head
            pred = policy(imgs, states, noisy, t)
            
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
            eval_states = torch.randn(4, 9, device=DEVICE)
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
            # Save only flow_head (969K params, ~4MB vs 500MB full model)
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
        
        # Save checkpoint (flow_head only)
        torch.save({
            'flow_head_state_dict': policy.flow_head.state_dict(),
            'epoch': epoch,
            'loss': avg_loss,
            'history': history
        }, OUT_DIR + f'/epoch_{epoch}.pt')
    
    print(f'\nTraining complete. Best epoch: {best_epoch+1} (loss={best_loss:.6f})')
    print(f'Output: {OUT_DIR}')
    
    with open(OUT_DIR + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main()
