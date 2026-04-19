#!/usr/bin/env python3
"""
Phase 179: Train VLA on symmetrized dataset.
Dataset: phase179_symmetrized.h5 (20k frames, balanced +Y/-Y, w1 sign corrected)
Goal: Fix w1 sign bias → balanced SR across all quadrants.
"""
import sys, os, h5py, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

DEVICE = 'cpu'  # No GPU on this machine
STATE_DIM = 9   # arm_pos(6) + wheel_vel(3)
ACTION_DIM = 9
HIDDEN = 512
LR = 2e-5
EPOCHS = 10
BATCH_SIZE = 32

DATA_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/data/phase179_symmetrized.h5'
OUT_DIR = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase179_symmetrized_train'
os.makedirs(OUT_DIR, exist_ok=True)

class SymDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.goals = f['goal_positions'][:]
            self.actions = f['actions'][:]
            self.states = f['states'][:]
            self.rewards = f['rewards'][:]
        self.n = len(self.actions)
        print(f'Dataset: {self.n} frames')
        print(f'  +Y: {(self.goals[:,1]>=0).sum()}, -Y: {(self.goals[:,1]<0).sum()}')
        print(f'  w1 mean +Y: {self.actions[self.goals[:,1]>=0, 6].mean():+.4f}')
        print(f'  w1 mean -Y: {self.actions[self.goals[:,1]<0, 6].mean():+.4f}')
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        state = torch.from_numpy(self.states[idx]).float()      # (9,)
        action = torch.from_numpy(self.actions[idx]).float()    # (9,)
        goal = torch.from_numpy(self.goals[idx]).float()        # (2,)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32)
        return state, action, goal, reward, torch.tensor(idx)

def main():
    print(f'Device: {DEVICE}')
    print(f'Learning rate: {LR}, Epochs: {EPOCHS}, Batch: {BATCH_SIZE}')
    
    # Load data
    dataset = SymDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Policy
    policy = CLIPFlowMatchingPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                                    hidden=HIDDEN, device=DEVICE)
    optimizer = torch.optim.AdamW(policy.flow_head.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_loss = float('inf')
    history = []
    
    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        policy.flow_head.train()
        
        for batch_idx, (states, actions, goals, rewards, indices) in enumerate(loader):
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            t = torch.rand(len(states), 1).to(DEVICE)  # random timestep in [0,1)
            
            # Flow matching: predict flow from noisy action to target
            # noise ~ N(0,I), noisy = t*target + (1-t)*noise
            noise = torch.randn_like(actions)
            t_3d = t.expand_as(actions)  # (B, 1) -> (B, 9)
            noisy = t_3d * actions + (1 - t_3d) * noise
            target = actions - noise  # flow matching velocity
            
            # Dummy image (224x224)
            dummy_img = torch.zeros(len(states), 3, 224, 224, dtype=torch.float32, device=DEVICE)
            
            # Forward: CLIPFlowMatchingPolicy.forward handles vision encoding
            pred = policy(dummy_img, states, noisy, t)
            
            # Loss: MSE on flow
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.flow_head.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss, 'lr': scheduler.get_last_lr()[0]})
        elapsed = time.time() - t0
        
        print(f'Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.2e}, elapsed={elapsed:.0f}s')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'policy_state_dict': policy.state_dict(), 'epoch': epoch,
                        'loss': avg_loss, 'history': history},
                       OUT_DIR + '/best_policy.pt')
            best_epoch = epoch
        
        torch.save({'policy_state_dict': policy.state_dict(), 'epoch': epoch,
                    'loss': avg_loss, 'history': history},
                   OUT_DIR + f'/epoch_{epoch}.pt')
    
    # Final save
    torch.save({'policy_state_dict': policy.state_dict(), 'epoch': EPOCHS-1,
                'loss': avg_loss, 'history': history},
               OUT_DIR + '/final_policy.pt')
    
    print(f'\nTraining complete. Best epoch: {best_epoch+1} (loss={best_loss:.6f})')
    print(f'Output: {OUT_DIR}')
    
    with open(OUT_DIR + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main()
