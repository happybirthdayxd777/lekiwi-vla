#!/usr/bin/env python3
"""
Phase 179: Create symmetrized training dataset.
For each frame, create a Y-mirrored counterpart: negate goal_y, flip w1 sign.
Result: 10000 original + 10000 mirrored = 20000 balanced frames.
"""
import sys, os, h5py, numpy as np
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')

ROOT = '/Users/i_am_ai/hermes_research/lekiwi_vla'
src_path = ROOT + '/data/jacobian_pctrl_50ep_kP01.h5'
dst_path = ROOT + '/data/phase179_symmetrized.h5'

print(f'Reading {src_path}...')
with h5py.File(src_path, 'r') as f:
    goals = f['goal_positions'][:]
    actions = f['actions'][:]
    states = f['states'][:]
    rewards = f['rewards'][:]
    ep_starts = f['episode_starts'][:]

print(f'Source: {len(goals)} frames, {len(ep_starts)} episodes')
print(f'  +Y goals: {(goals[:,1] >= 0).sum()}, -Y goals: {(goals[:,1] < 0).sum()}')

# Create Y-mirror: negate goal_y, flip w1 (index 6)
mirrored_goals = goals.copy()
mirrored_goals[:, 1] = -mirrored_goals[:, 1]  # negate Y
mirrored_actions = actions.copy()
mirrored_actions[:, 6] = -actions[:, 6]  # flip w1 sign
mirrored_states = states.copy()
mirrored_rewards = rewards.copy()

# Combine original + mirrored
sym_goals = np.concatenate([goals, mirrored_goals], axis=0)
sym_actions = np.concatenate([actions, mirrored_actions], axis=0)
sym_states = np.concatenate([states, mirrored_states], axis=0)
sym_rewards = np.concatenate([rewards, mirrored_rewards], axis=0)

print(f'\nAfter full symmetrization:')
print(f'  Total: {len(sym_goals)} frames')
print(f'  goal_y>=0: {(sym_goals[:,1] >= 0).sum()}, goal_y<0: {(sym_goals[:,1] < 0).sum()}')

# Stats by quadrant
n_orig = len(goals)
print(f'\nOriginal data ({n_orig} frames):')
print(f'  +Y: w1 mean={actions[goals[:,1]>=0, 6].mean():+.4f}')
print(f'  -Y: w1 mean={actions[goals[:,1]<0, 6].mean():+.4f}')

print(f'\nMirrored data ({n_orig} frames):')
print(f'  +Y: w1 mean={mirrored_actions[mirrored_goals[:,1]>=0, 6].mean():+.4f}')
print(f'  -Y: w1 mean={mirrored_actions[mirrored_goals[:,1]<0, 6].mean():+.4f}')

# Save
print(f'\nWriting {dst_path}...')
if os.path.exists(dst_path):
    os.remove(dst_path)
with h5py.File(dst_path, 'w') as f:
    f.create_dataset('goal_positions', data=sym_goals, compression='gzip')
    f.create_dataset('actions', data=sym_actions, compression='gzip')
    f.create_dataset('states', data=sym_states, compression='gzip')
    f.create_dataset('rewards', data=sym_rewards, compression='gzip')
    f.create_dataset('episode_starts', data=np.concatenate([ep_starts, ep_starts + n_orig]))
    f.attrs['symmetrized'] = True
    f.attrs['source'] = 'jacobian_pctrl_50ep_kP01.h5'
    f.attrs['note'] = 'Full Y-mirror: negate goal_y, flip w1, double dataset size'

print(f'Saved: {dst_path} ({os.path.getsize(dst_path)/1e6:.1f} MB)')
