#!/usr/bin/env python3
"""
Phase 181: Create symmetrized image dataset from phase59 + train vision VLA.

Problem: phase179_symmetrized.h5 has 20k frames but NO images.
         phase59_urdf_goal_5k.h5 has 5k frames WITH images.

Solution:
  1. Take phase59_urdf_goal_5k (5k frames, images, normalized states [-1,1])
  2. Y-mirror ALL frames: negate goal_y, flip w1 sign
  3. Result: 10k frames with images, balanced +Y/-Y

Dataset structure (phase59):
  states: (5000, 9) — raw MuJoCo qpos [arm(6) + wheel(3)]
  actions: (5000, 9) — normalized [-1, 1]
  images: (5000, 224, 224, 3) — uint8 RGB
  goals: (5000, 2) — world coordinates

Y-Mirror transform:
  goal_y → -goal_y  (mirror Y axis)
  w1 → -w1  (w1 direction reverses with Y mirror)
  keep: arm actions, images (top-down view is Y-symmetric)
"""
import sys, os, h5py, numpy as np
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

SRC = 'data/phase59_urdf_goal_5k.h5'
OUT = 'data/phase181_symmetrized_10k.h5'

print(f"Loading {SRC}...")
with h5py.File(SRC, 'r') as f:
    images  = f['images'][:]
    states  = f['states'][:]
    actions  = f['actions'][:]
    goals    = f['goal_positions'][:]
    rewards  = f['rewards'][:]

N = len(actions)
print(f"Original: {N} frames")
print(f"  States: {states.shape}, range [{states.min():.2f}, {states.max():.2f}]")
print(f"  Actions: {actions.shape}, range [{actions.min():.4f}, {actions.max():.4f}]")
print(f"  Images: {images.shape}, range [{images.min()}, {images.max()}]")
print(f"  Goals: {goals.shape}")

# ── Normalize states ──────────────────────────────────────────────────────────
# phase59 states are raw MuJoCo qpos. We need to normalize to [-1, 1].
# From analysis: dim[0-5] = arm joints, dim[6-8] = wheel joints
# BUT dim[6-8] values like 152, 131, 96 are way outside [-1, 1]
# These are accumulated wheel rotations (not normalized velocities!)
# 
# Actually, looking at phase59 state sample: [0.065, 0.152, -0.023, -0.023, 0.158, 0.077, 0, 0]
# Wait, that's from the symmetrized dataset, not phase59.
# Let me check the actual phase59 raw values...

# From earlier output:
#   state[6]: mean=+152.758, min=-241.876, max=+208.751  ← wheel1 accumulated angle
#   state[7]: mean=+131.369, min=-77.208, max=+599.022   ← wheel2 accumulated angle
#   state[8]: mean=+96.638, min=-113.439, max=+208.767   ← wheel3 accumulated angle
# These are wheel joint POSITIONS (qpos), not velocities (qvel).

# For training, we need normalized states. Let's normalize per dimension.
# State normalization: zero-mean, unit-variance (per-dimension)
state_mean = states.mean(axis=0, keepdims=True)  # (1, 9)
state_std  = states.std(axis=0, keepdims=True)   # (1, 9)
# Avoid division by zero
state_std = np.where(state_std < 1e-6, 1.0, state_std)
states_norm = (states - state_mean) / state_std

print(f"\nState normalization:")
for i in range(9):
    col = states_norm[:, i]
    print(f"  [{i}]: mean={col.mean():+.4f}, std={col.std():.4f}, min={col.min():+.4f}, max={col.max():+.4f}")

# ── Y-Mirror transform ───────────────────────────────────────────────────────
# Mirror goal_y: goal[:, 1] → -goal[:, 1]
goals_mirrored = goals.copy()
goals_mirrored[:, 1] = -goals_mirrored[:, 1]

# Mirror w1 (wheel 1 action): actions[:, 6] → -actions[:, 6]
# (w1 axis points in Y direction, so Y-mirror reverses it)
actions_mirrored = actions.copy()
actions_mirrored[:, 6] = -actions_mirrored[:, 6]

# States stay the same (the robot state is symmetric)
states_norm_mirrored = states_norm.copy()

# Images stay the same (top-down view, Y-mirror is symmetric)
images_mirrored = images.copy()

# Rewards stay the same (mirrored trajectory has same reward)
rewards_mirrored = rewards.copy()

print(f"\nY-mirror transform:")
print(f"  Original goals w1: mean={actions[:, 6].mean():+.4f}")
print(f"  Mirrored goals w1: mean={actions_mirrored[:, 6].mean():+.4f}")
print(f"  Original goal_y: mean={goals[:, 1].mean():+.4f}")
print(f"  Mirrored goal_y: mean={goals_mirrored[:, 1].mean():+.4f}")

# ── Combine original + mirrored ─────────────────────────────────────────────
images_combined   = np.concatenate([images, images_mirrored], axis=0)
states_combined   = np.concatenate([states_norm, states_norm_mirrored], axis=0)
actions_combined  = np.concatenate([actions, actions_mirrored], axis=0)
goals_combined    = np.concatenate([goals, goals_mirrored], axis=0)
rewards_combined  = np.concatenate([rewards, rewards_mirrored], axis=0)

N_total = len(actions_combined)
print(f"\nCombined dataset: {N_total} frames")
print(f"  +Y goals: {(goals_combined[:, 1] >= 0).sum()}, -Y goals: {(goals_combined[:, 1] < 0).sum()}")
print(f"  w1 mean +Y: {actions_combined[goals_combined[:, 1]>=0, 6].mean():+.4f}")
print(f"  w1 mean -Y: {actions_combined[goals_combined[:, 1]<0, 6].mean():+.4f}")
print(f"  Images: {images_combined.shape}")
print(f"  States: {states_combined.shape}")
print(f"  Actions: {actions_combined.shape}")

# ── Save ─────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT}...")
with h5py.File(OUT, 'w') as f:
    f.create_dataset('images', data=images_combined, compression='gzip', compression_opts=4)
    f.create_dataset('states', data=states_combined.astype(np.float32))
    f.create_dataset('actions', data=actions_combined.astype(np.float32))
    f.create_dataset('goal_positions', data=goals_combined.astype(np.float32))
    f.create_dataset('rewards', data=rewards_combined.astype(np.float32))
    
    # Save normalization stats for training
    f.create_dataset('state_mean', data=state_mean.astype(np.float32))
    f.create_dataset('state_std', data=state_std.astype(np.float32))

print(f"Done! Dataset saved: {OUT}")
print(f"File size: {os.path.getsize(OUT) / 1e6:.1f} MB")
