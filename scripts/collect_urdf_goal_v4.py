#!/usr/bin/env python3
"""
Phase 109: Fixed Goal-Aware Data Collection for k_omni Physics
=============================================================
KEY INSIGHT from Phase 108 root cause analysis:

ROOT CAUSE of 100% negative rewards in v3:
  1. M7=[0,1,-1] forward speed: ~0.0027 m/step (very slow)
  2. 200 steps × 0.0027 m/step = 0.54m max travel per episode
  3. goals at 0.6-0.7m → robot falls SHORT by 0.1-0.2m → negative rewards
  4. steps_per_move=20 → direction updates only 10× per episode → poor tracking

Phase 109 fixes:
  - steps_per_move: 20 → 5 (40 direction updates per episode, better tracking)
  - goal_max: 0.7m → 0.4m (within robot's reachable range)
  - goal_threshold: 0.1m → 0.2m (robot reliably reaches this)
  - goal_min: 0.3m → 0.15m (closer goals for faster learning)

This controller uses GOAL DIRECTION to select the best primitive:
  - Goal near +X axis: M7 (pure forward)
  - Goal near +Y axis: M9/M4 (diagonal forward+right)
  - Goal at 45deg: M6
  - For other directions: rotate first with M3, then move

Usage:
  python3 scripts/collect_urdf_goal_v4.py --episodes 200 --output data/phase109_urdf_goal_v4.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import json

TARGET_SIZE = (224, 224)


class GoalAwareController:
    """
    Goal-aware controller for LeKiWi URDF with k_omni physics.
    
    Key discovery (Phase 107): k_omni force direction mapping
    Measured empirical results (50 steps, k_omni=15):
      M7=[0,1,-1]:   +X=0.130m, +Y=0.035m,  dir=4.1deg   (97% forward!) ← BEST FOR +X GOALS
      M4=[1,0.5,-0.5]: +X=0.095m, +Y=0.021m,  dir=38.7deg
      M9=[0.5,0.5,-0.5]: identical to M4
      M6=[2,1,0]:   +X=0.035m, +Y=0.049m,  dir=43.4deg
      M2=[1,1,1]:   -X=0.023m, +Y=0.052m,  minimal motion
      M3=[0.5,-0.5,0]: rotation only, no translation
    
    For goal at angle θ from +X axis:
      θ in [-15, +15]:  M7 (pure forward)
      θ in [+15, +55]:  M9/M4 (38.7deg diagonal)
      θ in [-15, -55]:  M1 or mirror of M4
      θ in [+55, +125]: rotate CCW with M3, then M7 when aligned
      θ in [-55, -125]: rotate CW with M3, then M7
    """
    
    # [w1, w2, w3] — normalized to [-1, 1] range
    PRIMITIVES = {
        'fwd_pure':    np.array([0.0,  1.0, -1.0]),   # +X forward (4.1deg)
        'diag_right':  np.array([0.5,  0.5, -0.5]),   # +X+0.9Y (38.7deg)
        'diag_right2': np.array([1.0,  0.5, -0.5]),   # same as diag_right
        'forward':     np.array([2.0,  1.0,  0.0]),   # +X+0.95Y (43.4deg)
        'back_pure':   np.array([3.0,  2.0,  1.0]),   # backward-ish
        'rot_ccw':     np.array([0.5, -0.5,  0.0]),   # CCW rotation
        'rot_cw':      np.array([1.0, -1.0,  0.0]),   # CW rotation
        'idle':        np.array([0.0,  0.0,  0.0]),   # no motion
    }
    
    # Angle thresholds (degrees from +X axis)
    ANGLE_THRESHOLDS = {
        'fwd_pure':    15,    # |θ| < 15deg → M7
        'diag_right':  50,    # 15 < |θ| < 50deg → M9
        'forward':     80,    # 50 < |θ| < 80deg → M6
    }
    
    def __init__(self, steps_per_move=20, exploration_noise=0.1):
        self.steps_per_move = steps_per_move
        self.exploration_noise = exploration_noise
        self._counter = 0
        self._mode = 'idle'  # 'move' or 'rotate'
        self._rotate_start_yaw = None
        
    def reset(self):
        self._counter = 0
        self._mode = 'idle'
        self._rotate_start_yaw = None
    
    def compute_action(self, base_pos, base_yaw, goal_pos):
        """
        Return normalized wheel action (3,) in [-1, 1] range.
        
        Args:
            base_pos: [x, y, z] world position
            base_yaw: yaw angle in radians
            goal_pos: [gx, gy] world position
        """
        # Vector from base to goal (in world frame)
        dx = goal_pos[0] - base_pos[0]
        dy = goal_pos[1] - base_pos[1]
        dist = np.linalg.norm([dx, dy])
        
        if dist < 0.05:
            return self.PRIMITIVES['idle'].copy()
        
        # Goal angle in world frame (from +X axis)
        world_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Goal angle in body frame (relative to robot's forward direction)
        body_angle = world_angle - base_yaw * 180 / np.pi
        body_angle = ((body_angle + 180) % 360) - 180  # normalize to [-180, 180]
        
        self._counter += 1
        
        # Select primitive based on goal direction
        if self._counter % self.steps_per_move == 0:
            abs_angle = abs(body_angle)
            sign_angle = np.sign(body_angle)
            
            if abs_angle < self.ANGLE_THRESHOLDS['fwd_pure']:
                self._mode = 'fwd_pure'
            elif abs_angle < self.ANGLE_THRESHOLDS['diag_right']:
                if sign_angle > 0:
                    self._mode = 'diag_right'
                else:
                    self._mode = 'back_pure'  # mirror direction
            elif abs_angle < self.ANGLE_THRESHOLDS['forward']:
                if sign_angle > 0:
                    self._mode = 'forward'
                else:
                    self._mode = 'back_pure'
            else:
                # Need to rotate first
                if sign_angle > 0:
                    self._mode = 'rot_ccw'
                else:
                    self._mode = 'rot_cw'
                self._rotate_start_yaw = base_yaw
        
        # Get current primitive
        primitive = self.PRIMITIVES[self._mode].copy()
        
        # Scale by distance (closer = slower)
        if dist < 0.2:
            primitive *= 0.5
        elif dist < 0.1:
            primitive *= 0.3
        
        # Add exploration noise
        if self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=3)
            primitive = np.clip(primitive + noise, -1, 1)
        
        return primitive.astype(np.float32)


def compute_reward(base_pos, base_pos_next, goal_pos, threshold=0.1):
    """Sparse + shaped reward for goal-directed navigation."""
    dist_t = np.linalg.norm(base_pos[:2] - goal_pos[:2])
    dist_tp1 = np.linalg.norm(base_pos_next[:2] - goal_pos[:2])
    
    is_goal = dist_tp1 < threshold
    
    if is_goal and dist_tp1 < dist_t:
        reward = 1.0
    elif is_goal:
        reward = 0.5
    else:
        reward = (dist_t - dist_tp1) * 2.0
    
    return reward, is_goal, dist_t, dist_tp1


def collect_episode(sim, controller, max_steps=200, goal_min=0.15, goal_max=0.4,
                   goal_threshold=0.2, seed=None):
    """Collect one goal-directed episode."""
    if seed is not None:
        np.random.seed(seed)
    
    # Random goal position (polar coordinates for uniform distribution)
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(goal_min, goal_max)
    goal_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
    
    obs = sim.reset(target=goal_pos)
    
    controller.reset()
    
    all_images = []
    all_states = []
    all_actions = []
    all_rewards = []
    all_goals = []
    goal_reached = False
    
    for step in range(max_steps):
        # Get current base state
        base_pos = obs['base_position'].copy()  # [x, y, z]
        base_quat = obs['base_quaternion'].copy()  # [qx, qy, qz, qw]
        
        # Extract yaw from quaternion (rotation around Z axis)
        # For quaternion [qx, qy, qz, qw], yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        # But simpler: use the freejoint qpos directly
        base_yaw = sim.data.qpos[2]  # 3rd element of freejoint qpos = yaw
        
        # Controller selects action based on goal direction
        wheel_action = controller.compute_action(base_pos, base_yaw, goal_pos)
        
        # Full action: arm (6) + wheel (3) = 9D
        arm_action = np.zeros(6)
        action = np.concatenate([arm_action, wheel_action])
        
        # Render image
        img = sim.render()
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img).resize(TARGET_SIZE, Image.BILINEAR)
        
        obs_next, reward, done, info = sim.step(action)
        
        # State: arm (6) + wheel velocities (3) = 9D
        arm_pos = obs['arm_positions']  # 6 joints
        wheel_vel = obs['wheel_velocities']  # 3 wheels
        state = np.concatenate([arm_pos, wheel_vel])
        
        all_images.append(np.array(img))
        all_states.append(state.astype(np.float32))
        all_actions.append(action.astype(np.float32))
        all_rewards.append(reward)
        all_goals.append(goal_pos.copy())
        
        # Phase 109 FIX: check dist < threshold (reward >= 1.0 requires dist <= 0, impossible)
        final_dist = np.linalg.norm(obs_next['base_position'][:2] - goal_pos)
        if final_dist < goal_threshold:
            goal_reached = True
        
        obs = obs_next
        
        if done:
            break
    
    return {
        'image': np.stack(all_images),
        'state': np.stack(all_states),
        'action': np.stack(all_actions),
        'reward': np.array(all_rewards),
        'goal_position': np.stack(all_goals),
        'goal_reached': goal_reached,
    }


def main():
    parser = argparse.ArgumentParser(description='Goal-aware data collection for k_omni physics')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--goal-min', type=float, default=0.3)
    parser.add_argument('--goal-max', type=float, default=0.7)
    parser.add_argument('--output', type=str, default='data/phase108_urdf_goal_v3.h5')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    print(f"[INFO] Phase 109: Fixed Goal-Aware Data Collection for k_omni Physics")
    print(f"[INFO] Using GoalAwareController (steps_per_move=5, goal_max=0.4m)")
    
    # Import here to avoid circular dependency
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    print("[INFO] Creating LeKiWiSimURDF (Phase 86 physics, k_omni=15)")
    sim = LeKiWiSimURDF()
    print(f"[LeKiWiSimURDF] bodies={sim.model.nbody}, meshes={26}, joints={sim.model.njnt}, geoms={sim.model.ngeom}")
    controller = GoalAwareController(steps_per_move=5, exploration_noise=0.02)
    
    all_images = []
    all_states = []
    all_actions = []
    all_rewards = []
    all_goals = []
    total_frames = 0
    goal_reached = 0
    
    for ep in range(args.episodes):
        seed = args.seed + ep if args.seed is not None else None
        result = collect_episode(
            sim,
            controller,
            max_steps=args.steps,
            goal_min=args.goal_min,
            goal_max=args.goal_max,
            seed=seed,
        )
        
        all_images.append(result['image'])
        all_states.append(result['state'])
        all_actions.append(result['action'])
        all_rewards.append(result['reward'])
        all_goals.append(result['goal_position'])
        total_frames += len(result['reward'])
        
        if result['goal_reached']:
            goal_reached += 1
        
        if (ep + 1) % 20 == 0:
            print(f"[INFO] Episode {ep+1}/{args.episodes}: {total_frames} frames, "
                  f"goals reached: {goal_reached}")
    
    # Concatenate
    images = np.concatenate(all_images, axis=0)
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    goals = np.concatenate(all_goals, axis=0)
    
    print(f"\n[INFO] Total: {total_frames} frames from {args.episodes} episodes")
    print(f"[INFO] States: {states.shape}, Actions: {actions.shape}")
    print(f"[INFO] Rewards: mean={rewards.mean():.3f}, max={rewards.max():.3f}, "
          f"frac>0={(rewards>0).mean():.3f}")
    print(f"[INFO] Goals reached: {goal_reached}/{args.episodes} ({goal_reached/args.episodes*100:.1f}%)")
    
    # Save to HDF5
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('images', data=images, compression='lzf')
        f.create_dataset('states', data=states, compression='lzf')
        f.create_dataset('actions', data=actions, compression='lzf')
        f.create_dataset('rewards', data=rewards, compression='lzf')
        f.create_dataset('goal_positions', data=goals, compression='lzf')
    
    meta = {
        'episodes': args.episodes,
        'steps_per_episode': args.steps,
        'total_frames': total_frames,
        'physics': 'Phase 86 URDF (k_omni=15)',
        'controller': 'GoalAwareController (Phase 108)',
        'state_dim': 9,
        'action_dim': 9,
        'wheel_action_mean': actions[:, 6:].mean(axis=0).tolist(),
        'wheel_action_std': actions[:, 6:].std(axis=0).tolist(),
        'goals_reached': goal_reached,
        'frac_positive_reward': float((rewards > 0).mean()),
    }
    meta_path = args.output.replace('.h5', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[INFO] Saved to {args.output}")
    print(f"[INFO] Metadata: {meta_path}")


if __name__ == '__main__':
    main()
