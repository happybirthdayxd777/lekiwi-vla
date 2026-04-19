#!/usr/bin/env python3
"""
Phase 187: Clean Goal-Directed Data Collection with CORRECT 11D State
====================================================================
The phase181_symmetrized_10k.h5 data has CRITICAL contamination:
  - w2/w3 wheel actions are IDENTICAL for +Y and -Y goals
  - The data collector was NOT goal-directed for Y-axis locomotion
  - Result: VLA trained on this data has 10% SR vs P-ctrl 45%

This script collects clean data with CORRECT goal-conditioned wheel actions:
  - State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D [MATCHES TRAINING/EVAL]
  - Action: arm(6) + wheel(3) = 9D [normalized to [-1, 1]]
  - Images: rendered from sim
  - Wheel actions computed from DIRECT goal_position → twist_to_contact_wheel_speeds()

Data correlation check after collection:
  - w1/w2 SHOULD have strong positive/negative correlation with goal_x
  - w0 should have positive correlation with goal_y
  - If correlations are weak, data is contaminated (discard and recollect)

Usage:
  python3 scripts/collect_phase187_clean.py --episodes 50 --output data/phase187_clean_50ep.h5
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from datetime import datetime

from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

TARGET_SIZE = (224, 224)
ARM_POS_SCALE = 2.0      # arm positions normalized to [-1, 1] at ±2.0
WHEEL_VEL_SCALE = 0.5   # wheel velocities normalized at ±0.5 rad/s


class CleanJacobianController:
    """
    Proportional controller using CORRECT contact Jacobian IK.
    This is the SAME controller used in eval — ensures perfect train/eval match.
    """
    def __init__(self, kP=0.5, wheel_clip=0.5):
        self.kP = kP
        self.wheel_clip = wheel_clip

    def compute_wheel_velocities(self, base_xy, goal_xy):
        dx = goal_xy[0] - base_xy[0]
        dy = goal_xy[1] - base_xy[1]
        dist = np.linalg.norm([dx, dy])
        if dist < 0.005:
            return np.zeros(3, dtype=np.float32)
        # Same formula as eval: constant kP speed
        vx = self.kP * dx
        vy = self.kP * dy
        wz = 0.0
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy, wz)
        if self.wheel_clip is not None:
            wheel_speeds = np.clip(wheel_speeds, -self.wheel_clip, self.wheel_clip)
        return np.array(wheel_speeds, dtype=np.float32)


def collect_episode_clean(sim, controller, goal_pos, max_steps=200,
                          arm_action_scale=0.05, base_body_id=None):
    """
    Collect one goal-directed episode with CORRECT 11D state format.
    
    State: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
      - arm_pos: current arm joint positions (scaled to [-1,1])
      - wheel_vel: wheel angular velocities (scaled to [-1,1])
      - goal_norm: goal position (scaled to [-1,1])
    
    Action: arm(6) + wheel(3) = 9D (normalized to [-1,1])
    """
    obs = sim.reset()
    for _ in range(15):
        sim.step([0]*9)

    if base_body_id is None:
        base_body_id = sim.model.body('base').id

    # Normalize goal once at start
    goal_norm = np.clip(goal_pos / 0.5, -1, 1).astype(np.float32)

    states_list, actions_list, rewards_list, goals_list = [], [], [], []

    # Arm: independent random walk
    arm_pos = np.zeros(6, dtype=np.float32)

    for step in range(max_steps):
        # Get current base position from world frame
        base_xy = sim.data.xpos[base_body_id, :2]

        # Compute wheel velocities from goal (THE CORRECT METHOD)
        wheel_vels = controller.compute_wheel_velocities(base_xy, goal_pos)

        # Arm random walk
        arm_delta = np.random.normal(0, arm_action_scale, size=6).astype(np.float32)
        arm_pos = np.clip(arm_pos + arm_delta, -1.0, 1.0)

        # Normalize action to [-1, 1]
        arm_action = arm_pos.astype(np.float32)
        wheel_action = (wheel_vels / WHEEL_VEL_SCALE).astype(np.float32)
        wheel_action = np.clip(wheel_action, -1.0, 1.0)
        action = np.concatenate([arm_action, wheel_action])

        # Step simulation
        obs, reward, done, info = sim.step(action)

        # Build 11D state: arm_pos + wheel_vel + goal_norm
        wheel_vels_raw = obs['wheel_velocities']
        wheel_vels_norm = np.clip(wheel_vels_raw / WHEEL_VEL_SCALE, -1, 1).astype(np.float32)
        arm_pos_norm = np.clip(arm_pos / ARM_POS_SCALE, -1, 1).astype(np.float32)
        state11 = np.concatenate([arm_pos_norm, wheel_vels_norm, goal_norm])

        # Render image
        img = sim.render().astype(np.uint8)

        states_list.append(state11)
        actions_list.append(action)
        rewards_list.append(reward)
        goals_list.append(goal_norm)

        if done:
            break

    return {
        'states': np.array(states_list, dtype=np.float32),
        'actions': np.array(actions_list, dtype=np.float32),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'goal_norm': np.array(goals_list, dtype=np.float32),
        'goal_world': np.array([goal_pos] * len(states_list), dtype=np.float32),
        'image': img,  # (H, W, 3) uint8
    }


def collect_episode_batch(episode_data, start_idx=0):
    """Flatten episode data into batch format."""
    n = len(episode_data['states'])
    indices = np.arange(start_idx, start_idx + n)
    return indices, np.array(episode_data['states']), np.array(episode_data['actions']), \
           np.array(episode_data['rewards']), np.array(episode_data['goal_norm']), \
           np.array(episode_data['goal_world']), episode_data.get('images')


def main():
    parser = argparse.ArgumentParser(description='Phase 187: Clean goal-directed data collection')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=str, default='data/phase187_clean_50ep.h5')
    parser.add_argument('--goal_min', type=float, default=0.2)
    parser.add_argument('--goal_max', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    controller = CleanJacobianController(kP=0.5, wheel_clip=0.5)

    # Pre-create sim to get base_body_id
    temp_sim = LeKiWiSimURDF()
    temp_sim.reset()
    base_body_id = temp_sim.model.body('base').id
    del temp_sim

    all_states, all_actions, all_rewards, all_goals_norm = [], [], [], []
    all_goals_world, all_images = [], []
    episode_starts = [0]

    total_frames = 0
    successes = 0

    print(f"Phase 187: Collecting {args.episodes} episodes with CLEAN Jacobian P-controller...")
    print(f"  kP=0.5, wheel_clip=0.5, goal range: [{args.goal_min}, {args.goal_max}]m")
    print(f"  State format: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D [MATCHES TRAINING/EVAL]")
    print()

    sim = LeKiWiSimURDF()

    for ep in range(args.episodes):
        # Random goal in polar coords for uniform distribution
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(args.goal_min, args.goal_max)
        goal_pos = np.array([r*np.cos(angle), r*np.sin(angle)], dtype=np.float32)

        ep_data = collect_episode_clean(
            sim, controller, goal_pos, max_steps=args.steps,
            arm_action_scale=0.05, base_body_id=base_body_id
        )

        n = len(ep_data['states'])
        all_states.append(ep_data['states'])
        all_actions.append(ep_data['actions'])
        all_rewards.append(ep_data['rewards'])
        all_goals_norm.append(ep_data['goal_norm'])
        all_goals_world.append(ep_data['goal_world'])
        all_images.append(ep_data['image'])  # (H, W, 3) uint8

        episode_starts.append(total_frames + n)
        total_frames += n

        ep_success = ep_data['rewards'].sum() > 0
        if ep_success:
            successes += 1

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{args.episodes}: {n} frames, cumulative={total_frames}, successes={successes}")

    print(f"\nTotal frames: {total_frames}, Success rate: {successes}/{args.episodes}")

    # Concatenate all
    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    all_rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    all_goals_norm = np.concatenate(all_goals_norm, axis=0).astype(np.float32)
    all_goals_world = np.concatenate(all_goals_world, axis=0).astype(np.float32)

    # DATA QUALITY CHECK
    print("\n=== DATA QUALITY CHECK ===")
    print("Wheel-goal correlations (SHOULD BE STRONG for goal-directed data):")
    wheel_actions = all_actions[:, 6:9]
    goals_world = all_goals_world

    for i, name in enumerate(['w0 (lateral)', 'w1 (fwd)', 'w2 (fwd)']):
        cx = np.corrcoef(wheel_actions[:, i], goals_world[:, 0])[0, 1]
        cy = np.corrcoef(wheel_actions[:, i], goals_world[:, 1])[0, 1]
        print(f"  {name}: corr_goal_x={cx:+.3f}, corr_goal_y={cy:+.3f}")

    # Critical check: w1/w2 should have strong correlation with goal_x
    w1x_corr = np.corrcoef(wheel_actions[:, 1], goals_world[:, 0])[0, 1]
    w2x_corr = np.corrcoef(wheel_actions[:, 2], goals_world[:, 0])[0, 1]
    if abs(w1x_corr) < 0.3 or abs(w2x_corr) < 0.3:
        print("  WARNING: Weak correlation detected — data may be contaminated!")
    else:
        print("  OK: Strong correlations detected — data is goal-directed ✓")

    # Save
    print(f"\nSaving to {args.output}...")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        f.create_dataset('rewards', data=all_rewards, compression='gzip')
        f.create_dataset('goal_positions', data=all_goals_world, compression='gzip')
        f.create_dataset('goal_norm', data=all_goals_norm, compression='gzip')
        f.create_dataset('episode_starts', data=np.array(episode_starts))
        if all_images:
            all_images_stacked = np.stack(all_images, axis=0)  # (N, H, W, 3)
            f.create_dataset('images', data=all_images_stacked, compression='gzip')

        f.attrs['phase'] = 187
        f.attrs['controller'] = 'CleanJacobianController (kP=0.5, k_omni=15.0)'
        f.attrs['state_format'] = 'arm_pos(6)+wheel_vel(3)+goal_norm(2)=11D'
        f.attrs['action_format'] = 'arm(6)+wheel(3)=9D normalized [-1,1]'
        f.attrs['created'] = datetime.now().isoformat()

    # Stats
    print(f"\nDataset stats:")
    print(f"  States: {all_states.shape}, range=[{all_states.min():.3f}, {all_states.max():.3f}]")
    print(f"  Actions: {all_actions.shape}, range=[{all_actions.min():.3f}, {all_actions.max():.3f}]")
    print(f"  Rewards: pos={int((all_rewards>0).sum())}, neg={int((all_rewards<=0).sum())}")
    if all_images:
        print(f"  Images: {all_images_stacked.shape}")
    print(f"\nDone! Saved to {args.output}")


if __name__ == '__main__':
    main()
