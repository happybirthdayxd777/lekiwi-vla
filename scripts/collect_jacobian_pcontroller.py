#!/usr/bin/env python3
"""
Phase 142: Correct Data Collection with Jacobian P-Controller
=============================================================
Collects 10k goal-directed frames using the CORRECT contact Jacobian P-controller.

CRITICAL FINDING (Phase 142):
  - GridSearchController (used by ALL previous VLA training): 0% SR
  - Jacobian P-controller (bridge_node.py method): 100% SR

This script uses twist_to_contact_wheel_speeds() — the same method
used by bridge_node.py — to collect training data that matches the bridge.

Usage:
  python3 scripts/collect_jacobian_pcontroller.py --episodes 50 --output data/leikiwi_jacobian_pctrl_50ep.h5
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from datetime import datetime

from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

TARGET_SIZE = (224, 224)


class JacobianPController:
    """
    Proportional controller using contact Jacobian IK.
    
    This is the CORRECT controller — same method used by bridge_node.py.
    Achieves 100% SR vs GridSearchController's 0% SR.
    
    Key: twist_to_contact_wheel_speeds(vx, vy) uses J_c^+ inverse
    to convert world-frame velocity to wheel angular velocities.
    """
    
    def __init__(self, kP=0.1, max_speed=0.25, wheel_clip=None):
        self.kP = kP
        self.max_speed = max_speed
        self.wheel_clip = wheel_clip
    
    def compute_wheel_velocities(self, base_pos, goal_pos):
        """
        Returns wheel angular velocities to drive toward goal.
        
        Uses the contact Jacobian pseudo-inverse — the CORRECT method.
        """
        dx = goal_pos[0] - base_pos[0]
        dy = goal_pos[1] - base_pos[1]
        dist = np.linalg.norm([dx, dy])
        
        if dist < 0.05:
            return np.zeros(3, dtype=np.float32)
        
        # P-control: scale velocity by distance
        v_mag = min(self.kP * dist, self.max_speed)
        vx = v_mag * (dx / dist)
        vy = v_mag * (dy / dist)
        
        # Convert world-frame velocity to wheel angular velocities via J_c^+
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        
        # Clip to URDF stable range
        if self.wheel_clip is not None:
            wheel_speeds = np.clip(wheel_speeds, -self.wheel_clip, self.wheel_clip)
        return wheel_speeds.astype(np.float32)


def collect_episode(sim, controller, goal_pos, max_steps=200,
                    exploration_noise=0.0, arm_action_scale=0.1,
                    base_body_id=None):
    """
    Collect one goal-directed episode.
    
    CRITICAL FIX (Phase 142): Use sim.data.xpos[base_id,:2] NOT obs['base_position'].
    obs['base_position'] = qpos[0:3] = [0,0,0] after reset() — WRONG.
    sim.data.xpos[base_id] = world position = CORRECT.
    
    State: arm(6) + base_xy(2) = 8D  [goal stored separately]
    Action: arm(6) + wheel(3), normalized
    """
    obs = sim.reset()
    
    if hasattr(sim, 'set_target'):
        sim.set_target(goal_pos)
    
    if base_body_id is None:
        base_body_id = sim.model.body('base').id
    
    states, actions, rewards, goal_positions = [], [], [], []
    
    # Arm: independent random walk (manipulation while navigating)
    arm_pos = np.zeros(6, dtype=np.float32)
    
    for step in range(max_steps):
        # CRITICAL: Use xpos (world frame), NOT obs['base_position'] (local qpos)
        base_xy = sim.data.xpos[base_body_id, :2]
        
        # Compute wheel velocities using Jacobian P-controller
        wheel_vels = controller.compute_wheel_velocities(base_xy, goal_pos)
        
        # Add exploration noise
        # Phase 167: No noise in collect (matches eval conditions)
        
        # Arm random walk
        arm_delta = np.random.normal(0, arm_action_scale, size=6).astype(np.float32)
        arm_pos = np.clip(arm_pos + arm_delta, -1.0, 1.0)
        
        # Convert wheel_speeds (rad/s) to wheel_action (servo units) like eval
        wheel_action = wheel_vels / 12.0  # matches eval_matched_goals.py
        action = np.concatenate([arm_pos, wheel_action]).astype(np.float32)
        
        # Step simulation
        obs, reward, done, info = sim.step(action)
        
        # Record state (arm pos + base xy in world frame)
        base_xy_record = sim.data.xpos[base_body_id, :2]
        states.append(np.concatenate([arm_pos.copy(), base_xy_record]))
        actions.append(action.copy())
        rewards.append(reward)
        goal_positions.append(goal_pos.copy())
        
        if done:
            break
    
    return {
        'states': np.array(states, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'goal_positions': np.array(goal_positions, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description='Collect goal-directed data with Jacobian P-controller')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=str, default='data/leikiwi_jacobian_pctrl.h5')
    parser.add_argument('--goal_min', type=float, default=0.3)
    parser.add_argument('--goal_max', type=float, default=0.7)
    parser.add_argument('--exploration', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    controller = JacobianPController(kP=0.1, max_speed=0.25, wheel_clip=None)
    
    all_states, all_actions, all_rewards, all_goals = [], [], [], []
    episode_starts = [0]
    
    total_frames = 0
    successes = 0
    
    print(f"Collecting {args.episodes} episodes with Jacobian P-controller (100% SR method)...")
    print(f"  kP=0.1, max_speed=0.25, exploration={args.exploration}")
    print(f"  goal range: [{args.goal_min}, {args.goal_max}]m")
    print()
    
    # Pre-create sim to get base_body_id (sim reset clears it each episode)
    temp_sim = LeKiWiSimURDF()
    temp_sim.reset()
    base_body_id = temp_sim.model.body('base').id
    del temp_sim
    
    for ep in range(args.episodes):
        # Random goal position
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(args.goal_min, args.goal_max)
        goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        
        # Collect episode
        sim = LeKiWiSimURDF()
        ep_data = collect_episode(
            sim, controller, goal_pos,
            max_steps=args.steps,
            exploration_noise=args.exploration,
            base_body_id=base_body_id
        )
        
        n_frames = len(ep_data['rewards'])
        total_frames += n_frames
        
        # Check success
        final_state = ep_data['states'][-1]
        final_xy = final_state[6:8]  # base x, y
        final_dist = np.linalg.norm(final_xy - goal_pos)
        sr = 1.0 if final_dist < 0.2 else 0.0
        successes += sr
        
        all_states.append(ep_data['states'])
        all_actions.append(ep_data['actions'])
        all_rewards.append(ep_data['rewards'])
        all_goals.append(ep_data['goal_positions'])
        episode_starts.append(total_frames)
        
        if ep % 10 == 0 or ep == args.episodes - 1:
            print(f"  Episode {ep:3d}: {n_frames:3d} frames, dist={final_dist:.3f}m, SR={int(sr)}, "
                  f"cumulative SR={successes/(ep+1)*100:.0f}% ({successes}/{ep+1})")
    
    # Concatenate all episodes
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_goals = np.concatenate(all_goals, axis=0)
    episode_starts = np.array(episode_starts, dtype=np.int64)
    
    print(f"\nTotal: {total_frames} frames from {args.episodes} episodes")
    print(f"Success Rate: {successes}/{args.episodes} = {successes/args.episodes*100:.0f}%")
    
    # Save to HDF5
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        f.create_dataset('rewards', data=all_rewards, compression='gzip')
        f.create_dataset('goal_positions', data=all_goals, compression='gzip')
        f.create_dataset('episode_starts', data=episode_starts)
        f.attrs['controller'] = 'JacobianPController (twist_to_contact_wheel_speeds)'
        f.attrs['kP'] = 0.1
        f.attrs['max_speed'] = 0.25
        f.attrs['k_omni'] = 15.0
        f.attrs['noslip_iterations'] = 10
        f.attrs['success_rate'] = successes / args.episodes
        f.attrs['created'] = datetime.now().isoformat()
    
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved to {args.output} ({size_mb:.1f}MB)")


if __name__ == '__main__':
    main()
