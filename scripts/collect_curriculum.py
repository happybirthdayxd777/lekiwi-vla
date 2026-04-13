#!/usr/bin/env python3
"""
Curriculum Learning Data Collection for LeKiWi
===============================================
Collects goal-directed navigation data with PROGRESSIVE DIFFICULTY:
  Stage 1: goals at 0.1-0.2m  (very easy, guarantees arrivals)
  Stage 2: goals at 0.2-0.4m  (moderate)
  Stage 3: goals at 0.4-0.6m  (harder)
  Stage 4: goals at 0.6-0.8m  (difficult, final challenge)

Each episode is 200 steps. The P-controller has HIGHER GAINS than
the original script (which produced actions in ±0.01 range — too weak).

The key fix: Kp=3.0 (not 0.5) so actions are actually in a useful range.

Usage:
  python3 scripts/collect_curriculum.py --stage 1 --episodes 20 --output data/curriculum_s1.h5
  python3 scripts/collect_curriculum.py --stage 2 --episodes 20 --output data/curriculum_s2.h5
  # ... then merge for training
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

from sim_lekiwi_urdf import LeKiWiSimURDF


# ── Curriculum stage configurations ────────────────────────────────────────────
CURRICULUM_STAGES = {
    1: {"goal_min": 0.10, "goal_max": 0.20, "label": "very_easy"},
    2: {"goal_min": 0.20, "goal_max": 0.40, "label": "easy"},
    3: {"goal_min": 0.40, "goal_max": 0.60, "label": "medium"},
    4: {"goal_min": 0.60, "goal_max": 0.80, "label": "hard"},
}


def collect_episode(sim, goal_pos, max_steps=200):
    """
    Collect one episode of trajectory data.
    Returns (images, states, actions, rewards, goal_positions, done)
    
    Uses a P-controller with Kp=3.0 (key fix vs original Kp=0.5).
    """
    images = []
    states = []
    actions = []
    rewards = []
    goal_positions = []
    
    obs = sim.reset(target=goal_pos)
    done = False
    total_reward = 0.0
    goal_arrived = False
    
    # Initial render
    img = sim.render()
    
    for step in range(max_steps):
        # ── Record state ──────────────────────────────────────────────────────
        joint_pos = sim.data.qpos[:]
        joint_vel = sim.data.qvel[:]
        base_pos = sim.data.qpos[:3]  # x, y, theta
        wheel_pos = sim.data.qpos[3:6]
        
        state = np.concatenate([base_pos, wheel_pos, joint_vel])
        images.append(img.copy())
        states.append(state)
        goal_positions.append(goal_pos.copy())
        
        # ── P-controller: steer toward goal ──────────────────────────────────
        # FIXED: Kp=3.0 (original was ~0.5 which produced tiny actions in ±0.01)
        Kp = 3.0
        v_max = 0.5  # max forward speed m/s
        
        dx = goal_pos[0] - base_pos[0]
        dy = goal_pos[1] - base_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)
        angle_err = angle_to_goal - base_pos[2]
        
        # Normalize angle error to [-pi, pi]
        while angle_err > np.pi:
            angle_err -= 2 * np.pi
        while angle_err < -np.pi:
            angle_err += 2 * np.pi
        
        # Proportional control
        v_cmd = Kp * dist  # speed proportional to distance
        w_cmd = Kp * angle_err  # angular velocity
        
        # Clamp to physical limits
        v_cmd = np.clip(v_cmd, -v_max, v_max)
        w_cmd = np.clip(w_cmd, -2.0, 2.0)
        
        # Convert to wheel velocities (omni-wheel inverse kinematics)
        # Assumes 3 omnis at 120° intervals
        wheel_speeds = omni_diff_drive_to_wheels(v_cmd, w_cmd)
        
        # Arm stays at home position (0s)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + list(wheel_speeds), dtype=np.float32)
        actions.append(action)
        
        # ── Step simulation ───────────────────────────────────────────────────
        obs, reward, done, info = sim.step(action)
        img = sim.render()
        total_reward += reward
        rewards.append(reward)
        
        # Check goal arrival
        if dist < 0.10:  # 10cm threshold for data collection
            goal_arrived = True
        
        if done:
            # Pad remaining steps
            for _ in range(step + 1, max_steps):
                images.append(img.copy())
                states.append(state.copy())
                actions.append(action.copy())
                rewards.append(0.0)
                goal_positions.append(goal_pos.copy())
            break
    
    return {
        "images": np.array(images, dtype=np.uint8),
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "goal_positions": np.array(goal_positions, dtype=np.float32),
        "goal_arrived": goal_arrived,
        "total_reward": total_reward,
        "final_dist": dist if 'dist' in dir() else 0.0,
    }


def omni_diff_drive_to_wheels(vx, vy):
    """
    Convert diff-drive (v, omega) to 3x omni-wheel velocities.
    Wheels at angles 0°, 120°, 240°.
    """
    # Assume robot frame: x=forward, y=left
    angles = np.array([0, 120, 240]) * np.pi / 180
    wheel_speeds = []
    for ang in angles:
        # Omni-wheel: each wheel can provide force in its direction
        # v_wheel = vx*cos(theta) + vy*sin(theta) + omega*r
        wheel_speeds.append(vx * np.cos(ang) + vy * np.sin(ang))
    return np.array(wheel_speeds)


def save_hdf5(trajectories, output_path):
    """Save collected trajectories to HDF5 with goal_positions dataset."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        # Concatenate all episodes
        all_images = np.concatenate([t["images"] for t in trajectories], axis=0)
        all_states = np.concatenate([t["states"] for t in trajectories], axis=0)
        all_actions = np.concatenate([t["actions"] for t in trajectories], axis=0)
        all_rewards = np.concatenate([t["rewards"] for t in trajectories], axis=0)
        all_goals = np.concatenate([t["goal_positions"] for t in trajectories], axis=0)
        
        f.create_dataset("images", data=all_images, compression="lzf")
        f.create_dataset("robot_state", data=all_states, compression="lzf")
        f.create_dataset("action", data=all_actions, compression="lzf")
        f.create_dataset("reward", data=all_rewards, compression="lzf")
        f.create_dataset("goal_positions", data=all_goals, compression="lzf")
        
        # Metadata
        f.attrs["num_episodes"] = len(trajectories)
        f.attrs["total_frames"] = len(all_images)
        f.attrs["created"] = datetime.now().isoformat()
        
        # Episode boundaries
        ep_sizes = [len(t["images"]) for t in trajectories]
        ep_start = [0] + list(np.cumsum(ep_sizes)[:-1])
        f.attrs["episode_sizes"] = ep_sizes
        f.attrs["episode_starts"] = ep_start
        
        # Per-episode stats
        for i, t in enumerate(trajectories):
            grp = f.create_group(f"episode_{i}")
            grp.attrs["goal_arrived"] = t["goal_arrived"]
            grp.attrs["total_reward"] = t["total_reward"]
            grp.attrs["final_dist"] = t["final_dist"]
        
        print(f"  Saved {len(trajectories)} episodes, {len(all_images)} frames to {output_path}")
        arrived = sum(t["goal_arrived"] for t in trajectories)
        print(f"  Goal arrivals: {arrived}/{len(trajectories)} ({100*arrived/len(trajectories):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Curriculum learning data collection")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Curriculum stage (1=easiest, 4=hardest)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--output", type=str, default="",
                        help="Output HDF5 path (auto-generates if empty)")
    parser.add_argument("--sim_type", type=str, default="urdf",
                        choices=["primitive", "urdf"])
    args = parser.parse_args()
    
    stage_cfg = CURRICULUM_STAGES[args.stage]
    goal_min = stage_cfg["goal_min"]
    goal_max = stage_cfg["goal_max"]
    label = stage_cfg["label"]
    
    if not args.output:
        args.output = f"data/curriculum_s{args.stage}_{label}.h5"
    
    print(f"[Stage {args.stage}] Curriculum: goals at {goal_min}-{goal_max}m radius")
    print(f"[Stage {args.stage}] Collecting {args.episodes} episodes x {args.steps} steps")
    
    # Use URDF sim (matches evaluation environment)
    sim = LeKiWiSimURDF()
    
    trajectories = []
    arrived_total = 0
    
    for ep in range(args.episodes):
        # Random goal position in polar coords (angle uniformly [0, 2pi])
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(goal_min, goal_max)
        goal_pos = np.array([
            radius * np.cos(angle),  # x
            radius * np.sin(angle),  # y
        ])
        
        # Clamp goal to sim bounds (assuming 2m x 2m arena)
        goal_pos = np.clip(goal_pos, -0.9, 0.9)
        
        traj = collect_episode(sim, goal_pos, max_steps=args.steps)
        trajectories.append(traj)
        
        arrived_total += traj["goal_arrived"]
        
        if (ep + 1) % 5 == 0:
            arrived_pct = 100 * arrived_total / (ep + 1)
            print(f"  Episodes {ep-3}-{ep+1}: arrival rate={arrived_pct:.1f}%, "
                  f"latest reward={traj['total_reward']:.1f}, dist={traj['final_dist']:.3f}m")
    
    # Save
    save_hdf5(trajectories, args.output)
    
    # Summary
    arrived = sum(t["goal_arrived"] for t in trajectories)
    mean_reward = np.mean([t["total_reward"] for t in trajectories])
    mean_dist = np.mean([t["final_dist"] for t in trajectories])
    print(f"\n[Stage {args.stage}] Collection complete:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Goal arrivals: {arrived}/{args.episodes} ({100*arrived/args.episodes:.1f}%)")
    print(f"  Mean reward: {mean_reward:.1f}")
    print(f"  Mean final distance: {mean_dist:.3f}m")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
