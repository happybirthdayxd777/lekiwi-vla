#!/usr/bin/env python3
"""
Curriculum Learning Data Collection for LeKiWi — FIXED
=======================================================
Phase 19 fixes:
  1. Planar joints: qpos[2] = theta directly (no quaternion needed)
  2. Direct holonomic IK for P-controller

Stages:
  Stage 1: goals at 0.1-0.2m  (very easy, guarantees arrivals)
  Stage 2: goals at 0.2-0.4m  (moderate)
  Stage 3: goals at 0.4-0.6m  (harder)
  Stage 4: goals at 0.6-0.8m  (difficult)
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


CURRICULUM_STAGES = {
    1: {"goal_min": 0.10, "goal_max": 0.20, "label": "very_easy"},
    2: {"goal_min": 0.20, "goal_max": 0.40, "label": "easy"},
    3: {"goal_min": 0.40, "goal_max": 0.60, "label": "medium"},
    4: {"goal_min": 0.60, "goal_max": 0.80, "label": "hard"},
}


def quat_to_yaw(quat):
    """Extract yaw (rotation around Z) from MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = quat
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def omni_ik(vx_world, vy_world, omega, yaw):
    """
    Omni-wheel inverse kinematics.
    Converts world-frame velocity (vx, vy, omega) to 3 wheel velocities.
    
    Wheel layout (MuJoCo URDF):
      w1: front-right  at 0.0866, 0.10  — axis approx [-0.866, 0, 0.5]
      w2: back-left    at -0.0866, 0.10 — axis approx [0.866, 0, 0.5]
      w3: back-right   at -0.0866, -0.10 — axis approx [0, 0, -1]
    
    Steps: 
      1. Rotate world vel to robot frame
      2. Apply omni IK in robot frame
    """
    # 1. World → robot frame rotation
    c, s = np.cos(-yaw), np.sin(-yaw)
    vx_r = c * vx_world - s * vy_world
    vy_r = s * vx_world + c * vy_world
    
    # 2. Omni IK: wheels at 30°, 150°, 270° in robot frame (approx)
    # Standard 3-wheel omni: wheel_i = vx*cos(phi_i) + vy*sin(phi_i) + omega*r
    r = 0.12  # distance from center to wheel ~12cm
    phi = np.array([np.pi/6, 5*np.pi/6, 3*np.pi/2])  # 30°, 150°, 270°
    
    wheel = np.array([
        vx_r * np.cos(phi[i]) + vy_r * np.sin(phi[i]) + omega * r
        for i in range(3)
    ])
    # Scale to [-1, 1] for sim (sim expects normalized, applies *5.0 rad/s)
    wheel = wheel / 5.0
    return np.clip(wheel, -1.0, 1.0)


def collect_episode(sim, goal_pos, max_steps=200):
    """
    Collect one episode with a fixed P-controller.
    Returns dict of arrays: images, states, actions, rewards, goal_positions, goal_arrived.
    """
    images, states, actions, rewards, goal_positions = [], [], [], [], []
    
    obs = sim.reset(target=goal_pos)
    done = False
    goal_arrived = False
    total_reward = 0.0
    img = sim.render()
    
    for step in range(max_steps):
        # ── Get state ────────────────────────────────────────────────────────
        qpos = sim.data.qpos
        base_xy = qpos[:2]              # (x, y)
        yaw = qpos[2]                   # Planar joints: theta directly at qpos[2]
        
        # Wheel velocities are at qvel[3:6] (direct indexing with planar joints)
        wheel_vel = sim.data.qvel[3:6]
        arm_pos   = np.array([qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]])
        
        state = np.concatenate([arm_pos, wheel_vel, goal_pos])  # 6+3+2 = 11D goal-aware
        
        images.append(img.copy())
        states.append(state)
        goal_positions.append(goal_pos.copy())
        
        # ── P-controller ────────────────────────────────────────────────────
        dx = goal_pos[0] - base_xy[0]
        dy = goal_pos[1] - base_xy[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        Kp_v = 2.0       # position gain → controls forward speed
        Kp_w = 3.0       # heading gain → controls turning
        
        # Desired heading
        angle_to_goal = np.arctan2(dy, dx)
        angle_err = angle_to_goal - yaw
        # Normalize to [-pi, pi]
        angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
        
        # If not facing goal, prioritize turning (reduce forward speed)
        heading_factor = np.clip(np.cos(angle_err), 0.1, 1.0)
        vx_world = Kp_v * dist * np.cos(angle_to_goal) * heading_factor
        vy_world = Kp_v * dist * np.sin(angle_to_goal) * heading_factor
        omega    = Kp_w * angle_err
        
        # Clamp
        vx_world = np.clip(vx_world, -0.5, 0.5)
        vy_world = np.clip(vy_world, -0.5, 0.5)
        omega    = np.clip(omega, -2.0, 2.0)
        
        wheel_speeds = omni_ik(vx_world, vy_world, omega, yaw)
        
        # Arm stays at home
        action = np.concatenate([[0.0]*6, wheel_speeds]).astype(np.float32)
        actions.append(action)
        
        # ── Step ─────────────────────────────────────────────────────────────
        obs, reward, done, info = sim.step(action)
        img = sim.render()
        total_reward += reward
        rewards.append(float(reward))
        
        if dist < 0.10:
            goal_arrived = True
        
        if done:
            for _ in range(step + 1, max_steps):
                images.append(img.copy())
                states.append(state.copy())
                actions.append(action.copy())
                rewards.append(0.0)
                goal_positions.append(goal_pos.copy())
            break
    
    return {
        "images":         np.array(images,         dtype=np.uint8),
        "states":         np.array(states,         dtype=np.float32),
        "actions":        np.array(actions,        dtype=np.float32),
        "rewards":        np.array(rewards,        dtype=np.float32),
        "goal_positions": np.array(goal_positions, dtype=np.float32),
        "goal_arrived":   goal_arrived,
        "total_reward":   total_reward,
        "final_dist":     dist,
    }


def save_hdf5(trajectories, output_path):
    """Save collected trajectories to HDF5."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, "w") as f:
        all_images   = np.concatenate([t["images"]         for t in trajectories])
        all_states   = np.concatenate([t["states"]         for t in trajectories])
        all_actions  = np.concatenate([t["actions"]        for t in trajectories])
        all_rewards  = np.concatenate([t["rewards"]        for t in trajectories])
        all_goals    = np.concatenate([t["goal_positions"] for t in trajectories])
        
        f.create_dataset("images",         data=all_images,  compression="gzip")
        f.create_dataset("states",         data=all_states,  compression="gzip")
        f.create_dataset("actions",        data=all_actions, compression="gzip")
        f.create_dataset("rewards",        data=all_rewards, compression="gzip")
        f.create_dataset("goal_positions", data=all_goals,   compression="gzip")
        
        ep_len = trajectories[0]["images"].shape[0]
        f.attrs["num_episodes"]  = len(trajectories)
        f.attrs["episode_length"] = ep_len
        f.attrs["total_frames"]  = len(all_images)
        f.attrs["goal_arrivals"] = sum(t["goal_arrived"] for t in trajectories)
        f.attrs["state_dim"]     = all_states.shape[1]
        f.attrs["action_dim"]    = all_actions.shape[1]
        f.attrs["created_at"]    = datetime.now().isoformat()
    
    print(f"  Saved {len(trajectories)} episodes, {len(all_images)} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",    type=int, default=1, choices=[1,2,3,4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output",   type=str, required=True)
    args = parser.parse_args()
    
    cfg = CURRICULUM_STAGES[args.stage]
    print(f"[Stage {args.stage}] Curriculum: goals at {cfg['goal_min']}-{cfg['goal_max']}m radius — {cfg['label']}")
    print(f"[Stage {args.stage}] Collecting {args.episodes} episodes x 200 steps")
    
    sim = LeKiWiSimURDF()
    rng = np.random.default_rng(42)
    
    trajectories = []
    arrivals = 0
    
    for ep in range(args.episodes):
        # Sample goal in a random direction at the specified distance
        dist_target = rng.uniform(cfg["goal_min"], cfg["goal_max"])
        angle = rng.uniform(0, 2 * np.pi)
        goal = np.array([dist_target * np.cos(angle), dist_target * np.sin(angle)])
        
        traj = collect_episode(sim, goal)
        trajectories.append(traj)
        if traj["goal_arrived"]:
            arrivals += 1
        
        if (ep + 1) % 5 == 0:
            sr = arrivals / (ep + 1) * 100
            print(f"  Episodes {ep-3}-{ep+1}: arrival_rate={sr:.1f}%, "
                  f"latest reward={traj['total_reward']:.1f}, dist={traj['final_dist']:.3f}m")
    
    save_hdf5(trajectories, args.output)
    
    print(f"\n[Stage {args.stage}] Collection complete:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Goal arrivals: {arrivals}/{args.episodes} ({arrivals/args.episodes*100:.1f}%)")
    print(f"  Mean reward: {np.mean([t['total_reward'] for t in trajectories]):.1f}")
    print(f"  Mean final dist: {np.mean([t['final_dist'] for t in trajectories]):.3f}m")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
