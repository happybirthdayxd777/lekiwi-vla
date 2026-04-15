#!/usr/bin/env python3
"""
Phase 106: Collect goal-directed training data on Phase 86 URDF physics
=========================================================================
Collects (image, state, action, reward, goal_position) tuples using LeKiWiSimURDF
(k_omni=15 physics) so the resulting policy is trained on correct physics.

This is the ROOT CAUSE FIX for Phase 85→86 mismatch:
  - Phase 85/goal_aware_50ep: trained on LeKiWiSim (primitive) physics
  - Phase 86 URDF sim: k_omni overlay, different locomotion model
  - Solution: retrain on Phase 86 physics → policy-physics match → SR > 0%

Key differences from collect_goal_directed.py:
  1. Uses LeKiWiSimURDF (k_omni=15) not LeKiWiSim
  2. GridSearchController actions scaled to [-1, 1] for policy compatibility
  3. 200 steps per episode, random goals in reachable range
  4. Records arm (6) + wheel (3) actions normalized to [-1, 1]

Usage:
  python3 scripts/collect_urdf_goal.py --episodes 100 --output data/phase106_urdf_goal_10k.h5

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


class GridSearchController:
    """
    Grid-search adaptive controller for LeKiWi omni-wheel base.
    Since the wheel→velocity relationship is complex, non-linear, and noisy,
    we use a GRID SEARCH approach to find which wheel command moves the robot
    toward the goal.

    Phase 106: This controller is used for DATA COLLECTION on Phase 86 URDF physics.
    The PRIMITIVES are calibrated for Phase 86 k_omni physics.
    """
    
    PRIMITIVES = [
        # [w1, w2, w3] — wheel velocities in rad/s
        # These produce the best locomotion in Phase 86 URDF k_omni physics
        [3.0, 2.0, 1.0],    # M1: asymmetric — high forward + rotation
        [1.0, 1.0, 1.0],    # M2: symmetric — vy motion only (no forward!)
        [0.5, -0.5, 0.0],   # M3: rotation — pure yaw (no translation!)
        [1.0, 0.5, -0.5],   # M4: partial asymmetric
        [0.3, -0.1, -0.3],  # M5: best grid search result
        [2.0, 1.0, 0.0],    # M6: forward-biased
        [0.0, 1.0, -1.0],   # M7: sideways
        [1.0, -1.0, 0.0],   # M8: opposite rotation
        [0.5, 0.5, -0.5],   # M9: diagonal
    ]
    
    def __init__(self, steps_per_move=20, exploration_noise=0.08):
        self.steps_per_move = steps_per_move
        self.exploration_noise = exploration_noise
        self._counter = 0
        self._current_primitive = 0
    
    def reset(self):
        self._counter = 0
        self._current_primitive = 0
    
    def compute_wheel_velocities(self, base_pos, goal_pos, base_yaw):
        """
        Return wheel velocity command (3,) for Phase 86 URDF k_omni physics.
        Values are in rad/s (to be applied directly as MuJoCo ctrl).
        """
        # Direction to goal (world frame)
        dx = goal_pos[0] - base_pos[0]
        dy = goal_pos[1] - base_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 0.05:
            return np.array([0.0, 0.0, 0.0])
        
        # Rotate goal direction to body frame
        yaw = base_yaw  # qw quaternion gives yaw
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        body_dx = cos_y * dx + sin_y * dy
        body_dy = -sin_y * dx + cos_y * dy
        
        # Move counter forward
        self._counter += 1
        
        # Try each primitive for steps_per_move steps
        if self._counter >= self.steps_per_move:
            self._counter = 0
            # Pick the primitive most likely to move toward goal
            # For Phase 86: [3,2,1] produces +X force (via w2 dominant vx)
            # and [0.3,-0.1,-0.3] produces +X (small but present)
            best_prim = 0  # default to M1
            if body_dx < -0.1:
                # Need to go backward
                best_prim = 4  # M5 has some backward component
            elif abs(body_dy) > abs(body_dx) * 2:
                # Strong lateral component needed
                best_prim = 2  # M3 rotation
            self._current_primitive = best_prim
            if self._current_primitive < len(self.PRIMITIVES) - 1:
                self._current_primitive += 1
        
        # Get current primitive
        primitive = np.array(self.PRIMITIVES[self._current_primitive])
        
        # Scale by distance (closer = slower)
        if dist < 0.2:
            primitive *= 0.5
        elif dist < 0.1:
            primitive *= 0.3
        
        # Add noise for exploration
        if self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=3)
            primitive = primitive + noise
        
        return primitive.astype(np.float32)


def compute_reward(base_pos, base_pos_next, goal_pos, threshold=0.1):
    """Sparse + shaped reward for goal-directed navigation."""
    dist_t = np.linalg.norm(base_pos - goal_pos)
    dist_tp1 = np.linalg.norm(base_pos_next - goal_pos)
    
    is_goal = dist_tp1 < threshold
    
    # Sparse: +1.0 at goal
    if is_goal and dist_tp1 < dist_t:  # actually moving closer
        reward = 1.0
    elif is_goal:
        reward = 0.5  # at goal but not improving
    else:
        # Shaped: reward based on distance improvement
        reward = (dist_t - dist_tp1) * 2.0
    
    return reward, is_goal, dist_t, dist_tp1


def collect_episode(sim, max_steps=200, goal_min=0.3, goal_max=0.7,
                   goal_threshold=0.1, seed=None):
    """
    Collect one goal-directed episode on Phase 86 URDF physics.
    
    State: arm positions (6) + wheel velocities (3) = 9D
    Action: arm (6) + wheel (3) = 9D normalized [-1, 1]
    
    Returns dict with keys: image, state, action, reward, goal_position
    """
    if seed is not None:
        np.random.seed(seed)
    
    sim.reset()
    
    # Sample random goal
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(goal_min, goal_max)
    goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    if hasattr(sim, 'set_target'):
        sim.set_target(goal_pos)
    
    controller = GridSearchController(steps_per_move=20, exploration_noise=0.1)
    controller.reset()
    
    imgs, states, actions, rewards, goal_positions = [], [], [], [], []
    
    arm_action = np.zeros(6, dtype=np.float32)
    max_wheel_ctrl = 0.5  # MuJoCo wheel ctrl range (URDF sim)
    
    for step in range(max_steps):
        # Render image
        img_arr = sim.render()
        if img_arr is None:
            img_arr = np.zeros((640, 480, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img_arr).resize(TARGET_SIZE, Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.uint8)
        
        # Get state
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_vel = obs["wheel_velocities"]
        state = np.concatenate([arm_pos, wheel_vel]).astype(np.float32)
        
        # Base position from free joint
        base_pos = sim.data.qpos[:2].copy()
        
        # P-controller wheel command (in rad/s for URDF sim ctrl)
        base_yaw = sim.data.qpos[3] if hasattr(sim, 'data') else 0.0
        wheel_vel_cmd = controller.compute_wheel_velocities(base_pos, goal_pos, base_yaw)
        
        # Normalize wheel velocities to [-1, 1] for policy action
        # max_wheel_ctrl = 0.5 rad/s (URDF sim wheel clamp)
        # But controller outputs up to 3.0 rad/s → scale
        wheel_action_norm = np.clip(wheel_vel_cmd / 3.0, -1.0, 1.0).astype(np.float32)
        
        # Arm: smooth random walk
        arm_delta = np.random.normal(0, 0.1, size=6).astype(np.float32)
        arm_action = np.clip(arm_action + arm_delta, -1.0, 1.0).astype(np.float32)
        
        # Combined action (normalized)
        action = np.concatenate([arm_action, wheel_action_norm]).astype(np.float32)
        
        # Apply to URDF sim (needs ctrl in actual units)
        ctrl_arm = arm_action * 3.14  # arm: ±3.14 rad
        ctrl_wheel = wheel_vel_cmd  # wheel: raw rad/s (clamped in sim)
        ctrl = np.concatenate([ctrl_arm, ctrl_wheel]).astype(np.float64)
        sim.step(ctrl)
        
        # Next base position for reward
        base_pos_next = sim.data.qpos[:2].copy()
        reward, is_goal, dist_t, dist_tp1 = compute_reward(base_pos, base_pos_next, goal_pos, goal_threshold)
        
        imgs.append(img_arr)
        states.append(state)
        actions.append(action.copy())
        rewards.append(reward)
        goal_positions.append(goal_pos.copy())
        
        if is_goal and dist_tp1 < dist_t:
            # At goal and improving — could exit early
            pass
    
    return {
        "image": np.stack(imgs),
        "state": np.stack(states),
        "action": np.stack(actions),
        "reward": np.array(rewards, dtype=np.float32),
        "goal_position": np.stack(goal_positions),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect goal-directed data on URDF physics")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--goal-min", type=float, default=0.3)
    parser.add_argument("--goal-max", type=float, default=0.7)
    parser.add_argument("--output", type=str, default="data/phase106_urdf_goal_10k.h5")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Import sim
    from sim_lekiwi_urdf import LeKiWiSimURDF
    print(f"[INFO] Creating LeKiWiSimURDF (Phase 86 physics, k_omni=15)")
    sim = LeKiWiSimURDF()
    
    all_images = []
    all_states = []
    all_actions = []
    all_rewards = []
    all_goals = []
    
    total_frames = 0
    for ep in range(args.episodes):
        seed = args.seed + ep if args.seed is not None else None
        result = collect_episode(
            sim,
            max_steps=args.steps,
            goal_min=args.goal_min,
            goal_max=args.goal_max,
            seed=seed,
        )
        
        n = len(result["reward"])
        all_images.append(result["image"])
        all_states.append(result["state"])
        all_actions.append(result["action"])
        all_rewards.append(result["reward"])
        all_goals.append(result["goal_position"])
        total_frames += n
        
        if (ep + 1) % 10 == 0:
            goal_reached = (np.concatenate(all_rewards) >= 1.0).sum()
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
    
    # Save to HDF5
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip')
        f.create_dataset('states', data=states)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('goal_positions', data=goals)
    
    print(f"[INFO] Saved to {args.output}")
    
    # Save metadata
    meta = {
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "total_frames": total_frames,
        "physics": "Phase 86 URDF (k_omni=15)",
        "sim_type": "LeKiWiSimURDF",
        "state_dim": 9,
        "action_dim": 9,
        "goal_dim": 2,
    }
    meta_path = args.output.replace('.h5', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Metadata: {meta_path}")


if __name__ == "__main__":
    main()
