#!/usr/bin/env python3
"""
Phase 63: Fix Training Data Quadrant Bias
==========================================

ROOT CAUSE (Phase 62):
  The robot can ONLY move in +X direction (M7=[1,1,1] and M8=[-1,-1,-1] 
  both produce +X motion). Goals in -X quadrants are UNREACHABLE, 
  resulting in 55% negative reward frames and a policy that cannot 
  handle goals in the -X hemisphere.

FIX:
  Only sample goals in the +X hemisphere (angles from -90° to +90°).
  This ensures all goals are reachable with M7, producing higher-quality
  training data with better reward signals.

Usage:
  python3 scripts/collect_reachable_goals.py \
    --episodes 100 \
    --steps 200 \
    --output data/phase63_reachable_10k.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

TARGET_SIZE = (224, 224)

# ─── GridSearchController (from collect_goal_directed.py, unchanged) ──────────

class GridSearchController:
    """Grid-search adaptive controller for LeKiWi omni-wheel base."""
    
    PRIMITIVES = np.array([
        [0.0,  0.0,  0.0 ],  # M0: stop
        [1.0,  0.0,  0.0 ],  # M1: w1 → +Y
        [0.0,  1.0,  0.0 ],  # M2: w2 → -XY diagonal
        [0.0,  0.0,  1.0 ],  # M3: w3 → -Y
        [-1.0, 0.0,  0.0 ],  # M4: w1 rev → -Y
        [0.0, -1.0,  0.0 ],  # M5: w2 rev → +Y
        [0.0,  0.0, -1.0 ],  # M6: w3 rev → +Y
        [1.0,  1.0,  1.0 ],  # M7: all forward → +X (PRIMARY)
        [-1.0,-1.0, -1.0 ],  # M8: all backward → +X (slow)
    ], dtype=np.float32)
    
    def __init__(self, steps_per_move=20, exploration_noise=0.05):
        self.steps_per_move = steps_per_move
        self.exploration_noise = exploration_noise
        self._step_count = 0
        self._current_primitive = 0
        self._best_primitive = 0
        self._pos_start = None
        self._best_dist = float('inf')
        
    def reset(self):
        self._step_count = 0
        self._current_primitive = 0
        self._best_primitive = 0
        self._pos_start = None
        self._best_dist = float('inf')
    
    def compute_wheel_velocities(self, base_pos, goal_pos, base_yaw=0.0):
        self._step_count += 1
        
        if self._step_count == 1 or self._step_count % self.steps_per_move == 1:
            if self._step_count > 1:
                dist = np.linalg.norm(base_pos - goal_pos)
                if dist < self._best_dist:
                    self._best_dist = dist
                    self._best_primitive = self._current_primitive
                else:
                    self._current_primitive = self._best_primitive
            
            error = goal_pos - base_pos
            dist = np.linalg.norm(error)
            
            if dist < 0.01:
                self._current_primitive = 0
            else:
                # Phase 63 FIX: Goals are always in +X hemisphere
                # M7 is the PRIMARY +X primitive (1.606m/200steps)
                if error[0] >= 0 and error[1] >= 0:
                    self._current_primitive = 7  # +X +Y: M7
                elif error[0] >= 0 and error[1] < 0:
                    self._current_primitive = 7  # +X -Y: M7
                elif error[0] < 0 and error[1] >= 0:
                    self._current_primitive = 1  # -X +Y: M1 (pure +Y)
                else:
                    self._current_primitive = 2  # -X -Y: M2 (-XY diagonal)
                
                self._best_primitive = self._current_primitive
                self._best_dist = dist
        
        base_cmd = self.PRIMITIVES[self._current_primitive]
        noise = np.random.normal(0, self.exploration_noise, size=3)
        wheel_cmd = np.clip(base_cmd + noise, -1.0, 1.0)
        
        return wheel_cmd.astype(np.float32)


# ─── Simulation Factory ────────────────────────────────────────────────────────

def make_sim(sim_type: str):
    if sim_type == "urdf":
        from sim_lekiwi_urdf import LeKiWiSimURDF
        return LeKiWiSimURDF()
    else:
        from sim_lekiwi import LeKiWiSim
        return LeKiWiSim()


# ─── Reward Computation ─────────────────────────────────────────────────────────

def compute_reward(base_pos_t, base_pos_tp1, goal_pos, threshold=0.1):
    GOAL_POS = np.array(goal_pos)
    dist_t  = np.linalg.norm(base_pos_t  - GOAL_POS)
    dist_tp1 = np.linalg.norm(base_pos_tp1 - GOAL_POS)
    
    if dist_tp1 < threshold and dist_t >= threshold:
        reward = 1.0
        is_goal = True
    else:
        improvement = dist_t - dist_tp1
        reward = np.clip(improvement / 0.1, -0.1, 0.1)
        is_goal = False
    
    return reward, is_goal, dist_t, dist_tp1


# ─── Phase 63: REACHABLE Goal Sampling ────────────────────────────────────────

def sample_reachable_goal(goal_min=0.3, goal_max=0.7, seed=None):
    """
    Phase 63 FIX: Only sample goals in the +X hemisphere.
    
    The robot can ONLY move in +X direction (M7=[1,1,1] and M8=[-1,-1,-1]).
    Goals in the -X hemisphere are UNREACHABLE, producing only negative rewards.
    
    Solution: Sample goals uniformly from angles [-90°, +90°] relative to +X axis.
    This gives full coverage of the reachable workspace.
    
    This replaces uniform circle sampling [0, 2π] which put 50% of goals
    in the unreachable -X hemisphere.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample angle uniformly from [-90°, +90°] = [-π/2, +π/2]
    # This ensures all goals are in the +X hemisphere
    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    radius = np.random.uniform(goal_min, goal_max)
    
    goal_pos = np.array([
        radius * np.cos(angle),  # Always >= 0 (reachable with M7)
        radius * np.sin(angle),  # Can be +Y or -Y (reachable)
    ])
    
    return goal_pos


# ─── Episode Collection ────────────────────────────────────────────────────────

def collect_episode_reachable(sim, max_steps=200,
                               goal_min=0.3, goal_max=0.7,
                               goal_threshold=0.1,
                               record_wrist=False,
                               seed=None):
    """Collect one episode with goals only in the reachable +X hemisphere."""
    if seed is not None:
        np.random.seed(seed)
    
    try:
        sim.reset()
    except AttributeError:
        pass
    
    # Phase 63 FIX: Use reachable goal sampling
    goal_pos = sample_reachable_goal(goal_min, goal_max, seed)
    
    if hasattr(sim, 'set_target'):
        sim.set_target(goal_pos)
    
    imgs, states, actions, rewards = [], [], [], []
    wrist_imgs = [] if record_wrist else None
    goal_positions = []
    distances = []
    
    controller = GridSearchController(steps_per_move=20, exploration_noise=0.08)
    controller.reset()
    
    arm_action = np.zeros(6, dtype=np.float32)
    wheel_action = np.zeros(3, dtype=np.float32)
    
    arrived = False
    steps_at_goal = 0
    max_steps_at_goal = 10
    
    for step in range(max_steps):
        # Render image
        img_arr = sim.render()
        if img_arr is None:
            img_arr = np.zeros((640, 480, 3), dtype=np.uint8)
        elif isinstance(img_arr, np.ndarray):
            img_arr = img_arr
        else:
            img_arr = np.array(img_arr)
        img_pil = Image.fromarray(img_arr).resize(TARGET_SIZE, Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.uint8)
        
        # State: arm positions + wheel velocities
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_vel = obs["wheel_velocities"]
        state = np.concatenate([arm_pos, wheel_vel]).astype(np.float32)
        
        # Base position for reward
        base_pos = sim.data.qpos[:2].copy() if hasattr(sim.data, 'qpos') else np.zeros(2)
        
        # P-controller
        base_yaw = sim.data.qpos[3] if hasattr(sim.data, 'qpos') else 0.0
        wheel_cmd = controller.compute_wheel_velocities(base_pos, goal_pos, base_yaw)
        
        noise = np.random.normal(0, 0.08, size=3)
        wheel_action = np.clip(wheel_cmd + noise, -1.0, 1.0).astype(np.float32)
        
        # Arm random walk
        arm_delta = np.random.normal(0, 0.1, size=6).astype(np.float32)
        arm_action = np.clip(arm_action + arm_delta, -1.0, 1.0).astype(np.float32)
        
        # Combined action
        action = np.concatenate([arm_action, wheel_action]).astype(np.float32)
        
        # Reward
        result = sim.step(action)
        base_pos_next = sim.data.qpos[:2].copy() if hasattr(sim.data, 'qpos') else np.zeros(2)
        reward, is_goal, dist_t, dist_tp1 = compute_reward(base_pos, base_pos_next, goal_pos, goal_threshold)
        
        if is_goal:
            arrived = True
            steps_at_goal += 1
            if steps_at_goal >= max_steps_at_goal:
                pass
        
        imgs.append(img_arr)
        states.append(state)
        actions.append(action.copy())
        rewards.append(reward)
        goal_positions.append(goal_pos.copy())
        distances.append(dist_t)
        
        if record_wrist and hasattr(sim, 'render_wrist'):
            wimg = sim.render_wrist()
            if wimg is not None:
                if isinstance(wimg, np.ndarray):
                    wimg_pil = Image.fromarray(wimg).resize(TARGET_SIZE, Image.BILINEAR)
                else:
                    wimg_pil = wimg.resize(TARGET_SIZE, Image.BILINEAR)
                wrist_imgs.append(np.array(wimg_pil, dtype=np.uint8))
        
        if isinstance(result, tuple) and len(result) >= 3:
            _, _, term, *_ = result
            if term:
                break
    
    result = {
        "image": np.stack(imgs),
        "state": np.stack(states),
        "action": np.stack(actions),
        "reward": np.array(rewards, dtype=np.float32),
        "goal_position": np.stack(goal_positions),
        "distance_to_goal": np.array(distances, dtype=np.float32),
        "goal_reached": arrived,
    }
    if record_wrist and wrist_imgs:
        result["wrist_image"] = np.stack(wrist_imgs)
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 63: Collect data with REACHABLE +X hemisphere goals only"
    )
    parser.add_argument("--episodes",      type=int,   default=50,
                        help="Number of episodes to collect")
    parser.add_argument("--steps",         type=int,   default=200,
                        help="Max steps per episode")
    parser.add_argument("--output",        type=str,
                        default="data/phase63_reachable_10k.h5",
                        help="Output HDF5 path")
    parser.add_argument("--sim_type",       type=str,   default="urdf",
                        choices=["primitive", "urdf"],
                        help="Simulation type")
    parser.add_argument("--goal_min",      type=float,  default=0.3,
                        help="Minimum goal radius (m)")
    parser.add_argument("--goal_max",      type=float,  default=0.7,
                        help="Maximum goal radius (m)")
    parser.add_argument("--goal_threshold",type=float,  default=0.1,
                        help="Goal arrival threshold (m)")
    parser.add_argument("--wrist",         action="store_true",
                        help="Record wrist camera")
    parser.add_argument("--validate",      action="store_true",
                        help="Validate one episode then exit")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--ep_offset",     type=int,   default=0,
                        help="Episode offset for seeding")
    
    args = parser.parse_args()
    
    print(f"=== Phase 63: REACHABLE Goal Data Collection ===")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Steps/ep:   {args.steps}")
    print(f"  Sim type:   {args.sim_type}")
    print(f"  Goal range: [{args.goal_min}, {args.goal_max}] m")
    print(f"  Seed:       {args.seed}")
    print()
    
    # Validate goal distribution first
    print("=== Validating REACHABLE goal sampling ===")
    test_goals = [sample_reachable_goal(args.goal_min, args.goal_max, args.seed + i)
                  for i in range(1000)]
    test_goals = np.array(test_goals)
    
    # Quadrant analysis
    q1 = ((test_goals[:,0] >= 0) & (test_goals[:,1] >= 0)).sum()
    q2 = ((test_goals[:,0] < 0)  & (test_goals[:,1] >= 0)).sum()
    q3 = ((test_goals[:,0] < 0)  & (test_goals[:,1] < 0)).sum()
    q4 = ((test_goals[:,0] >= 0) & (test_goals[:,1] < 0)).sum()
    
    print(f"  Goal distribution (1000 samples):")
    print(f"    Q1 (+X,+Y): {q1:4d} ({q1/10:.1f}%)  ← REACHABLE")
    print(f"    Q2 (-X,+Y): {q2:4d} ({q2/10:.1f}%)  ← UNREACHABLE")
    print(f"    Q3 (-X,-Y): {q3:4d} ({q3/10:.1f}%)  ← UNREACHABLE")
    print(f"    Q4 (+X,-Y): {q4:4d} ({q4/10:.1f}%)  ← REACHABLE")
    print(f"  Reachable: {q1+q4} ({(q1+q4)/10:.1f}%)")
    print(f"  Unreachable: {q2+q3} ({(q2+q3)/10:.1f}%)")
    print(f"  Mean goal position: ({test_goals[:,0].mean():+.3f}, {test_goals[:,1].mean():+.3f})")
    print()
    
    if args.validate:
        print("\n=== VALIDATION MODE (1 episode) ===")
        from sim_lekiwi_urdf import LeKiWiSimURDF
        sim = LeKiWiSimURDF()
        ep_data = collect_episode_reachable(
            sim, max_steps=200,
            goal_min=args.goal_min, goal_max=args.goal_max,
            goal_threshold=args.goal_threshold,
            record_wrist=args.wrist,
            seed=args.seed
        )
        
        actions = ep_data["action"]
        rewards = ep_data["reward"]
        distances = ep_data["distance_to_goal"]
        goals = ep_data["goal_position"]
        
        print(f"  Frames collected: {len(actions)}")
        print(f"  Goal reached: {ep_data['goal_reached']}")
        
        print(f"\n  Wheel action stats:")
        for i, name in enumerate(["w1", "w2", "w3"]):
            w_mean = actions[:, 6+i].mean()
            w_std  = actions[:, 6+i].std()
            print(f"    wheel_{i} ({name}): mean={w_mean:+.3f}, std={w_std:.3f}")
        
        print(f"\n  Reward stats:")
        print(f"    mean={rewards.mean():+.4f}, std={rewards.std():.4f}")
        print(f"    positive: {(rewards > 0).sum()} frames ({(rewards > 0).mean()*100:.1f}%)")
        print(f"    goal arrivals: {ep_data['goal_reached']}")
        
        print(f"\n  Distance to goal (start→end):")
        print(f"    start: {distances[0]:.3f}m, end: {distances[-1]:.3f}m")
        print(f"    improvement: {distances[0] - distances[-1]:+.3f}m")
        
        print(f"\n  Goal position analysis:")
        print(f"    mean: ({goals[:,0].mean():+.3f}, {goals[:,1].mean():+.3f})")
        print(f"    x >= 0: {(goals[:,0] >= 0).sum()}/{len(goals)} (all should be >= 0)")
        return
    
    # Actual collection
    from sim_lekiwi_urdf import LeKiWiSimURDF
    sim = LeKiWiSimURDF()
    
    all_images, all_states, all_actions = [], [], []
    all_rewards, all_goals = [], []
    all_wrist = []
    
    goals_reached_count = 0
    
    for ep in range(args.episodes):
        ep_seed = args.seed + (ep + args.ep_offset) * 137
        
        ep_data = collect_episode_reachable(
            sim, max_steps=args.steps,
            goal_min=args.goal_min, goal_max=args.goal_max,
            goal_threshold=args.goal_threshold,
            record_wrist=args.wrist,
            seed=ep_seed
        )
        
        all_images.append(ep_data["image"])
        all_states.append(ep_data["state"])
        all_actions.append(ep_data["action"])
        all_rewards.append(ep_data["reward"])
        all_goals.append(ep_data["goal_position"])
        if "wrist_image" in ep_data:
            all_wrist.append(ep_data["wrist_image"])
        
        n = len(ep_data["image"])
        arrived = ep_data["goal_reached"]
        if arrived:
            goals_reached_count += 1
        dist_start = ep_data["distance_to_goal"][0]
        dist_end   = ep_data["distance_to_goal"][-1]
        
        print(f"  Episode {ep+1:3d}/{args.episodes}: {n:4d} frames, "
              f"goal={'✓' if arrived else '✗'}, "
              f"dist: {dist_start:.2f}m→{dist_end:.2f}m "
              f"({dist_start-dist_end:+.2f}m)")
    
    print(f"\n=== SAVING to {args.output} ===")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    all_images   = np.concatenate(all_images, axis=0)
    all_states   = np.concatenate(all_states, axis=0)
    all_actions  = np.concatenate(all_actions, axis=0)
    all_rewards  = np.concatenate(all_rewards, axis=0)
    all_goals    = np.concatenate(all_goals, axis=0)
    
    with h5py.File(args.output, "w") as f:
        f.create_dataset("image",       data=all_images,   compression="lzf")
        f.create_dataset("state",       data=all_states,  compression="lzf")
        f.create_dataset("action",      data=all_actions,  compression="lzf")
        f.create_dataset("reward",      data=all_rewards,  compression="lzf")
        f.create_dataset("goal_position", data=all_goals, compression="lzf")
        if all_wrist:
            all_wrist = np.concatenate(all_wrist, axis=0)
            f.create_dataset("wrist_image", data=all_wrist, compression="lzf")
        
        f.attrs["episodes"]      = args.episodes
        f.attrs["steps"]          = args.steps
        f.attrs["goal_min"]       = args.goal_min
        f.attrs["goal_max"]       = args.goal_max
        f.attrs["goal_threshold"] = args.goal_threshold
        f.attrs["sim_type"]       = args.sim_type
        f.attrs["seed"]           = args.seed
        f.attrs["phase"]          = 63
        f.attrs["description"]     = "Phase 63: REACHABLE +X hemisphere goals only"
    
    print(f"  Saved: {all_images.shape[0]} frames")
    print(f"  Goals reached: {goals_reached_count}/{args.episodes} episodes")
    print(f"  Positive reward frames: {(all_rewards > 0).sum()}/{len(all_rewards)} ({(all_rewards > 0).mean()*100:.1f}%)")
    
    # Final quadrant analysis
    q1 = ((all_goals[:,0] >= 0) & (all_goals[:,1] >= 0)).sum()
    q2 = ((all_goals[:,0] < 0)  & (all_goals[:,1] >= 0)).sum()
    q3 = ((all_goals[:,0] < 0)  & (all_goals[:,1] < 0)).sum()
    q4 = ((all_goals[:,0] >= 0) & (all_goals[:,1] < 0)).sum()
    print(f"\n  Final goal quadrant distribution:")
    print(f"    Q1 (+X,+Y): {q1:5d} ({q1/len(all_goals)*100:.1f}%)")
    print(f"    Q2 (-X,+Y): {q2:5d} ({q2/len(all_goals)*100:.1f}%) ← should be ~0")
    print(f"    Q3 (-X,-Y): {q3:5d} ({q3/len(all_goals)*100:.1f}%) ← should be ~0")
    print(f"    Q4 (+X,-Y): {q4:5d} ({q4/len(all_goals)*100:.1f}%)")


if __name__ == "__main__":
    main()
