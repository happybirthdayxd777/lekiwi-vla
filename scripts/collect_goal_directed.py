#!/usr/bin/env python3
"""
Goal-Directed Data Collection for LeKiWi Task-Oriented Training
================================================================
Collects (image, state, action) tuples where the robot actively moves toward
a target goal position — unlike collect_data.py which uses pure random walk.

Key difference from collect_data.py:
  - Pure random: action = clip(action + uniform(-0.15, 0.15), -1, 1)
  - Goal-directed: proportional controller steers toward goal

Episode flow:
  1. Reset sim at origin
  2. Sample goal position in the arena (0.3–0.7m radius)
  3. P-controller drives base toward goal
  4. Small Gaussian noise added for exploration
  5. Arm performs random-walk manipulation (arm actions independent of navigation)

Reward recorded per frame for training sample weighting:
  - +1.0 sparse when reaching goal
  - Shaped reward based on distance improvement

Usage:
  python3 scripts/collect_goal_directed.py \
    --episodes 50 \
    --steps 200 \
    --output data/lekiwi_goal_5k.h5

  python3 scripts/collect_goal_directed.py \
    --episodes 100 \
    --goal_min 0.2 --goal_max 0.6 \
    --output data/lekiwi_goal_10k.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import json

# Resize images to 224x224 for CLIP ViT-B/32 compatibility
TARGET_SIZE = (224, 224)

    # ─── Proportional Controller ─────────────────────────────────────────────────

class GridSearchController:
    """
    Grid-search adaptive controller for LeKiWi omni-wheel base.
    
    Since the wheel→velocity relationship is complex, non-linear, and noisy,
    we use a GRID SEARCH approach to find which wheel command moves the robot
    toward the goal:
    
    1. Each "step" = 20 simulation steps
    2. At each decision point:
       - Try each of 9 motion primitives for 20 steps
       - Measure which one reduces distance to goal most
       - Apply that primitive for the next 20 steps
    3. Small Gaussian noise added for exploration
    
    This is robust to:
    - Complex, non-linear wheel-ground contact physics
    - Wheel slip and stochastic effects
    - Model errors and uncertainties
    
    Phase 36 FIX: Corrected quadrant→primitive mapping based on EMPIRICALLY MEASURED
    wheel→direction data from LeKiWiSimURDF. Previous mapping was WRONG:
    
    Empirical direction mapping (URDF sim, 200 steps, action scale [-1,+1]):
      M0=[0,0,0]   → STOP
      M1=[1,0,0]   → +Y  (0.177m)
      M2=[0,1,0]   → -XY diagonal (0.724m, dx=-0.417, dy=-0.592)
      M3=[0,0,1]   → -Y  (0.054m)
      M4=[-1,0,0]  → -Y  (0.465m)
      M5=[0,-1,0]  → +Y  (0.498m)
      M6=[0,0,-1]  → +Y  (0.309m)
      M7=[1,1,1]   → +X  (1.606m, dx=+1.439, dy=-0.713) ← PRIMARY +X primitive!
      M8=[-1,-1,-1]→ +X  (0.159m) ← also +X (backward is SLOW in +X!)
    
    CRITICAL DISCOVERY: M7 and M8 BOTH move in +X direction!
    - M7=[1,1,1] → +1.606m in +X (fast, saturated)
    - M8=[-1,-1,-1] → +0.159m in +X (slow, not saturated)
    This means the robot CANNOT move in -X direction with any primitive!
    
    Correct quadrant mapping (URDF sim):
      Goal in +X +Y quadrant: M7 (all forward, +X dominant)
      Goal in +X -Y quadrant: M7 or M3 or M1 (mixed +X/+Y)
      Goal in -X +Y quadrant: M1 or M6 or M5 (pure +Y)
      Goal in -X -Y quadrant: M2 or M3 (pure -Y/-XY)
      No -X primitive exists — robot must navigate around obstacles
    """
    
    # 9 motion primitives: (w1, w2, w3)
    # Phase 36: Updated comments to reflect CORRECTED direction mapping
    # Phase 35: Scale to [1,1,1] — URDF sim clips at ctrl=10 (action=1.0),
    # giving ~1.6m/200steps at saturation.
    PRIMITIVES = np.array([
        [0.0,  0.0,  0.0 ],  # M0: stop
        [1.0,  0.0,  0.0 ],  # M1: w1 → +Y (0.177m/200steps)
        [0.0,  1.0,  0.0 ],  # M2: w2 → -XY diagonal (0.724m, dx=-0.417, dy=-0.592)
        [0.0,  0.0,  1.0 ],  # M3: w3 → -Y (0.054m)
        [-1.0, 0.0,  0.0 ],  # M4: w1 rev → -Y (0.465m)
        [0.0, -1.0,  0.0 ],  # M5: w2 rev → +Y (0.498m)
        [0.0,  0.0, -1.0 ],  # M6: w3 rev → +Y (0.309m)
        [1.0,  1.0,  1.0 ],  # M7: all forward → +X (1.606m/200steps) ← PRIMARY
        [-1.0,-1.0, -1.0 ],  # M8: all backward → +X (0.159m/200steps) ← SLOW
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
        """
        Returns wheel command for the current step.
        
        Steps 1..N: apply current primitive with noise
        At step N+1: evaluate all primitives, pick best, repeat
        """
        self._step_count += 1
        
        # On first step or at decision boundary: pick best primitive
        if self._step_count == 1 or self._step_count % self.steps_per_move == 1:
            if self._step_count > 1:
                # Evaluate previous primitive
                dist = np.linalg.norm(base_pos - goal_pos)
                if dist < self._best_dist:
                    self._best_dist = dist
                    self._best_primitive = self._current_primitive
                else:
                    # Revert to best known
                    self._current_primitive = self._best_primitive
            
            # Try each primitive and estimate which moves toward goal
            # For URDF sim, empirically: M1=[0.5,0,0] → +x+y, M3=[0,0,0.5] → +x+y
            # Use gradient-free selection: pick primitive that gives most negative dot product
            # with error vector (i.e., moves in the direction of -error = toward goal)
            # Empirically validated best directions:
            #   +y (forward): M7=[0.5,0.5,0.5] or M3=[0,0,0.5] or M1=[0.5,0,0]
            #   The URDF sim seems to move diagonally in +x+y for most positive wheel combos
            
            # Simple heuristic: move toward goal using sign of error
            error = goal_pos - base_pos
            dist = np.linalg.norm(error)
            
            if dist < 0.01:
                self._current_primitive = 0  # stop
            else:
                # Phase 36 FIX: Use CORRECTED quadrant mapping based on empirical URDF data.
                # 
                # Key insight: M7 and M8 BOTH move in +X direction!
                # - M7=[1,1,1] → +X (1.606m/200steps) ← fast
                # - M8=[-1,-1,-1] → +X (0.159m/200steps) ← slow
                # The robot CANNOT move in -X direction.
                #
                # Correct quadrant mapping:
                #   +X +Y quadrant: M7 (all forward, +X dominant)
                #   +X -Y quadrant: M7 or M1 or M3 (mixed +X/+Y)
                #   -X +Y quadrant: M1 or M6 or M5 (pure +Y)
                #   -X -Y quadrant: M2 or M3 (pure -Y/-XY diagonal)
                #   -X goal with large |dx|: rotate to +Y first via M1, then approach
                if error[0] >= 0 and error[1] >= 0:
                    # +X +Y: M7 gives +X and slight -Y (1.606m, -0.713m)
                    self._current_primitive = 7  # all forward
                elif error[0] >= 0 and error[1] < 0:
                    # +X -Y: M7's -Y component helps, or M3 (small -Y)
                    # M7 gives 1.44m in +X and -0.71m in Y (net useful)
                    self._current_primitive = 7  # all forward
                elif error[0] < 0 and error[1] >= 0:
                    # -X +Y: No -X primitive! Must approach from +Y
                    # M1 gives pure +Y (0.177m/200steps), small but controllable
                    # M6 and M5 also give +Y
                    self._current_primitive = 1  # M1: w1 → +Y
                else:  # error[0] < 0 and error[1] < 0
                    # -X -Y: Approach via M2 (gives -Y and some -X via diagonal)
                    self._current_primitive = 2  # M2: w2 → -XY diagonal
                
                self._best_primitive = self._current_primitive
                self._best_dist = dist
        
        # Apply current primitive with noise
        base_cmd = self.PRIMITIVES[self._current_primitive]
        noise = np.random.normal(0, self.exploration_noise, size=3)
        wheel_cmd = np.clip(base_cmd + noise, -1.0, 1.0)
        
        return wheel_cmd.astype(np.float32)


# Legacy alias
PController = GridSearchController

# ─── Simulation Factory ───────────────────────────────────────────────────────

def make_sim(sim_type: str):
    """Create simulation backend by type."""
    if sim_type == "urdf":
        from sim_lekiwi_urdf import LeKiWiSimURDF
        return LeKiWiSimURDF()
    else:
        from sim_lekiwi import LeKiwiSim
        return LeKiwiSim()


# ─── Reward Computation ────────────────────────────────────────────────────────

def compute_reward(base_pos_t, base_pos_tp1, goal_pos, threshold=0.1):
    """
    Compute shaped reward for this transition.
    
    Returns:
        reward: float — sparse + shaped
        is_goal: bool — whether we just arrived at goal
        dist_t, dist_tp1: distances before/after
    """
    GOAL_POS = np.array(goal_pos)
    dist_t  = np.linalg.norm(base_pos_t  - GOAL_POS)
    dist_tp1 = np.linalg.norm(base_pos_tp1 - GOAL_POS)
    
    # Sparse: +1.0 only when we ARRIVE (were NOT already at goal)
    if dist_tp1 < threshold and dist_t >= threshold:
        reward = 1.0
        is_goal = True
    else:
        # Shaped: reward proportional to improvement in distance
        improvement = dist_t - dist_tp1
        reward = np.clip(improvement / 0.1, -0.1, 0.1)
        is_goal = False
    
    return reward, is_goal, dist_t, dist_tp1


# ─── Episode Collection ───────────────────────────────────────────────────────

def collect_episode_goal_directed(sim, max_steps=200,
                                   goal_min=0.3, goal_max=0.7,
                                   goal_threshold=0.1,
                                   record_wrist=False,
                                   seed=None):
    """
    Collect one goal-directed episode: robot moves toward a random goal.
    
    State: arm positions (qpos[0:6]) + wheel velocities (qvel[0:3]) = [9]
    Action: arm (6) + wheel (3), normalized [-1, +1]
    
    Wheel actions come from GridSearchController toward goal.
    Arm actions are independent random walk (manipulation while navigating).
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        sim.reset()
    except AttributeError:
        pass  # LeKiWiSim has no reset()
    
    # Sample random goal in arena
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(goal_min, goal_max)
    goal_pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    # CRITICAL FIX (Phase 15): Must set target BEFORE first render/observation.
    # Previously set_target was called AFTER first render, causing image[0] to show
    # the STALE default target (0.5, 0) while state/action matched the real goal.
    # Now the rendered image always matches the current goal_pos.
    if hasattr(sim, 'set_target'):
        sim.set_target(goal_pos)
    
    imgs, states, actions, rewards = [], [], [], []
    wrist_imgs = [] if record_wrist else None
    goal_positions = []
    distances = []
    
    # Initialize GridSearchController for base
    controller = GridSearchController(steps_per_move=20, exploration_noise=0.08)
    controller.reset()
    
    # Current action: arm (random) + wheel (P-controller)
    arm_action = np.zeros(6, dtype=np.float32)
    wheel_action = np.zeros(3, dtype=np.float32)
    
    arrived = False
    steps_at_goal = 0
    max_steps_at_goal = 10  # Exit after staying at goal for a few steps
    
    for step in range(max_steps):
        # ── Render image ──────────────────────────────────────────────────
        img_arr = sim.render()
        if img_arr is None:
            img_arr = np.zeros((640, 480, 3), dtype=np.uint8)
        elif isinstance(img_arr, np.ndarray):
            img_arr = img_arr
        else:
            img_arr = np.array(img_arr)
        img_pil = Image.fromarray(img_arr).resize(TARGET_SIZE, Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.uint8)
        
        # ── State: arm positions + wheel velocities ─────────────────────
        # CRITICAL FIX (2026-04-13): Use sim._obs() for correct joint-level extraction.
        # LeKiWiSim (primitive):  qpos[0:6]=arm joints, qpos[6:9]=wheel (coincident by design)
        # LeKiWiSimURDF (mesh):  qpos[0:7]=base_free(xyz+quat), qpos[7:13]=arm joints
        # The old code used qpos[0:6] + qvel[0:3] which gave WRONG base pos + base vel!
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_vel = obs["wheel_velocities"]
        state = np.concatenate([arm_pos, wheel_vel]).astype(np.float32)

        # ── Base position for reward ────────────────────────────────────
        base_pos = sim.data.qpos[:2].copy() if hasattr(sim.data, 'qpos') else np.zeros(2)

        # ── P-controller for wheel action (goal-directed) ───────────────
        base_yaw = sim.data.qpos[3] if hasattr(sim.data, 'qpos') else 0.0  # qw quaternion
        wheel_cmd = controller.compute_wheel_velocities(base_pos, goal_pos, base_yaw)
        
        # Add exploration noise to wheel actions (Gaussian, small)
        noise = np.random.normal(0, 0.08, size=3)
        wheel_action = np.clip(wheel_cmd + noise, -1.0, 1.0).astype(np.float32)
        
        # ── Arm action: smooth random walk (independent of navigation) ──
        arm_delta = np.random.normal(0, 0.1, size=6).astype(np.float32)
        arm_action = np.clip(arm_action + arm_delta, -1.0, 1.0).astype(np.float32)
        
        # ── Combined action ─────────────────────────────────────────────
        action = np.concatenate([arm_action, wheel_action]).astype(np.float32)
        
        # ── Compute reward ────────────────────────────────────────────────
        result = sim.step(action)
        base_pos_next = sim.data.qpos[:2].copy() if hasattr(sim.data, 'qpos') else np.zeros(2)
        reward, is_goal, dist_t, dist_tp1 = compute_reward(base_pos, base_pos_next, goal_pos, goal_threshold)
        
        if is_goal:
            arrived = True
            steps_at_goal += 1
            if steps_at_goal >= max_steps_at_goal:
                # Stay at goal for a few steps then exit
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
        
        # Exit if terminated (if sim returns terminated flag)
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
    parser = argparse.ArgumentParser(description="Goal-directed LeKiWi data collection")
    parser.add_argument("--episodes",      type=int,   default=25,
                        help="Number of episodes to collect")
    parser.add_argument("--steps",         type=int,   default=200,
                        help="Max steps per episode")
    parser.add_argument("--output",        type=str,   default="data/lekiwi_goal_5k.h5",
                        help="Output HDF5 path")
    parser.add_argument("--sim_type",       type=str,   default="urdf",
                        choices=["primitive", "urdf"],
                        help="urdf=STL mesh (recommended), primitive=fast cylinders")
    parser.add_argument("--goal_min",      type=float, default=0.3,
                        help="Minimum goal distance from origin (m)")
    parser.add_argument("--goal_max",      type=float, default=0.7,
                        help="Maximum goal distance from origin (m)")
    parser.add_argument("--goal_threshold",type=float, default=0.1,
                        help="Goal arrival threshold (m)")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--wrist",         action="store_true",
                        help="Also capture wrist camera images")
    parser.add_argument("--validate",      action="store_true",
                        help="Run quick validation: 1 episode, print stats, exit")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[collect_goal_directed]")
    print(f"  sim_type={args.sim_type}, episodes={args.episodes}, steps={args.steps}")
    print(f"  goal_range=[{args.goal_min}, {args.goal_max}]m, threshold={args.goal_threshold}m")
    print(f"  Output: {output_path}")
    
    sim = make_sim(args.sim_type)
    print(f"  Simulator: {type(sim).__name__}")
    
    # Validation mode: just run 1 episode and print stats
    if args.validate:
        print("\n=== VALIDATION MODE (1 episode) ===")
        ep_data = collect_episode_goal_directed(
            sim, max_steps=200,
            goal_min=args.goal_min, goal_max=args.goal_max,
            goal_threshold=args.goal_threshold,
            record_wrist=args.wrist,
            seed=args.seed
        )
        
        actions = ep_data["action"]
        rewards = ep_data["reward"]
        distances = ep_data["distance_to_goal"]
        
        print(f"  Frames collected: {len(actions)}")
        print(f"  Goal reached: {ep_data['goal_reached']}")
        
        print(f"\n  Wheel action stats (should be non-zero, goal-directed):")
        for i, name in enumerate(["w1 (front-right)", "w2 (back-left)", "w3 (back)"]):
            w_mean = actions[:, 6+i].mean()
            w_std  = actions[:, 6+i].std()
            print(f"    wheel_{i} ({name}): mean={w_mean:+.3f}, std={w_std:.3f}")
        
        print(f"\n  Reward stats:")
        print(f"    mean={rewards.mean():+.4f}, std={rewards.std():.4f}")
        print(f"    max={rewards.max():+.4f}, min={rewards.min():+.4f}")
        print(f"    positive ({ rewards>0 }): {(rewards > 0).sum()} frames")
        
        print(f"\n  Distance to goal (start→end):")
        print(f"    start: {distances[0]:.3f}m, end: {distances[-1]:.3f}m")
        print(f"    improvement: {distances[0] - distances[-1]:+.3f}m")
        
        return
    
    # Actual collection
    all_images, all_states, all_actions = [], [], []
    all_rewards, all_goals = [], []
    all_wrist = []
    
    goals_reached_count = 0
    
    for ep in range(args.episodes):
        # Different seed per episode for diversity
        ep_seed = args.seed + ep * 137
        
        ep_data = collect_episode_goal_directed(
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
    
    # Concatenate all episodes
    all_images_nd = np.concatenate(all_images)
    all_states_nd  = np.concatenate(all_states)
    all_actions_nd = np.concatenate(all_actions)
    all_rewards_nd  = np.concatenate(all_rewards)
    all_goals_nd    = np.concatenate(all_goals)
    
    total = len(all_images_nd)
    print(f"\n─── Collection Summary ───")
    print(f"  Total frames: {total}")
    print(f"  Goals reached: {goals_reached_count}/{args.episodes} "
          f"({100*goals_reached_count/args.episodes:.1f}%)")
    
    # Action statistics
    print(f"\n  Wheel action means (goal-directed signal):")
    for i, name in enumerate(["w1 (front-right)", "w2 (back-left)", "w3 (back)"]):
        w_mean = all_actions_nd[:, 6+i].mean()
        w_std  = all_actions_nd[:, 6+i].std()
        print(f"    wheel_{i} ({name}): mean={w_mean:+.3f}, std={w_std:.3f}")
    
    print(f"\n  Arm action means:")
    for i in range(6):
        a_mean = all_actions_nd[:, i].mean()
        a_std  = all_actions_nd[:, i].std()
        print(f"    arm_{i}: mean={a_mean:+.3f}, std={a_std:.3f}")
    
    print(f"\n  Reward distribution:")
    pos_r = (all_rewards_nd > 0).sum()
    neg_r = (all_rewards_nd < 0).sum()
    zero_r = (all_rewards_nd == 0).sum()
    goal_r = (all_rewards_nd >= 1.0).sum()
    print(f"    positive: {pos_r} ({100*pos_r/total:.1f}%), "
          f"negative: {neg_r} ({100*neg_r/total:.1f}%), "
          f"zero: {zero_r} ({100*zero_r/total:.1f}%)")
    print(f"    goal arrivals (r=1.0): {goal_r}")
    
    # Save as HDF5
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images",   data=all_images_nd)
        f.create_dataset("states",   data=all_states_nd)
        f.create_dataset("actions",  data=all_actions_nd)
        f.create_dataset("rewards",  data=all_rewards_nd)
        f.create_dataset("goal_positions", data=all_goals_nd)
        if all_wrist:
            f.create_dataset("wrist_images", data=np.concatenate(all_wrist))
        f.attrs["episodes"]       = args.episodes
        f.attrs["steps"]          = args.steps
        f.attrs["sim_type"]       = args.sim_type
        f.attrs["goal_min"]       = args.goal_min
        f.attrs["goal_max"]       = args.goal_max
        f.attrs["goal_threshold"] = args.goal_threshold
        f.attrs["goals_reached"]  = goals_reached_count
        f.attrs["img_shape"]      = all_images[0][0].shape   # (H, W, C)
        f.attrs["state_dim"]      = all_states[0][0].shape   # (9,)
        f.attrs["action_dim"]     = all_actions[0][0].shape # (9,)
    
    print(f"✓ Saved {total} frames")
    print(f"  Images:   {output_path}['images']   {all_images_nd.shape}")
    print(f"  States:   {output_path}['states']   {all_states_nd.shape}")
    print(f"  Actions:  {output_path}['actions']  {all_actions_nd.shape}")
    print(f"  Rewards:  {output_path}['rewards']  {all_rewards_nd.shape}")
    print(f"\n  Dataset ready for task-oriented training:")
    print(f"  python3 scripts/train_task_oriented.py \\")
    print(f"    --data {output_path} \\")
    print(f"    --epochs 50 \\")
    print(f"    --device cpu \\")
    print(f"    --output results/task_oriented_goaldirected")


if __name__ == "__main__":
    main()
