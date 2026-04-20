#!/usr/bin/env python3
"""
Phase 227: Collect Q2 Extended Data for LeKiWi VLA Training
============================================================

ROOT CAUSE from Phase 226:
  - Training Q2 goals: gy ∈ [0.020, 0.235]m (ONLY)
  - Eval Q2 failures: 9/10 had gy > 0.235m (OUT OF DISTRIBUTION)
  - FAIL pattern: large |gx| > 0.24 AND large |gy| > 0.24 simultaneously
  - This combination was NEVER seen in 13 Q2 training episodes

Strategy:
  - Collect 15 episodes specifically targeting Q2 with gy ∈ [0.24, 0.45]m
  - Use SAME Contact-Jacobian P-controller as Phase 196 (kP=2.0, 100% SR)
  - Goal distribution:
    * 5 episodes: gx ∈ [-0.35, -0.15], gy ∈ [0.28, 0.40]  (left-deep Q2)
    * 5 episodes: gx ∈ [-0.20, -0.05], gy ∈ [0.24, 0.38]  (front Q2)
    * 5 episodes: gx ∈ [-0.30, -0.20], gy ∈ [0.20, 0.35]  (mid Q2 diagonal)
  - Combine with existing 50 episodes from phase196_clean_50ep.h5
  - Result: phase227_extended_65ep.h5 with comprehensive Q2 coverage

Controller: Same as Phase 196 — Contact-Jacobian P-controller
  - v_desired = kP * err (kP=2.0)
  - wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
  - This achieves 100% SR in simulation

Usage:
  python3 scripts/collect_phase227_q2_extended.py --n_q2_episodes 15
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path

# Import the CORRECT Contact-Jacobian model
from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV

np.random.seed(None)


class Phase196Controller:
    """
    CORRECT Contact-Jacobian P-controller.
    Same as Phase 196 data collection.
    """
    def __init__(self, kP=2.0, wheel_clip=0.5):
        self.kP = kP
        self.wheel_clip = wheel_clip

    def compute(self, goal_xy, base_xy):
        """Compute wheel speed actions from goal and current base position."""
        err = goal_xy - base_xy  # [gx, gy]
        v_desired = self.kP * err  # [vx, vy]
        wheel_speeds = _CONTACT_JACOBIAN_PSEUDO_INV @ v_desired
        wheel_speeds = np.clip(wheel_speeds, -self.wheel_clip, self.wheel_clip)
        return np.array(wheel_speeds, dtype=np.float32)


def collect_episode(sim, controller, goal, goal_range=0.35, max_steps=250, seed=None):
    """Collect one episode with goal-directed navigation."""
    if seed is not None:
        np.random.seed(seed)

    goal = np.array(goal, dtype=np.float32)

    sim.reset()
    base_body_id = sim.model.body('base').id

    states_list = []
    actions_list = []
    images_list = []
    goals_list = []
    rewards_list = []

    # Fixed arm position (resting)
    arm_pos = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0], dtype=np.float32)

    for step in range(max_steps):
        # Get current base position
        base_xy = sim.data.xpos[base_body_id, :2].copy()

        # Compute wheel speeds using Contact-Jacobian controller
        wheel_speeds = controller.compute(goal, base_xy)

        # Full action: arm + wheels
        action = np.concatenate([arm_pos, wheel_speeds]).astype(np.float32)

        # Render image BEFORE step (current view — front camera)
        img = sim.render().astype(np.uint8)

        # Step simulation
        sim.step(action)

        # Record state: arm_pos(6) + wheel_vel(3) + goal_norm(2)
        wheel_vel = sim.data.qvel[6:9].copy()  # CORRECT: wheel joint velocities
        state = np.concatenate([
            arm_pos,                    # 6 arm positions
            wheel_vel,                  # 3 wheel velocities
            goal / (goal_range * 1.5)  # 2 normalized goal coords
        ]).astype(np.float32)

        # Compute reward
        dist = np.linalg.norm(goal - base_xy)
        reward = 1.0 if dist < 0.10 else 0.0

        # Record
        states_list.append(state)
        actions_list.append(action)
        images_list.append(img)
        goals_list.append(goal / (goal_range * 1.5))  # Normalized
        rewards_list.append(reward)

        if reward > 0.5:
            break  # Success!

    return {
        'states': np.array(states_list, dtype=np.float32),
        'actions': np.array(actions_list, dtype=np.float32),
        'images': np.array(images_list, dtype=np.uint8),
        'goals': np.array(goals_list, dtype=np.float32),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'goal_raw': goal,
    }


def collect_q2_episodes(n_episodes=15, goal_range=0.35, max_steps=250, seed_start=2024):
    """
    Collect episodes specifically targeting Q2 (gx<0, gy>0) with extended gy coverage.
    This fills the gy ∈ [0.24, 0.45]m gap in Phase 196 data.
    """
    controller = Phase196Controller(kP=2.0)

    # Q2 goal configurations:
    # - 5 episodes: gx ∈ [-0.35, -0.15], gy ∈ [0.28, 0.40] (left-deep Q2)
    # - 5 episodes: gx ∈ [-0.20, -0.05], gy ∈ [0.24, 0.38] (front Q2)
    # - 5 episodes: gx ∈ [-0.30, -0.20], gy ∈ [0.20, 0.35] (mid Q2 diagonal)
    np.random.seed(seed_start)

    all_data = []

    for i in range(n_episodes):
        if i < 5:
            # Left-deep Q2: large negative X, large positive Y
            gx = np.random.uniform(-0.35, -0.15)
            gy = np.random.uniform(0.28, 0.40)
        elif i < 10:
            # Front Q2: small negative X, large positive Y
            gx = np.random.uniform(-0.20, -0.05)
            gy = np.random.uniform(0.24, 0.38)
        else:
            # Mid Q2 diagonal: medium negative X, medium-positive Y
            gx = np.random.uniform(-0.30, -0.20)
            gy = np.random.uniform(0.20, 0.35)

        goal = np.array([gx, gy], dtype=np.float32)

        print(f"  Episode {i+1}/{n_episodes}: Q2 goal=({gx:.3f}, {gy:.3f}) |g|={np.linalg.norm(goal):.3f}m")

        sim = LeKiWiSimURDF()
        ep_data = collect_episode(sim, controller, goal, goal_range=goal_range,
                                 max_steps=max_steps, seed=seed_start + i)

        # Check success
        final_dist = np.linalg.norm(ep_data['goal_raw'] - sim.data.xpos[sim.model.body('base').id, :2])
        print(f"    → {len(ep_data['states'])} steps, final_dist={final_dist:.3f}m {'✓ SUCCESS' if final_dist < 0.10 else '✗ FAILED'}")

        all_data.append(ep_data)

    return all_data


def verify_correlations(data_list, name="Q2 Extended"):
    """Verify data has correct correlations."""
    print(f"\n[{name}] Correlation Verification")

    all_actions = np.concatenate([d['actions'][:, 6:9] for d in data_list], axis=0)
    all_goals = np.concatenate([d['goals'] for d in data_list], axis=0)

    wheel_actions = all_actions[:, 0:3]
    w0, w1, w2 = wheel_actions[:, 0], wheel_actions[:, 1], wheel_actions[:, 2]
    gx, gy = all_goals[:, 0], all_goals[:, 1]

    corr_x0 = np.corrcoef(w0, gx)[0, 1]
    corr_x1 = np.corrcoef(w1, gx)[0, 1]
    corr_x2 = np.corrcoef(w2, gx)[0, 1]
    corr_y0 = np.corrcoef(w0, gy)[0, 1]
    corr_y1 = np.corrcoef(w1, gy)[0, 1]
    corr_y2 = np.corrcoef(w2, gy)[0, 1]

    print(f"  Corr(w0, gx) = {corr_x0:+.3f}  (expected <0 for +X goal → w0 neg)")
    print(f"  Corr(w1, gx) = {corr_x1:+.3f}  (expected >0 for +X goal → w1 pos)")
    print(f"  Corr(w2, gx) = {corr_x2:+.3f}  (expected <0 for +X goal → w2 neg)")
    print(f"  Corr(w0, gy) = {corr_y0:+.3f}  (expected >0 for +Y goal → w0 pos)")
    print(f"  Corr(w1, gy) = {corr_y1:+.3f}  (expected >0 for +Y goal → w1 pos)")
    print(f"  Corr(w2, gy) = {corr_y2:+.3f}  (expected >0 for +Y goal → w2 pos)")

    return {
        'corr_x0': corr_x0, 'corr_x1': corr_x1, 'corr_x2': corr_x2,
        'corr_y0': corr_y0, 'corr_y1': corr_y1, 'corr_y2': corr_y2,
    }


def merge_and_save(q2_data, output_path, base_data_path):
    """Merge Q2 extended data with existing Phase 196 data."""
    print(f"\n[Merging Data]")

    # Load base data
    with h5py.File(base_data_path, 'r') as f:
        base_images = f['images'][:]
        base_states = f['states'][:]
        base_actions = f['actions'][:]
        base_goals = f['goals'][:]
        base_rewards = f['rewards'][:]
        base_ep_starts = f['episode_starts'][:]
        base_goal_raw = f['goal_raw'][:] if 'goal_raw' in f else None

    # Concatenate Q2 data
    q2_images = np.concatenate([d['images'] for d in q2_data], axis=0)
    q2_states = np.concatenate([d['states'] for d in q2_data], axis=0)
    q2_actions = np.concatenate([d['actions'] for d in q2_data], axis=0)
    q2_goals = np.concatenate([d['goals'] for d in q2_data], axis=0)
    q2_rewards = np.concatenate([d['rewards'] for d in q2_data], axis=0)
    q2_goal_raw = np.array([d['goal_raw'] for d in q2_data], dtype=np.float32)

    # Update episode_starts
    n_base_eps = len(base_ep_starts)
    q2_ep_starts = base_ep_starts[-1] + np.arange(1, len(q2_data) + 1) * 100  # approximate
    # Recompute exact episode boundaries
    q2_ep_boundaries = [0]
    for d in q2_data:
        q2_ep_boundaries.append(q2_ep_boundaries[-1] + len(d['states']))
    q2_ep_starts = np.array(q2_ep_boundaries[:-1]) + (base_ep_starts[-1] if base_ep_starts.size > 0 else 0)
    new_ep_starts = np.concatenate([base_ep_starts, q2_ep_starts])

    # Concatenate all
    all_images = np.concatenate([base_images, q2_images], axis=0)
    all_states = np.concatenate([base_states, q2_states], axis=0)
    all_actions = np.concatenate([base_actions, q2_actions], axis=0)
    all_goals = np.concatenate([base_goals, q2_goals], axis=0)
    all_rewards = np.concatenate([base_rewards, q2_rewards], axis=0)
    all_goal_raw = np.concatenate([base_goal_raw, q2_goal_raw], axis=0) if base_goal_raw is not None else q2_goal_raw

    print(f"  Base: {len(base_images)} images, {n_base_eps} episodes")
    print(f"  Q2 Extended: {len(q2_images)} images, {len(q2_data)} episodes")
    print(f"  Combined: {len(all_images)} images, {len(new_ep_starts)} episodes")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('images', data=all_images, compression='gzip')
        f.create_dataset('states', data=all_states, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        f.create_dataset('goals', data=all_goals, compression='gzip')
        f.create_dataset('rewards', data=all_rewards, compression='gzip')
        f.create_dataset('episode_starts', data=new_ep_starts)
        f.create_dataset('goal_raw', data=all_goal_raw)

    print(f"  Saved: {output_path}")
    return output_path


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect Q2 Extended Data')
    parser.add_argument('--n_q2_episodes', type=int, default=15,
                       help='Number of Q2 episodes to collect (default: 15)')
    parser.add_argument('--output', type=str, default='data/phase227_extended_65ep.h5',
                       help='Output merged HDF5 path')
    parser.add_argument('--base_data', type=str, default='data/phase196_clean_50ep.h5',
                       help='Base data to merge with (default: phase196_clean_50ep.h5)')
    parser.add_argument('--goal_range', type=float, default=0.35,
                       help='Goal range (default: 0.35)')
    parser.add_argument('--max_steps', type=int, default=250,
                       help='Max steps per episode (default: 250)')
    parser.add_argument('--seed_start', type=int, default=2024,
                       help='Random seed start (default: 2024)')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 227: Collect Q2 Extended Data")
    print("=" * 70)
    print(f"  Q2 episodes to collect: {args.n_q2_episodes}")
    print(f"  Output: {args.output}")
    print(f"  Base data: {args.base_data}")
    print(f"  Goal range: ±{args.goal_range}m")
    print(f"  Max steps: {args.max_steps}")
    print()

    # Verify base data exists
    if not Path(args.base_data).exists():
        print(f"ERROR: Base data not found: {args.base_data}")
        sys.exit(1)

    # Collect Q2 episodes
    print(f"[Collecting {args.n_q2_episodes} Q2 episodes targeting gy ∈ [0.20, 0.40]m]")
    q2_data = collect_q2_episodes(
        n_episodes=args.n_q2_episodes,
        goal_range=args.goal_range,
        max_steps=args.max_steps,
        seed_start=args.seed_start,
    )

    # Verify correlations
    verify_correlations(q2_data, name="Q2 Extended")

    # Merge and save
    merge_and_save(q2_data, args.output, args.base_data)

    # Summary
    total_steps = sum(len(d['states']) for d in q2_data)
    q2_success = sum(1 for d in q2_data if d['rewards'].max() > 0.5)
    print(f"\n[Summary]")
    print(f"  Q2 episodes collected: {len(q2_data)}")
    print(f"  Q2 success rate: {q2_success}/{len(q2_data)} = {q2_success/len(q2_data)*100:.0f}%")
    print(f"  Total Q2 steps: {total_steps}")
    print(f"  Output: {args.output}")
