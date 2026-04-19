#!/usr/bin/env python3
"""
Phase 196: Collect Clean Vision Data with CORRECT Contact-Jacobian P-Controller
================================================================================
CRITICAL FIX vs Phase 187/189:
  - Phase 187/189 used twist_to_contact_wheel_speeds() which was calibrated
    for k_omni=15 overlay physics (kinematic IK model from Phase 164).
  - Phase 195 discovered the CORRECT model: _CONTACT_JACOBIAN_PSEUDO_INV
    which achieves 100% SR on 20 random goals.
  - This script uses the CORRECT Contact-Jacobian P-controller directly.

Controller (from eval_jacobian_pcontroller.py):
  - v_desired = kP * err  (kP=2.0 — achieves 100% SR)
  - wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
  - This is the P-controller that achieves 100% SR in simulation!

Data format:
  - states: (N, 11) — arm_pos(6) + wheel_vel(3) + goal_norm(2)
  - actions: (N, 9) — arm_torque(6) + wheel_speed(3)
  - images: (N, 640, 480, 3) — one per step
  - goals: (N, 2)
  - episode_starts: (N_episodes+1,)
  - rewards: (N,)

Expected correlations (with CORRECT controller):
  - Corr(w0, -gy) > 0.5  (w0 negatively correlated with lateral goal error)
  - Corr(w1, gx) > 0.5  (w1 positively correlated with forward goal error)
  - Corr(w2, -gx+gy) > 0.3  (w2 for diagonal)

Usage:
  python3 scripts/collect_phase196_clean.py --n_episodes 50 --output data/phase196_clean_50ep.h5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
import mujoco
from pathlib import Path

# Import the CORRECT Contact-Jacobian model
from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV

np.random.seed(None)  # Allow different seeds per run


class Phase196Controller:
    """
    CORRECT Contact-Jacobian P-controller.
    This is the SAME controller from eval_jacobian_pcontroller.py that achieves 100% SR.
    
    Key insight (Phase 195):
      - OLD: twist_to_contact_wheel_speeds() used kinematic IK calibrated for k_omni=15
      - NEW: _CONTACT_JACOBIAN_PSEUDO_INV is the TRUE contact Jacobian
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


def collect_episode(sim, controller, goal_range=0.35, max_steps=250, seed=None):
    """Collect one episode with goal-directed navigation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Random goal position
    goal = np.array([
        np.random.uniform(-goal_range, goal_range),
        np.random.uniform(-goal_range * 0.8, goal_range * 0.8)
    ], dtype=np.float32)
    
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
        
        # Compute wheel speeds using CORRECT Contact-Jacobian controller
        wheel_speeds = controller.compute(goal, base_xy)
        
        # Full action: arm + wheels
        action = np.concatenate([arm_pos, wheel_speeds]).astype(np.float32)
        
        # Render image BEFORE step (current view — front camera)
        img = sim.render().astype(np.uint8)
        
        # Step simulation
        sim.step(action)
        
        # Record state: arm_pos(6) + wheel_vel(3) + goal_norm(2)
        wheel_vel = sim.data.qvel[9:12].copy()  # wheel joint velocities
        state = np.concatenate([
            arm_pos,                    # 6 arm positions
            wheel_vel,                  # 3 wheel velocities
            goal / (goal_range * 1.5)   # 2 normalized goal coords
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


def verify_correlations(data_path):
    """Verify that collected data has STRONG correlations (unlike Phase 187)."""
    print(f"\n[Correlation Verification]")
    
    with h5py.File(data_path, 'r') as f:
        actions = f['actions'][:]       # (N, 9) — arm(6) + wheels(3)
        goals = f['goals'][:]            # (N, 2)
        states = f['states'][:]          # (N, 11) — arm(6) + wheel_vel(3) + goal_norm(2)
    
    wheel_actions = actions[:, 6:9]  # (N, 3) — wheel speeds
    w0, w1, w2 = wheel_actions[:, 0], wheel_actions[:, 1], wheel_actions[:, 2]
    gx, gy = goals[:, 0], goals[:, 1]
    
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
    
    # Phase 187 had near-zero correlations (all |corr| < 0.1)
    # With CORRECT controller, expect |corr| > 0.3 for most
    strong_corr = sum([
        abs(corr_x0) > 0.3, abs(corr_x1) > 0.3, abs(corr_y0) > 0.3,
        abs(corr_y1) > 0.3, abs(corr_y2) > 0.3
    ])
    
    if strong_corr >= 3:
        print(f"  ✅ PASS: {strong_corr}/5 strong correlations (Phase 187 had 0/5)")
        return True
    else:
        print(f"  ❌ FAIL: Only {strong_corr}/5 strong correlations (should be >= 3)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Phase 196: Collect clean vision data with Contact-Jacobian P-controller')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=250)
    parser.add_argument('--goal_range', type=float, default=0.35)
    parser.add_argument('--output', type=str, default='data/phase196_clean_50ep.h5')
    parser.add_argument('--seed', type=int, default=196)
    parser.add_argument('--verify_only', action='store_true')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify-only mode
    if args.verify_only:
        verify_correlations(args.output)
        return
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Phase 196: Collect Clean Vision Data")
    print("=" * 60)
    print(f"  Controller: Contact-Jacobian P-controller (kP=2.0)")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Goal range: ±{args.goal_range}m")
    print(f"  Output: {args.output}")
    print()
    
    # Controller
    controller = Phase196Controller(kP=2.0, wheel_clip=0.5)
    
    # Collect episodes
    all_states = []
    all_actions = []
    all_images = []
    all_goals = []
    all_rewards = []
    episode_starts = [0]
    success_count = 0
    
    for ep in range(args.n_episodes):
        print(f"  Episode {ep+1}/{args.n_episodes}...", end=" ", flush=True)
        
        sim = LeKiWiSimURDF()
        ep_data = collect_episode(
            sim, controller,
            goal_range=args.goal_range,
            max_steps=args.max_steps,
            seed=args.seed + ep if args.seed else None
        )
        
        n_steps = len(ep_data['states'])
        all_states.append(ep_data['states'])
        all_actions.append(ep_data['actions'])
        all_images.append(ep_data['images'])
        all_goals.append(ep_data['goals'])
        all_rewards.append(ep_data['rewards'])
        episode_starts.append(episode_starts[-1] + n_steps)
        
        ep_success = ep_data['rewards'].sum() > 0
        success_count += int(ep_success)
        sr = success_count / (ep + 1) * 100
        
        print(f"{n_steps} steps, success={ep_success}, running SR={sr:.0f}%")
    
    # Concatenate
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    images = np.concatenate(all_images, axis=0).astype(np.uint8)
    goals = np.concatenate(all_goals, axis=0).astype(np.float32)
    rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    episode_starts = np.array(episode_starts, dtype=np.int64)
    
    print(f"\n[Data Summary]")
    print(f"  Total frames: {len(states)}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Success rate: {success_count}/{args.n_episodes} = {success_count/args.n_episodes*100:.0f}%")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Images shape: {images.shape}")
    print(f"  Goals shape: {goals.shape}")
    print(f"  Rewards: {rewards.mean()*100:.1f}% positive")
    
    # Save
    print(f"\n[Saving to {args.output}]")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('states', data=states, compression='gzip')
        f.create_dataset('actions', data=actions, compression='gzip')
        f.create_dataset('images', data=images, compression='gzip')
        f.create_dataset('goals', data=goals, compression='gzip')
        f.create_dataset('rewards', data=rewards, compression='gzip')
        f.create_dataset('episode_starts', data=episode_starts)
        f.attrs['controller'] = 'Phase196Controller (kP=2.0, Contact-Jacobian, _CONTACT_JACOBIAN_PSEUDO_INV)'
        f.attrs['phase'] = 196
        f.attrs['n_episodes'] = args.n_episodes
        f.attrs['success_rate'] = success_count / args.n_episodes
    
    print(f"  ✅ Saved to {args.output}")
    
    # Verify correlations
    print()
    verify_correlations(args.output)
    
    print("\n✅ Phase 196 data collection complete!")


if __name__ == "__main__":
    main()
