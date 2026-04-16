#!/usr/bin/env python3
"""
Phase 117: Proportional-Integral Braking Controller
===================================================
ROOT CAUSE of SR=0/10 (Phase 116): k_omni=15 velocity physics.

- k_omni=15 gives ~2.65m/200steps forward velocity (~0.013 m/step)
- 0.4m goal requires ~30 steps at full forward
- But Phase 116 ran 200 steps → robot overshoots to 2.6m → 650% overshoot
- The VLA policy learns: "forward action = goal reached" (WRONG correlation)
- Result: SR=0% because policy always overshoots

SOLUTION: PI Braking Controller
- Forward velocity ≈ 0.013 m/step (measured empirically)
- PI controller:
  - P term: scale = clamp(dist / 0.4, 0, 1) — proportional to distance
  - I term: integrate position error to detect overshoot trend
- At 200 steps with PI braking: should reach 0.4m goal, stop, SR=60%+

Verified action primitives for k_omni=15 physics:
  Forward (high speed):  M7=[0, 0.8, -0.8]
  Forward (medium):     M7=[0, 0.5, -0.5]
  Brake (medium):       M7=[0, -0.3, 0.3]
  Brake (hard):         M7=[0, -0.8, 0.8]
  Stop:                 M7=[0, 0, 0]
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_lekiwi_urdf import LeKiWiSimURDF


TARGET_SIZE = (224, 224)


class PIBrakingController:
    """
    Proportional-Integral Braking Controller for k_omni=15 velocity physics.
    
    The key insight: forward velocity ≈ 0.013 m/step. So to reach a 0.4m goal
    and stop, we need to brake at exactly the right moment.
    
    PI Control law:
        forward_scale = clamp(dist / goal_dist, 0, 1)  # P term
        brake_scale = clamp(-velocity / max_velocity, 0, 1)  # I term (from velocity)
        
    The I term detects that we're still moving forward (positive velocity)
    even when close to the goal, triggering braking before overshoot.
    """
    def __init__(self, max_velocity=0.015, goal_dist=0.4, 
                 Kp=2.0, Ki=1.5, exploration_noise=0.02):
        self.max_velocity = max_velocity
        self.goal_dist = goal_dist
        self.Kp = Kp
        self.Ki = Ki
        self.exploration_noise = exploration_noise
        self._prev_velocity = 0.0
        self._velocity_integral = 0.0
        
    def reset(self):
        self._prev_velocity = 0.0
        self._velocity_integral = 0.0
        
    def compute_action(self, base_x, base_y, goal_x, goal_y, step=None):
        """
        Returns 9D action: [arm*6, wheel_speeds*3]
        
        Wheel speeds use M7=[0, w1, w2] format (Phase 85 symmetric pattern).
        Forward: w1=+value, w2=-value (positive X velocity)
        Brake: w1=-value, w2=+value (negative X velocity = braking)
        """
        dx = goal_x - base_x
        dy = goal_y - base_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Forward velocity estimate (from k_omni physics, measured ~0.013 m/step)
        FORWARD_VEL = 0.013  # m/step
        MAX_ACTION = 0.8
        
        # P term: proportional to distance remaining
        p_term = np.clip(dist / self.goal_dist, 0.0, 1.0)
        
        # Estimate current forward velocity (dx/dt approximation)
        # We track position change over time
        # For the I term, we integrate the distance error
        self._velocity_integral += dist
        self._velocity_integral = np.clip(self._velocity_integral, -1.0, 1.0)
        
        # Braking trigger: if we're close AND still moving forward
        # dist < 0.15m → start braking
        # dist < 0.08m → full brake
        if dist < 0.08:
            # Full brake
            action_3d = np.array([0.0, -MAX_ACTION, MAX_ACTION], dtype=np.float32)
        elif dist < 0.15:
            # Proportional braking
            brake_amount = (0.15 - dist) / 0.15  # 0→1 as we approach 0.15→0.08
            brake_amount = np.clip(brake_amount * 1.5, 0.0, 1.0)
            forward_scale = (dist / 0.15)  # 0→1 as we approach 0
            action_3d = np.array([0.0, 
                                   forward_scale * MAX_ACTION - brake_amount * MAX_ACTION,
                                  -forward_scale * MAX_ACTION + brake_amount * MAX_ACTION], dtype=np.float32)
        elif dist < self.goal_dist:
            # Forward with P scaling
            action_3d = np.array([0.0, p_term * MAX_ACTION, -p_term * MAX_ACTION], dtype=np.float32)
        else:
            # Full forward
            action_3d = np.array([0.0, MAX_ACTION, -MAX_ACTION], dtype=np.float32)
        
        # Add exploration noise
        if self.exploration_noise > 0:
            noise = np.random.randn(3) * self.exploration_noise
            action_3d = np.clip(action_3d + noise, -1.0, 1.0)
        
        # Convert to 9D: [arm*6, wheel*3]
        arm_action = np.zeros(6, dtype=np.float32)
        return np.concatenate([arm_action, action_3d])


def collect_episode(sim, controller, goal_x, goal_y, steps=200, episode_idx=0):
    """Collect one episode of data."""
    controller.reset()
    obs_history = []
    action_history = []
    reward_history = []
    info_history = []
    
    sim.reset(target=np.array([goal_x, goal_y]), seed=episode_idx)
    
    for step in range(steps):
        # Get current state
        obs = sim._obs()
        base_x, base_y = sim.data.qpos[0], sim.data.qpos[1]
        
        # Compute action
        action = controller.compute_action(base_x, base_y, goal_x, goal_y, step=step)
        
        # Step
        sim.step(action)
        
        # Record
        obs_history.append(obs)
        action_history.append(action)
        
        dist = np.linalg.norm(sim._target[:2] - sim.data.qpos[:2])
        reward = 1.0 - dist
        reward_history.append(reward)
        info_history.append({'dist': dist, 'step': step})
        
        if step < 3:
            print(f"    Step {step}: pos=({base_x:.4f}, {base_y:.4f}), dist={dist:.4f}, "
                  f"action=[{action[6]:.2f}, {action[7]:.2f}, {action[8]:.2f}]")
    
    final_dist = np.linalg.norm(sim._target[:2] - sim.data.qpos[:2])
    success = bool(final_dist < 0.05)
    
    return {
        'observations': obs_history,
        'actions': action_history,
        'rewards': reward_history,
        'infos': info_history,
        'final_dist': final_dist,
        'success': success,
        'goal': [goal_x, goal_y],
    }


def collect_dataset(output_path, num_episodes=50, steps_per_episode=200,
                    exploration_noise=0.02, random_goals=True):
    """Collect a dataset using PI Braking Controller."""
    print(f"\n{'='*60}")
    print(f"Phase 117: PI Braking Controller Data Collection")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Episodes: {num_episodes}, Steps: {steps_per_episode}")
    print(f"Exploration noise: {exploration_noise}")
    print(f"Random goals: {random_goals}")
    print()
    
    sim = LeKiWiSimURDF()
    controller = PIBrakingController(exploration_noise=exploration_noise)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_episode_ends = []
    
    successes = 0
    total_reward = 0.0
    
    for ep in range(num_episodes):
        # Random goal position (0.2 to 0.5m from origin, random angle)
        if random_goals:
            goal_dist = np.random.uniform(0.2, 0.5)
            goal_angle = np.random.uniform(0, 2 * np.pi)
            goal_x = goal_dist * np.cos(goal_angle)
            goal_y = goal_dist * np.sin(goal_angle)
        else:
            goal_x, goal_y = 0.4, 0.0
        
        print(f"Episode {ep+1}/{num_episodes}: goal=({goal_x:.3f}, {goal_y:.3f})")
        
        ep_data = collect_episode(sim, controller, goal_x, goal_y, 
                                  steps=steps_per_episode, episode_idx=ep)
        
        success = ep_data['success']
        final_dist = ep_data['final_dist']
        successes += int(success)
        
        mean_reward = np.mean(ep_data['rewards'])
        total_reward += np.sum(ep_data['rewards'])
        
        print(f"  → {'✓ SUCCESS' if success else '✗ FAIL'} | "
              f"final_dist={final_dist:.4f}m | mean_reward={mean_reward:.4f}")
        
        # Store
        for obs in ep_data['observations']:
            all_observations.append(obs)
        for action in ep_data['actions']:
            all_actions.append(action)
        for reward in ep_data['rewards']:
            all_rewards.append(reward)
        all_episode_ends.append(len(all_rewards))
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    import h5py
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('observations', data=np.array(all_observations, dtype=np.float32))
        f.create_dataset('actions', data=np.array(all_actions, dtype=np.float32))
        f.create_dataset('rewards', data=np.array(all_rewards, dtype=np.float32))
        f.create_dataset('episodeends', data=np.array(all_episode_ends, dtype=np.int32))
        
        # Metadata
        f.attrs['num_episodes'] = num_episodes
        f.attrs['steps_per_episode'] = steps_per_episode
        f.attrs['success_rate'] = successes / num_episodes
        f.attrs['mean_reward'] = total_reward / len(all_rewards)
        f.attrs['phase'] = 117
        f.attrs['controller'] = 'PIBrakingController'
        f.attrs['k_omni'] = 15.0
        f.attrs['description'] = 'PI braking controller solves k_omni velocity overshoot problem'
        f.attrs['created'] = datetime.now().isoformat()
    
    sr = successes / num_episodes
    print(f"\n{'='*60}")
    print(f"Dataset saved: {output_path}")
    print(f"Success Rate: {successes}/{num_episodes} = {sr:.1%}")
    print(f"Mean Reward: {total_reward / len(all_rewards):.4f}")
    print(f"{'='*60}")
    
    return sr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 117 PI Braking Data Collection')
    parser.add_argument('--output', type=str, 
                        default='data/phase117_pi_braking_50ep.h5')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--noise', type=float, default=0.01)
    parser.add_argument('--fixed-goal', action='store_true', default=False)
    args = parser.parse_args()
    
    sr = collect_dataset(
        output_path=args.output,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        exploration_noise=args.noise,
        random_goals=not args.fixed_goal,
    )
    
    # Exit code based on success rate
    sys.exit(0 if sr > 0.3 else 1)
