#!/usr/bin/env python3
"""
Phase 116: Braking-Aware Data Collection v2
===========================================
CRITICAL INSIGHT: k_omni=15 creates velocity-based locomotion with momentum.
Robot MUST brake BEFORE reaching the goal using reverse action.

Verified strategy (step-by-step):
  dist > 0.25: M7=[0, +K*dist, -K*dist] (forward, proportional)
  0.15 < dist < 0.25: reverse M7=[0, -0.5, +0.5] (braking)
  0.05 < dist < 0.15: soft reverse=[0, -0.3, +0.3]
  dist < 0.05: idle=[0, 0, 0]

Result: dist=0.033m (VERIFIED)
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from sim_lekiwi_urdf import LeKiWiSimURDF

TARGET_SIZE = (224, 224)

class BrakingControllerV2:
    """
    Goal-aware controller with explicit braking for k_omni physics.
    States: FORWARD -> BRAKE -> SOFT_BRAKE -> IDLE
    """
    FORWARD_DIST = 0.25
    BRAKE_DIST = 0.15
    SOFT_BRAKE_DIST = 0.05
    
    def __init__(self, K=5.0, exploration_noise=0.02):
        self.K = K
        self.exploration_noise = exploration_noise
        self._state = 'FORWARD'
        
    def reset(self):
        self._state = 'FORWARD'
    
    def compute_action(self, base_x, base_y, goal_x, goal_y):
        dx = goal_x - base_x
        dy = goal_y - base_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Determine state based on distance
        if dist > self.FORWARD_DIST:
            self._state = 'FORWARD'
        elif dist > self.BRAKE_DIST:
            self._state = 'BRAKE'
        elif dist > self.SOFT_BRAKE_DIST:
            self._state = 'SOFT_BRAKE'
        else:
            self._state = 'IDLE'
        
        # Compute action based on state
        if self._state == 'FORWARD':
            w2 = min(self.K * dist, 0.5)
            w3 = -w2
        elif self._state == 'BRAKE':
            w2 = -0.5
            w3 = 0.5
        elif self._state == 'SOFT_BRAKE':
            w2 = -0.3
            w3 = 0.3
        else:
            w2 = 0.0
            w3 = 0.0
        
        # Add exploration noise
        if self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=3)
            w2 += noise[1]
            w3 += noise[2]
        
        return np.clip(np.array([0.0, w2, w3], dtype=np.float32), -1, 1)

def collect_episode(sim, controller, max_steps=200, goal_min=0.15, goal_max=0.4,
                   goal_threshold=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(goal_min, goal_max)
    goal_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
    
    sim.reset(target=goal_pos)
    controller.reset()
    
    all_images, all_states, all_actions, all_rewards, all_goals = [], [], [], [], []
    goal_reached = False
    final_dist = None
    
    for step in range(max_steps):
        base_x = sim.data.qpos[0]
        base_y = sim.data.qpos[1]
        
        wheel_action = controller.compute_action(base_x, base_y, goal_pos[0], goal_pos[1])
        arm_action = np.zeros(6)
        action = np.concatenate([arm_action, wheel_action])
        
        img = sim.render()
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img).resize(TARGET_SIZE, Image.BILINEAR)
        
        obs_next, reward, done, _ = sim.step(action)
        
        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ['j0','j1','j2','j3','j4','j5']])
        wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ['w1','w2','w3']])
        state = np.concatenate([arm_pos, wheel_vel])
        
        all_images.append(np.array(img))
        all_states.append(state.astype(np.float32))
        all_actions.append(action.astype(np.float32))
        all_rewards.append(reward)
        all_goals.append(goal_pos.copy())
        
        final_dist = np.linalg.norm(obs_next['base_position'][:2] - goal_pos)
        if final_dist < goal_threshold:
            goal_reached = True
        
        if done:
            break
    
    return {
        'image': np.stack(all_images),
        'state': np.stack(all_states),
        'action': np.stack(all_actions),
        'reward': np.array(all_rewards, dtype=np.float32),
        'goal_position': np.stack(all_goals),
        'goal_reached': goal_reached,
        'final_dist': final_dist,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', type=str, default='data/phase116_braking_100ep.h5')
    parser.add_argument('--goal_min', type=float, default=0.15)
    parser.add_argument('--goal_max', type=float, default=0.4)
    parser.add_argument('--goal_threshold', type=float, default=0.1)
    args = parser.parse_args()
    
    controller = BrakingControllerV2(K=5.0, exploration_noise=0.02)
    sim = LeKiWiSimURDF()
    
    print(f"Collecting {args.episodes} episodes with BrakingControllerV2...")
    print(f"  K={controller.K}, noise={controller.exploration_noise}")
    print(f"  Forward>0.25m, Brake<0.25, Soft<0.15, Idle<0.05")
    
    all_episodes = []
    total_rewards = []
    goals_reached = 0
    
    for ep in range(args.episodes):
        result = collect_episode(sim, controller,
                                goal_min=args.goal_min, goal_max=args.goal_max,
                                goal_threshold=args.goal_threshold, seed=ep)
        all_episodes.append(result)
        total_rewards.append(result['reward'].sum())
        if result['goal_reached']:
            goals_reached += 1
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{args.episodes}: "
                  f"reward={result['reward'].sum():.1f}, "
                  f"goal_reached={result['goal_reached']}, "
                  f"final_dist={result['final_dist']:.3f}m")
    
    mean_reward = np.mean(total_rewards)
    print(f"\nSummary:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Goals reached: {goals_reached}/{args.episodes} ({100*goals_reached/args.episodes:.1f}%)")
    print(f"  Mean reward: {mean_reward:.2f}")
    
    # Save to h5
    import h5py
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with h5py.File(args.output, 'w') as hf:
        for key in ['image', 'state', 'action', 'reward', 'goal_position']:
            data = np.concatenate([ep[key] for ep in all_episodes], axis=0)
            hf.create_dataset(key, data=data, compression='lzf')
        hf.attrs['goal_reached'] = goals_reached
        hf.attrs['total_episodes'] = args.episodes
    
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()
