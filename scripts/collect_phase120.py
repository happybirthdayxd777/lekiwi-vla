#!/usr/bin/env python3
"""
Phase 120: Rotate+Forward 2D Controller
======================================
KEY INSIGHT (Phase 119): 
- M7-forward [0,+a,-a] moves in +X BASE FRAME (not world frame)
- After rotating base by yaw, M7-forward pushes in different world direction
- For 2D goals: ROTATE so M7-forward points at goal, then FORWARD

APPROACH:
1. Compute angle to goal: yaw_to_goal = atan2(goal_y - base_y, goal_x - base_x)
2. Compute yaw_error = yaw_to_goal - current_yaw  (how far to rotate)
3. ROTATE until |yaw_error| < threshold
4. FORWARD using M7-forward [0,+a,-a] in +X direction

Author: LeKiWi Researcher
"""
import os, sys, numpy as np, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_lekiwi_urdf import LeKiWiSimURDF


def quaternion_to_yaw(q):
    """Extract yaw from quaternion [qx, qy, qz, qw]."""
    tx = 2.0 * (q[3] * q[2] + q[0] * q[1])
    ty = 1.0 - 2.0 * (q[1]**2 + q[2]**2)
    return np.arctan2(tx, ty)


class RotateForwardController:
    """Rotate+Forward 2D Controller. Phase 119 insight: M7-forward is base-frame."""
    def __init__(self, rotate_speed=0.3, forward_speed=0.3, yaw_threshold=0.15):
        self.rotate_speed = rotate_speed
        self.forward_speed = forward_speed
        self.yaw_threshold = yaw_threshold
        self.state = "rotate"
        
    def reset(self):
        self.state = "rotate"
        
    def compute_action(self, base_x, base_y, base_yaw, goal_x, goal_y):
        dx = goal_x - base_x; dy = goal_y - base_y; dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.02:
            self.state = "rotate"
            return np.concatenate([np.zeros(6, dtype=np.float32), np.zeros(3, dtype=np.float32)])
        yaw_to_goal = np.arctan2(dy, dx)
        yaw_error = yaw_to_goal - base_yaw
        while yaw_error > np.pi: yaw_error -= 2*np.pi
        while yaw_error < -np.pi: yaw_error += 2*np.pi
        if self.state == "rotate":
            if abs(yaw_error) < self.yaw_threshold:
                self.state = "forward"
            else:
                sign = 1.0 if yaw_error > 0 else -1.0
                return np.concatenate([np.zeros(6), np.array([sign*self.rotate_speed]*3)])
        if dist > 0.05:
            return np.concatenate([np.zeros(6), np.array([0.0, self.forward_speed, -self.forward_speed])])
        else:
            self.state = "rotate"
            return np.concatenate([np.zeros(6), np.zeros(3)])


def quick_eval(controller, num_episodes=10, max_steps=200):
    """Quick eval without image rendering."""
    successes = 0; distances = []
    for ep in range(num_episodes):
        np.random.seed(ep + 100)
        goal_x = np.random.uniform(-0.4, 0.4)
        goal_y = np.random.uniform(-0.4, 0.4)
        dist0 = np.sqrt(goal_x**2 + goal_y**2)
        if dist0 < 0.1: continue
        
        sim = LeKiWiSimURDF()
        controller.reset()
        sim.reset(target=np.array([goal_x, goal_y]), seed=ep+42)
        success = False
        
        for _ in range(max_steps):
            obs = sim._obs()
            bp = obs['base_position']; bq = obs['base_quaternion']
            action = controller.compute_action(bp[0], bp[1], quaternion_to_yaw(bq), goal_x, goal_y)
            obs, reward, done, info = sim.step(action)
            if reward > 0.9: success = True
            if done: break
        
        fp = sim._obs()['base_position']
        fd = np.sqrt((goal_x-fp[0])**2 + (goal_y-fp[1])**2)
        distances.append(fd)
        if success: successes += 1
        print(f"  Ep {ep+1}: goal=({goal_x:.3f},{goal_y:.3f}), dist0={dist0:.3f}, "
              f"final=({fp[0]:.3f},{fp[1]:.3f}), fd={fd:.3f}, ok={success}")
    
    sr = successes / num_episodes * 100 if num_episodes > 0 else 0
    md = np.mean(distances) if distances else 999
    print(f"\nSR: {successes}/{num_episodes} = {sr:.1f}%")
    print(f"Mean final dist: {md:.4f}m")
    return sr, md


def main():
    print("=" * 60)
    print("Phase 120: Rotate+Forward 2D Controller")
    print("=" * 60)
    print()
    print("KEY INSIGHT (Phase 119): M7-forward is BASE FRAME")
    print("- rotate_speed=0.3 ([a,a,a] rotation)")
    print("- forward_speed=0.3 (M7-forward [0,+a,-a])")
    print("- yaw_threshold=0.15 rad (~8.6 deg)")
    print()
    
    controller = RotateForwardController(rotate_speed=0.3, forward_speed=0.3, yaw_threshold=0.15)
    
    # Single episode test (no render for speed)
    print("Single Episode Test:")
    print("-" * 40)
    t0 = time.time()
    sim = LeKiWiSimURDF()
    controller.reset()
    sim.reset(target=np.array([0.3, 0.2]), seed=42)
    
    rewards = []
    for step in range(200):
        obs = sim._obs()
        bp = obs['base_position']; bq = obs['base_quaternion']
        action = controller.compute_action(bp[0], bp[1], quaternion_to_yaw(bq), 0.3, 0.2)
        obs, reward, done, info = sim.step(action)
        rewards.append(reward)
        if done: break
    
    elapsed = time.time() - t0
    fp = sim._obs()['base_position']
    fd = np.sqrt((0.3-fp[0])**2 + (0.2-fp[1])**2)
    max_r = max(rewards) if rewards else 0
    
    print(f"Goal: (0.3, 0.2)")
    print(f"Final: ({fp[0]:.3f}, {fp[1]:.3f})")
    print(f"Final dist: {fd:.4f}m")
    print(f"Max reward: {max_r:.4f}")
    print(f"Steps: {len(rewards)}, time: {elapsed:.1f}s")
    print()
    
    # Quick eval: 10 episodes without rendering
    print("Quick Evaluation (10 episodes, no render):")
    print("-" * 40)
    t0 = time.time()
    sr, md = quick_eval(controller, num_episodes=10, max_steps=200)
    print(f"\nEval time: {time.time()-t0:.1f}s")
    print()
    print("=" * 60)
    print(f"RESULT: SR={sr:.1f}%, MeanDist={md:.4f}m")
    print("=" * 60)


if __name__ == "__main__":
    main()