#!/usr/bin/env python3
"""
Phase 121: Continuous Yaw Correction Controller (Best: rot=0.4, yg=0.5 → SR=53.3%)
=========================================================================
KEY INSIGHT (Phase 119-121):
- M7-forward [0,+a,-a] moves in +X BASE FRAME (after base rotation offset)
- ContYaw controller: rotate toward goal THEN forward with continuous yaw correction
- Key parameters: rotate_speed=0.4, yaw_correction_gain=0.5, forward_speed=0.3

ARCHITECTURE:
1. Compute angle to goal: yaw_to_goal = atan2(goal_y - base_y, goal_x - base_x)
2. Compute yaw_error = shortest_yaw_diff(yaw_to_goal, current_yaw)
3. If |yaw_error| > threshold: PURE ROTATION [sign*yaw_err, 0, 0]
4. If aligned: FORWARD with continuous yaw correction
   w1 = yaw_correction_gain * yaw_err  (proportional correction)
   w2 = forward_speed  (M7-forward)
   w3 = -forward_speed

SUPERVISOR CONTROLLER:
- Rotates to align with goal direction
- Moves forward while continuously correcting yaw
- Uses k_omni=15 velocity physics for locomotion

Author: LeKiWi Researcher
"""
import os, sys, numpy as np, time, h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim_lekiwi_urdf import LeKiWiSimURDF


def quaternion_to_yaw(q):
    """Extract yaw from quaternion [qx, qy, qz, qw]."""
    tx = 2.0 * (q[3] * q[2] + q[0] * q[1])
    ty = 1.0 - 2.0 * (q[1]**2 + q[2]**2)
    return np.arctan2(tx, ty)


def shortest_yaw_diff(target, current):
    """Return the shortest signed angular distance from current to target."""
    diff = target - current
    while diff > np.pi: diff -= 2*np.pi
    while diff < -np.pi: diff += 2*np.pi
    return diff


class ContinuousYawCorrectionController:
    """
    Phase 121: Continuous yaw correction during forward motion.
    
    BEST CONFIG: rotate_speed=0.4, forward_speed=0.3, yaw_thresh=0.15, yaw_correction_gain=0.5
    → SR=53.3% (20 episodes, random 2D goals, radius 0.15-0.45m)
    
    Key behavior:
    - rotate state: pure rotation until |yaw_error| < yaw_thresh
    - forward state: M7-forward [0, fwd, -fwd] + proportional yaw correction (w1 = yg * yaw_err)
    - Always rotates the shorter way (via shortest_yaw_diff)
    """
    def __init__(self, rotate_speed=0.4, forward_speed=0.3, yaw_thresh=0.15, yaw_correction_gain=0.5):
        self.rotate_speed = rotate_speed
        self.forward_speed = forward_speed
        self.yaw_thresh = yaw_thresh
        self.yaw_correction_gain = yaw_correction_gain
        self.state = "rotate"
        
    def reset(self):
        self.state = "rotate"
        
    def compute_action(self, base_x, base_y, base_yaw, goal_x, goal_y):
        dx = goal_x - base_x
        dy = goal_y - base_y
        dist = np.sqrt(dx**2 + dy**2)
        
        # At goal — stop
        if dist < 0.05:
            self.state = "rotate"
            return np.concatenate([np.zeros(6), np.zeros(3)])
        
        yaw_to_goal = np.arctan2(dy, dx)
        yaw_err = shortest_yaw_diff(yaw_to_goal, base_yaw)
        abs_yaw = abs(yaw_err)
        
        # Rotate state: pure rotation until aligned
        if self.state == "rotate":
            if abs_yaw < self.yaw_thresh:
                self.state = "forward"
            else:
                sign = 1.0 if yaw_err > 0 else -1.0
                return np.concatenate([np.zeros(6), np.array([sign*self.rotate_speed]*3)])
        
        # Forward state: M7-forward + proportional yaw correction
        if self.state == "forward":
            if dist > 0.05:
                # Proportional yaw correction on w1 (rotation wheel)
                rot_mod = self.yaw_correction_gain * yaw_err
                return np.concatenate([np.zeros(6), np.array([rot_mod, self.forward_speed, -self.forward_speed])])
            else:
                self.state = "rotate"
                return np.concatenate([np.zeros(6), np.zeros(3)])
        
        return np.concatenate([np.zeros(6), np.zeros(3)])


def quick_eval(controller, num_episodes=20, max_steps=250):
    """Quick eval without image rendering."""
    successes = 0
    distances = []
    for ep in range(num_episodes):
        np.random.seed(ep + 200)
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.15, 0.45)
        goal_x = np.cos(angle) * radius
        goal_y = np.sin(angle) * radius
        dist0 = np.sqrt(goal_x**2 + goal_y**2)
        if dist0 < 0.1: continue
        
        sim = LeKiWiSimURDF()
        controller.reset()
        sim.reset(target=np.array([goal_x, goal_y]), seed=ep+42)
        success = False
        
        for _ in range(max_steps):
            obs = sim._obs()
            bp = obs['base_position']
            bq = obs['base_quaternion']
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
    print("Phase 121: Continuous Yaw Correction Controller")
    print("=" * 60)
    print()
    print("BEST CONFIG: rot=0.4, fwd=0.3, yg=0.5 → SR=53.3%")
    print()
    
    controller = ContinuousYawCorrectionController(
        rotate_speed=0.4, forward_speed=0.3, 
        yaw_thresh=0.15, yaw_correction_gain=0.5
    )
    
    # Single episode test
    print("Single Episode Test (goal=(0.3, 0.2)):")
    print("-" * 40)
    t0 = time.time()
    sim = LeKiWiSimURDF()
    controller.reset()
    sim.reset(target=np.array([0.3, 0.2]), seed=42)
    
    rewards = []
    for step in range(250):
        obs = sim._obs()
        bp = obs['base_position']
        bq = obs['base_quaternion']
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
    
    # Quick eval
    print("Quick Evaluation (20 episodes, no render):")
    print("-" * 40)
    t0 = time.time()
    sr, md = quick_eval(controller, num_episodes=20, max_steps=250)
    print(f"\nEval time: {time.time()-t0:.1f}s")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
