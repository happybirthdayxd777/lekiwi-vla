#!/usr/bin/env python3
"""
Task-Oriented Reward Evaluation
==============================
Tests whether a policy can complete a task (move to target),
not just minimize a distance metric.

Tasks:
  1. Reach target: move within 0.1m of (0.5, 0.0)
  2. Follow path: visit 3 waypoints in order
  3. Grasp object: arm tip within 0.05m of object position

Success = task completed. Not just "less negative reward".
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from sim_lekiwi import LeKiwiSim


class TaskEvaluator:
    """Task-oriented evaluation for LeKiWi robot."""

    def __init__(self, sim, policy=None, device="cpu"):
        """
        Args:
            sim: LeKiWiSim or LeKiWiSimURDF instance
            policy: Policy with .infer(image, state) method. If None → random baseline.
            device: Device for policy inference (cpu/mps/cuda)
        """
        self.sim = sim
        self.policy = policy
        self.device = device

    def _get_action(self, img):
        """Infer action from policy, or random if no policy."""
        if self.policy is None:
            return np.random.uniform(-1, 1, size=9).astype(np.float32)
        # Handle both PIL Image (LeKiWiSim) and np.ndarray (LeKiWiSimURDF)
        if isinstance(img, np.ndarray):
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img.astype(np.uint8))
        else:
            img_pil = img
        # Normalize image
        img_np = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        # State: arm_pos(6) + wheel_vel(3) — must match the qpos/qvel layout of the sim
        arm_pos = self.sim.data.qpos[0:6]
        wheel_v = self.sim.data.qvel[0:3]
        state_t = torch.from_numpy(np.concatenate([arm_pos, wheel_v])).float().unsqueeze(0).to(self.device)
        action = self.policy.infer(img_t, state_t, num_steps=4)
        return np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)

    def reach_target(self, target=(0.5, 0.0), threshold=0.1, max_steps=200):
        """
        Task: Move the robot base to within `threshold` meters of target.
        Returns: (success: bool, steps_taken: int, final_dist: float)
        """
        self.sim.reset()
        target = np.array(target)

        for step in range(max_steps):
            pos = self.sim.data.qpos[:2]  # x, y of base
            dist = np.linalg.norm(pos - target)

            if dist < threshold:
                return True, step, dist

            # Render and get policy action
            img_pil = self.sim.render()
            action = self._get_action(img_pil)
            self.sim.step(action)

        final_dist = np.linalg.norm(self.sim.data.qpos[:2] - target)
        return False, max_steps, final_dist

    def follow_waypoints(self, waypoints=[(0.3, 0.2), (-0.2, 0.3), (0.0, -0.3)],
                         threshold=0.15, max_steps=600):
        """
        Task: Visit each waypoint in order.
        Returns: (fraction_complete: float, waypoints_visited: int)
        """
        self.sim.reset()
        visited = 0

        for wp_idx, wp in enumerate(waypoints):
            wp_target = np.array(wp)
            for step in range(max_steps // len(waypoints)):
                pos = self.sim.data.qpos[:2]
                dist = np.linalg.norm(pos - wp_target)

                if dist < threshold:
                    visited += 1
                    break

                img_pil = self.sim.render()
                action = self._get_action(img_pil)
                self.sim.step(action)

        return visited / len(waypoints), visited

    def arm_reach(self, target_pos=(0.3, 0.0, 0.15), threshold=0.05, max_steps=200):
        """
        Task: Move arm tip within `threshold` of target position.
        Only arm moves, base stays.
        """
        self.sim.reset()
        target = np.array(target_pos)

        for step in range(max_steps):
            # Arm tip position (approximate - last joint position)
            arm_tip = self.sim.data.qpos[3:6]  # rough proxy
            dist = np.linalg.norm(arm_tip - target[:3])

            if dist < threshold:
                return True, step, dist

            # Arm-only action (wheels = 0)
            action = np.zeros(9, dtype=np.float32)
            action[:6] = self._get_action(self.sim.render())[:6]
            self.sim.step(action)

        final_tip = self.sim.data.qpos[3:6]
        final_dist = np.linalg.norm(final_tip - target[:3])
        return False, max_steps, final_dist


def evaluate_policy(policy=None, device="cpu", episodes=20, task="reach", sim=None):
    """Evaluate a policy on task-oriented metrics."""
    if sim is None:
        sim = LeKiWiSim()
    evaluator = TaskEvaluator(sim, policy=policy, device=device)

    if task == "reach":
        results = [evaluator.reach_target() for _ in range(episodes)]
        successes = sum(1 for r in results if r[0])
        avg_steps = np.mean([r[1] for r in results])
        avg_dist = np.mean([r[2] for r in results])
        print(f"\n=== Reach Target Task ({episodes} episodes) ===")
        print(f"  Success rate: {successes}/{episodes} ({100*successes/episodes:.0f}%)")
        print(f"  Avg steps:     {avg_steps:.1f}")
        print(f"  Avg final dist: {avg_dist:.3f}m")

    elif task == "waypoints":
        results = [evaluator.follow_waypoints() for _ in range(episodes)]
        fractions = [r[0] for r in results]
        print(f"\n=== Waypoint Following ({episodes} episodes) ===")
        print(f"  Avg completion: {np.mean(fractions)*100:.0f}%")
        print(f"  Perfect runs:   {sum(1 for f in fractions if f >= 1.0)}")

    elif task == "arm":
        results = [evaluator.arm_reach() for _ in range(episodes)]
        successes = sum(1 for r in results if r[0])
        print(f"\n=== Arm Reach Task ({episodes} episodes) ===")
        print(f"  Success rate: {successes}/{episodes} ({100*successes/episodes:.0f}%)")
        print(f"  Avg final dist: {np.mean([r[2] for r in results]):.3f}m")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    type=str, default="reach", choices=["reach", "waypoints", "arm"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy",  type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--device",   type=str, default="cpu")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Task-Oriented Evaluation | {args.task}")
    print(f"{'='*60}")

    policy = None
    if args.policy:
        from eval_policy import FlowMatchingPolicy
        policy = FlowMatchingPolicy()
        state_dict = torch.load(args.policy, map_location=args.device, weights_only=True)
        policy.load_state_dict(state_dict)
        policy.to(args.device)
        policy.eval()
        print(f"Loaded policy: {args.policy}")

    evaluate_policy(policy=policy, device=args.device, episodes=args.episodes, task=args.task)

if __name__ == "__main__":
    main()
