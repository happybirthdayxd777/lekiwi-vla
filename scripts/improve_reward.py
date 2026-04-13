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

CRITICAL FIX (2026-04-13): reach_target was resetting robot AT the goal
and testing if it STAYS there. This tests station-keeping, not goal-seeking.
Fixed: reset at start position, measure if robot REACHES the target.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from sim_lekiwi import LeKiwiSim
from sim_lekiwi_urdf import LeKiWiSimURDF


class TaskEvaluator:
    """Task-oriented evaluation for LeKiWi robot.

    Phase 16 Goal-Aware:
      Policy state_dim=11 → [arm_pos(6), wheel_vel(3), goal_xy(2)]
      If self.goal_pos is set, _get_action embeds goal_xy into state.
      Otherwise falls back to 9D state for legacy policy compatibility.
    """

    def __init__(self, sim, policy=None, device="cpu", goal_pos=None):
        """
        Args:
            sim: LeKiWiSim or LeKiWiSimURDF instance
            policy: Policy with .infer(image, state) method. If None → random baseline.
            device: Device for policy inference (cpu/mps/cuda)
            goal_pos: (x, y) goal in meters. If set, appended to state as goal_xy(2).
        """
        self.sim = sim
        self.policy = policy
        self.device = device
        self.goal_pos = goal_pos  # Set by reach_target() before evaluation loop

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
        # State: arm_pos(6) + wheel_vel(3) — MUST match the sim's joint layout.
        #
        # LeKiWiSim (primitive):     qpos[0:6]=arm_joints, qpos[6:9]=wheel_pos, qvel same
        # LeKiWiSimURDF (STL mesh): qpos[0:7]=base_free(xyz+quat), qpos[7:13]=arm_joints,
        #                            qvel[0:6]=base_linang, qvel[6:9]=arm_vel, qvel[9:12]=wheel_vel
        from sim_lekiwi_urdf import LeKiWiSimURDF
        if isinstance(self.sim, LeKiWiSimURDF):
            arm_pos = np.array([self.sim.data.qpos[self.sim._jpos_idx[n]]
                                for n in ['j0','j1','j2','j3','j4','j5']])
            wheel_v = np.array([self.sim.data.qvel[self.sim._jvel_idx[n]]
                                for n in ['w1','w2','w3']])
        else:
            arm_pos = self.sim.data.qpos[0:6]
            wheel_v = self.sim.data.qvel[6:9]

        # Phase 16: append goal_xy if goal_pos is set (goal-aware policy)
        if self.goal_pos is not None:
            goal_norm = np.clip(np.array(self.goal_pos) / 1.0, -1.0, 1.0).astype(np.float32)
            state_t = torch.from_numpy(
                np.concatenate([arm_pos, wheel_v, goal_norm])
            ).float().unsqueeze(0).to(self.device)
        else:
            state_t = torch.from_numpy(
                np.concatenate([arm_pos, wheel_v])
            ).float().unsqueeze(0).to(self.device)

        action = self.policy.infer(img_t, state_t, num_steps=4)
        return np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)

    def reach_target(self, target=(0.5, 0.0), start=(0.0, 0.0), threshold=0.1, max_steps=300):
        """
        Task: Move the robot base FROM start position TO within `threshold` meters of target.
        
        CRITICAL FIX: Robot now resets at START position (not the goal), then we test
        if it successfully navigates TO the target.
        
        Args:
            target: Goal position (x, y) to reach
            start: Starting position (x, y) for the robot (default: origin)
            threshold: Success radius in meters
            max_steps: Maximum steps before giving up
            
        Returns: (success: bool, steps_taken: int, final_dist: float)
        """
        target_arr = np.array(target)
        start_arr  = np.array(start)

        # ── Phase 16: Set goal_pos so _get_action passes 11D state to policy ────
        self.goal_pos = (float(target_arr[0]), float(target_arr[1]))

        # Reset sim first, then set target marker at goal
        self.sim.reset()  # Reset at default (origin) first
        if hasattr(self.sim, 'set_target'):
            self.sim.set_target(target_arr)  # Show marker at goal position

        # Move robot to START position (if sim supports it)
        # LeKiWiSimURDF uses qpos[:2] = base x, y
        if hasattr(self.sim, '_jpos_idx'):
            # URDF sim: set base x, y
            base_joint_id = self.sim._jpos_idx.get('j0', 0)
            # For URDF, qpos[0:2] are x, y of base (via freejoint)
            self.sim.data.qpos[0] = start_arr[0]
            self.sim.data.qpos[1] = start_arr[1]
        else:
            # Primitive sim: qpos[0:2] are x, y
            self.sim.data.qpos[0] = start_arr[0]
            self.sim.data.qpos[1] = start_arr[1]

        for step in range(max_steps):
            pos = self.sim.data.qpos[:2]  # x, y of base
            dist = np.linalg.norm(pos - target_arr)

            if dist < threshold:
                return True, step, dist

            # Render and get policy action
            img_pil = self.sim.render()
            action = self._get_action(img_pil)
            self.sim.step(action)

        final_dist = np.linalg.norm(self.sim.data.qpos[:2] - target_arr)
        return False, max_steps, final_dist

    def follow_waypoints(self, waypoints=[(0.3, 0.2), (-0.2, 0.3), (0.0, -0.3)],
                         threshold=0.15, max_steps=600):
        """
        Task: Visit each waypoint in order.
        Returns: (fraction_complete: float, waypoints_visited: int)
        """
        self.sim.reset()
        visited = 0
        # Set initial target to first waypoint
        if waypoints:
            self.sim.set_target(np.array(waypoints[0]))

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

            # Arm only action (wheels = 0)
            action = np.zeros(9, dtype=np.float32)
            action[:6] = self._get_action(self.sim.render())[:6]
            self.sim.step(action)

        final_tip = self.sim.data.qpos[3:6]
        final_dist = np.linalg.norm(final_tip - target[:3])
        return False, max_steps, final_dist


def evaluate_policy(policy=None, device="cpu", episodes=20, task="reach", sim=None,
                   threshold=0.1, max_steps=300, start_pos=(0.0, 0.0), goal_pos=(0.5, 0.0)):
    """Evaluate a policy on task-oriented metrics."""
    if sim is None:
        sim = LeKiWiSim()
    evaluator = TaskEvaluator(sim, policy=policy, device=device)

    if task == "reach":
        results = [evaluator.reach_target(
            target=goal_pos, start=start_pos, threshold=threshold, max_steps=max_steps
        ) for _ in range(episodes)]
        successes = sum(1 for r in results if r[0])
        avg_steps = np.mean([r[1] for r in results])
        avg_dist = np.mean([r[2] for r in results])
        print(f"\n=== Reach Target Task ({episodes} episodes) ===")
        print(f"  Start: {start_pos} → Goal: {goal_pos} (threshold={threshold}m)")
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
    parser.add_argument("--sim_type", type=str, default="urdf",
                        choices=["primitive", "urdf"],
                        help="Simulation type: urdf=STL mesh (matches training), primitive=cylinders")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Success threshold in meters (default: 0.1 for primitive, 0.3 for urdf)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: 200 for primitive, 300 for urdf)")
    parser.add_argument("--start_x",  type=float, default=0.0,
                        help="Start X position (default: 0.0)")
    parser.add_argument("--start_y",  type=float, default=0.0,
                        help="Start Y position (default: 0.0)")
    parser.add_argument("--goal_x",   type=float, default=0.5,
                        help="Goal X position (default: 0.5)")
    parser.add_argument("--goal_y",   type=float, default=0.0,
                        help="Goal Y position (default: 0.0)")
    args = parser.parse_args()

    start_pos = (args.start_x, args.start_y)
    goal_pos  = (args.goal_x, args.goal_y)

    print(f"\n{'='*60}")
    print(f"  Task-Oriented Evaluation | {args.task}")
    print(f"  Sim: {args.sim_type} | Policy: {args.policy or 'random'}")
    print(f"  Start: {start_pos} → Goal: {goal_pos}")
    print(f"{'='*60}")

    # Create simulation (MUST match what was used during training)
    if args.sim_type == "urdf":
        sim = LeKiWiSimURDF()
        default_threshold = 0.1  # 10cm precision navigation
        default_max_steps = 300
    else:
        sim = LeKiwiSim()
        default_threshold = 0.1
        default_max_steps = 200

    threshold = args.threshold if args.threshold is not None else default_threshold
    max_steps = args.max_steps if args.max_steps is not None else default_max_steps
    print(f"  Threshold: {threshold}m | Max steps: {max_steps}")

    policy = None
    if args.policy:
        from scripts.train_task_oriented import CLIPFlowMatchingPolicy
        # Phase 16: goal-aware policy uses state_dim=11
        policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device=args.device)
        state_dict = torch.load(args.policy, map_location=args.device, weights_only=False)
        policy.load_state_dict(state_dict.get("policy_state_dict", state_dict), strict=False)
        policy.to(args.device)
        policy.eval()
        print(f"Loaded policy: {args.policy} (goal-aware, state_dim=11)")

    evaluate_policy(policy=policy, device=args.device, episodes=args.episodes,
                    task=args.task, sim=sim,
                    threshold=threshold, max_steps=max_steps,
                    start_pos=start_pos, goal_pos=goal_pos)

if __name__ == "__main__":
    main()
