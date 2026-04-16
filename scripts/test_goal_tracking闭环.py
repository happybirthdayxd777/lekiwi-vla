#!/usr/bin/env python3
"""
Phase 124: Closed-Loop Goal Tracking Test
==========================================
Simulates the bridge receiving cmd_vel commands aimed at a world-frame goal
and tests whether the robot successfully reaches the goal.

Tests 4 goals at different positions in the world frame.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

import numpy as np
from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds


class ClosedLoopController:
    """P-controller for world-frame goal tracking using contact Jacobian."""

    def __init__(self, kP=1.5, max_speed=0.3):
        self.kP = kP
        self.max_speed = max_speed

    def compute_cmd_vel(self, base_pos, goal):
        """Compute cmd_vel (vx, vy) to move from base_pos toward goal."""
        dx = goal[0] - base_pos[0]
        dy = goal[1] - base_pos[1]
        dist = np.linalg.norm([dx, dy])
        if dist < 0.05:
            return 0.0, 0.0
        # P-control: scale velocity by distance
        v_mag = min(self.kP * dist, self.max_speed)
        vx = v_mag * (dx / dist)
        vy = v_mag * (dy / dist)
        return vx, vy


def test_goal_tracking():
    print("=" * 60)
    print("Phase 124: Closed-Loop Goal Tracking Test")
    print("=" * 60)

    controller = ClosedLoopController(kP=1.5, max_speed=0.3)
    goals = [
        ("Right",     np.array([0.5,  0.0])),
        ("Top-right", np.array([0.35, 0.35])),
        ("Top",       np.array([0.0,  0.5])),
        ("Left-top",  np.array([-0.3, 0.4])),
    ]

    all_reached = []
    for label, goal in goals:
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        reached = False
        rewards = []
        for step in range(200):
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(goal - base_pos)

            # Check goal reached
            if dist < 0.20:
                reached = True
                print(f"\n{label}: goal={goal} → REACHED at step {step} (dist={dist:.3f})")
                break

            # Compute cmd_vel toward goal
            vx, vy = controller.compute_cmd_vel(base_pos, goal)

            # Convert to wheel speeds via contact Jacobian
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            action = np.zeros(9)
            action[6:9] = np.clip(wheel_speeds, -0.5, 0.5)
            sim.step(action)
            rewards.append(sim._reward())

        if not reached:
            final_pos = sim.data.xpos[base_id, :2]
            final_dist = np.linalg.norm(goal - final_pos)
            mean_r = np.mean(rewards) if rewards else 0.0
            print(f"\n{label}: goal={goal} → NOT reached (final_dist={final_dist:.3f}m, mean_reward={mean_r:.3f})")
        all_reached.append(reached)

    n_reached = sum(all_reached)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_reached}/{len(goals)} goals reached ({100*n_reached/len(goals):.0f}%)")
    print(f"{'=' * 60}")
    return n_reached == len(goals)


if __name__ == "__main__":
    success = test_goal_tracking()
    sys.exit(0 if success else 1)
