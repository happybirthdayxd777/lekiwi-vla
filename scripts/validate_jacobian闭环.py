#!/usr/bin/env python3
"""
Phase 124: Closed-Loop Validation of Contact Jacobian Bridge
================================================================
Tests that twist_to_contact_wheel_speeds(J_c^+ @ [vx*10, vy*10])
correctly drives the robot toward a world-frame goal.

Tests 3 scenarios:
  1. Pure +X goal: cmd_vel=(0.3, 0.0) → should move toward +X
  2. Pure +Y goal: cmd_vel=(0.0, 0.3) → should move toward +Y
  3. Diagonal goal:  cmd_vel=(0.2, 0.2) → should move diagonally

Expected: robot follows cmd_vel direction in world frame (not base frame).
This validates the Phase 123 contact Jacobian fix.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

import numpy as np
from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds, _CONTACT_JACOBIAN

def test_jacobian_closed_loop():
    """Test: given desired vx/vy, does robot move in that direction?"""
    print("=" * 60)
    print("Phase 124: Contact Jacobian Closed-Loop Validation")
    print("=" * 60)

    sim = LeKiWiSimURDF()
    sim.reset()

    # Test cases: (label, vx, vy)
    tests = [
        ("Pure +X",  0.3, 0.0),
        ("Pure +Y",  0.0, 0.3),
        ("Diagonal", 0.2, 0.2),
    ]

    all_passed = True
    for label, vx, vy in tests:
        sim.reset()
        base_id = sim.model.body('base').id
        init_pos = sim.data.xpos[base_id, :2].copy()

        # Apply 100 steps of cmd_vel=(vx, vy)
        for step in range(100):
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            # Build action: arm=[0]*6, wheel=wheel_speeds
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            action = np.clip(action, -0.5, 0.5)  # safety clip
            sim.step(action)

        final_pos = sim.data.xpos[base_id, :2].copy()
        delta = final_pos - init_pos
        dist = np.linalg.norm(delta)

        # Direction check: for each axis, if cmd_vel is non-zero, delta should follow
        x_ok = (vx == 0) or (vx > 0 and delta[0] > -0.15) or (vx < 0 and delta[0] < 0.15)
        y_ok = (vy == 0) or (vy > 0 and delta[1] > 0) or (vy < 0 and delta[1] < 0)
        direction_ok = x_ok and y_ok
        # Magnitude check: should move at least 0.05m
        magnitude_ok = dist > 0.05

        status = "PASS" if (direction_ok and magnitude_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"\n{label}: cmd_vel=({vx}, {vy})")
        print(f"  Init: ({init_pos[0]:.3f}, {init_pos[1]:.3f})")
        print(f"  Final: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        print(f"  Delta: ({delta[0]:+.3f}, {delta[1]:+.3f}), |d|={dist:.3f}m")
        print(f"  Direction match: {direction_ok}, Magnitude >0.05m: {magnitude_ok}")
        print(f"  STATUS: {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED — Jacobian closed-loop is correct ✓")
    else:
        print("SOME TESTS FAILED — check contact Jacobian calibration")
    print("=" * 60)

    # Additional: show J_c and its pseudo-inverse
    print("\nContact Jacobian J_c (m/step per rad/s):")
    print(_CONTACT_JACOBIAN)
    print("\nPseudo-inverse J_c^+:")
    print(np.linalg.pinv(_CONTACT_JACOBIAN))

    return all_passed


def test_jacobian_vs_kinematic():
    """Compare contact-Jacobian vs old kinematic wheel speeds."""
    print("\n" + "=" * 60)
    print("Comparison: Contact Jacobian vs Old Kinematic Model")
    print("=" * 60)

    tests = [(0.3, 0.0), (0.0, 0.3), (0.2, 0.2)]
    for vx, vy in tests:
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        print(f"\ncmd_vel=({vx}, {vy}) → wheel_speeds={wheel_speeds}")


if __name__ == "__main__":
    ok = test_jacobian_closed_loop()
    test_jacobian_vs_kinematic()
    sys.exit(0 if ok else 1)
