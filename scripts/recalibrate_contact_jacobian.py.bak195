#!/usr/bin/env python3
"""
Recalibrate contact Jacobian for k_omni=0 physics.

Phase 161: The k_omni=15.0 overlay has been DISABLED because it uses the wrong
kinematic model (equilateral wheels) for the URDF's isosceles wheel geometry.
This script recalibrates the contact Jacobian using pure contact physics (k_omni=0).

Usage:
    python scripts/recalibrate_contact_jacobian.py
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from sim_lekiwi_urdf import LeKiWiSimURDF

def calibrate_contact_jacobian(n_steps=200, n_warmup=50):
    """Measure the contact Jacobian empirically with k_omni=0."""
    sim = LeKiWiSimURDF()
    base_body_id = sim.model.body('base').id

    def reset_and_warmup():
        sim.reset()
        sim.data.xfrc_applied[base_body_id, :] = 0
        for _ in range(n_warmup):
            sim.data.ctrl[6:9] = 0
            sim.data.xfrc_applied[base_body_id, :] = 0
            mujoco.mj_step(sim.model, sim.data)

    def measure_displacement(wheel_cmd, n_steps=n_steps):
        reset_and_warmup()
        for _ in range(n_steps):
            sim.data.ctrl[6:9] = np.clip(wheel_cmd, -0.5, 0.5) * 10.0
            sim.data.xfrc_applied[base_body_id, :] = 0
            mujoco.mj_step(sim.model, sim.data)
        return sim.data.xpos[base_body_id, :2].copy()

    print(f"Calibrating contact Jacobian (k_omni=0, {n_steps} steps, {n_warmup} warmup)...")

    # Measure single-wheel displacements
    d1 = measure_displacement([0.5, 0, 0])  # w1
    d2 = measure_displacement([0, 0.5, 0])  # w2
    d3 = measure_displacement([0, 0, 0.5])  # w3

    print(f"\nSingle-wheel displacements (200 steps):")
    print(f"  w1=+0.5: ({d1[0]:+.4f}, {d1[1]:+.4f})")
    print(f"  w2=+0.5: ({d2[0]:+.4f}, {d2[1]:+.4f})")
    print(f"  w3=+0.5: ({d3[0]:+.4f}, {d3[1]:+.4f})")

    # Build Jacobian (dx, dy per unit wheel speed per 200 steps)
    J_c = np.array([
        [d1[0]/0.5, d2[0]/0.5, d3[0]/0.5],
        [d1[1]/0.5, d2[1]/0.5, d3[1]/0.5],
    ])

    print(f"\nContact Jacobian J_c (per unit wheel vel per 200 steps):")
    print(J_c)

    J_pinv = np.linalg.pinv(J_c)
    print(f"\nPseudo-inverse J_c^+:")
    print(J_pinv)

    return J_c, J_pinv


def test_p_controller(J_pinv, n_goals=20, seed=42):
    """Test P-controller with recalibrated J_c on k_omni=0 physics."""
    import mujoco
    from sim_lekiwi_urdf import LeKiWiSimURDF

    sim = LeKiWiSimURDF()
    base_body_id = sim.model.body('base').id

    def reset_sim():
        sim.reset()
        sim.data.xfrc_applied[base_body_id, :] = 0

    def step_sim(wheels):
        sim.data.ctrl[6:9] = np.clip(wheels, -0.5, 0.5) * 10.0
        sim.data.xfrc_applied[base_body_id, :] = 0
        mujoco.mj_step(sim.model, sim.data)

    np.random.seed(seed)
    successes = 0

    for i in range(n_goals):
        goal = np.array([
            np.random.uniform(-0.3, 0.4),
            np.random.uniform(-0.3, 0.3)
        ])
        reset_sim()
        for _ in range(50): step_sim([0,0,0])  # warmup

        for step in range(200):
            base = sim.data.xpos[base_body_id, :2]
            err = goal - base
            if np.linalg.norm(err) < 0.15:
                successes += 1
                break
            # Scale error to m/200steps (J_c units)
            v_des = np.clip(err * 1.5, -0.05, 0.05) * 200  # scale to per-200steps
            wheels = np.clip(J_pinv @ v_des * 2.0, -0.5, 0.5)
            step_sim(wheels)

    sr = 100 * successes / n_goals
    print(f"\nP-controller eval: {successes}/{n_goals} = {sr:.1f}% SR")
    return successes, sr


if __name__ == "__main__":
    J_c, J_pinv = calibrate_contact_jacobian(n_steps=200, n_warmup=50)

    print("\n" + "="*60)
    print("P-controller test with recalibrated J_c (k_omni=0)")
    print("="*60)
    successes, sr = test_p_controller(J_pinv, n_goals=30, seed=42)

    print(f"\nRecalibrated J_c (for sim_lekiwi_urdf.py):")
    print(f"_CONTACT_JACOBIAN_KOMNI0 = np.array([")
    print(f"    [{J_c[0,0]:.4f}, {J_c[0,1]:.4f}, {J_c[0,2]:.4f}],  # dx")
    print(f"    [{J_c[1,0]:.4f}, {J_c[1,1]:.4f}, {J_c[1,2]:.4f}],  # dy")
    print(f"], dtype=np.float64)")
