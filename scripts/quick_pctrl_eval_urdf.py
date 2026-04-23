#!/usr/bin/env python3
"""
Phase 273: Quick P-controller eval on URDF sim — 10 goals, 200 steps each.
Purpose: Establish ground truth for URDF locomotion capability.

If P-controller SR > 50% on URDF sim → URDF sim is viable for bridge.
If P-controller SR < 30% → URDF sim locomotion is broken (needs fixing).
"""
import sys, numpy as np, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

DEVICE = 'cpu'
np.random.seed(42)

sim = LeKiWiSimURDF()
sim.reset()

# Goal positions (all within |r|<0.40m for fair comparison)
goals = [
    np.array([0.20,  0.15]),
    np.array([-0.15,  0.25]),
    np.array([0.30, -0.10]),
    np.array([-0.25, -0.20]),
    np.array([0.10, -0.30]),
    np.array([0.35,  0.20]),
    np.array([-0.30,  0.15]),
    np.array([0.15,  0.35]),
    np.array([-0.20, -0.35]),
    np.array([0.25,  0.00]),
]

K_P = 2.0
MAX_STEPS = 200
SUCCESS_RADIUS = 0.10

results = []
for g_idx, goal in enumerate(goals):
    sim.reset()
    sim._goal_xy = goal.copy()

    for step in range(MAX_STEPS):
        # Get base position
        base_xy = sim.data.xpos[sim.model.body('base').id, :2]
        # P-controller: desired velocity toward goal
        err = goal - base_xy
        vx = K_P * err[0]
        vy = K_P * err[1]
        # Convert to wheel speeds using contact Jacobian
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        # Build action: arm(6) + wheel(3)
        action = np.concatenate([np.zeros(6), wheel_speeds])
        sim.step(action)

        dist = np.linalg.norm(sim.data.xpos[sim.model.body('base').id, :2] - goal)
        if dist < SUCCESS_RADIUS:
            print(f"  Goal {g_idx}: SUCCESS @{step}, dist={dist:.4f}m, steps={step}")
            results.append((g_idx, 'success', step, dist))
            break
    else:
        final_dist = np.linalg.norm(sim.data.xpos[sim.model.body('base').id, :2] - goal)
        print(f"  Goal {g_idx}: FAILED, final_dist={final_dist:.3f}m, max_steps={MAX_STEPS}")
        results.append((g_idx, 'fail', MAX_STEPS, final_dist))

    # Print step-0 base state for diagnosis
    if g_idx == 0:
        base_pos_0 = sim.data.xpos[sim.model.body('base').id, :2]
        print(f"  [DIAG] Goal {g_idx} start base_xy=({base_pos_0[0]:.4f}, {base_pos_0[1]:.4f})")

sr = sum(1 for r in results if r[1] == 'success') / len(results)
print(f"\nP-controller on URDF sim: {sr*100:.0f}% SR ({sum(1 for r in results if r[1]=='success')}/{len(results)})")
print(f"Mean steps to success: {np.mean([r[2] for r in results if r[1]=='success']):.1f}")
