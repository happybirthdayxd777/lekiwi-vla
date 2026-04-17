#!/usr/bin/env python3
"""Test asymmetric wheel actions for forward locomotion"""
import mujoco
import numpy as np
import sys
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF

print('=== Phase 137: Asymmetric Wheel Actions for Forward Locomotion ===\n')

# Test various asymmetric wheel actions
test_cases = [
    ("[0.5, 0.5, 0.5]", [0.5, 0.5, 0.5]),       # M7-forward (symmetric)
    ("[0.5, 0.0, 0.0]", [0.5, 0.0, 0.0]),       # w1 only
    ("[0.0, 0.5, 0.0]", [0.0, 0.5, 0.0]),       # w2 only
    ("[0.0, 0.0, 0.5]", [0.0, 0.0, 0.5]),       # w3 only
    ("[0.5, 0.3, 0.3]", [0.5, 0.3, 0.3]),       # asymmetric (Phase 136 suggestion)
    ("[0.3, 0.5, 0.3]", [0.3, 0.5, 0.3]),       # w2 dominant
    ("[0.3, 0.3, 0.5]", [0.3, 0.3, 0.5]),       # w3 dominant
    ("[0.5, 0.3, 0.0]", [0.5, 0.3, 0.0]),       # w1+w2
    ("[0.5, 0.0, 0.3]", [0.5, 0.0, 0.3]),       # w1+w3
    ("[0.3, 0.5, 0.0]", [0.3, 0.5, 0.0]),       # w2 dominant asymmetric
]

results = []
for name, wheel_action in test_cases:
    ep_results = []
    for ep in range(5):
        sim = LeKiWiSimURDF()
        for t in range(200):
            action = [0.0]*6 + wheel_action
            wheel_action_clipped = np.clip(action[6:9], -0.5, 0.5)
            wheel_torque = wheel_action_clipped * 10.0
            ctrl = np.array([*action[:6], *wheel_torque], dtype=np.float64)
            ctrl = np.clip(ctrl, -10.0, 10.0)
            sim.data.ctrl[:] = ctrl
            if sim.data.ncon == 0 and sim.data.qvel[2] > 0:
                drag = sim.data.qvel[2] * 15.0
                sim.data.xfrc_applied[sim.model.body('base').id][2] -= drag
            mujoco.mj_step(sim.model, sim.data)
        
        base_pos = sim.data.body('base').xpos.copy()
        dx = base_pos[0]  # +X is forward
        dy = base_pos[1]
        dist = sim.get_reward()
        ep_results.append((dist, dx, dy))
    
    mean_dist = np.mean([r[0] for r in ep_results])
    mean_dx = np.mean([r[1] for r in ep_results])
    mean_dy = np.mean([r[2] for r in ep_results])
    sr = sum(1 for r in ep_results if r[0] < 0.2) / len(ep_results) * 100
    
    results.append((name, mean_dist, mean_dx, mean_dy, sr))
    print(f'{name:20s}: dist={mean_dist:.3f}m, dx={mean_dx:+.3f}, dy={mean_dy:+.3f}, SR={sr:.0f}%')

print('\n=== Summary: Best forward (dx) actions ===')
for name, dist, dx, dy, sr in sorted(results, key=lambda x: -abs(x[2]))[:5]:
    print(f'{name:20s}: dx={dx:+.3f}m, dy={dy:+.3f}m, dist={dist:.3f}m')