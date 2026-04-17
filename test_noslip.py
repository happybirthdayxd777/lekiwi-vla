#!/usr/bin/env python3
"""Test noslip_iterations=10 effect on pure contact physics (k_omni=0)"""
import mujoco
import numpy as np
from sim_lekiwi_urdf import LeKiWiSimURDF

print('Testing noslip_iterations=10 on pure contact physics (k_omni=0)')
print('='*60)

results = []
for ep in range(10):
    sim = LeKiWiSimURDF()
    
    for t in range(200):
        # 9-element action: arm joints 0-5 = 0, wheel torques 6-8 = 0.5
        action = [0.0]*6 + [0.5, 0.5, 0.5]
        
        wheel_action_clipped = np.clip(action[6:9], -0.5, 0.5)
        wheel_torque = wheel_action_clipped * 10.0
        ctrl = np.array([*action[:6], *wheel_torque], dtype=np.float64)
        ctrl = np.clip(ctrl, -10.0, 10.0)
        
        sim.data.ctrl[:] = ctrl
        
        # Air resistance logic (copied from step)
        if sim.data.ncon == 0 and sim.data.qvel[2] > 0:
            drag = sim.data.qvel[2] * 15.0
            sim.data.xfrc_applied[sim.model.body('base').id][2] -= drag
        
        mujoco.mj_step(sim.model, sim.data)
    
    final_dist = sim.get_reward()
    results.append(final_dist)
    print(f'  EP{ep}: dist={final_dist:.3f}m')

mean_dist = np.mean(results)
sr = sum(1 for d in results if d < 0.2) / len(results) * 100
print()
print(f'noslip_iterations=10, k_omni=0:')
print(f'  mean_distance: {mean_dist:.3f}m')
print(f'  success_rate (< 0.2m): {sr:.1f}%')
print(f'  all distances: {[round(d,3) for d in results]}')