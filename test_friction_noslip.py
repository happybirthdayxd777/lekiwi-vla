#!/usr/bin/env python3
"""Test friction=2.7 + noslip_iterations=10 effect on pure contact locomotion"""
import mujoco
import numpy as np
import re
import sys

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')

print('=== Phase 136 Priority: Restore friction=2.7 + noslip=10 ===')

# Read sim_lekiwi_urdf.py
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py', 'r') as f:
    content = f.read()

# Count occurrences of friction=1.5
friction_15_count = content.count('friction="1.5 0.15 0.01"')
print(f'friction="1.5 0.15 0.01" occurrences: {friction_15_count}')

# Save backup
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py.bak136', 'w') as f:
    f.write(content)

# Replace friction values for wheel/base contact geoms
new_content = content.replace('friction="1.5 0.15 0.01"', 'friction="2.7 0.27 0.02"')

# Verify the change
friction_new = re.findall(r'friction="[0-9.]+ [0-9.]+ [0-9.]+"', new_content)
print(f'New friction values: {set(friction_new)}')

# Write updated content
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py', 'w') as f:
    f.write(new_content)

# Clear cached modules
for mod in list(sys.modules.keys()):
    if 'sim_lekiwi' in mod or 'lekiwi' in mod:
        del sys.modules[mod]

# Now test with noslip_iterations=10 and friction=2.7
from sim_lekiwi_urdf import LeKiWiSimURDF

print('\n=== Testing with noslip_iterations=10 + friction=2.7 ===')

# Test M7-forward [0.5, 0.5, 0.5] which should move mostly in +X
results = []
for ep in range(5):
    sim = LeKiWiSimURDF()
    for t in range(200):
        action = [0.0]*6 + [0.5, 0.5, 0.5]
        wheel_action_clipped = np.clip(action[6:9], -0.5, 0.5)
        wheel_torque = wheel_action_clipped * 10.0
        ctrl = np.array([*action[:6], *wheel_torque], dtype=np.float64)
        ctrl = np.clip(ctrl, -10.0, 10.0)
        sim.data.ctrl[:] = ctrl
        if sim.data.ncon == 0 and sim.data.qvel[2] > 0:
            drag = sim.data.qvel[2] * 15.0
            sim.data.xfrc_applied[sim.model.body('base').id][2] -= drag
        mujoco.mj_step(sim.model, sim.data)
    dist = sim.get_reward()
    results.append(dist)
    base_pos = sim.data.body('base').xpos.copy()
    print(f'  EP{ep}: dist={dist:.3f}m, base_pos=({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f})')

print(f'\nMean dist: {np.mean(results):.3f}m')
print(f'SR (<0.2m): {sum(1 for r in results if r < 0.2)/len(results)*100:.0f}%')

# Restore backup
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py.bak136', 'r') as f:
    original = f.read()
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py', 'w') as f:
    f.write(original)
print('\nRestored original (backup saved as .bak136)')