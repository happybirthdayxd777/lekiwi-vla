#!/usr/bin/env python3
"""Phase 93: Sweep wheel body Z to find optimal contact geometry."""
import numpy as np
import mujoco
import re
import sys

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')

# Read the actual source file
with open('/Users/i_am_ai/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py') as f:
    src = f.read()

# Extract the XML template
start = src.find('LEKIWI_URDF_XML = f"""') + len('LEKIWI_URDF_XML = f"""')
end = src.find('"""', start)
xml_template = src[start:end]

_3DP = '/Users/i_am_ai/hermes_research/lekiwi_modular/src/lekiwi_description/3DPrintMeshes'
_URDFM = '/Users/i_am_ai/hermes_research/lekiwi_modular/src/lekiwi_description/urdf/meshes'

def expand_xml(template):
    result = template
    result = re.sub(r'{_mp\(([^)]+)\)}',
                   lambda m: f'"{_3DP}/{m.group(1).strip(chr(39))}"', result)
    result = re.sub(r'{_mp2\(([^)]+)\)}',
                   lambda m: f'"{_URDFM}/{m.group(1).strip(chr(39))}"', result)
    return result

original_xml = expand_xml(xml_template)

# Verify original loads
model = mujoco.MjModel.from_xml_string(original_xml)
data = mujoco.MjData(model)
action = np.zeros(9)
action[6:9] = [1.0, 1.0, 1.0]

# Test ORIGINAL geometry with k_omni DISABLED (zero xfrc after each step)
for _ in range(200):
    data.ctrl[:] = np.clip(action, -5, 5)
    mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0  # Disable k_omni overlay

orig_dist = np.linalg.norm(data.qpos[:2])
print(f'Original (-0.060), k_omni=0: dist={orig_dist:.4f}m, contacts={data.ncon}, z={data.qpos[2]:.4f}')

# Test ORIGINAL with k_omni ENABLED
model2 = mujoco.MjModel.from_xml_string(original_xml)
data2 = mujoco.MjData(model2)
for _ in range(200):
    data2.ctrl[:] = np.clip(action, -5, 5)
    mujoco.mj_step(model2, data2)
with_komni = np.linalg.norm(data2.qpos[:2])
print(f'Original (-0.060), WITH k_omni: dist={with_komni:.4f}m, contacts={data2.ncon}, z={data2.qpos[2]:.4f}')

print()
print('=== SWEEP: wheel body Z vs contact locomotion (k_omni=0) ===')
print(f'{"Body Z":>10} | {"CylBot_Z":>9} | {"Dist":>8} | {"Contacts":>9} | {"Base_Z":>8}')
print('-' * 60)

best_z, best_dist = -0.060, orig_dist
for body_z in [-0.060, -0.062, -0.064, -0.066, -0.068, -0.070]:
    xml_test = expand_xml(xml_template)
    for old in ['0.0866 0.10 -0.060', '-0.0866 0.10 -0.060', '-0.0866 -0.10 -0.060']:
        xml_test = xml_test.replace(old, old.replace('-0.060', f'{body_z:.3f}'))

    m = mujoco.MjModel.from_xml_string(xml_test)
    d = mujoco.MjData(m)

    for _ in range(200):
        d.ctrl[:] = np.clip(action, -5, 5)
        mujoco.mj_step(m, d)
        d.xfrc_applied[:] = 0  # Disable k_omni

    dist = np.linalg.norm(d.qpos[:2])
    cyl_bot = 0.075 + body_z + (-0.015) - 0.008
    print(f'{body_z:>10.3f} | {cyl_bot:>9.4f} | {dist:>8.4f} | {d.ncon:>9} | {d.qpos[2]:>8.4f}')
    if dist > best_dist:
        best_dist = dist
        best_z = body_z

print()
print(f'Best body Z: {best_z:.3f} with dist={best_dist:.4f}m')
print(f'Phase 92 baseline (k_omni=0): 0.118m')
print(f'Improvement over Phase 92: {best_dist/0.118:.2f}x')
