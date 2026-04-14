#!/usr/bin/env python3
import sys, os
sys.path.insert(0, "/Users/i_am_ai/hermes_research/lekiwi_vla")
os.chdir("/Users/i_am_ai/hermes_research/lekiwi_vla")

# First run the generator
import urdf2mujoco
xml_str = urdf2mujoco.generate_xml()

# Validate
import xml.etree.ElementTree as ET
try:
    ET.fromstring(xml_str)
    print("XML valid: YES")
except ET.ParseError as e:
    print(f"XML valid: NO - {e}")

# Then load with MuJoCo via from_xml_path (uses absolute mesh paths in models/)
import mujoco
try:
    m = mujoco.MjModel.from_xml_path(str(urdf2mujoco.OUT_FILE))
    print(f"MuJoCo load: SUCCESS")
    print(f"  nq={m.nq} nv={m.nv} nbody={m.nbody} njnt={m.njnt} nctrl={m.nctrl} nmesh={m.nmesh}")
    print()
    print("Joints:")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = m.jnt_type[i]
        print(f"  [{i}] {name} (type={jtype})")
    print()
    print("Bodies:")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  [{i}] {name}")
    print()
    print("Actuators:")
    for i in range(m.na):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")
except Exception as e:
    print(f"MuJoCo load: FAILED - {e}")
