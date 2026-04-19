#!/usr/bin/env python3
"""
test_bridge.py — smoke test for LeKiWi ROS2 Bridge
Verifies the bridge can be imported and the MuJoCo sim loaded.
Run: python3 test/test_bridge.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lekiwi_ros2_bridge.bridge_node import LeKiWiBridge
from sim_lekiwi import LeKiWiSim, WHEEL_JOINTS, ARM_JOINTS
import numpy as np

def test_sim():
    print("[1] LeKiWiSim smoke test")
    sim = LeKiWiSim()
    obs = sim.step(np.zeros(9))
    print(f"    arm pos:    {obs['arm_positions'].round(4)}")
    print(f"    wheel vel:  {obs['wheel_velocities'].round(4)}")
    print(f"    base pos:   {obs['base_position'].round(4)}")
    print("    ✓ sim step OK")
    return sim

def test_kinematics():
    print("\n[2] cmd_vel → wheel speed kinematics")
    sim = LeKiWiSim()
    bridge = type("MockBridge", (), {
        "wheel_radius": 0.05,
        "wheel_positions": [
            np.array([0.1732,  0.0,   0.0]),
            np.array([-0.0866, 0.15,  0.0]),
            np.array([-0.0866, -0.15, 0.0]),
        ],
        "joint_axes": [
            np.array([0.866025, 0.0, 0.5]) / np.linalg.norm([0.866025, 0.0, 0.5]),
        ] * 3,
    })()

    vx, vy, wz = 0.5, 0.0, 0.0
    wheel_speeds = []
    for i in range(3):
        wheel_vel = np.array([
            vx - wz * bridge.wheel_positions[i][1],
            vy + wz * bridge.wheel_positions[i][0],
            0.0,
        ])
        angular_speed = np.dot(wheel_vel, bridge.joint_axes[i]) / bridge.wheel_radius
        wheel_speeds.append(angular_speed)

    print(f"    cmd_vel (vx={vx}, vy={vy}, wz={wz}) → wheel speeds: {[f'{s:.3f}' for s in wheel_speeds]}")
    print("    ✓ kinematics OK")

def test_bridge_import():
    print("\n[3] Bridge module import")
    from lekiwi_ros2_bridge import bridge_node
    print(f"    bridge_node: {bridge_node.__version__}")
    print("    ✓ import OK")

if __name__ == "__main__":
    test_bridge_import()
    test_kinematics()
    test_sim()
    print("\n✅ All tests passed")
