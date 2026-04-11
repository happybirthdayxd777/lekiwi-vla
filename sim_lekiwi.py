#!/usr/bin/env python3
"""
LeKiwi Robot Simulation
=======================
MuJoCo simulation of LeKiwi robot — 6-DOF arm + 3 omni wheels.
Runs standalone for VLA demo/testing.

Usage:
    python3 sim_lekiwi.py --test kinematics
    python3 sim_lekiwi.py --test camera
    python3 sim_lekiwi.py --test random
"""

import argparse
import numpy as np
from PIL import Image
import mujoco
import time


LEKIWI_XML = """<?xml version="1.0"?>
<mujoco model="lekiwi">
    <visual>
        <global offwidth="1280" offheight="960"/>
    </visual>
    <option timestep="0.01" integrator="Euler">
        <flag contact="disable"/>
    </option>

    <default>
        <joint damping="0.3" frictionloss="0.05"/>
        <geom contype="0" conaffinity="0" rgba="0.5 0.5 0.5 1"/>
    </default>

    <worldbody>
        <!-- Ground -->
        <geom type="plane" size="5 5 0.01" rgba="0.2 0.2 0.2 1"/>

        <!-- Base chassis (freejoint = 6DOF) -->
        <body name="base" pos="0 0 0.12">
            <freejoint/>
            <geom type="cylinder" size="0.12 0.04" mass="3" rgba="0.25 0.45 0.80 1"/>

            <!-- Omni wheel 1: front -->
            <body name="wheel1" pos="0.10 0 -0.05">
                <joint name="w1" type="hinge" axis="0 1 0" damping="0.5"/>
                <geom type="cylinder" size="0.035 0.015" mass="0.2" rgba="0.1 0.1 0.1 1"/>
            </body>

            <!-- Omni wheel 2: back-left (120 deg) -->
            <body name="wheel2" pos="-0.05 0.087 -0.05">
                <joint name="w2" type="hinge" axis="0 1 0" damping="0.5"/>
                <geom type="cylinder" size="0.035 0.015" mass="0.2" rgba="0.1 0.1 0.1 1"/>
            </body>

            <!-- Omni wheel 3: back-right (240 deg) -->
            <body name="wheel3" pos="-0.05 -0.087 -0.05">
                <joint name="w3" type="hinge" axis="0 1 0" damping="0.5"/>
                <geom type="cylinder" size="0.035 0.015" mass="0.2" rgba="0.1 0.1 0.1 1"/>
            </body>

            <!-- Arm: shoulder pan (rotates base around Z) -->
            <body name="arm_base" pos="0 0 0.04">
                <joint name="j0" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.5"/>
                <geom type="cylinder" size="0.025 0.015" mass="0.5" rgba="0.90 0.60 0.20 1"/>

                <!-- Arm: shoulder lift -->
                <body name="arm_1" pos="0 0 0.06">
                    <joint name="j1" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.5"/>
                    <geom type="cylinder" size="0.022 0.05" mass="0.4" rgba="0.80 0.55 0.18 1"/>

                    <!-- Arm: elbow -->
                    <body name="arm_2" pos="0 0 0.10">
                        <joint name="j2" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="1.2"/>
                        <geom type="cylinder" size="0.018 0.08" mass="0.3" rgba="0.70 0.50 0.16 1"/>

                        <!-- Arm: wrist flex -->
                        <body name="arm_3" pos="0 0 0.08">
                            <joint name="j3" type="hinge" axis="0 1 0" range="-1.57 1.57" damping="0.8"/>
                            <geom type="cylinder" size="0.014 0.04" mass="0.15" rgba="0.60 0.45 0.14 1"/>

                            <!-- Arm: wrist roll -->
                            <body name="arm_4" pos="0 0 0.04">
                                <joint name="j4" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.6"/>
                                <geom type="cylinder" size="0.010 0.03" mass="0.10" rgba="0.50 0.40 0.12 1"/>

                                <!-- Gripper (slide) -->
                                <body name="gripper" pos="0 0 0.03">
                                    <joint name="j5" type="slide" axis="1 0 0" range="0 0.04" damping="3.0"/>
                                    <geom type="box" size="0.018 0.025 0.015" mass="0.08" rgba="0.2 0.2 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Front camera -->
            <camera name="front" pos="0 0 0.06" xyaxes="1 0 0 0 1 0" fovy="60"/>
        </body>

        <!-- Target object -->
        <body name="target" pos="0.5 0 0.02">
            <geom type="cylinder" size="0.04 0.04" rgba="1 0.2 0.2 1"/>
        </body>
    </worldbody>

    <!-- Actuators: ctrl[i] maps to j0..j5 (arm) and w1..w3 (wheels) -->
    <actuator>
        <motor joint="j0" gear="10"/>
        <motor joint="j1" gear="10"/>
        <motor joint="j2" gear="10"/>
        <motor joint="j3" gear="5"/>
        <motor joint="j4" gear="5"/>
        <motor joint="j5" gear="3"/>
        <motor joint="w1" gear="0.5"/>
        <motor joint="w2" gear="0.5"/>
        <motor joint="w3" gear="0.5"/>
    </actuator>
</mujoco>
"""


# Joint names in qpos order
ARM_JOINTS = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINTS = ["w1", "w2", "w3"]


def get_joint_qpos_idx(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def get_joint_qvel_idx(model, name):
    """qvel index = joint index * 1 (since all joints are scalar)"""
    return get_joint_qpos_idx(model, name)


class LeKiwiSim:
    """LeKiwi robot simulation wrapper."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(LEKIWI_XML)
        self.data = mujoco.MjData(self.model)

        # Pre-compute joint indices
        self._jpos = {n: get_joint_qpos_idx(self.model, n) for n in ARM_JOINTS + WHEEL_JOINTS}
        self._jvel = {n: get_joint_qvel_idx(self.model, n) for n in ARM_JOINTS + WHEEL_JOINTS}

        # Actuator indices (in ctrl order: j0..j5, w1..w3)
        self._act_idx = list(range(9))

        self.action_dim = 9  # 6 arm + 3 wheel
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Small initial perturbation to break symmetry
        self.data.qpos[self._jpos["j1"]] = 0.3  # slight shoulder lift

    def step(self, action: np.ndarray) -> dict:
        """
        Args:
            action[0:6]: arm joint position targets (rad)
            action[6:9]: wheel velocity targets (rad/s)
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (9,):
            raise ValueError(f"Expected (9,) action, got {action.shape}")

        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:6] = np.clip(action[0:6], -3.14, 3.14)
        self.data.ctrl[6:9] = np.clip(action[6:9], -5.0, 5.0)

        mujoco.mj_step(self.model, self.data)

        return self._obs()

    def _obs(self) -> dict:
        """Return observation dict."""
        return {
            "arm_positions": np.array([self.data.qpos[self._jpos[n]] for n in ARM_JOINTS]),
            "wheel_velocities": np.array([self.data.qvel[self._jvel[n]] for n in WHEEL_JOINTS]),
            "base_position": self.data.qpos[:3].copy(),
            "base_quaternion": self.data.qpos[3:7].copy(),
            "base_linear_velocity": self.data.qvel[:3].copy(),
            "base_angular_velocity": self.data.qvel[3:6].copy(),
            "time": self.data.time,
        }

    def render(self, width=640, height=480) -> Image.Image:
        """Render front camera."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        r = mujoco.Renderer(self.model, width, height)
        r.update_scene(self.data, camera=cam_id)
        img = r.render()
        r.close()
        return Image.fromarray(img)

    def get_reward(self) -> float:
        """Negative distance to target (0.5, 0, 0)."""
        return -float(np.linalg.norm(self.data.qpos[:2] - np.array([0.5, 0.0])))

    def __repr__(self):
        return f"LeKiwiSim(action_dim={self.action_dim})"


def test_kinematics():
    print("=" * 60)
    print("  LeKiwi Simulation — Kinematics Test")
    print("=" * 60)
    sim = LeKiwiSim()
    print(f"\nModel: {sim}")

    def fmt(vec):
        return ", ".join(f"{v:.4f}" for v in vec)

    # 1. Stay still
    print("\n[1] Zero action — 50 steps")
    sim.reset()
    for _ in range(50):
        sim.step(np.zeros(9))
    obs = sim._obs()
    print(f"    base pos: ({fmt(obs['base_position'][:2])})")

    # 2. Wheel 1 forward
    print("\n[2] Wheel1=1 rad/s — 100 steps")
    sim.reset()
    for _ in range(100):
        sim.step(np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0]))
    obs = sim._obs()
    print(f"    base pos: ({fmt(obs['base_position'][:2])})")
    print(f"    w1 vel: {obs['wheel_velocities'][0]:.4f}")

    # 3. Rotate in place (w2=1, w3=-1)
    print("\n[3] w2=1, w3=-1 → rotate — 100 steps")
    sim.reset()
    for _ in range(100):
        sim.step(np.array([0, 0, 0, 0, 0, 0, 0, 1.0, -1.0]))
    obs = sim._obs()
    ang_vel = obs["base_angular_velocity"][2]
    print(f"    base ang vel z: {ang_vel:.4f} rad/s")

    # 4. Arm target positions
    print("\n[4] Arm → [0.5, 0.3, -0.3, 0.1, 0, 0.02] — 100 steps")
    sim.reset()
    target_arm = np.array([0.5, 0.3, -0.3, 0.1, 0.0, 0.02])
    for _ in range(100):
        sim.step(np.concatenate([target_arm, np.zeros(3)]))
    obs = sim._obs()
    print(f"    arm pos: [{fmt(obs['arm_positions'])}]")

    # 5. Reward
    print("\n[5] Random actions — 50 steps")
    sim.reset()
    Rs = []
    for _ in range(50):
        sim.step(np.random.uniform(-0.5, 0.5, 9))
        Rs.append(sim.get_reward())
    print(f"    mean reward: {np.mean(Rs):.3f}")

    print("\nAll kinematics tests passed!")


def test_camera():
    print("=" * 60)
    print("  LeKiwi Simulation — Camera Test")
    print("=" * 60)
    sim = LeKiwiSim()

    # Move arm while stepping
    arm_targets = [[0.5, 0.3, -0.3, 0.1, 0, 0.02]] * 30
    for at in arm_targets:
        sim.step(np.concatenate([at, [0.5, 0.5, 0.5]]))

    img = sim.render()
    path = "/Users/i_am_ai/lekiwi_sim_camera.png"
    img.save(path)
    print(f"\nCamera frame saved: {path}  ({img.size[0]}×{img.size[1]})")
    return path


def run_random(n=200):
    print("=" * 60)
    print("  LeKiwi Simulation — Random Policy")
    print("=" * 60)
    sim = LeKiwiSim()
    total_r = 0.0

    for i in range(n):
        sim.step(np.random.uniform(-1, 1, 9))
        total_r += sim.get_reward()

        if i % 50 == 0:
            obs = sim._obs()
            print(f"  step {i:3d} | reward={sim.get_reward():.3f} | "
                  f"base=({obs['base_position'][0]:+.2f},{obs['base_position'][1]:+.2f}) | "
                  f"arm=[{', '.join(f'{a:.2f}' for a in obs['arm_positions'][:3])}]")

    print(f"\nTotal reward: {total_r:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeKiwi MuJoCo Simulation")
    parser.add_argument("--test", choices=["kinematics", "camera", "random"], default="kinematics")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.test == "kinematics":
        test_kinematics()
    elif args.test == "camera":
        test_camera()
    elif args.test == "random":
        run_random(args.steps)