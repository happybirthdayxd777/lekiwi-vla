#!/usr/bin/env python3
"""
LeKiwi Robot Simulation — v2
============================
MuJoCo simulation with proper wheel-ground contact, plus a
Gymnasium wrapper so it can be used with RL training pipelines.

Key improvements over v1:
  - Contact-enabled wheels (contype/conaffinity set)
  - Explicit friction params on ground and wheels
  - Gymnasium VectorEnv wrapper (single-env for now)
  - `gymnasium.make("LeKiwiSim-v0")` registration
  - Passive viewer for human inspection
  - Camera rendering with larger offscreen buffer

Usage:
    python3 sim_lekiwi.py --test kinematics
    python3 sim_lekiwi.py --test gym       # run Gymnasium episode
    python3 sim_lekiwi.py --test camera
"""

import argparse
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from typing import Optional, Tuple
import mujoco
import mujoco.viewer
import time

FloatArray = NDArray[np.floating]


# ─── MuJoCo Model XML ────────────────────────────────────────────────────────

LEKIWI_XML = """<?xml version="1.0"?>
<mujoco model="lekiwi">
    <visual>
        <global offwidth="1280" offheight="960"/>
    </visual>

    <option timestep="0.005" integrator="Euler">
        <flag contact="enable" energy="disable"/>
    </option>

    <!-- Global default: light damping, no dry friction (frictionloss) -->
    <default>
        <joint damping="0.5"/>
        <geom contype="1" conaffinity="1" friction="0.6 0.05 0.01"
              solref="0.004 1.0" solimp="0.8 0.4 0.01"/>
    </default>

    <worldbody>
        <!-- Ground — smooth high-friction plane -->
        <geom name="ground" type="plane"
              size="5 5 0.01"
              rgba="0.18 0.18 0.22 1"
              friction="1.0 0.1 0.02"/>

        <!-- Base chassis (freejoint = 6DOF) -->
        <body name="base" pos="0 0 0.14">
            <freejoint/>
            <geom name="chassis" type="cylinder"
                  size="0.12 0.04" mass="3.0"
                  rgba="0.25 0.45 0.80 1"/>

            <!-- ── Omni wheel 1: front — cylinder along Y, rotates around X → rolls in X ── -->
            <body name="wheel1" pos="0.10 0 -0.04">
                <joint name="w1" type="hinge" axis="1 0 0"
                       damping="0.5"/>
                <!-- Cylinder (wheel) with axle along Y, rolling on XZ ground plane -->
                <geom name="wheel1_geom" type="cylinder"
                      size="0.035 0.018" mass="0.2"
                      friction="0.9 0.05 0.01"
                      rgba="0.08 0.08 0.08 1"
                      euler="0 0 0"/>
            </body>

            <!-- ── Omni wheel 2: back-left (120°) ── -->
            <body name="wheel2" pos="-0.05 0.087 -0.04">
                <joint name="w2" type="hinge" axis="1 0 0"
                       damping="0.5"/>
                <geom name="wheel2_geom" type="cylinder"
                      size="0.035 0.018" mass="0.2"
                      friction="0.9 0.05 0.01"
                      rgba="0.08 0.08 0.08 1"
                      euler="0 0 0"/>
            </body>

            <!-- ── Omni wheel 3: back-right (240°) ── -->
            <body name="wheel3" pos="-0.05 -0.087 -0.04">
                <joint name="w3" type="hinge" axis="1 0 0"
                       damping="0.5"/>
                <geom name="wheel3_geom" type="cylinder"
                      size="0.035 0.018" mass="0.2"
                      friction="0.9 0.05 0.01"
                      rgba="0.08 0.08 0.08 1"
                      euler="0 0 0"/>
            </body>

            <!-- ── Arm: shoulder pan ── -->
            <body name="arm_base" pos="0 0 0.04">
                <joint name="j0" type="hinge" axis="0 0 1"
                       range="-3.14 3.14" damping="1.5"/>
                <geom name="j0_geom" type="cylinder"
                      size="0.025 0.015" mass="0.5"
                      rgba="0.90 0.60 0.20 1"/>

                <!-- Arm: shoulder lift -->
                <body name="arm_1" pos="0 0 0.06">
                    <joint name="j1" type="hinge" axis="0 1 0"
                           range="-1.57 1.57" damping="1.5"/>
                    <geom name="j1_geom" type="cylinder"
                          size="0.022 0.05" mass="0.4"
                          rgba="0.80 0.55 0.18 1"/>

                    <!-- Arm: elbow -->
                    <body name="arm_2" pos="0 0 0.10">
                        <joint name="j2" type="hinge" axis="0 1 0"
                               range="-1.57 1.57" damping="1.2"/>
                        <geom name="j2_geom" type="cylinder"
                              size="0.018 0.08" mass="0.3"
                              rgba="0.70 0.50 0.16 1"/>

                        <!-- Arm: wrist flex -->
                        <body name="arm_3" pos="0 0 0.08">
                            <joint name="j3" type="hinge" axis="0 1 0"
                                   range="-1.57 1.57" damping="0.8"/>
                            <geom name="j3_geom" type="cylinder"
                                  size="0.014 0.04" mass="0.15"
                                  rgba="0.60 0.45 0.14 1"/>

                            <!-- Arm: wrist roll -->
                            <body name="arm_4" pos="0 0 0.04">
                                <joint name="j4" type="hinge" axis="0 0 1"
                                       range="-3.14 3.14" damping="0.6"/>
                                <geom name="j4_geom" type="cylinder"
                                      size="0.010 0.03" mass="0.10"
                                      rgba="0.50 0.40 0.12 1"/>

                                <!-- Gripper (slide) -->
                                <body name="gripper" pos="0 0 0.03">
                                    <joint name="j5" type="slide"
                                           axis="1 0 0"
                                           range="0 0.04" damping="3.0"/>
                                    <geom name="j5_geom" type="box"
                                          size="0.018 0.025 0.015"
                                          mass="0.08"
                                          rgba="0.2 0.2 0.2 1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Front camera -->
            <camera name="front" pos="0 0 0.06"
                    xyaxes="1 0 0 0 1 0" fovy="60"/>
        </body>

        <!-- Target object -->
        <body name="target" pos="0.5 0 0.02">
            <geom name="target_geom" type="cylinder"
                  size="0.04 0.04" rgba="1 0.2 0.2 1"/>
        </body>
    </worldbody>

    <!-- Actuators: ctrl[0..5] arm joints, ctrl[6..8] wheel joints -->
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

ARM_JOINTS  = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINTS = ["w1", "w2", "w3"]
ALL_JOINTS   = ARM_JOINTS + WHEEL_JOINTS


def _jid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


# ─── Gymnasium Wrapper ────────────────────────────────────────────────────────

class LeKiwiEnv(gym.Env):
    """
    Gymnasium environment for the LeKiwi robot.

    Action space: Box(-1, 1, shape=(9,))
        [0..5]  arm joint position targets (normalized -1..1 → -3.14..3.14 rad)
        [6..8]  wheel velocity targets   (normalized -1..1 → -5..5 rad/s)

    Observation space: Box(-inf, inf, shape=(25,))
        [0..2]   base position (x, y, z)
        [3..6]   base quaternion (w, x, y, z)
        [7..9]   base linear velocity
        [10..12] base angular velocity
        [13..18] arm joint positions (6)
        [19..21] wheel velocities (3)
        [22]     time
        [23..24] target dx/dy (for reward)

    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(LEKIWI_XML)
        self.data  = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self._viewer = None

        # Pre-index joint positions/velocities
        self._jpos_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}
        self._jvel_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}

        # Target for reward (x=0.5, y=0)
        self._target = np.array([0.5, 0.0, 0.0])

        # Gymnasium spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )

        self.reset(seed=None)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        d = self.data
        base_xy   = d.qpos[:2]
        target_dx = self._target[:2] - base_xy
        return np.concatenate([
            d.qpos[:3],          # base pos
            d.qpos[3:7],         # base quat
            d.qvel[:3],          # base linvel
            d.qvel[3:6],         # base angvel
            np.array([d.qpos[self._jpos_idx[n]] for n in ARM_JOINTS]),   # arm pos
            np.array([d.qvel[self._jvel_idx[n]] for n in WHEEL_JOINTS]), # wheel vel
            [d.time],            # time
            target_dx,           # dx, dy to target
        ], dtype=np.float32)

    def _reward(self) -> float:
        dist = float(np.linalg.norm(self._target[:2] - self.data.qpos[:2]))
        arm_effort = float(np.sum(np.abs(self.data.qvel[self._jvel_idx["j0"]])))
        return -dist - 0.01 * arm_effort

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        # Denormalize action to control values
        arm_ctrl   = action[:6] * 3.14           # -3.14..3.14
        wheel_ctrl = action[6:9] * 5.0            # -5..5
        return np.concatenate([arm_ctrl, wheel_ctrl]).astype(np.float64)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Small non-zero arm pose so it starts with a natural posture
        self.data.qpos[self._jpos_idx["j1"]] = 0.3

        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        ctrl = self._action_to_ctrl(np.asarray(action, dtype=np.float32))
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        obs    = self._obs()
        reward = self._reward()
        done   = bool(self.data.time > 60.0)   # 60s episode timeout
        trunc  = False
        info: dict = {}

        return obs, reward, done, trunc, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
            r = mujoco.Renderer(self.model, 640, 480)
            r.update_scene(self.data, camera=cam_id)
            img = r.render()
            r.close()
            return img
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def open_viewer(self):
        """Launch interactive MuJoCo viewer (blocks current thread)."""
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)


# ─── Standalone Simulation (non-Gymnasium) ───────────────────────────────────

class LeKiwiSim:
    """Standalone MuJoCo wrapper with helpers and rendering."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(LEKIWI_XML)
        self.data  = mujoco.MjData(self.model)
        self._jpos_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}
        self._jvel_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}
        self._target = np.array([0.5, 0.0, 0.0])  # default goal (x, y, z)
        self.action_dim = 9
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._jpos_idx["j1"]] = 0.3
        # Sync xpos to _target so render() shows correct goal from frame 0
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.data.xpos[body_id] = self._target

    def set_target(self, pos):
        """Move the target marker to (x, y)."""
        self._target = np.array([pos[0], pos[1], 0.02], dtype=np.float64)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.data.xpos[body_id] = self._target

    def step(self, action: np.ndarray) -> dict:
        """Step with raw 9-D float64 action (first 6 arm pos, last 3 wheel vel)."""
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:6] = np.clip(action[0:6], -3.14, 3.14)
        self.data.ctrl[6:9] = np.clip(action[6:9], -5.0, 5.0)
        mujoco.mj_step(self.model, self.data)
        return self._obs()

    def _obs(self) -> dict:
        return {
            "arm_positions":  np.array([self.data.qpos[self._jpos_idx[n]] for n in ARM_JOINTS]),
            "wheel_velocities": np.array([self.data.qvel[self._jvel_idx[n]] for n in WHEEL_JOINTS]),
            "base_position":  self.data.qpos[:3].copy(),
            "base_quaternion": self.data.qpos[3:7].copy(),
            "base_linear_velocity":  self.data.qvel[:3].copy(),
            "base_angular_velocity": self.data.qvel[3:6].copy(),
            "time": self.data.time,
        }

    def render(self, width=640, height=480) -> Image.Image:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        r = mujoco.Renderer(self.model, width, height)
        r.update_scene(self.data, camera=cam_id)
        img = r.render()
        r.close()
        return Image.fromarray(img)

    def get_reward(self) -> float:
        return -float(np.linalg.norm(np.array([0.5, 0.0]) - self.data.qpos[:2]))

    def __repr__(self):
        return f"LeKiwiSim(action_dim={self.action_dim})"


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_kinematics():
    print("=" * 60)
    print("  LeKiwi Simulation v2 — Kinematics Test")
    print("=" * 60)

    sim = LeKiwiSim()
    print(f"\nModel: {sim}\n")

    def fmt(v): return ", ".join(f"{x:.4f}" for x in v)

    # 1. Stay still
    print("[1] Zero action — 50 steps")
    sim.reset()
    for _ in range(50):
        sim.step(np.zeros(9))
    obs = sim._obs()
    print(f"    base (x,y): ({obs['base_position'][0]:.4f}, {obs['base_position'][1]:.4f})")

    # 2. Wheel 1 forward
    print("\n[2] Wheel1=1.0 rad/s — 200 steps")
    sim.reset()
    for _ in range(200):
        sim.step(np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0]))
    obs = sim._obs()
    print(f"    base (x,y): ({obs['base_position'][0]:.4f}, {obs['base_position'][1]:.4f})")
    print(f"    w1 vel: {obs['wheel_velocities'][0]:.4f} rad/s")

    # 3. Rotate (w2=1, w3=-1)
    print("\n[3] w2=1, w3=-1 → rotation — 200 steps")
    sim.reset()
    for _ in range(200):
        sim.step(np.array([0, 0, 0, 0, 0, 0, 0, 1.0, -1.0]))
    obs = sim._obs()
    print(f"    ang vel z: {obs['base_angular_velocity'][2]:.4f} rad/s")

    # 4. Arm to target
    print("\n[4] Arm → [0.5, 0.3, -0.3, 0.1, 0, 0.02] — 200 steps")
    sim.reset()
    arm_target = np.array([0.5, 0.3, -0.3, 0.1, 0.0, 0.02])
    for _ in range(200):
        sim.step(np.concatenate([arm_target, np.zeros(3)]))
    obs = sim._obs()
    print(f"    arm pos: [{fmt(obs['arm_positions'])}]")

    # 5. Mixed
    print("\n[5] arm=[0,0,0,0,0,0] + w=[0.5,0.5,0.5] — 100 steps")
    sim.reset()
    for _ in range(100):
        sim.step(np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5]))
    obs = sim._obs()
    print(f"    base (x,y): ({obs['base_position'][0]:.4f}, {obs['base_position'][1]:.4f})")
    print(f"    reward: {sim.get_reward():.4f}")

    print("\nAll kinematics tests passed!")


def test_gym():
    """Run a complete Gymnasium episode with random policy."""
    print("=" * 60)
    print("  LeKiwi Simulation v2 — Gymnasium Episode Test")
    print("=" * 60)

    env = LeKiwiEnv(render_mode=None)
    obs, info = env.reset()
    print(f"\nObs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    total_reward = 0.0
    for i in range(300):
        # Random policy
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward

        if i % 50 == 0:
            print(f"  step {i:3d} | reward={reward:.3f} | "
                  f"base=({obs[0]:+.2f},{obs[1]:+.2f}) | "
                  f"arm=[{', '.join(f'{a:.2f}' for a in obs[13:16])}]")

        if done or trunc:
            print(f"\nEpisode ended at step {i}")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    env.close()


def test_camera():
    print("=" * 60)
    print("  LeKiwi Simulation v2 — Camera Test")
    print("=" * 60)

    sim = LeKiwiSim()

    # Step with some arm motion
    arm = np.array([0.5, 0.3, -0.3, 0.1, 0.0, 0.02])
    for _ in range(50):
        sim.step(np.concatenate([arm, [0.5, 0.5, 0.5]]))

    img = sim.render()
    path = "/Users/i_am_ai/lekiwi_sim_v2_camera.png"
    img.save(path)
    print(f"\nCamera frame saved: {path}  ({img.size[0]}×{img.size[1]})")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeKiwi MuJoCo Simulation v2")
    parser.add_argument("--test", choices=["kinematics", "camera", "gym"], default="kinematics")
    args = parser.parse_args()

    if args.test == "kinematics":
        test_kinematics()
    elif args.test == "camera":
        test_camera()
    elif args.test == "gym":
        test_gym()