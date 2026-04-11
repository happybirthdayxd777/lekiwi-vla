#!/usr/bin/env python3
"""
LeKiWi Robot Simulation — Real STL Mesh Geometry
================================================
MuJoCo sim using STL meshes from lekiwi_modular URDF where feasible.
Hybrid approach: meshes for simple/medium parts, primitives for
complex meshes (>200k faces).

Usage:
    python3 sim_lekiwi_urdf.py --test all
    python3 sim_lekiwi_urdf.py --render
"""

import argparse
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from typing import Optional, Tuple
import mujoco
import mujoco.viewer
import os

FloatArray = NDArray[np.floating]

# ── Paths ──────────────────────────────────────────────────────────────────────
_3DP   = os.path.expanduser("~/hermes_research/lekiwi_modular/src/lekiwi_description/3DPrintMeshes")
_URDFM = os.path.expanduser("~/hermes_research/lekiwi_modular/src/lekiwi_description/urdf/meshes")

def _mp(n: str) -> str: return os.path.join(_3DP, n)
def _mp2(n: str) -> str: return os.path.join(_URDFM, n)


# ── Kinematics from LeKiWi.urdf ───────────────────────────────────────────────
# Arm joints (6-DOF):
#   j0: arm_base pan     axis=[0,0,1]   range=[-3.14, 3.14]
#   j1: arm_1 lift       axis=[0,1,0]   range=[-1.57, 1.57]
#   j2: arm_2 elbow      axis=[1,0,0]   range=[-1.57, 1.57]
#   j3: arm_3 wrist_flex axis=[1,0,0]   range=[-1.57, 1.57]
#   j4: arm_4 wrist_roll axis=[0,0,1]   range=[-3.14, 3.14]
#   j5: gripper slide   axis=[1,0,0]   range=[0, 0.04]
#
# Wheel joints (3 omni, 120° apart):
#   w1 (wheel0): Revolute-64  axis=[-0.866, 0, 0.5]
#   w2 (wheel1): Revolute-62  axis=[ 0.866, 0, 0.5]
#   w3 (wheel2): Revolute-60  axis=[ 0, 0, -1]
#
# Control layout: ctrl[0:6]=arm joints, ctrl[6:9]=wheel velocities


# ── MuJoCo XML ────────────────────────────────────────────────────────────────

LEKIWI_URDF_XML = f"""<?xml version="1.0"?>
<mujoco model="lekiwi_urdf">
    <visual>
        <global offwidth="1280" offheight="960"/>
    </visual>

    <option timestep="0.005" integrator="Euler">
        <flag contact="enable" energy="disable"/>
    </option>

    <!-- ── Asset library: STL meshes (mm-scale, scale=0.001) ── -->
    <asset>
        <!-- Base structural -->
        <mesh name="base_plate_1"   file="{_mp('base_plate_layer1.stl')}"    scale="0.001 0.001 0.001"/>
        <mesh name="base_plate_2"   file="{_mp('base_plate_layer2.stl')}"    scale="0.001 0.001 0.001"/>
        <mesh name="base_cam_mount" file="{_mp('base_camera_mount.stl')}"    scale="0.001 0.001 0.001"/>
        <mesh name="battery_mount"   file="{_mp('battery_mount.stl')}"       scale="0.001 0.001 0.001"/>

        <!-- Arm servos -->
        <mesh name="servo_base"    file="{_mp2('STS3215_03a-v1.stl')}"       scale="0.001 0.001 0.001"/>
        <mesh name="servo_j2"      file="{_mp2('STS3215_03a-v1-1.stl')}"     scale="0.001 0.001 0.001"/>
        <mesh name="servo_j3"      file="{_mp2('STS3215_03a-v1-2.stl')}"     scale="0.001 0.001 0.001"/>
        <mesh name="servo_j4"      file="{_mp2('STS3215_03a-v1-3.stl')}"     scale="0.001 0.001 0.001"/>
        <mesh name="servo_wrist"   file="{_mp2('STS3215_03a_Wrist_Roll-v1.stl')}" scale="0.001 0.001 0.001"/>

        <!-- Arm structural parts -->
        <mesh name="base_q"        file="{_mp2('Base_08q-v1.stl')}"          scale="0.001 0.001 0.001"/>
        <mesh name="wave_plate"     file="{_mp2('WaveShare_Mounting_Plate_01d-v1.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="arm_clip"       file="{_mp2('SO_ARM100_08k_Asym_Mirror_Clip-v1.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="arm_square"     file="{_mp2('SO_ARM100_08k_116_Square-v1.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="arm_mirror"     file="{_mp2('SO_ARM100_08k_Mirror-v1.stl')}" scale="0.001 0.001 0.001"/>

        <!-- Wrist -->
        <mesh name="wrist_pitch"   file="{_mp2('Wrist_Roll_Pitch_08i-v1.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="wrist_horn"    file="{_mp2('Wrist_Roll_08c-v1.stl')}"   scale="0.001 0.001 0.001"/>
        <mesh name="wrist_servo"   file="{_mp2('STS3215_03a-v1-3.stl')}"    scale="0.001 0.001 0.001"/>
        <mesh name="horn_fixed"    file="{_mp2('Passive_Horn_01-v1.stl')}"   scale="0.001 0.001 0.001"/>

        <!-- Gripper -->
        <mesh name="servo_gripper"  file="{_mp2('STS3215_03a-v1-4.stl')}"    scale="0.001 0.001 0.001"/>
        <mesh name="moving_jaw"    file="{_mp2('Moving_Jaw_08d-v1.stl')}"   scale="0.001 0.001 0.001"/>
        <mesh name="gripper_horn"  file="{_mp2('Passive_Horn_01-v1.stl')}"   scale="0.001 0.001 0.001"/>

        <!-- Camera -->
        <mesh name="wrist_cam_mount" file="{_mp2('Wrist-Camera-Mount-v11.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="wrist_cam_body"  file="{_mp2('Camera-Model-v3.stl')}"     scale="0.001 0.001 0.001"/>
    </asset>

    <default>
        <joint damping="0.5"/>
        <!-- Ground / base contact: standard friction cone -->
        <geom friction="0.6 0.05 0.01"
              solref="0.004 1.0" solimp="0.8 0.4 0.01"/>
    </default>

    <worldbody>
        <!-- Ground -->
        <geom name="ground" type="plane"
              size="5 5 0.01"
              rgba="0.18 0.18 0.22 1"
              friction="1.0 0.1 0.02"/>

        <!-- ══ Base (free 6-DOF) ══ -->
        <body name="base" pos="0 0 0.035">
            <freejoint/>
            <inertial pos="0 0 0.01" mass="2.0" diaginertia="0.01 0.01 0.015"/>

            <!-- Base plate STL layers -->
            <geom name="base_p1" type="mesh" mesh="base_plate_1" rgba="0 0 0.9 1"/>
            <geom name="base_p2" type="mesh" mesh="base_plate_2"
                  rgba="0 0 0.8 1" pos="0 0 0.006"/>
            <!-- Battery mount -->
            <geom name="batt_m" type="mesh" mesh="battery_mount"
                  rgba="0.1 0.1 0.1 1" pos="-0.04 0 0.01"/>
            <!-- Camera mount -->
            <geom name="cam_m" type="mesh" mesh="base_cam_mount"
                  rgba="0.5 0.5 0.5 1" pos="0 0 0.08"/>

            <!-- ══ Wheel 0: front-right ─ collider only (wheel geometry) ══ -->
            <body name="wheel0" pos="0.0866 0.10 -0.06">
                <joint name="w1" type="hinge" axis="1 0 0" damping="0.5"/>
                <geom name="wheel0_geom" type="cylinder"
                      size="0.035 0.018" mass="0.1"
                      contype="1" conaffinity="1"
                      friction="0.9 0.05 0.01" rgba="0.05 0.05 0.05 1"/>
            </body>

            <!-- ══ Wheel 1: back-left ══ -->
            <body name="wheel1" pos="-0.0866 0.10 -0.06">
                <joint name="w2" type="hinge" axis="1 0 0" damping="0.5"/>
                <geom name="wheel1_geom" type="cylinder"
                      size="0.035 0.018" mass="0.1"
                      contype="1" conaffinity="1"
                      friction="0.9 0.05 0.01" rgba="0.05 0.05 0.05 1"/>
            </body>

            <!-- ══ Wheel 2: back-right ══ -->
            <body name="wheel2" pos="-0.0866 -0.10 -0.06">
                <joint name="w3" type="hinge" axis="1 0 0" damping="0.5"/>
                <geom name="wheel2_geom" type="cylinder"
                      size="0.035 0.018" mass="0.1"
                      contype="1" conaffinity="1"
                      friction="0.9 0.05 0.01" rgba="0.05 0.05 0.05 1"/>
            </body>

            <!-- ══ Arm base ══ -->
            <body name="arm_base" pos="0 0 0.09">
                <!-- Shoulder pan j0 -->
                <joint name="j0" type="hinge" axis="0 0 1"
                       range="-3.14 3.14" damping="1.5"/>
                <!-- Shoulder pan geom: flat STL + horn -->
                <geom name="base_q_geom" type="mesh" mesh="base_q"
                      rgba="0.9 0.6 0.2 1" mass="0.5"/>

                <!-- ══ Arm segment 1: j1 shoulder lift ══ -->
                <body name="arm_1" pos="0 0 0.015">
                    <joint name="j1" type="hinge" axis="0 1 0"
                           range="-1.57 1.57" damping="1.5"/>
                    <!-- Shoulder servo STL -->
                    <geom name="s1_geom" type="mesh" mesh="servo_base"
                          rgba="0.8 0.5 0.2 1" mass="0.065"/>
                    <!-- Arm link 1: square tube + mirror clip -->
                    <geom name="arm1_link" type="mesh" mesh="arm_square"
                          rgba="0.7 0.55 0.18 1" mass="0.3" pos="0 0 0.05"/>

                    <!-- ══ Arm segment 2: j2 elbow ══ -->
                    <body name="arm_2" pos="0 0 0.10">
                        <joint name="j2" type="hinge" axis="1 0 0"
                               range="-1.57 1.57" damping="1.2"/>
                        <!-- Elbow servo + link -->
                        <geom name="s2_geom" type="mesh" mesh="servo_j2"
                              rgba="0.8 0.5 0.2 1" mass="0.065"/>
                        <geom name="arm2_link" type="mesh" mesh="arm_mirror"
                              rgba="0.7 0.5 0.16 1" mass="0.25" pos="0 0 0.06"/>

                        <!-- ══ Arm segment 3: j3 wrist flex ══ -->
                        <body name="arm_3" pos="0 0 0.12">
                            <joint name="j3" type="hinge" axis="1 0 0"
                                   range="-1.57 1.57" damping="0.8"/>
                            <geom name="s3_geom" type="mesh" mesh="servo_j3"
                                  rgba="0.8 0.5 0.2 1" mass="0.065"/>
                            <geom name="arm3_link" type="mesh" mesh="arm_clip"
                                  rgba="0.6 0.45 0.14 1" mass="0.2" pos="0 0 0.04"/>

                            <!-- ══ Wrist: j4 roll ══ -->
                            <body name="arm_4" pos="0 0 0.08">
                                <joint name="j4" type="hinge" axis="0 0 1"
                                       range="-3.14 3.14" damping="0.6"/>
                                <geom name="wrist_s_geom" type="mesh" mesh="wrist_servo"
                                      rgba="0.6 0.4 0.12 1" mass="0.04"/>
                                <geom name="wrist_horn_geom" type="mesh" mesh="wrist_horn"
                                      rgba="0.5 0.35 0.1 1" mass="0.02"/>

                                <!-- ══ Gripper: j5 slide ══ -->
                                <body name="gripper" pos="0 0 0.03">
                                    <joint name="j5" type="slide"
                                           axis="1 0 0"
                                           range="0 0.04" damping="3.0"/>
                                    <!-- Gripper servo + moving jaw -->
                                    <geom name="gripper_base" type="mesh" mesh="servo_gripper"
                                          rgba="0.2 0.2 0.2 1" mass="0.1"/>
                                    <geom name="gripper_jaw" type="mesh" mesh="moving_jaw"
                                          rgba="0.3 0.3 0.3 1" mass="0.05" pos="0.02 0 0"/>
                                </body>

                                <!-- ══ Wrist camera ══ -->
                                <body name="wrist_cam" pos="0.01 0 -0.04">
                                    <geom name="wcm_geom" type="mesh" mesh="wrist_cam_mount"
                                          rgba="0.4 0.4 0.4 1" mass="0.03"/>
                                    <geom name="wcm_body" type="mesh" mesh="wrist_cam_body"
                                          rgba="0.35 0.35 0.35 1" mass="0.02"/>
                                    <!-- Wrist camera sensor: 80° FOV, follows arm_j4 rotation -->
                                    <camera name="wrist" pos="0.008 0 -0.018"
                                            xyaxes="1 0 0 0 1 0" fovy="80"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>

            <!-- Front camera sensor -->
            <camera name="front" pos="0 0 0.12"
                    xyaxes="1 0 0 0 1 0" fovy="60"/>
        </body>

        <!-- Target object -->
        <body name="target" pos="0.5 0 0.02">
            <geom name="target_geom" type="cylinder"
                  size="0.04 0.04" rgba="1 0.2 0.2 1"/>
        </body>
    </worldbody>

    <!-- Actuators: ctrl[0..5]=arm, ctrl[6..8]=wheels -->
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


# ── Helpers ─────────────────────────────────────────────────────────────────

ARM_JOINTS   = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINTS = ["w1", "w2", "w3"]
ALL_JOINTS   = ARM_JOINTS + WHEEL_JOINTS


def _jid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


# ── Simulation ───────────────────────────────────────────────────────────────

class LeKiWiSimURDF:
    """MuJoCo sim with real STL mesh geometry (hybrid: meshes + primitives)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(LEKIWI_URDF_XML)
        self.data  = mujoco.MjData(self.model)
        self._jpos_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}
        self._jvel_idx = {n: _jid(self.model, n) for n in ALL_JOINTS}
        self._target   = np.array([0.5, 0.0, 0.0])
        print(f"[LeKiWiSimURDF] bodies={self.model.nbody}, "
              f"meshes={self.model.nmesh}, joints={self.model.njnt}, "
              f"geoms={self.model.ngeom}")

    def _obs(self) -> np.ndarray:
        d = self.data
        return np.concatenate([
            d.qpos[:3], d.qpos[3:7], d.qvel[:3], d.qvel[3:6],
            np.array([d.qpos[self._jpos_idx[n]] for n in ARM_JOINTS]),
            np.array([d.qvel[self._jvel_idx[n]] for n in WHEEL_JOINTS]),
            [d.time],
            self._target[:2] - d.qpos[:2],
        ], dtype=np.float32)

    def _action_to_ctrl(self, action):
        arm   = np.clip(action[:6], -1, 1) * 3.14
        wheel = np.clip(action[6:9], -1, 1) * 5.0
        return np.concatenate([arm, wheel]).astype(np.float64)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._jpos_idx["j1"]] = 0.3
        self.data.qpos[self._jpos_idx["j2"]] = -0.3
        return self._obs()

    def step(self, action):
        self.data.ctrl[:] = self._action_to_ctrl(np.asarray(action, dtype=np.float32))
        mujoco.mj_step(self.model, self.data)
        return self._obs(), float(self._reward()), bool(self.data.time > 60), {}

    def _reward(self) -> float:
        return -float(np.linalg.norm(self._target[:2] - self.data.qpos[:2]))

    def render(self) -> Optional[np.ndarray]:
        """Render from front camera (640x480)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        r = mujoco.Renderer(self.model, 640, 480)
        r.update_scene(self.data, camera=cam_id)
        img = r.render()
        r.close()
        return img

    def render_wrist(self) -> Optional[np.ndarray]:
        """Render from wrist camera (640x480, follows arm_j4 rotation)."""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        r = mujoco.Renderer(self.model, 640, 480)
        r.update_scene(self.data, camera=cam_id)
        img = r.render()
        r.close()
        return img

    def render_window(self):
        return mujoco.viewer.launch_passive(self.model, self.data)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_physics():
    print("\n=== Physics Sanity ===")
    sim = LeKiWiSimURDF()
    sim.reset()
    for _ in range(100):
        sim.step(np.zeros(9, dtype=np.float32))
    print(f"  base pos: {sim.data.qpos[:3]}")
    print(f"  time:     {sim.data.time:.3f}s")
    print(f"  arm_j0:   {sim.data.qpos[sim._jpos_idx['j0']]:.3f}")
    print(f"  wheel_w1: {sim.data.qvel[sim._jvel_idx['w1']]:.3f}")
    ok = not (np.any(np.isnan(sim.data.qpos)) or np.any(np.isinf(sim.data.qpos)))
    print(f"  {'✓ Physics stable' if ok else '✗ NaN/Inf'}")
    return ok

def test_meshes():
    print("\n=== Mesh Loading ===")
    sim = LeKiWiSimURDF()
    names = ['base_p1', 's1_geom', 'gripper_j_geom', 'wcm_geom']
    all_ok = True
    for n in names:
        try:
            gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            print(f"  ✓ {n} id={gid}")
        except Exception as e:
            print(f"  ✗ {n}: {e}")
            all_ok = False
    img = sim.render()
    if img is not None:
        out = "/tmp/lekiwi_urdf_mesh_test.png"
        Image.fromarray(img).save(out)
        print(f"  ✓ Render saved: {out} ({img.shape})")
    return all_ok and img is not None

def test_camera():
    print("\n=== Camera ===")
    sim = LeKiWiSimURDF()
    sim.reset()
    img = sim.render()
    if img is not None:
        print(f"  ✓ front shape={img.shape}, dtype={img.dtype}")
    wrist_img = sim.render_wrist()
    if wrist_img is not None:
        print(f"  ✓ wrist  shape={wrist_img.shape}, dtype={wrist_img.dtype}")
    return (img is not None) and (wrist_img is not None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["physics", "meshes", "camera", "all"], default="all")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    ok = True
    if args.test in ("physics", "all"): ok &= test_physics()
    if args.test in ("meshes",  "all"): ok &= test_meshes()
    if args.test in ("camera",  "all"): ok &= test_camera()
    if args.render:
        print("\n=== Interactive Viewer ===")
        sim = LeKiWiSimURDF()
        sim.reset()
        sim.render_window()

    print("\n" + ("ALL TESTS PASSED ✓" if ok else "SOME TESTS FAILED ✗"))
