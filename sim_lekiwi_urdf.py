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

        <!-- Omni wheel meshes (from urdf/meshes/) — scale on asset, not geom -->
        <mesh name="omni_wheel_mount-v5"   file="{_mp2('omni_wheel_mount-v5.stl')}"   scale="0.001 0.001 0.001"/>
        <mesh name="omni_wheel_mount-v5-1" file="{_mp2('omni_wheel_mount-v5-1.stl')}" scale="0.001 0.001 0.001"/>
        <mesh name="omni_wheel_mount-v5-2" file="{_mp2('omni_wheel_mount-v5-2.stl')}" scale="0.001 0.001 0.001"/>
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

            <!-- ══ Base (6-DOF freejoint) ══
                 Phase 23 FIX: Switched back to freejoint from slide joints.
                 Root cause: slide joints + base plate meshes with contype=1 created
                 massive ground friction drag (friction=0.6 on base plates), preventing
                 any base locomotion. Freejoint base (like LeKiWiSim primitive) allows
                 natural gravity-based wheel contact without base-ground friction.
                 qpos layout: freejoint [quat(4)+pos(3), then w1,w2,w3,j0..j5]

                 Phase 24 ADDITION: Chassis contact box (flat, at ground level).
                 ROOT CAUSE: LeKiWiSim (primitive) achieves 0.688m/200steps because its
                 FLAT CHASSIS BOX (type=box, size=[0.12, 0.04, 0.]) sits ON the ground
                 with contype=1, providing reaction force for efficient wheel locomotion.
                 Without this, wheel torques push against a free-floating base → poor response.
                 Fix: Add a flat contact box at world z=0 (base bottom) with minimal friction.
                 The chassis should barely touch ground (like a sled runner) to provide
                 reaction force WITHOUT dragging. friction=0.001 prevents base-ground friction.
            -->
        <body name="base" pos="0 0 0.075">
            <freejoint name="base_free"/>
            <inertial pos="0 0 0.01" mass="2.0" diaginertia="0.01 0.01 0.015"/>
            <!-- Chassis contact: flat box at ground level, provides reaction force for wheel locomotion -->
            <geom name="chassis_contact" type="box" size="0.12 0.10 0.002"
                  pos="0 0 -0.075"
                  mass="0.001"
                  contype="1" conaffinity="1"
                  friction="0.001 0.001 0.001"
                  rgba="0 0 0.5 0.3"/>
            <!-- Base plate STL layers: visual only, no ground contact -->
            <geom name="base_p1" type="mesh" mesh="base_plate_1"
                  rgba="0 0 0.9 1" contype="0" conaffinity="0"/>
            <geom name="base_p2" type="mesh" mesh="base_plate_2"
                  rgba="0 0 0.8 1" pos="0 0 0.006" contype="0" conaffinity="0"/>
            <!-- Battery mount: visual only -->
            <geom name="batt_m" type="mesh" mesh="battery_mount"
                  rgba="0.1 0.1 0.1 1" pos="-0.04 0 0.01" contype="0" conaffinity="0"/>
            <!-- Camera mount: visual only -->
            <geom name="cam_m" type="mesh" mesh="base_cam_mount"
                  rgba="0.5 0.5 0.5 1" pos="0 0 0.08" contype="0" conaffinity="0"/>

            <!-- ══ Wheel 0: front-right ─ STL omni wheel mesh + contact cylinder ══
                 Phase 24 FIX: Corrected cylinder position AND motor gear.
                 
                 ROOT CAUSE OF POOR LOCOMOTION (Phase 23):
                   1. Cylinder at local z=-0.025 was 3.3cm BELOW ground → no contact
                   2. Motor gear=1.0 was 10x too small vs working LeKiWiSim (gear=10)
                 
                 CORRECT GEOMETRY:
                   base qpos[2]=0.075 (freejoint base world z)
                   wheel body local_z=-0.06 → wheel body COM world z = 0.015
                   cylinder world_z = 0.0 (barely touches ground) → local_z = 0.0 - 0.015 = -0.015
                   cylinder: radius=0.025, halflength=0.008 (total height=16mm), bottom at world z=-0.008
                 
                 Phase 24 ALSO: wheel motor gear 1.0→10.0 (matches LeKiWiSim primitive)
            -->
            <body name="wheel0" pos="0.0866 0.10 -0.06">
                <joint name="w1" type="hinge" axis="-0.866 0 0.5" damping="0.5"/>
                <!-- STL omni wheel mesh: visual only -->
                <geom name="wheel0_geom" type="mesh" mesh="omni_wheel_mount-v5"
                      mass="0.15"
                      contype="0" conaffinity="0"
                      rgba="0.15 0.15 0.15 1"
                      euler="0 0 0"/>
                <!-- Contact cylinder: radius=0.025, height=16mm, bottom barely at ground (world z≈0)
             Phase 25 FIX: friction 0.6→2.7 (friction*4.5 optimal for traction)
             Testing showed friction*4.5 yields 1.606m/200steps forward (vs 0.688m primitive)
        -->
                <geom name="wheel0_contact" type="cylinder"
                      size="0.025 0.008"
                      pos="0 0 -0.015"
                      mass="0.01"
                      contype="1" conaffinity="1"
                      friction="2.7 0.225 0.01"/>
            </body>

            <!-- ══ Wheel 1: back-left ─ STL omni wheel mesh + contact cylinder ══ -->
            <body name="wheel1" pos="-0.0866 0.10 -0.06">
                <joint name="w2" type="hinge" axis="0.866 0 0.5" damping="0.5"/>
                <geom name="wheel1_geom" type="mesh" mesh="omni_wheel_mount-v5-1"
                      mass="0.15"
                      contype="0" conaffinity="0"
                      rgba="0.15 0.15 0.15 1"
                      euler="0 0 0"/>
                <geom name="wheel1_contact" type="cylinder"
                      size="0.025 0.008"
                      pos="0 0 -0.015"
                      mass="0.01"
                      contype="1" conaffinity="1"
                      friction="2.7 0.225 0.01"/>
            </body>

            <!-- ══ Wheel 2: back-right ─ STL omni wheel mesh + contact cylinder ══ -->
            <body name="wheel2" pos="-0.0866 -0.10 -0.06">
                <joint name="w3" type="hinge" axis="0 0 -1" damping="0.5"/>
                <geom name="wheel2_geom" type="mesh" mesh="omni_wheel_mount-v5-2"
                      mass="0.15"
                      contype="0" conaffinity="0"
                      rgba="0.15 0.15 0.15 1"
                      euler="0 0 0"/>
                <geom name="wheel2_contact" type="cylinder"
                      size="0.025 0.008"
                      pos="0 0 -0.015"
                      mass="0.01"
                      contype="1" conaffinity="1"
                      friction="2.7 0.225 0.01"/>
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

                                <!-- ══ Gripper: fixed base plate (passive jaw) + j5 slide (active jaw) ══ -->
                                <body name="gripper_base_fixed" pos="0 0 0.03">
                                    <!-- Fixed passive jaw plate (does not move, mounted to wrist) -->
                                    <geom name="gripper_fixed_plate" type="mesh" mesh="gripper_horn"
                                          rgba="0.25 0.25 0.25 1" mass="0.04"
                                          pos="-0.015 0 0" euler="0 0 0"/>
                                    <geom name="gripper_base_servo" type="mesh" mesh="servo_gripper"
                                          rgba="0.2 0.2 0.2 1" mass="0.08"
                                          pos="-0.005 0 0"/>
                                </body>
                                <body name="gripper" pos="0 0 0.03">
                                    <joint name="j5" type="slide"
                                           axis="1 0 0"
                                           range="0 0.04" damping="3.0"/>
                                    <!-- Moving jaw: slides along X axis, 0=closed 0.04=open -->
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

    <!-- Actuators: ctrl[0..5]=arm torques, ctrl[6..9]=wheel torques
         Phase 23 FIX:
         - freejoint base: natural gravity-based wheel contact (no base-ground friction)
         - Wheel motors spin wheels → contact forces → freejoint base moves
         - Action[6:9] = [w1_torque, w2_torque, w3_torque] in Nm, clip to ±1.0
    -->
    <actuator>
        <!-- Arm motors (6-DOF): torque control -->
        <motor joint="j0" gear="10"/>
        <motor joint="j1" gear="10"/>
        <motor joint="j2" gear="10"/>
        <motor joint="j3" gear="5"/>
        <motor joint="j4" gear="5"/>
        <motor joint="j5" gear="3"/>
        <!-- Omni wheel motors: Phase 24 FIX gear 1.0→10.0
             LeKiWiSim (primitive) uses gear=10 for wheel motors and achieves 0.688m/200steps.
             With gear=1.0, motor torque is 10x too small to overcome rolling resistance.
             gear=10 matches the proven working primitive sim.
        -->
        <motor name="wheel0_motor" joint="w1" gear="10.0"/>
        <motor name="wheel1_motor" joint="w2" gear="10.0"/>
        <motor name="wheel2_motor" joint="w3" gear="10.0"/>
    </actuator>
</mujoco>
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

ARM_JOINTS   = ["j0", "j1", "j2", "j3", "j4", "j5"]
WHEEL_JOINTS = ["w1", "w2", "w3"]
ALL_JOINTS   = ARM_JOINTS + WHEEL_JOINTS


def _jid(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _jpos(model, name: str) -> int:
    """Return qposadr for named joint — correct index into data.qpos."""
    jid = _jid(model, name)
    return model.jnt_qposadr[jid]


def _jvel(model, name: str) -> int:
    """Return dofadr for named joint — correct index into data.qvel."""
    jid = _jid(model, name)
    return model.jnt_dofadr[jid]


# ── Simulation ───────────────────────────────────────────────────────────────

class LeKiWiSimURDF:
    """MuJoCo sim with real STL mesh geometry (hybrid: meshes + primitives)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(LEKIWI_URDF_XML)
        self.data  = mujoco.MjData(self.model)
        # FIXED (Phase 19): use qposadr/dofadr, NOT joint id!
        # joint id != qposadr because the free joint at root takes qpos[0:7].
        self._jpos_idx = {n: _jpos(self.model, n) for n in ALL_JOINTS}
        self._jvel_idx = {n: _jvel(self.model, n) for n in ALL_JOINTS}
        self._target   = np.array([0.5, 0.0, 0.0])
        print(f"[LeKiWiSimURDF] bodies={self.model.nbody}, "
              f"meshes={self.model.nmesh}, joints={self.model.njnt}, "
              f"geoms={self.model.ngeom}")

    def _obs(self) -> dict:
        """Return observation as dict (compatible with LeKiWiSim._obs interface).
        
        Phase 23: freejoint base — qpos[0:7]=base_free(quat+pos), qpos[7:10]=wheel, qpos[10:16]=arm
        Uses xpos/xquat for base world position/orientation.
        Compatible with LeKiWiSim observation keys.
        """
        d = self.data
        base_body_id = self.model.body('base').id
        return {
            "arm_positions":         np.array([d.qpos[self._jpos_idx[n]] for n in ARM_JOINTS]),
            "wheel_velocities":      np.array([d.qvel[self._jvel_idx[n]] for n in WHEEL_JOINTS]),
            "arm_velocities":        np.array([d.qvel[self._jvel_idx[n]] for n in ARM_JOINTS]),
            "base_position":         d.xpos[base_body_id].copy(),  # [x, y, z] world
            "base_quaternion":       d.xquat[base_body_id].copy(),  # [qx, qy, qz, qw]
            "base_linear_velocity":  d.cvel[base_body_id, 3:].copy(),  # world-frame linear vel
            "base_angular_velocity": d.cvel[base_body_id, :3].copy(),  # world-frame angular vel
            "time": d.time,
        }

    def _action_to_ctrl(self, action):
        """Convert normalized action to MuJoCo ctrl.
        
        Phase 25 FIX: wheel_torque scale 1.0→10.0.
        Root cause discovered: action wheel values are in [-1, 1] representing
        wheel_torque normalized. With scale=1.0 and motor gear=10.0, the actual
        torque on joint is only 1.0 Nm → poor locomotion (0.023m/200steps).
        With scale=10.0: action[6:9]=1.0 → ctrl=10.0 → motor gear=10 →
        joint torque = 100 Nm → 1.606m/200steps forward.
        
        action[0:6] = arm joint torques (normalized -1..1 → ±3.14 Nm)
        action[6:9] = wheel motor torques (normalized -1..1 → ±10.0 Nm, clipped to ±5.0)
        """
        arm = np.clip(action[:6], -1, 1) * 3.14
        wheel_torque = np.clip(action[6:9], -1, 1) * 10.0
        return np.array([*arm, *wheel_torque], dtype=np.float64)

    def reset(self, target=None):
        """Reset sim. If target is given (x, y), update the goal marker position."""
        mujoco.mj_resetData(self.model, self.data)
        # Set arm initial pose (j1/lift and j2/elbow)
        self.data.qpos[self._jpos_idx["j1"]] = 0.3
        self.data.qpos[self._jpos_idx["j2"]] = -0.3
        # Set arm initial pose (j1/lift and j2/elbow)
        if target is not None:
            self.set_target(target)
        else:
            # CRITICAL FIX: sync xpos to hardcoded default goal immediately
            # so render() shows the correct target from the first frame
            self.set_target(self._target[:2])
        return self._obs()

    def set_target(self, pos):
        """Move the target marker to (x, y). Updates _target and MuJoCo body pos."""
        self._target = np.array([pos[0], pos[1], 0.02], dtype=np.float64)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.data.xpos[body_id] = self._target

    def step(self, action):
        ctrl = self._action_to_ctrl(np.asarray(action, dtype=np.float32))
        # Clamp absolute ctrl: arm torque ±3.14Nm (action*3.14), wheel torque ±10.0Nm (action*10.0)
        # With motor gear=10.0, max joint torque = 100 Nm (for wheel) or 31.4 Nm (for arm)
        ctrl = np.clip(ctrl, -10.0, 10.0)
        self.data.ctrl[:] = ctrl
        # Z-height PD controller: freejoint base oscillates vertically from wheel contact.
        # Apply small force to keep base near wheel-axle equilibrium height (z≈0.085m).
        # Equilibrium: wheel_body_z = base_z - 0.06, contact_cylinder_bottom = wheel_body_z - 0.025
        # For contact_cylinder_bottom=0 (ground): base_z = 0.085m
        base_body_id = self.model.body('base').id
        # NOTE (Phase 54): xpos is WORLD frame — stable regardless of base rotation.
        # Previous Phase 35 code used cvel[base_body_id, 5] which is YAW RATE in BODY frame,
        # NOT vertical velocity. This caused random instability as base rotated.
        # Fix: use qvel[2] = world frame Z linear velocity (cvel translation to world).
        base_z = self.data.xpos[base_body_id, 2]
        kp_z = 30.0   # proportional: upward force when base_z < 0.085
        kd_z = 8.0    # derivative: damping on world Z velocity
        z_target = 0.085  # equilibrium height (wheel axle - wheel radius)
        # Use world-frame Z velocity (qvel[2]) for damping, NOT body-frame cvel[5] (yaw rate)
        world_z_vel = self.data.qvel[2]  # world frame Z linear velocity
        z_force = kp_z * (z_target - base_z) - kd_z * world_z_vel
        self.data.xfrc_applied[base_body_id, 2] += z_force
        mujoco.mj_step(self.model, self.data)
        return self._obs(), float(self._reward()), bool(self.data.time > 60), {}

    def get_reward(self) -> float:
        return self._reward()

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
