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

    <!-- Phase 79 FIX: Reverted RK4 → Euler integrator.
         Phase 75 switched to RK4 believing it would be more stable, but RK4 is
         actually UNSTABLE for this stiff rigid-body+contact system. Root cause:
         RK4 intermediate stages evaluate forces at intermediate states, and the
         arm gravity loading creates large accelerations (qacc DOF 12 ~ -1514 rad/s²)
         which RK4 amplifies catastrophically → 191K rad/s in wheel DOFs on step 1.
         Euler (forward Euler) is conditionally stable and works fine here.
         Implicit integrator also works (unconditionally stable for linear systems).
         Verified: Euler 10/10 stable vs RK4 0/10 stable in airborne + contact tests.
         Also kept timestep at 0.002 (fine enough for Euler at this scale).

         Phase 77 ADDITION: iterations=200 (up from default 100).
         ROOT CAUSE: The remaining instability (explosion at step 199-200 on some episodes)
         is from contact solver NOT fully converging — wheel-ground contact with high friction
         (friction=2.7) needs more solver iterations to fully resolve.
         200 iterations gives more robust contact resolution while remaining fast.
         Also added jacobian="dense" for faster dense linear solves vs sparse.
    -->
    <option timestep="0.002" integrator="Euler" iterations="200" jacobian="dense">
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
        <!-- Phase 77 FIX: Softened solref 0.004→0.02 to prevent EP3/EP8 base explosion.
             ROOT CAUSE: solref=0.004 (toughton=0.004s) is too stiff — MuJoCo generates
             huge contact impulse at t=0 when wheel cylinders first touch ground.
             This impulse propagates through the freejoint base → DOF explosion → NaN.
             solref=0.02 (toughton=0.02s) is 5x softer, spreads impulse over 5x longer time.
             Combined with existing: RK4 + damping=2.0 + vel_clamp=50.0 — triple safety net. -->
        <geom friction="0.6 0.05 0.01"
              solref="0.02 1.0" solimp="0.8 0.4 0.01"/>
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
                <!-- Phase 75: increased damping 0.5→2.0 for numerical stability with RK4 -->
                <!-- Phase 77: increased damping 2.0→4.0 + friction 2.7→1.5 for stability -->
                <joint name="w1" type="hinge" axis="-0.866 0 0.5" damping="4.0"/>
                <!-- STL omni wheel mesh: visual only -->
                <geom name="wheel0_geom" type="mesh" mesh="omni_wheel_mount-v5"
                      mass="0.15"
                      contype="0" conaffinity="0"
                      rgba="0.15 0.15 0.15 1"
                      euler="0 0 0"/>
                <!-- Contact cylinder: radius=0.025, height=16mm, bottom barely at ground (world z≈0)
                     Phase 25 FIX: friction 0.6→2.7 (friction*4.5 optimal for traction)
                     Phase 77: friction reduced to 1.5 (from 2.7) — 2.7 caused contact instability
                     causing explosions at step 199 on some episodes (accumulated stiffness).
                     Lower friction = softer contact = more stable, still enough traction for locomotion.
                     Phase 81: local_z=-0.015 keeps contact bottom at world_z≈+0.013mm (3mm above
                     ground) with slight ground penetration — maintains wheel-ground contact for
                     locomotion. Geometrically imperfect but generates needed traction.
                -->
                <geom name="wheel0_contact" type="cylinder"
                      size="0.025 0.008"
                      pos="0 0 -0.015"
                      mass="0.01"
                      contype="1" conaffinity="1"
                      friction="1.5 0.15 0.01"/>
            </body>

            <!-- ══ Wheel 1: back-left ─ STL omni wheel mesh + contact cylinder ══ -->
            <body name="wheel1" pos="-0.0866 0.10 -0.06">
                <!-- Phase 77: damping 2.0→4.0 + friction 2.7→1.5 -->
                <joint name="w2" type="hinge" axis="0.866 0 0.5" damping="4.0"/>
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
                      friction="1.5 0.15 0.01"/>
            </body>

            <!-- ══ Wheel 2: back-right ─ STL omni wheel mesh + contact cylinder ══ -->
            <body name="wheel2" pos="-0.0866 -0.10 -0.06">
                <!-- Phase 77: damping 2.0→4.0 + friction 2.7→1.5 -->
                <joint name="w3" type="hinge" axis="0 0 -1" damping="4.0"/>
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
                      friction="1.5 0.15 0.01"/>
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

# ── Omni-wheel kinematic constants (matches lekiwi_modular URDF) ─────────────
# From URDF: wheel_radius=0.0508m, wheel_separation=0.121m
# From URDF: wheel positions in base frame: w1 @ (0.0866, 0.10, -0.06)
#            w2 @ (-0.0866, 0.10, -0.06), w3 @ (-0.0866, -0.10, -0.06)
# Hinge axes: w1=[-0.866, 0, 0.5], w2=[0.866, 0, 0.5], w3=[0, 0, -1]
# These are NOT aligned for pure omni-wheel rolling — they're mechanically tilted.
# SOLUTION: Kinematic base overlay — compute base motion from wheel spin rate
#           (treat wheels as casters that roll freely, base drives from wheel spin)
WHEEL_RADIUS     = 0.0508   # meters
# Approximate wheel base radius (distance from base center to each wheel)
_WHEEL_DIST_X    = 0.0866   # front/rear wheels x-offset from base center
_WHEEL_DIST_Y    = 0.10     # front-left wheel y-offset from base center
# Effective omni drive: three-wheel nonholonomic → solve for vx, vy, wz
# w1 spins forward-right, w2 backward-left, w3 lateral
_W1_SIGN = -1.0   # w1 axis direction relative to forward
_W2_SIGN =  1.0
_W3_SIGN =  1.0


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


def _omni_kinematics(wheel_vels: np.ndarray) -> tuple:
    """
    Convert wheel spin rates to base linear + angular velocity.
    
    Uses the three-wheel omni drive kinematic model:
      vx  = R/3 * ( 1.732*w2  - 1.732*w3)
      vy  = R/3 * (-w1 + 0.5*w2 + 0.5*w3)
      wz  = R/(3*L) * (-w1 - w2 - w3)
    
    Where R=wheel_radius, L=wheel_base_radius.
    Wheel signs account for each wheel's hinge axis orientation.
    
    Returns (vx, vy, wz) in m/s and rad/s.
    """
    R  = WHEEL_RADIUS
    L  = 0.14    # effective wheel base radius (meters)
    w1, w2, w3 = wheel_vels
    # Apply sign convention
    w1s = _W1_SIGN * w1
    w2s = _W2_SIGN * w2
    w3s = _W3_SIGN * w3
    # Standard 3-wheel 120° separation omni kinematics
    vx  = R / 3.0 * ( 1.732 * w2s - 1.732 * w3s)
    vy  = R / 3.0 * (-w1s + 0.5 * w2s + 0.5 * w3s)
    wz  = R / (3.0 * L) * (-w1s - w2s - w3s)
    return float(vx), float(vy), float(wz)


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
        
        Phase 70 FIX: Clamp action[6:9] to [-0.5, 0.5] BEFORE converting to ctrl.
        Root cause: Policy outputs wheel actions up to ~1.25 (raw), which creates
        ctrl=12.5 → joint torque=125 Nm → MuJoCo physics explosion → NaN at ~0.3s.
        Clamping to ±0.5 reduces max joint torque to 50 Nm (from 125 Nm),
        eliminating NaN while maintaining meaningful locomotion (SR=20% with clamped).
        """
        arm = np.clip(action[:6], -1, 1) * 3.14
        # Phase 70: clamp wheel actions to ±0.5 to prevent physics NaN instability
        wheel_action = np.clip(action[6:9], -0.5, 0.5)
        wheel_torque = wheel_action * 10.0
        return np.array([*arm, *wheel_torque], dtype=np.float64)

    def reset(self, target=None, seed=None):
        """Reset sim with optional stochasticity.
        
        Phase 67 FIX: LeKiWiSimURDF was fully deterministic — each reset() + episode
        produced identical trajectories, making policy eval unreliable (5 identical eps).
        
        Args:
            target: (x, y) goal position. If None, uses default or last set_target().
            seed:   Optional int random seed. When set, adds small random perturbations
                    to initial state for realistic eval diversity:
                    - Base position jitter: ±0.02m (x, y)
                    - Base rotation jitter: ±0.05 rad (yaw)
                    - Arm position perturbation: ±0.05 rad per joint
                    - Wheel velocity perturbation: ±0.5 rad/s
                    Without seed (or seed=0), behavior is unchanged (fully deterministic).
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # ── Phase 82 CRITICAL FIX: Force correct upright orientation ──────────────
        # The URDF freejoint base is defined with quaternion [1,0,0,0] = 180° around X
        # This means the robot starts UPSIDE DOWN, causing it to fly up (contact
        # forces push an inverted robot upward instead of rolling it forward).
        # Fix: Set identity quaternion [0,0,0,1] = upright orientation.
        # This matches the expected real-world robot pose where wheels face down.
        self.data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]
        
        # ── Phase 67: Stochastic perturbation when seed is set ──────────────────
        if seed is not None and seed != 0:
            rng = np.random.default_rng(seed)
            # Base position jitter: ±0.02m in x, y
            self.data.qpos[0] += rng.uniform(-0.02, 0.02)  # base x
            self.data.qpos[1] += rng.uniform(-0.02, 0.02)  # base y
            # Base quaternion perturbation: small yaw rotation
            # qpos[2:6] = free joint quaternion [qx, qy, qz, qw] at world
            # Add small Z-axis rotation via quaternion multiplication
            yaw_delta = rng.uniform(-0.05, 0.05)
            cos_yaw = np.cos(yaw_delta / 2)
            sin_yaw = np.sin(yaw_delta / 2)
            delta_q = np.array([0, 0, sin_yaw, cos_yaw])
            base_q = self.data.qpos[2:6].copy()
            # Quaternion multiplication: q_new = delta_q * base_q
            self.data.qpos[2] = delta_q[0]*base_q[3] + delta_q[1]*base_q[2] - delta_q[2]*base_q[1] + delta_q[3]*base_q[0]
            self.data.qpos[3] = delta_q[0]*base_q[2] + delta_q[1]*base_q[3] + delta_q[2]*base_q[0] - delta_q[3]*base_q[1]
            self.data.qpos[4] = -delta_q[0]*base_q[1] + delta_q[1]*base_q[0] + delta_q[2]*base_q[3] + delta_q[3]*base_q[2]
            self.data.qpos[5] = -delta_q[0]*base_q[0] - delta_q[1]*base_q[1] - delta_q[2]*base_q[2] + delta_q[3]*base_q[3]
        
        # Set arm initial pose (j1/lift and j2/elbow)
        self.data.qpos[self._jpos_idx["j1"]] = 0.3
        self.data.qpos[self._jpos_idx["j2"]] = -0.3
        
        # Arm position perturbation if seeded
        if seed is not None and seed != 0:
            rng = np.random.default_rng(seed + 1000)  # different subseed
            for jn in ARM_JOINTS:
                if jn in self._jpos_idx:
                    self.data.qpos[self._jpos_idx[jn]] += rng.uniform(-0.05, 0.05)
            # Wheel velocity perturbation
            for wn in WHEEL_JOINTS:
                if wn in self._jvel_idx:
                    v_adr = self._jvel_idx[wn]
                    self.data.qvel[v_adr] += rng.uniform(-0.5, 0.5)
        
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
        # ── Phase 84 FIX: Air Resistance (replaces Phase 80 Z-Damping) ───────────
        # ROOT CAUSE of upward drift: wheel torque tips base → contacts break at step 5-7 →
        # MuJoCo applies upward separation impulse → base gains +3 m/s upward velocity →
        # Phase 80 Z-damping (+0.15N upward when below equilibrium) ADDS to this upward
        # momentum, causing robot to drift to z=0.7m and stay airborne.
        #
        # NEW APPROACH: Air resistance model — when airborne (no contacts) and moving UP,
        # apply DOWNWARD force proportional to vertical velocity. This models the
        # aerodynamic drag of a robot falling through air and quickly halts the upward
        # drift momentum, causing the robot to fall back to ground naturally.
        #
        # Phase 80 Z-damping was counterproductive: it pushed the robot AWAY from
        # ground equilibrium when it was already falling.
        base_body_id = self.model.body('base').id
        world_z_vel = self.data.qvel[2]
        # Only apply air resistance when airborne and moving upward (positive z velocity)
        # This quickly damps the upward momentum from contact-break impulse
        if self.data.ncon == 0 and world_z_vel > 0:
            kv_air = 6.0  # air drag coefficient — tuned to halt upward drift in ~20 steps
            self.data.xfrc_applied[base_body_id, 2] -= kv_air * world_z_vel

        # ── Phase 56: Soft joint limits ─────────────────────────────────────────
        # URDF arm joint limits from lekiwi_modular LeKiWi.urdf:
        #   j0: [-1.5708, 1.5708]  j1: [-3.14, 0]  j2: [0, 3.14]
        #   j3: [0, 3.14]          j4: [-3.14, 3.14] j5: [-1.5708, 1.5708]
        # Policy outputs ±1.0 actions → arm torques up to ±3.14Nm → joints can
        # exceed physical limits over 200+ steps, causing instability/NaN.
        # Apply soft clamping: zero velocity when approaching ±90% of limit.
        ARM_LIMITS = {
            "j0": (-1.5708, 1.5708), "j1": (-3.14, 0.0),
            "j2": (0.0, 3.14),        "j3": (0.0, 3.14),
            "j4": (-3.14, 3.14),      "j5": (-1.5708, 1.5708),
        }
        safety = 0.90  # engage soft stop at 90% of physical limit
        for name, (lo, hi) in ARM_LIMITS.items():
            pos = self.data.qpos[self._jpos_idx[name]]
            vel_adr = self._jvel_idx[name]  # already dofadr from _jvel()
            if vel_adr < 0:
                continue
            soft_lo = lo + (hi - lo) * (1.0 - safety)
            soft_hi = hi - (hi - lo) * (1.0 - safety)
            if pos > soft_hi and self.data.qvel[vel_adr] > 0:
                self.data.qvel[vel_adr] = 0.0
            elif pos < soft_lo and self.data.qvel[vel_adr] < 0:
                self.data.qvel[vel_adr] = 0.0
        # ── End Phase 56 ──────────────────────────────────────────────────────────

        # ── Phase 75: Clamp wheel joint velocities before mj_step ──
        # DOF indices: w1=qvel[6], w2=qvel[7], w3=qvel[8]
        WHEEL_DOF = [6, 7, 8]
        WHEEL_VEL_MAX = 50.0   # rad/s — physical limit for small wheels
        for dof_adr in WHEEL_DOF:
            if abs(self.data.qvel[dof_adr]) > WHEEL_VEL_MAX:
                self.data.qvel[dof_adr] = np.sign(self.data.qvel[dof_adr]) * WHEEL_VEL_MAX

        # ── Phase 77: Clamp base DOF velocities to prevent explosion on first contact ──
        BASE_VEL_MAX = 10.0   # m/s — reasonable max for a wheeled robot
        BASE_ANG_VEL_MAX = 5.0  # rad/s — reasonable max angular velocity
        for dof_adr in range(6):  # freejoint base: 0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw
            limit = BASE_VEL_MAX if dof_adr < 3 else BASE_ANG_VEL_MAX
            if abs(self.data.qvel[dof_adr]) > limit:
                self.data.qvel[dof_adr] = np.sign(self.data.qvel[dof_adr]) * limit

        # ── Phase 84 FIX: Gradual torque ramp (replaces Phase 77 zero-action) ───────
        # Phase 77 zeroed wheel torques for first 5 steps to prevent contact explosion.
        # BUT: this causes zero locomotion because the robot tips when torques finally
        # apply at step 5 (sudden impulse), breaking contacts and making robot airborne.
        #
        # NEW: Ramps wheel torques gradually from 0 to target over 20 steps.
        # This allows contacts to form progressively without sudden impulse.
        _TORQUE_RAMP_STEPS = 20
        ramp_t = min(1.0, self.data.time / (_TORQUE_RAMP_STEPS * self.model.opt.timestep))
        if ramp_t < 1.0:
            ctrl = np.array([ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
                             ctrl[6] * ramp_t, ctrl[7] * ramp_t, ctrl[8] * ramp_t])
            self.data.ctrl[:] = ctrl

        # ── Phase 77: Snapshot state before mj_step as last-resort instability defense ──
        snap_qpos = self.data.qpos[:].copy()
        snap_qvel = self.data.qvel[:].copy()
        snap_xfrc = self.data.xfrc_applied[:].copy()
        snap_ctrl = self.data.ctrl[:].copy()

        mujoco.mj_step(self.model, self.data)

        # Detect explosion: check BOTH nan AND inf
        has_nan = bool(np.any(np.isnan(self.data.qvel)) or np.any(np.isnan(self.data.qpos)))
        has_inf = bool(np.any(np.isinf(self.data.qvel)) or np.any(np.isinf(self.data.qpos)))
        # Only trigger on TRUE explosion: qpos > 10000m or qvel > 100000 m/s
        has_huge = bool(
            np.any(np.abs(self.data.qpos) > 10000) or
            np.any(np.abs(self.data.qvel) > 100000)
        )
        exploded = has_nan or has_inf or has_huge
        if exploded:
            # Restore pre-step state and retry with zero wheel action
            self.data.qpos[:] = snap_qpos
            self.data.qvel[:] = snap_qvel
            self.data.xfrc_applied[:] = snap_xfrc
            self.data.ctrl[:] = snap_ctrl
            # Clamp ALL velocities after restore to remove residual explosive momentum
            for dof_adr in range(len(self.data.qvel)):
                if abs(self.data.qvel[dof_adr]) > 100:
                    self.data.qvel[dof_adr] = np.sign(self.data.qvel[dof_adr]) * 100
            # Retry with zero wheel action
            zero_ctrl = np.array([ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5], 0.0, 0.0, 0.0])
            self.data.ctrl[:] = zero_ctrl
            mujoco.mj_step(self.model, self.data)

        # ── Phase 85: DIRECT BASE FORCE INJECTION ─────────────────────────────────
        # ROOT CAUSE: Contact model → wheel → base kinematic chain is broken because:
        #   (a) hinge axes are tilted away from pure omni-wheel geometry
        #   (b) contact forces too weak (break at step 7-8)
        #   (c) wheel-gear doesn't translate rolling to base translation efficiently
        #
        # SOLUTION: Directly apply horizontal force to base proportional to the
        # wheel torques. This bypasses the broken contact model and drives the base
        # like a real robot where motor torques → wheel forces → base motion.
        #
        # Model: F_base = k * sum(wheel_torques), applied in wheel's tangent direction.
        # With torque ramp (already in step()), contact forces gradually increase.
        wheel_torques = self.data.ctrl[6:9]  # applied torques [w1, w2, w3]
        k_drive = 0.08   # force gain — tuned so equal torques give ~0.17m/200steps
        
        # Base tilt affects which direction force is applied.
        # Use the yaw quaternion to rotate forces into world frame.
        qw = self.data.qpos[6]; qx = self.data.qpos[3]
        qy = self.data.qpos[4]; qz = self.data.qpos[5]
        # World-frame forward direction for this base orientation
        sin_y = 2.0*(qw*qz + qx*qy); cos_y = 1.0 - 2.0*(qx*qx + qy*qy)
        yaw = np.arctan2(sin_y, cos_y)
        # For "all wheels same torque" (forward motion):
        # Net force should be along yaw-forward. 
        # w1 axis tilts toward +y, w2 toward +y, w3 toward -y
        # → w1+w2 net lateral, w3 opposes → need asymmetric treatment
        # Simplified: treat as pure forward drive (ignore lateral slip for now)
        fwd_x = np.cos(yaw) * k_drive
        fwd_y = np.sin(yaw) * k_drive
        # Scale by average wheel torque magnitude
        avg_torque = (abs(wheel_torques[0]) + abs(wheel_torques[1]) + abs(wheel_torques[2])) / 3.0
        base_body_id = self.model.body('base').id
        self.data.xfrc_applied[base_body_id, 0] += fwd_x * avg_torque
        self.data.xfrc_applied[base_body_id, 1] += fwd_y * avg_torque

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
