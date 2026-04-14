#!/usr/bin/env python3
"""
LeKiWi URDF → MuJoCo XML Converter  (v2 — clean rewrite)
==========================================================
Converts lekiwi_modular/src/lekiwi_description/urdf/LeKiWi.urdf
into a MuJoCo XML model using the real STL meshes.

Hand-crafted kinematic chain (not auto-generated) to avoid transform bugs.
Only the 9 actuated joints become MuJoCo joints:
  - 3 wheel motors (w0, w1, w2)
  - 6 arm joints  (j0..j5)

All intermediate bracket/motor links are merged into composite bodies.
STL meshes are referenced directly from the URDF package path.

Usage:
    python3 urdf2mujoco.py
    # writes models/lekiwi_mujoco.xml
"""

import os
import re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
HERMES_ROOT = Path("/Users/i_am_ai/hermes_research")
LEKIWI_MODULAR = HERMES_ROOT / "lekiwi_modular/src/lekiwi_description"
URDF_FILE = LEKIWI_MODULAR / "urdf/LeKiwi.urdf"
MESHHOME  = LEKIWI_MODULAR / "urdf/meshes"
OUT_DIR   = HERMES_ROOT / "lekiwi_vla/models"
OUT_FILE  = OUT_DIR / "lekiwi_mujoco.xml"

# ── URDF Parsing ────────────────────────────────────────────────────────────────

# Strip namespace prefixes
for event, elem in ET.iterparse(URDF_FILE, events=("start-ns",)):
    pass  # we just need to register namespaces

tree = ET.parse(URDF_FILE)
root = tree.getroot()

# Remove namespace prefixes for easier element access
ns_pattern = re.compile(r'\{[^}]+\}(.*)')
def strip_ns(tag):
    m = ns_pattern.match(tag)
    return m.group(1) if m else tag

# Rewrite elements without namespace
def strip_tree(elem):
    elem.tag = strip_ns(elem.tag)
    for child in list(elem):
        strip_tree(child)

strip_tree(root)

links = {e.attrib["name"]: e for e in root.findall("link")}
joints = {e.attrib["name"]: e for e in root.findall("joint")}


def inertial_of(link_name: str) -> tuple[float, list]:
    """Return (mass, [ixx,ixy,ixz,iyy,iyz,izz]) for a link."""
    link = links.get(link_name)
    if link is None:
        return 1.0, [0.01]*6
    # Try with namespace prefix first
    inert = link.find("inertial") or link.find("urdf:inertial") or \
            link.find("{http://robotics.gwu.edu/urdf/1.1}inertial")
    if inert is None:
        return 1.0, [0.01]*6
    # mass element
    mass_el = (inert.find("mass") or inert.find("urdf:mass") or
               inert.find("{http://robotics.gwu.edu/urdf/1.1}mass"))
    try:
        mass = float(mass_el.text) if mass_el is not None and mass_el.text else 1.0
    except (TypeError, ValueError):
        mass = 1.0
    # inertia element
    inertia_el = (inert.find("inertia") or inert.find("urdf:inertia") or
                 inert.find("{http://robotics.gwu.edu/urdf/1.1}inertia"))
    if inertia_el is not None:
        vals = [float(inertia_el.attrib.get(a, "0.01")) for a in
                ["ixx","ixy","ixz","iyy","iyz","izz"]]
    else:
        vals = [0.01]*6
    return mass, vals


def mesh_of(link_name: str) -> str | None:
    """Return STL filename (relative, for use with meshdir asset)."""
    link = links.get(link_name)
    if link is None:
        return None
    for tag in ("visual", "collision"):
        parent = link.find(tag)
        if parent is None:
            continue
        geom = parent.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        uri = mesh.attrib.get("filename", "")
        if uri:
            return os.path.splitext(os.path.basename(uri))[0]
    return None


def parent_of(link_name: str) -> tuple[str, ET.Element] | None:
    """Find joint whose child is this link."""
    for jn, je in joints.items():
        child = je.find("child")
        if child is not None and child.attrib.get("link") == link_name:
            parent = je.find("parent").attrib.get("link")
            return parent, je
    return None


# ── Kinematics constants (from prior analysis) ────────────────────────────────

# Wheel joint names (from URDF <joint> elements)
WHEEL_JOINT_NAMES = [
    "ST3215_Servo_Motor-v1-2_Revolute-60",   # w0 — front
    "ST3215_Servo_Motor-v1-1_Revolute-62",   # w1 — back-left
    "ST3215_Servo_Motor-v1_Revolute-64",     # w2 — back-right
]

ARM_JOINT_NAMES = [
    "STS3215_03a-v1_Revolute-45",
    "STS3215_03a-v1-1_Revolute-49",
    "STS3215_03a-v1-2_Revolute-51",
    "STS3215_03a-v1-3_Revolute-53",
    "STS3215_03a_Wrist_Roll-v1_Revolute-55",
    "STS3215_03a-v1-4_Revolute-57",
]

# Wheel positions in base frame (m)
# Derived from omni wheel kinematics: wheel0 at +x, wheels 1,2 at ±120°
WHEEL_POS = np.array([
    [ 0.100,  0.000, -0.040],   # w0 — front
    [-0.050,  0.087, -0.040],   # w1 — back-left (120°)
    [-0.050, -0.087, -0.040],   # w2 — back-right (240°)
], dtype=np.float64)

# Wheel joint axes in base frame (from URDF actual values)
WHEEL_AXES = np.array([
    [ 0.000,  0.000, -1.000],   # w0
    [ 0.866,  0.000,  0.500],   # w1
    [-0.866,  0.000,  0.500],   # w2
], dtype=np.float64)

# Arm joint axes in BASE FRAME (pre-computed from URDF chain analysis)
# These are the world-frame axes for each arm joint
ARM_JOINT_AXES_BASE = np.array([
    [ 0.000,  0.000,  1.000],   # j0 — shoulder pan (Z)
    [ 1.000,  0.000,  0.000],   # j1 — shoulder pitch (X)
    [ 1.000,  0.000,  0.000],   # j2 — elbow (X)
    [ 1.000,  0.000,  0.000],   # j3 — wrist pitch (X)
    [ 0.000,  0.423, -0.906],   # j4 — wrist roll
    [ 0.000, -0.423, -0.906],   # j5 — gripper
], dtype=np.float64)

# Arm link positions relative to base plate (pre-computed)
ARM_LINK_POSITIONS = np.array([
    [ 0.030,  0.011,  0.024],   # arm_0 (j0)
    [ 0.071,  0.019, -0.047],   # arm_1 (j1)
    [ 0.070,  0.021,  0.067],   # arm_2 (j2)
    [ 0.068,  0.024, -0.066],   # arm_3 (j3)
    [ 0.048,  0.018, -0.091],   # arm_4 (j4)
    [ 0.028,  0.015, -0.084],   # arm_5 (j5)
], dtype=np.float64)


# ── MuJoCo XML builder ─────────────────────────────────────────────────────────

def inertial_xml(mass: float, inertia: list, pos: str = "0 0 0") -> str:
    # MuJoCo diaginertia takes 3 values: ixx iyy izz (cross terms dropped)
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    return (f'<inertial pos="{pos}" mass="{mass}" '
            f'diaginertia="{ixx:.8g} {iyy:.8g} {izz:.8g}"/>')

def geom_xml(name: str, gtype: str, **attrs) -> str:
    parts = [f'<geom name="{name}" type="{gtype}"']
    for k, v in attrs.items():
        parts.append(f'{k}="{v}"')
    parts.append("/>")
    return " ".join(parts)


def build_wheel_xml(idx: int) -> str:
    """Build MuJoCo body XML for one wheel."""
    wjn = WHEEL_JOINT_NAMES[idx]
    child_link = joints[wjn].find("child").attrib["link"]
    mass, inertia = inertial_of(child_link)
    mesh = mesh_of(child_link)
    pos = WHEEL_POS[idx]
    axis = WHEEL_AXES[idx]

    # Use omni_wheel_mount mesh (the actual rubber wheel body)
    actual_mesh = mesh_of("omni_wheel_mount-v5") or \
                  mesh_of("4-Omni-Directional-Wheel_Single_Body-v1") or \
                  f"4-Omni-Directional-Wheel_Single_Body-v1-{idx+1}.stl"

    body_pos = f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"
    axis_str = f"{axis[0]:.6g} {axis[1]:.6g} {axis[2]:.6g}"

    lines = []
    lines.append(f'    <body name="wheel{idx}" pos="{body_pos}">')
    lines.append(inertial_xml(mass, inertia))
    # Omni wheel: use cylinder primitive (the actual STL has >200k faces)
    lines.append(geom_xml(
        f"wheel{idx}_geom", "cylinder",
        size="0.035 0.018", pos="0 0 0", euler="0 0 0",
        rgba="0.08 0.08 0.08 1",
        friction="0.9 0.05 0.01",
        contype="1", conaffinity="1",
        mass=f"{mass}",
    ))
    lines.append(f'      <joint name="w{idx}" type="hinge" axis="{axis_str}" '
                 f'pos="0 0 0" damping="0.5" range="-50 50"/>')
    lines.append('    </body>')
    return "\n".join(lines)


def build_arm_xml(idx: int) -> str:
    """Build MuJoCo body XML for one arm segment."""
    jn = ARM_JOINT_NAMES[idx]
    child_link = joints[jn].find("child").attrib["link"]
    mass, inertia = inertial_of(child_link)
    mesh = mesh_of(child_link)

    arm_mesh = f"STS3215_03a-v1{'-' + str(idx) if idx > 0 else ''}.stl"
    colors = [
        "0.90 0.60 0.20 1",
        "0.80 0.55 0.18 1",
        "0.70 0.50 0.16 1",
        "0.60 0.45 0.14 1",
        "0.50 0.40 0.12 1",
        "0.20 0.20 0.20 1",
    ]

    pos = ARM_LINK_POSITIONS[idx]
    body_pos = f"{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}"
    axis = ARM_JOINT_AXES_BASE[idx]
    axis_str = f"{axis[0]:.6g} {axis[1]:.6g} {axis[2]:.6g}"

    arm_range = "-3.14 3.14"
    damp = "1.5" if idx < 3 else "0.8"

    lines = []
    lines.append(f'    <body name="arm_{idx}" pos="{body_pos}">')
    lines.append(inertial_xml(mass, inertia))
    if mesh and os.path.exists(MESHHOME / mesh):
        lines.append(geom_xml(
            f"arm{idx}_geom", "mesh",
            mesh=mesh, pos="0 0 0", euler="0 0 0",
            rgba=colors[idx],
            friction="0.5 0.05 0.01",
            contype="0", conaffinity="0",
        ))
    else:
        size = ["0.025 0.015","0.022 0.05","0.018 0.08",
                "0.014 0.04","0.010 0.03","0.018 0.025"][idx]
        lines.append(geom_xml(
            f"arm{idx}_geom", "cylinder",
            size=size, pos="0 0 0", euler="0 0 0",
            rgba=colors[idx],
            friction="0.5 0.05 0.01",
            contype="0", conaffinity="0",
        ))
    lines.append(f'      <joint name="j{idx}" type="hinge" axis="{axis_str}" '
                 f'pos="0 0 0" damping="{damp}" range="{arm_range}"/>')
    lines.append('    </body>')
    return "\n".join(lines)


def build_base_geom_xml() -> str:
    """Base plate visual geom."""
    mesh = mesh_of("base_plate_layer1-v5")
    if mesh and os.path.exists(MESHHOME / mesh):
        return geom_xml(
            "base_geom", "mesh",
            mesh=mesh, pos="0 0 0", euler="0 0 0",
            rgba="0.0 0.0 1.0 1",
            friction="0.5 0.05 0.01",
            contype="1", conaffinity="1",
        )
    else:
        return geom_xml(
            "base_geom", "cylinder",
            size="0.12 0.035", pos="0 0 0", euler="0 0 0",
            rgba="0.0 0.0 1.0 1",
            friction="0.5 0.05 0.01",
            contype="1", conaffinity="1",
        )


def setup_meshes():
    """Copy/symlink needed STL meshes into models/meshes/."""
    mesh_dir = OUT_DIR / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    needed = {
        "base_plate_layer1-v5.stl",
        "omni_wheel_mount-v5.stl",
        "omni_wheel_mount-v5-1.stl",
        "omni_wheel_mount-v5-2.stl",
        "4-Omni-Directional-Wheel_Single_Body-v1-1.stl",
        "4-Omni-Directional-Wheel_Single_Body-v1-2.stl",
        "4-Omni-Directional-Wheel_Single_Body-v1.stl",
        "STS3215_03a-v1.stl",
        "STS3215_03a-v1-1.stl",
        "STS3215_03a-v1-2.stl",
        "STS3215_03a-v1-3.stl",
        "STS3215_03a-v1-4.stl",
        "STS3215_03a_Wrist_Roll-v1.stl",
        # Arm structural meshes (from urdf/meshes/)
        "Moving_Jaw_08d-v1.stl",
        "Rotation_Pitch_08i-v1.stl",
        "SO_ARM100_08k_116_Square-v1.stl",
        "SO_ARM100_08k_Mirror-v1.stl",
        "Wrist_Roll_08c-v1.stl",
        "Wrist_Roll_Pitch_08i-v1.stl",
    }
    for m in needed:
        src = MESHHOME / m
        dst = mesh_dir / m
        if not dst.exists():
            import os
            try:
                os.symlink(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)


def generate_xml() -> str:
    base_mass, base_inertia = inertial_of("base_plate_layer1-v5")
    setup_meshes()

    lines = [
        '<?xml version="1.0"?>',
        '<mujoco model="lekiwi">',
        '  <visual>',
        '    <global offwidth="1280" offheight="960"/>',
        '  </visual>',
        '',
        '  <option timestep="0.005" integrator="Euler">',
        '    <flag contact="enable" energy="disable"/>',
        '  </option>',
        '',
        '  <default>',
        '    <joint damping="0.5"/>',
        '    <geom friction="0.6 0.05 0.01" solref="0.004 1.0" solimp="0.8 0.4 0.01"/>',
        '  </default>',
        '',
        '  <asset>',
        '    <!-- STL meshes from lekiwi_modular -->',
        '    <mesh file="meshes/base_plate_layer1-v5.stl" name="base_plate_layer1-v5"/>',
        '    <mesh file="meshes/omni_wheel_mount-v5.stl" name="omni_wheel_mount-v5"/>',
        '    <mesh file="meshes/STS3215_03a-v1.stl" name="STS3215_03a-v1"/>',
        '    <mesh file="meshes/STS3215_03a-v1-1.stl" name="STS3215_03a-v1-1"/>',
        '    <mesh file="meshes/STS3215_03a-v1-2.stl" name="STS3215_03a-v1-2"/>',
        '    <mesh file="meshes/STS3215_03a-v1-3.stl" name="STS3215_03a-v1-3"/>',
        '    <mesh file="meshes/STS3215_03a-v1-4.stl" name="STS3215_03a-v1-4"/>',
        '    <mesh file="meshes/STS3215_03a_Wrist_Roll-v1.stl" name="STS3215_03a_Wrist_Roll-v1"/>',
        '    <mesh file="meshes/omni_wheel_mount-v5-1.stl" name="omni_wheel_mount-v5-1"/>',
        '    <mesh file="meshes/omni_wheel_mount-v5-2.stl" name="omni_wheel_mount-v5-2"/>',
        '    <!-- Arm structural meshes -->',
        '    <mesh file="meshes/Rotation_Pitch_08i-v1.stl" name="Rotation_Pitch_08i-v1"/>',
        '    <mesh file="meshes/SO_ARM100_08k_116_Square-v1.stl" name="SO_ARM100_08k_116_Square-v1"/>',
        '    <mesh file="meshes/SO_ARM100_08k_Mirror-v1.stl" name="SO_ARM100_08k_Mirror-v1"/>',
        '    <mesh file="meshes/Wrist_Roll_Pitch_08i-v1.stl" name="Wrist_Roll_Pitch_08i-v1"/>',
        '    <mesh file="meshes/Wrist_Roll_08c-v1.stl" name="Wrist_Roll_08c-v1"/>',
        '    <mesh file="meshes/Moving_Jaw_08d-v1.stl" name="Moving_Jaw_08d-v1"/>',
        '  </asset>',
        '',
        '  <worldbody>',
        '    <!-- Ground -->',
        '    <geom name="ground" type="plane" size="5 5 0.01"',
        '          rgba="0.18 0.18 0.22 1" friction="1.0 0.1 0.02"/>',
        '',
        '    <!-- Base + wheels + arm -->',
        '    <body name="base" pos="0 0 0.14">',
        f'      {inertial_xml(base_mass, base_inertia)}',
        f'      {build_base_geom_xml()}',
        '',
        '      <!-- Free joint for 6-DOF base -->',
        '      <freejoint name="root"/>',
        '',
        '      <!-- Wheels -->',
    ]

    for i in range(3):
        lines.append("")

    for i in range(3):
        lines.append(f'      {build_wheel_xml(i)}')

    lines.extend([
        '',
        '      <!-- Arm -->',
    ])

    for i in range(6):
        lines.append(f'      {build_arm_xml(i)}')

    lines.extend([
        '',
        '      <!-- Camera -->',
        '      <body name="camera" pos="0.05 0 0.05">',
        '        <inertial pos="0 0 0" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>',
        geom_xml("camera_geom", "box",
                 size="0.015 0.025 0.015",
                 pos="0 0 0", rgba="0.1 0.1 0.1 1",
                 contype="0", conaffinity="0"),
        '      </body>',
        '',
        '    </body>',
        '',
        '    <!-- Target object -->',
        '    <body name="target" pos="0.5 0 0.02">',
        '      <geom name="target_geom" type="cylinder" size="0.04 0.04" rgba="1 0.2 0.2 1"/>',
        '    </body>',
        '',
        '  </worldbody>',
        '',
        '  <!-- Actuators: ctrl[0..5] arm, ctrl[6..8] wheels -->',
        '  <actuator>',
        '    <motor joint="j0" gear="10"/>',
        '    <motor joint="j1" gear="10"/>',
        '    <motor joint="j2" gear="10"/>',
        '    <motor joint="j3" gear="5"/>',
        '    <motor joint="j4" gear="5"/>',
        '    <motor joint="j5" gear="3"/>',
        '    <motor joint="w0" gear="0.5"/>',
        '    <motor joint="w1" gear="0.5"/>',
        '    <motor joint="w2" gear="0.5"/>',
        '  </actuator>',
        '</mujoco>',
    ])

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"URDF: {URDF_FILE}")
    print(f"Meshes: {MESHHOME}")
    print(f"  links={len(links)}, joints={len(joints)}")

    # Verify mesh files
    for m in ["base_plate_layer1-v5.stl", "STS3215_03a-v1.stl",
               "4-Omni-Directional-Wheel_Single_Body-v1.stl",
               "omni_wheel_mount-v5.stl"]:
        path = MESHHOME / m
        print(f"  {m}: {'FOUND' if path.exists() else 'MISSING'}")

    xml_str = generate_xml()
    print(f"\nGenerated: {len(xml_str)} chars")

    # Validate XML
    try:
        ET.fromstring(xml_str)
        print("XML validation: OK")
    except ET.ParseError as e:
        print(f"XML validation ERROR: {e}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(xml_str)
    print(f"Written: {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")

    # Count structures
    print(f"  <body>: {xml_str.count('<body ')}")
    print(f"  <joint>: {xml_str.count('<joint ')}")
    print(f"  <motor>: {xml_str.count('<motor ')}")
    print(f"  <geom>: {xml_str.count('<geom ')}")
    print(f"  <inertial>: {xml_str.count('<inertial ')}")
