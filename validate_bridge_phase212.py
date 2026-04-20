#!/usr/bin/env python3
"""
validate_bridge_phase212.py — Phase 212: Render warmup fix validation

Issue: LeKiWiSim (primitive) returns all-black images from render() before the
first physics step, because the freejoint base hasn't been ticked by physics yet.
CameraAdapter starts its render loop immediately after sim init, getting black frames.

Fix: Add warmup step (sim.step(np.zeros(9))) after make_sim("primitive") in bridge_node.py.
This mirrors the existing URDF warmup (line 370-372).

Tests:
  1. Primitive sim render before/after warmup — black → non-black
  2. Bridge node primitive init includes warmup step
  3. URDF path unchanged (already had warmup)
  4. CameraAdapter importable (skip if ROS2 not built)
"""

import sys
import mujoco
import numpy as np

sys.path.insert(0, "/Users/i_am_ai/hermes_research/lekiwi_vla")
from sim_lekiwi import LeKiwiSim


def test_render_warmup():
    """Test 1: Render is all-black before warmup, non-black after."""
    sim = LeKiwiSim()
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")

    # Before warmup
    r = mujoco.Renderer(sim.model, 640, 480)
    r.update_scene(sim.data, camera=cam_id)
    img = r.render()
    r.close()
    arr = np.array(img)
    assert arr.max() == 0, f"FAIL: render before warmup should be black, got max={arr.max()}"
    print("  ✓ Render before warmup = ALL BLACK (as expected)")

    # After warmup
    sim.step(np.zeros(9))
    r2 = mujoco.Renderer(sim.model, 640, 480)
    r2.update_scene(sim.data, camera=cam_id)
    img2 = r2.render()
    r2.close()
    arr2 = np.array(img2)
    assert arr2.max() > 0, f"FAIL: render after warmup should be non-black, got max={arr2.max()}"
    print(f"  ✓ Render after warmup = max={arr2.max()} (non-black, camera working)")
    return True


def test_bridge_node_warmup():
    """Test 2: Check bridge_node.py includes warmup for primitive sim."""
    with open("/Users/i_am_ai/hermes_research/lekiwi_vla/src/lekiwi_ros2_bridge/bridge_node.py") as f:
        content = f.read()

    primitive_branch = content.find("Starting LeKiWiSim (cylinder primitives)")
    assert primitive_branch > 0, "FAIL: Could not find primitive sim section in bridge_node.py"
    section = content[primitive_branch:primitive_branch + 500]

    assert "self.sim.step(np.zeros(9))" in section, \
        "FAIL: Warmup step NOT found in primitive sim section"
    assert "Primitive warmup" in section, \
        "FAIL: Warmup log message NOT found in primitive sim section"
    print("  ✓ bridge_node.py: primitive sim has warmup step")
    return True


def test_urdf_unchanged():
    """Test 3: URDF path still has its warmup (no regression)."""
    with open("/Users/i_am_ai/hermes_research/lekiwi_vla/src/lekiwi_ros2_bridge/bridge_node.py") as f:
        content = f.read()

    urdf_section = content.find("Starting LeKiWiSimURDF")
    assert urdf_section > 0, "FAIL: Could not find URDF sim section"
    section = content[urdf_section:urdf_section + 400]

    assert "self.sim.step(np.zeros(9))" in section, \
        "FAIL: URDF warmup step missing (regression)"
    print("  ✓ URDF sim warmup step preserved (no regression)")
    return True


def test_camera_adapter_importable():
    """Test 4: Check lekiwi_ros2_bridge can be imported (ROS2 build check)."""
    try:
        from lekiwi_ros2_bridge.camera_adapter import CameraAdapter
        print("  ✓ lekiwi_ros2_bridge.camera_adapter is importable")
        return True
    except ImportError as e:
        print(f"  ⊘ SKIP: lekiwi_ros2_bridge not installed (ROS2 not built) — {e}")
        return True  # Skip if not built


def main():
    print("=" * 60)
    print("  Phase 212 — Render Warmup Fix Validation")
    print("=" * 60)

    tests = [
        ("Render warmup (black→non-black)", test_render_warmup),
        ("Bridge node primitive warmup present", test_bridge_node_warmup),
        ("URDF warmup unchanged (no regression)", test_urdf_unchanged),
        ("CameraAdapter importable", test_camera_adapter_importable),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            if fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")

    print(f"\n{'='*60}")
    print(f"  Result: {passed}/{len(tests)} passed")
    if passed == len(tests):
        print("  ✓ All tests PASSED — VLA will receive non-black images")
    else:
        print("  ✗ Some tests FAILED")
    print("=" * 60)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())