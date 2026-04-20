#!/usr/bin/env python3
"""
validate_hybrid_bridge.py
=========================
Phase 210: Validates Hybrid Bridge P-controller fallback WITHOUT requiring ROS2.

Tests:
  1. P-controller fallback computes correct wheel speeds for 8 goal directions
  2. VLA magnitude threshold triggers fallback correctly
  3. Blend factor: small VLA → mostly P-controller; large VLA → mostly VLA
  4. No goal set → fallback degrades gracefully to zero

Run:
    python3 scripts/validate_hybrid_bridge.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi_urdf import twist_to_contact_wheel_speeds, LeKiWiSimURDF

# ── Constants (copied from bridge_node.py) ─────────────────────────────────────
_HYBRID_WHEEL_FALLBACK_THRESHOLD = 0.15
kP_FALLBACK = 2.0

def hybrid_wheel_speeds(vla_wheel_raw, base_xy, goal_xy):
    """
    Simulate the hybrid bridge logic from bridge_node.py _on_cmd_vel.
    Returns wheel_speeds.
    """
    vla_mag = np.linalg.norm(vla_wheel_raw)
    if vla_mag < _HYBRID_WHEEL_FALLBACK_THRESHOLD:
        blend = 1.0 - min(vla_mag / _HYBRID_WHEEL_FALLBACK_THRESHOLD, 1.0)
        if goal_xy is not None:
            err = np.array(goal_xy) - np.array(base_xy)
            vx_fb = kP_FALLBACK * err[0]
            vy_fb = kP_FALLBACK * err[1]
            pctrl_ws = twist_to_contact_wheel_speeds(vx_fb, vy_fb, 0.0)
            pctrl_scale = 0.20 / max(np.linalg.norm(pctrl_ws), 0.01)
            pctrl_ws = pctrl_ws * pctrl_scale
        else:
            pctrl_ws = np.zeros(3)
            blend = 1.0
        wheel_speeds = blend * pctrl_ws + (1.0 - blend) * vla_wheel_raw
    else:
        wheel_speeds = vla_wheel_raw
    return wheel_speeds


def test_pcontroller_fallback():
    """Test 1: P-controller fallback produces non-trivial wheel speeds."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]
    goal_xy = np.array([0.5, 0.0])

    # Zero VLA input → pure P-controller
    vla_zero = np.zeros(3)
    ws = hybrid_wheel_speeds(vla_zero, base_xy, goal_xy)
    ok = np.linalg.norm(ws) > 0.01
    print(f"[Test 1] P-controller fallback (vla=0): |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_blend_factor():
    """Test 2: Blend=1.0 when VLA=0, blend→0 as VLA→threshold."""
    base_xy = np.zeros(2)
    goal_xy = np.array([0.5, 0.0])

    # vla=0 → blend=1.0 (pure P-controller)
    ws_zero = hybrid_wheel_speeds(np.zeros(3), base_xy, goal_xy)
    pctrl_norm = np.linalg.norm(ws_zero)

    # vla at threshold → blend=0 (pure VLA passthrough)
    vla_threshold = np.array([_HYBRID_WHEEL_FALLBACK_THRESHOLD, 0.0, 0.0])
    ws_threshold = hybrid_wheel_speeds(vla_threshold, base_xy, goal_xy)
    vla_norm = np.linalg.norm(ws_threshold)

    # When vla=0, ws should be ~P-controller; when vla=threshold, ws should be ~vla
    ok1 = np.linalg.norm(ws_zero) > 0.05  # P-controller active
    ok2 = np.allclose(ws_threshold, vla_threshold, atol=0.01)  # pure VLA passthrough
    print(f"[Test 2a] vla=0 → P-controller active: |ws|={pctrl_norm:.4f}  {'✓' if ok1 else '✗'}")
    print(f"[Test 2b] vla=threshold → pure VLA passthrough: {'✓' if ok2 else '✗'}")
    return ok1 and ok2


def test_no_goal_degrades_to_zero():
    """Test 3: No goal set → fallback produces zero wheel speeds."""
    base_xy = np.zeros(2)
    vla_small = np.array([0.05, 0.03, 0.01])  # below threshold
    ws = hybrid_wheel_speeds(vla_small, base_xy, None)
    ok = np.linalg.norm(ws) < 0.01
    print(f"[Test 3] No goal → zero fallback: |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_8direction_pcontroller():
    """Test 4: P-controller fallback works for 8 goal directions."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]

    # 8 goal directions (same as eval_jacobian_pcontroller.py)
    goals = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.4), (0.0, -0.4),
             (0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]

    vla_zero = np.zeros(3)
    all_nonzero = all(np.linalg.norm(hybrid_wheel_speeds(vla_zero, base_xy, g)) > 0.01
                     for g in goals)
    print(f"[Test 4] P-controller active for all 8 directions: {'✓' if all_nonzero else '✗'}")
    return all_nonzero


def test_pcontroller_vs_eval():
    """Test 5: P-controller fallback produces directionally correct wheel speeds."""
    # This tests the DIRECTION correctness, not closed-loop SR.
    # Full SR test requires actual MuJoCo sim with hybrid bridge (ROS2 required).
    # Here we just verify: P-controller direction matches expected base motion.
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]
    goal_xy = np.array([0.5, 0.0])

    # P-controller should produce wheel speeds that move toward goal
    err = goal_xy - base_xy
    vx = kP_FALLBACK * err[0]
    vy = kP_FALLBACK * err[1]
    ws = twist_to_contact_wheel_speeds(vx, vy, 0.0)
    pctrl_scale = 0.20 / max(np.linalg.norm(ws), 0.01)
    ws_scaled = ws * pctrl_scale

    # Apply ONE step and check robot moved in correct direction
    action = np.concatenate([np.zeros(6), ws_scaled])
    sim.step(action)
    obs2 = sim._obs()
    new_xy = obs2["base_position"][:2]
    delta = new_xy - base_xy

    # Forward motion (goal is at +X) should have positive delta x
    ok = delta[0] > 0.0  # robot moved toward +X (goal direction)
    print(f"[Test 5] P-controller direction correct: delta=({delta[0]:+.3f}, {delta[1]:+.3f})  {'✓' if ok else '✗'}")
    return ok


def main():
    print("=" * 60)
    print("Phase 210: Hybrid Bridge Validation")
    print("=" * 60)
    results = [
        test_pcontroller_fallback(),
        test_blend_factor(),
        test_no_goal_degrades_to_zero(),
        test_8direction_pcontroller(),
        test_pcontroller_vs_eval(),
    ]
    passed = sum(results)
    print(f"\nResults: {passed}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
