#!/usr/bin/env python3
"""
validate_hybrid_bridge.py
=========================
Phase 212: Validates Direction-Agreement Hybrid Bridge (replaces Phase 210 blending).

Tests:
  1. P-controller fallback produces non-trivial wheel speeds
  2. Direction-agreement: VLA direction correct → amplified VLA (2.5x)
  3. Direction-disagreement: VLA direction wrong → P-controller used
  4. No goal set → VLA used as-is (no P-controller reference)
  5. 8 goal directions all trigger P-controller fallback when VLA is conservative

Run:
    python3 scripts/validate_hybrid_bridge.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi_urdf import twist_to_contact_wheel_speeds, LeKiWiSimURDF

# ── Constants (copied from bridge_node.py) ─────────────────────────────────────
_HYBRID_WHEEL_FALLBACK_THRESHOLD = 0.15
_WHEEL_AMPLIFICATION_FACTOR = 2.5
_DIRECTION_AGREEMENT_THRESHOLD = 0.5
kP_FALLBACK = 2.0


def _wheel_direction_agrees(vla_wheels: np.ndarray, pctrl_ws: np.ndarray,
                            threshold: float = _DIRECTION_AGREEMENT_THRESHOLD) -> bool:
    """Phase 212: Check if VLA and P-controller wheel directions agree."""
    vla_norm = np.linalg.norm(vla_wheels)
    pctrl_norm = np.linalg.norm(pctrl_ws)
    if pctrl_norm < 0.01 or vla_norm < 0.01:
        return False
    vla_dir = vla_wheels / vla_norm
    pctrl_dir = pctrl_ws / pctrl_norm
    return np.dot(vla_dir, pctrl_dir) > threshold


def hybrid_wheel_speeds_phase212(vla_wheel_raw, base_xy, goal_xy):
    """
    Phase 212 direction-agreement hybrid logic.
    Returns wheel_speeds (np.ndarray, shape=(3,)).
    """
    vla_mag = np.linalg.norm(vla_wheel_raw)
    if vla_mag < _HYBRID_WHEEL_FALLBACK_THRESHOLD:
        if goal_xy is not None:
            err = np.array(goal_xy) - np.array(base_xy)
            vx_fb = kP_FALLBACK * err[0]
            vy_fb = kP_FALLBACK * err[1]
            pctrl_ws = twist_to_contact_wheel_speeds(vx_fb, vy_fb, 0.0)

            if _wheel_direction_agrees(vla_wheel_raw, pctrl_ws,
                                       _DIRECTION_AGREEMENT_THRESHOLD):
                # VLA direction correct → amplify and use VLA
                wheel_speeds = vla_wheel_raw * _WHEEL_AMPLIFICATION_FACTOR
            else:
                # VLA direction wrong → use P-controller
                wheel_speeds = pctrl_ws
        else:
            wheel_speeds = vla_wheel_raw  # no goal → use VLA as-is
    else:
        wheel_speeds = vla_wheel_raw  # VLA confident → use directly
    return wheel_speeds


def test_pcontroller_fallback():
    """Test 1: Zero VLA → P-controller fallback produces non-trivial wheel speeds."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]
    goal_xy = np.array([0.5, 0.0])

    vla_zero = np.zeros(3)
    ws = hybrid_wheel_speeds_phase212(vla_zero, base_xy, goal_xy)
    ok = np.linalg.norm(ws) > 0.01
    print(f"[Test 1] P-controller fallback (vla=0): |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_direction_agreement_amplification():
    """Test 2: VLA direction agrees with P-controller → VLA amplified 2.5x."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]
    goal_xy = np.array([0.5, 0.0])

    # VLA small magnitude but same direction as P-controller
    vla_small = np.array([0.05, 0.10, -0.08])  # below threshold, ~0.13 mag
    ws = hybrid_wheel_speeds_phase212(vla_small, base_xy, goal_xy)
    expected = vla_small * _WHEEL_AMPLIFICATION_FACTOR
    ok = np.allclose(ws, expected, atol=0.01)
    print(f"[Test 2] Direction-agreement → VLA amplified: |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_direction_disagreement_uses_pctrl():
    """Test 3: VLA direction disagrees with P-controller → P-controller used."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]
    goal_xy = np.array([0.5, 0.0])

    # VLA opposite direction from P-controller (wrong)
    vla_opposite = np.array([-0.05, -0.05, 0.05])  # below threshold
    ws = hybrid_wheel_speeds_phase212(vla_opposite, base_xy, goal_xy)

    # P-controller for goal (0.5, 0.0) from base (0,0)
    pctrl_ws = twist_to_contact_wheel_speeds(kP_FALLBACK * 0.5, kP_FALLBACK * 0.0, 0.0)
    ok = np.allclose(ws, pctrl_ws, atol=0.01)
    print(f"[Test 3] Direction-disagreement → P-controller: |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_no_goal_uses_vla():
    """Test 4: No goal set → VLA used as-is (no P-controller reference)."""
    base_xy = np.zeros(2)
    vla_small = np.array([0.05, 0.03, 0.01])  # below threshold
    ws = hybrid_wheel_speeds_phase212(vla_small, base_xy, None)
    ok = np.allclose(ws, vla_small, atol=0.01)
    print(f"[Test 4] No goal → VLA used as-is: |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def test_8direction_pcontroller():
    """Test 5: P-controller active for all 8 goal directions."""
    sim = LeKiWiSimURDF()
    sim.step(np.zeros(9))
    obs = sim._obs()
    base_xy = obs["base_position"][:2]

    goals = [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.4), (0.0, -0.4),
             (0.35, 0.35), (-0.35, 0.35), (0.35, -0.35), (-0.35, -0.35)]

    vla_zero = np.zeros(3)
    all_nonzero = all(np.linalg.norm(hybrid_wheel_speeds_phase212(vla_zero, base_xy, g)) > 0.01
                     for g in goals)
    print(f"[Test 5] P-controller active for all 8 directions: {'✓' if all_nonzero else '✗'}")
    return all_nonzero


def test_vla_confident_passthrough():
    """Test 6: VLA magnitude >= threshold → VLA used directly."""
    base_xy = np.zeros(2)
    goal_xy = np.array([0.5, 0.0])
    vla_confident = np.array([0.20, 0.0, 0.0])  # >= threshold (0.15)
    ws = hybrid_wheel_speeds_phase212(vla_confident, base_xy, goal_xy)
    ok = np.allclose(ws, vla_confident, atol=0.01)
    print(f"[Test 6] VLA confident → VLA passthrough: |ws|={np.linalg.norm(ws):.4f}  {'✓' if ok else '✗'}")
    return ok


def main():
    print("=" * 60)
    print("Phase 212: Direction-Agreement Hybrid Bridge Validation")
    print("=" * 60)
    results = [
        test_pcontroller_fallback(),
        test_direction_agreement_amplification(),
        test_direction_disagreement_uses_pctrl(),
        test_no_goal_uses_vla(),
        test_8direction_pcontroller(),
        test_vla_confident_passthrough(),
    ]
    passed = sum(results)
    print(f"\nResults: {passed}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())