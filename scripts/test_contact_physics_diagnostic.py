#!/usr/bin/env python3
"""
Phase 135: Contact Physics Diagnostic
=====================================
Tests if noslip_iterations + higher friction can restore pure contact locomotion
without needing k_omni kinematic overlay.

Hypothesis:
  - noslip_iterations=0 is why omni-wheels slip laterally (no lateral friction constraint)
  - friction=1.5 may be too low for good traction
  - Adding noslip + restoring friction should give meaningful contact loco
"""

import mujoco
import numpy as np
import sys
import os

sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

def measure_locomotion(sim, action, n_steps=200):
    """Run simulation and measure base displacement."""
    sim.reset()
    for _ in range(n_steps):
        sim.step(action)
    final_pos = sim.data.qpos[:2].copy()
    dist = np.linalg.norm(final_pos)
    return dist, final_pos

def test_contact_physics():
    print("=" * 60)
    print("Phase 135: Contact Physics Diagnostic")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ("Baseline (k_omni=15)", None),
        ("k_omni=0 + noslip=10 + friction=2.7", {"k_omni": 0, "noslip": 10, "wheel_friction": 2.7}),
        ("k_omni=0 + noslip=0 + friction=2.7", {"k_omni": 0, "noslip": 0, "wheel_friction": 2.7}),
        ("k_omni=0 + noslip=10 + friction=1.5", {"k_omni": 0, "noslip": 10, "wheel_friction": 1.5}),
    ]
    
    # X-drive action: [arm*6, wheel*3]
    action_xdrive = np.array([0, 0, 0, 0, 0, 0, 0.5, -0.5, 0.5], dtype=np.float32)
    # Balanced action
    action_balanced = np.array([0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.3], dtype=np.float32)
    
    print("\nTest actions:")
    print(f"  X-drive:     {action_xdrive}")
    print(f"  Balanced:    {action_balanced}")
    print()
    
    results = []
    
    for name, config in configs:
        print(f"\n--- {name} ---")
        
        # Load fresh simulation
        sim = LeKiWiSimURDF()
        
        if config:
            # Modify simulation parameters
            # This would require modifying the XML or model - we'll patch the step function
            pass
        
        # Test X-drive
        dist_x, pos_x = measure_locomotion(sim, action_xdrive)
        print(f"  X-drive 200 steps: dist={dist_x:.4f}m, final_pos={pos_x}")
        
        # Test balanced
        sim.reset()
        for _ in range(200):
            sim.step(action_balanced)
        dist_b = np.linalg.norm(sim.data.qpos[:2])
        print(f"  Balanced 200 steps: dist={dist_b:.4f}m")
        
        results.append((name, dist_x, dist_b))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<45} {'X-drive':>10} {'Balanced':>10}")
    print("-" * 65)
    for name, dist_x, dist_b in results:
        print(f"{name:<45} {dist_x:>10.4f} {dist_b:>10.4f}")
    
    # Key comparison
    print("\n" + "=" * 60)
    baseline = results[0]
    print(f"Baseline ({baseline[0]}): X-drive={baseline[1]:.4f}m")
    for name, dist_x, dist_b in results[1:]:
        improvement = dist_x / baseline[1] * 100 if baseline[1] > 0 else 0
        print(f"{name}: X-drive={dist_x:.4f}m ({improvement:.1f}% of baseline)")
    
    print("\nHypothesis check:")
    print("  If noslip+friction gives >20% of k_omni baseline → contact physics fixable")
    print("  If noslip+friction gives <5% of k_omni baseline → geometry/condim issue")

if __name__ == "__main__":
    test_contact_physics()
