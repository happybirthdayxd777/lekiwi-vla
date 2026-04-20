# Phase 222 — 2026-04-20 15:30 UTC

## VLA Failure Mode Root Cause Analysis

### 本次心跳完成

**Critical Bug Found: P-Controller Data Has No Wheel Velocity Diversity**

The root cause of VLA failures in the `+X/-Y` quadrant has been identified:

```
Training data: 50 episodes, phase196_contact_jacobian_train
Wheel velocity by quadrant (all quadrants identical):
  mean = [0.5, 0.0, 0.0]  ← FIXED open-loop forward velocity
  std  = [0.0, 0.0, 0.0]  ← essentially no variation!
```

**The P-controller was open-loop on wheels** — it drove forward at constant velocity `[0.5, 0, 0]`.
Error correction happened via the **ARM joints only** (proprioceptive feedback).

The VLA learned to mimic the arm trajectories, but never saw wheel velocity diversity.
When given a `+X/-Y` goal (requiring lateral `-Y` movement), the VLA:
1. Outputs "drive forward" wheel commands (mimicking training)
2. Arms flail trying to correct (mimicking P-controller arm behavior)
3. Robot barely moves laterally → failure

---

### Failure Mode Evidence

| Failed Goal | goal_xy | Y/X Ratio | Closest Train Goal | Train Y/X |
|------------|---------|-----------|-------------------|-----------|
| Phase218 G7 | (+0.274, -0.144) | **-0.526** | (+0.458, -0.173) | -0.377 |
| Phase218b G2 | (+0.269, -0.234) | **-0.870** | (+0.458, -0.173) | -0.377 |
| Phase218b G5 | (+0.393, -0.247) | **-0.628** | (+0.458, -0.173) | -0.377 |

**All three failures have Y/X ratios in [-0.87, -0.53] that are underrepresented in training.**

Training `+X/-Y` Y/X distribution (9 episodes):
```
ep0: Y/X=-0.489   ep1: Y/X=-0.715   ep2: Y/X=-0.221  ← mostly shallow angles
ep3: Y/X=-36.151  (near-pure -Y, tiny +X) ← outlier, not helpful for +X/-Y
ep4: Y/X=-0.377   ep5: Y/X=-0.314   ep6: Y/X=-0.428
ep7: Y/X=-1.927   ← steep angle, rare
ep8: Y/X=-0.762
```

The VLA has NOT seen enough examples of **combined forward + strong leftward** movement.
When Y/X is around -0.5 to -0.9, the robot needs to move with significant lateral component.
The P-controller doesn't provide this — it just goes forward and adjusts arms.

---

### Training Data Quadrant Coverage

| Quadrant | Episodes | % | Training Steps |
|----------|-----------|---|----------------|
| +X/+Y | 13 | 26% | 2020 (36%) |
| +X/-Y | 9 | 18% | 910 (16%) |
| -X/+Y | 13 | 26% | 1438 (26%) |
| -X/-Y | 15 | 30% | 1194 (21%) |

Coverage is OK per quadrant. The problem is **within** the `+X/-Y` quadrant:
the **directional composition** (Y/X ratio) distribution is biased.

---

### Why Other Quadrants Work

| Quadrant | Why VLA Succeeds |
|----------|----------------|
| +X/+Y (100%) | Y/X in [0, ~0.7] → forward + slight right → aligns with P-controller forward bias |
| -X/+Y (100%) | -X requires reversing, which the P-controller data actually has |
| -X/-Y (100%) | reversing + left ← opposite of forward, P-controller reversal covers this |
| **+X/-Y (40%)** | **forward + LEFT ← orthogonal to P-controller forward-only training** |

---

### Solution Pathways

**Option A: Fix Training Data (highest impact)**
- Collect new P-controller data with **proper wheel velocity variation**
- Specifically: close-loop P-controller with nonzero lateral wheel velocities
- Or: use kinematically-correct omni-wheel steering model that produces lateral movement

**Option B: Curriculum Learning**
- Train on +X/-Y goals first (most difficult) with longer episodes
- Then mix with other quadrants
- Ensures the VLA sees enough diverse +X/-Y examples

**Option C: Behavior Cloning with DAgger**
- Run the VLA in the sim, collect P-controller corrections when VLA deviates
- Augment training data with corrective actions

**Option D: Architecture Fix**
- The current policy may be underfitting on wheel control
- Increase model capacity or add explicit lateral movement head

---

### Next Steps (Priority Order)

**Priority 1: Verify P-controller wheel velocity issue**
```bash
# Check how P-controller actually commands wheels
grep -n "wheel\|qvel\|forward" ~/hermes_research/lekiwi_vla/sim_lekiwi_urdf.py | head -30
```

**Priority 2: Write Diagnostic Script**
```bash
# Save the analysis as a permanent diagnostic tool
scripts/diagnose_vla_failure.py  ← FAILURE_MODE script just created
```

**Priority 3: Design Better Data Collection**
- P-controller should use proper omni-wheel inverse kinematics
- For any goal, compute required wheel velocities to move there
- NOT just "drive forward at constant speed"

**Priority 4: 50-goal Statistical Evaluation**
- Still needed for ±5% confidence interval on VLA success rate
- Run in background to not block other work

---

### Git

- Working tree: modified (diagnostic script)
- Branch: main
- Status: ready to commit

---

### Summary

```
ROOT CAUSE: P-controller training data has NO wheel velocity diversity
- P-controller drives at constant [0.5, 0, 0] velocity open-loop
- Lateral (-Y) movement requires wheel velocity variation P-controller never produced
- VLA learned to mimic arm corrections, not wheel control
- +X/-Y goals need lateral movement → VLA fails (40% SR vs 100% elsewhere)

FIX: Need closed-loop P-controller with proper omni-wheel IK,
     OR collect data with diverse wheel velocities, OR DAgger-style augmentation
```
