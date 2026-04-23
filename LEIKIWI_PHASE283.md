# Phase 283 — VLA Wheel Action Magnitude Bug: Root Cause Confirmed

**Date**: 2026-04-23 16:30 CST

## 本次心跳完成

### Root Cause: VLA Wheel Actions ~26x Too Small

**Analysis of phase282 results: s3_epoch9 SR=2% (1/50 goals)**

Ran diagnostic episode with detailed per-step logging:

```
Goal (0.2, 0.3): P-ctrl wheel=[0.316, 0.294, -0.009] rad/s
                 VLA wheel (raw)=[0.022, -0.008, 0.030]
                 VLA wheel (denormed, [-0.5,0.5])=[0.011, -0.004, 0.015] rad/s
                 Ratio: 0.316/0.012 ≈ 26x
```

**After _action_to_ctrl() scaling**:
- VLA ctrl[6:9] = wheel_action * 10.0 = [0.11, -0.04, 0.15] Nm
- P-ctrl ctrl[6:9] = [3.16, 2.94, -0.09] Nm
- Ratio: ~28x (same order of magnitude)

**Simulation evidence**: After 10 steps with VLA actions:
- base_vel = [-0.096, -0.041] m/s (barely moving)
- After 10 steps P-ctrl would move ~0.5-1.0 m

The VLA is outputting wheel actions ~26x smaller than needed for locomotion.

### Why This Is NOT a normalize_action Bug

`normalize_action()` is correctly implemented:
- Input: raw VLA output in [-1, 1]
- Output: wheel native [-0.5, 0.5] rad/s

The problem is at the VLA policy output level — the network itself outputs tiny wheel actions.

### Why Stage2 Achieved 72% SR Despite Same Bug

Stage2 was trained with REPLACED wheel actions from P-controller:
- `action[6:9] = wheel_speeds` (P-controller commands directly replaced VLA wheel output)
- The VLA only learned arm actions; wheel locomotion was fully handled by P-controller
- Stage2's "VLA" label is misleading — it's a hybrid where P-controller does all mobile work

Stage3 training attempted to let the VLA learn wheel locomotion too (from dagger + phase227 data), but it failed to learn meaningful wheel commands.

### Architecture Impact

| Policy | Wheel Source | Expected | Actual | Result |
|--------|-------------|----------|--------|--------|
| Stage2 | P-controller (replaced) | P-ctrl | P-ctrl | 72% SR ✅ |
| Stage3 | VLA learned | VLA | VLA ~26x too small | 2% SR ❌ |
| DAgger | P-controller (30-step fallback) | P-ctrl | P-ctrl | 50% SR ⚠️ |

### What This Means for ROS2 Bridge

The `vla_policy_node.py` normalize_action fix (Phase 278) is CORRECT:
- VLA native-unit outputs → bridge applies _action_to_ctrl() scaling
- But since VLA outputs tiny values, the result is still insufficient locomotion

The bridge's Contact-Jacobian P-controller is the reliable fallback for locomotion.

---

## 下一步

- [ ] **Option A**: Retrain Stage3 with amplified wheel action loss (weight wheel loss × 20)
- [ ] **Option B**: Keep Stage3 as arm-only, use P-controller for locomotion in ROS2 bridge
- [ ] **Option C**: Investigate if s3_epoch6 has better wheel action magnitude
- [ ] **Phase 284**: Re-evaluate s3_epoch6 on 50 goals (was only quick-eval'd at 10 goals)

###阻礙

- VLA wheel action magnitude is a training issue, not a bridge/eval code issue
- No ROS2 environment to test real hardware integration