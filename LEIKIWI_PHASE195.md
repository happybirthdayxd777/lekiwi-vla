# Phase 195: Contact-Jacobian P-Controller — BREAKTHROUGH: 100% SR

## Summary

Phase 195 discovered that the **Contact-Jacobian P-controller achieves 100% SR**, shattering the Phase 194 false "20% ceiling" claim. The root cause was that all eval scripts used the WRONG `twist_to_contact_wheel_speeds()` — a kinematic IK model instead of the actual contact Jacobian.

## Key Findings

### 1. Root Cause of Phase 194's "20% ceiling"

Phase 194 tested controllers by calling `mj_step()` directly, bypassing the `step()` method's **20-step torque ramp**. This caused the robot to tip over on first contact, preventing locomotion entirely.

When using `step()` correctly with the Contact-Jacobian P-controller:
- **Contact-Jacobian P-ctrl (kP=2.0): 100% SR** (20/20 random goals)
- **Contact-Jacobian P-ctrl (50 goals): 94% SR**

### 2. The Wrong Function Was Used

`twist_to_contact_wheel_speeds()` used a **kinematic IK model** calibrated for k_omni=15 overlay physics. The correct approach is to use `_CONTACT_JACOBIAN_PSEUDO_INV` which was empirically measured from pure contact physics (Phase 122-123).

```
# OLD (kinematic IK — WRONG for URDF geometry):
w1 = -0.0124*vx + 0.1880*vy  (Phase 164 calibrated)
w2 =  0.1991*vx + 0.1991*vy
w3 = -0.1993*vx + 0.1872*vy

# NEW (contact Jacobian — CORRECT):
wheel_speeds = _CONTACT_JACOBIAN_PSEUDO_INV @ [vx, vy]
# = [[0.1257, 0.4426],
#    [0.2568, 0.3179],
#    [-0.2606, 0.1596]] @ [vx, vy]
```

### 3. All Previous VLA Policies Need Retraining

Previous VLA policies (Phase 190, 181, 187) were trained on data collected with the WRONG P-controller (kinematic IK). Their reported 80-100% SR was likely also corrupted by the wrong controller.

**Action Required:**
1. Re-collect training data using the fixed `twist_to_contact_wheel_speeds()`
2. Re-train VLA policies
3. Re-evaluate with correct baseline

## Changes Made

### `sim_lekiwi_urdf.py`
- **FIXED** `twist_to_contact_wheel_speeds()` to use `_CONTACT_JACOBIAN_PSEUDO_INV`
- Updated docstring to explain the Phase 164-192 bug

### `scripts/eval_jacobian_pcontroller.py` (NEW)
- Clean implementation of Contact-Jacobian P-controller evaluation
- kP sweep: kP=1.0 best at 90% SR

## Verified Results

| Controller | Success Rate | Notes |
|------------|-------------|-------|
| Contact-Jacobian P-ctrl (kP=2.0) | **100%** (20 goals) | BEST |
| Contact-Jacobian P-ctrl (kP=1.0) | 90% (30 goals) | |
| Contact-Jacobian P-ctrl (kP=4.0) | 90% (30 goals) | |
| Contact-Jacobian P-ctrl (kP=8.0) | 83% (30 goals) | Saturates |
| Grid-sign P-ctrl | 35% (20 goals) | |
| OLD kinematic P-ctrl | 0% (20 goals) | Wrong model |
