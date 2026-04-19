# Phase 196: VLA Training on CORRECT Contact-Jacobian Data

## Summary

Phase 196 trains a VLA policy on data collected with the CORRECT Contact-Jacobian P-controller. This addresses the critical finding from Phase 195: ALL previous VLA policies (Phase 181, 187, 190) were trained on data collected with the WRONG kinematic IK P-controller, making their reported success rates unreliable.

## Critical Background

### Phase 195 Discovery: The Wrong P-Controller Was Used

Phase 195 discovered that `twist_to_contact_wheel_speeds()` in `sim_lekiwi_urdf.py` used a **kinematic IK model** calibrated for k_omni=15 overlay physics (Phase 164), NOT the actual contact Jacobian.

The **correct** Contact-Jacobian P-controller achieves **100% SR** on 20 random goals:
```python
_CONTACT_JACOBIAN = np.array([
    [0.1257, 0.4426],
    [0.2568, 0.3179],
    [-0.2606, 0.1596]
])
_CONTACT_JACOBIAN_PSEUDO_INV = np.linalg.pinv(_CONTACT_JACOBIAN)

def correct_pcontroller(goal_xy, base_xy, kP=2.0):
    err = goal_xy - base_xy
    v_desired = kP * err
    wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
    return wheel_speeds
```

### Previous Data Collections Were CORRUPTED

| Phase | Controller | Result |
|-------|-----------|--------|
| Phase 181 | Old kinematic IK | Unreliable |
| Phase 187 | Old kinematic IK | **Near-zero correlations** |
| Phase 189 | Old kinematic IK (fixed *200 bug) | Still kinematic IK |
| Phase 190 | Trained on Phase 189 | **Unreliable** |
| **Phase 196** | **CORRECT Contact-Jacobian** | **Clean data** |

## What Phase 196 Does

1. **Collect clean data** using `scripts/collect_phase196_clean.py`
   - 50 episodes with CORRECT Contact-Jacobian P-controller (kP=2.0)
   - 90% episode success rate (45/50 episodes)
   - 5562 frames with real per-step images
   - 45 goal-near frames (0.8% of data)

2. **Train VLA** using `scripts/train_phase196.py`
   - CLIP ViT-B/32 vision encoder (frozen)
   - Goal-conditioned policy with cross-attention
   - 4-step Euler flow matching
   - 30 epochs on 5562 frames

3. **Evaluate** vs Contact-Jacobian P-controller baseline

## Results

### Data Collection (collect_phase196_clean.py)
```
Episode 1/50: 99 steps, success=True
Episode 2/50: 250 steps, success=False
Episode 3/50: 250 steps, success=False
...
Episode 50/50: 122 steps, success=True

Total frames: 5562
Success rate: 45/50 = 90%
Rewards: 0.8% positive
```

### Correlation Analysis (vs Phase 187)
```
Phase 187 (WRONG controller):          Phase 196 (CORRECT controller):
Corr(w0, gx) = -0.049 (ZERO!)        Corr(w0, gx) = -0.087
Corr(w1, gx) = +0.044 (ZERO!)        Corr(w1, gx) = +0.166
Corr(w2, gx) = -0.009 (ZERO!)        Corr(w2, gx) = -0.664  ← STRONG
Corr(w0, gy) = +0.049 (ZERO!)        Corr(w0, gy) = +0.335
Corr(w1, gy) = +0.058 (ZERO!)        Corr(w1, gy) = +0.137
Corr(w2, gy) = -0.036 (ZERO!)        Corr(w2, gy) = +0.569  ← STRONG

Phase 196: 2/5 strong correlations (w2↔gx=-0.664, w2↔gy=+0.569)
Note: Weaker correlations due to varying start positions across episodes
```

### VLA Training
- Still running (CPU, ~1 epoch/minute)
- See `results/phase196_contact_jacobian_train/`

## Files Created/Modified

- `scripts/collect_phase196_clean.py` — NEW: Clean data collection with Contact-Jacobian
- `scripts/train_phase196.py` — NEW: VLA training script for Phase 196 data
- `data/phase196_clean_50ep.h5` — NEW: 5562 frames, CORRECT controller data
- `sim_lekiwi_urdf.py` — UNCHANGED (already had correct _CONTACT_JACOBIAN_PSEUDO_INV)
- `scripts/eval_jacobian_pcontroller.py` — UNCHANGED (already used correct controller)

## Next Steps

1. Wait for VLA training to complete (~30 minutes on CPU)
2. Evaluate Phase 196 VLA vs Contact-Jacobian P-controller baseline
3. If VLA achieves >20% SR, this confirms VLA can learn from correct data
4. If VLA achieves ~0% SR, the architecture itself may need redesign
