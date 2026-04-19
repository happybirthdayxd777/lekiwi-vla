# LeKiWi ROS2-MuJoCo Platform Progress

## [Phase 190 - 2026-04-19 18:00 UTC] — CRITICAL: phase189 data has ZERO useful goal-wheel correlation (root cause: *200 scaling saturates all wheel speeds to ±0.5)

### ✅ 已完成

**ROOT CAUSE DEEP DIVE: The `*200` scaling in `twist_to_contact_wheel_speeds()` saturates ALL wheel speeds to ±0.5 for any meaningful goal.**

**Problem in `collect_phase189_fast.py` line 79-86:**
```python
def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    vx_200 = vx * 200.0   # ← SCALES UP by 200
    vy_200 = vy * 200.0
    w1 = -0.0124 * vx_200 + 0.1880 * vy_200   # coefficients calibrated for *200
    w2 =  0.1991 * vx_200 + 0.1991 * vy_200
    w3 = -0.1993 * vx_200 + 0.1872 * vy_200
    return np.clip(np.array([w1, w2, w3]), -0.5, 0.5)
```

With `kP=0.5` and `goal=(0.3, 0.3)`:
- `vx = 0.5 * 0.3 = 0.15`, `vy = 0.15`
- `vx_200 = 0.15 * 200 = 30`
- `w1 = -0.0124 * 30 + 0.1880 * 30 = -0.372 + 5.64 = 5.268 → clipped to 0.5`
- **ALL wheel speeds saturate to ±0.5 for any goal with |dx|,|dy| >= 0.1**

**This explains the near-zero correlations in phase189 data:**
- The P-controller produces IDENTICAL wheel patterns for all goals — just saturating ±0.5
- The policy sees no variation in wheel commands based on goal direction
- Only w1 has some variation (between +0.5 and -0.5) depending on goal quadrant

**Evidence from data analysis:**
```
phase189 data:
  Corr(w0, gx) = -0.0702  ← w0 independent of goal_x
  Corr(w0, gy) = +0.0260  ← w0 independent of goal_y
  Corr(w1, gx) = -0.0493  ← w1 independent of goal_x
  Corr(w1, gy) = -0.0639  ← w1 independent of goal_y
  Corr(w2, gx) = -0.0302  ← w2 independent of goal_x
  Corr(w2, gy) = +0.0488  ← w2 independent of goal_y
```

**The fix: REMOVE the `*200` scaling**
```python
def twist_to_contact_fixed(vx, vy, wz=0.0):
    '''WITHOUT *200 - proper small velocity scaling'''
    w1 = -0.0124 * vx + 0.1880 * vy
    w2 =  0.1991 * vx + 0.1991 * vy
    w3 = -0.1993 * vx + 0.1872 * vy
    return np.clip(np.array([w1, w2, w3]), -0.5, 0.5)
```

**With corrected formula + adaptive velocity (`v_mag = min(kP*dist, v_max)`):**
```
goal (+0.3,+0.3): v=(+0.212,+0.212) -> w=[+0.037, +0.084, -0.003]
goal (+0.3,-0.3): v=(+0.212,-0.212) -> w=[-0.043, +0.000, -0.082]
goal (-0.3,+0.3): v=(-0.212,+0.212) -> w=[+0.043, +0.000, +0.082]
goal (-0.3,-0.3): v=(-0.212,-0.212) -> w=[-0.037, -0.084, +0.003]
```

**Correlation test (200 random goals, adaptive vel + no *200):**
```
Corr(w0, gx)=-0.074, Corr(w0, gy)=0.958  ← w0 now encodes goal_y!
Corr(w1, gx)=0.676, Corr(w1, gy)=0.655   ← w1 encodes both goal_x and goal_y
Corr(w2, gx)=-0.717, Corr(w2, gy)=0.656  ← w2 encodes goal_x and goal_y
```

**TWO bugs need fixing:**
1. `collect_phase189_fast.py`: `*200` scaling causes wheel saturation → need REMOVE `*200`
2. `eval_phase188_quick.py`: uses WRONG old formula (Phase 122) instead of Phase 164 formula

### 🔍 架構現況
```
Phase 189 broken data flow:
  kP=0.5 → vx=0.15 → *200=30 → w1=5.268 → CLIPPED to 0.5
  ALL goals produce saturated wheel speeds → zero correlation

Fixed data flow:
  v_mag = min(1.5*dist, 0.3) → vx=0.212 → w1=0.037 → proper variation
```

### 🧭 下一步（下次心跳）

**PRIORITY 1: Re-collect phase190 data with FIXED controller**
1. Remove `*200` from `twist_to_contact_wheel_speeds`
2. Use adaptive velocity `v_mag = min(1.5*dist, 0.3)` (from eval scripts)
3. Collect 10k frames with proper goal-wheel correlation
4. Expect: Corr(w0,gy)=0.95+, Corr(w1,gx)=0.65+, Corr(w2,gx)=-0.70+

**PRIORITY 2: Create phase190 train script**
1. Use `GoalConditionedPolicy(state_dim=11)` architecture
2. Train 10-30 epochs on fixed phase190 data
3. Evaluate vs P-controller baseline

**PRIORITY 3: Fix eval script**
1. Update `eval_phase188_quick.py` to use CORRECT Phase 164 formula (no *200)
2. Verify P-controller 100% SR with fixed formula

### 🚫 阻礙
- **phase189 data: ALL wheel speeds saturate** → CORRUPTED, unusable for training
- **eval_phase188_quick.py: wrong twist_to_contact formula** → NEEDS FIX
- **Data collection uses wrong controller** → Need to re-collect with corrected formula

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p189 | Data: 10000 images | FIXED: per-step images (was 50) ✓ |
| p189 | Data: zero correlations | **ROOT CAUSE: `*200` saturates wheel speeds to ±0.5** |
| p189 | Corr(w0,gy)=0.026 | **Near-zero: w0 encodes nothing about goal** |
| p190 | **FIX identified** | **Remove `*200`, use adaptive vel** |
| p190 | Corr(w0,gy)=0.958 | **With fix: w0 NOW encodes goal_y** |

### Git
- New: `scripts/collect_phase189_fast.py` (Phase 189, broken — has *200 bug)
- Modified: `scripts/eval_phase188_quick.py` (pending: fix twist_to_contact)
- Commit pending: Phase 190 — ROOT CAUSE: `*200` scaling saturates ALL wheel speeds to ±0.5; fixed formula gives Corr(w0,gy)=0.958; need re-collect phase190 data
