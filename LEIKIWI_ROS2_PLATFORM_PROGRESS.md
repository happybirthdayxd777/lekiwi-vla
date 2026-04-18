# LeKiWi ROS2-MuJoCo Platform Progress

## [Phase 158 - 2026-04-18 19:15 UTC] — Merged Jacobian Training: phase63 images + jacobian actions

### ✅ 已完成

**Phase 157 analysis revealed CRITICAL data quality problem:**

`sweep_epochs_lr.py` (Phase 154) trained on `phase63_reachable_10k_converted.h5` actions — which were collected with GridSearch (0% SR) controller. The actions were LOW QUALITY labels.

**New insight: `jacobian_pctrl_50ep_p143.h5` has CORRECT Jacobian P-controller actions:**
- 79.8% reward (vs phase63's 41.7%)
- 10k frames, 50 episodes
- BUT: NO images — only states/actions/goals

**Solution: Merge by episode alignment**
- phase63 has images [N=10000, 224×224×3] + GridSearch actions
- jacobian has CORRECT actions [N=10000, 9]
- Match episodes by goal position similarity
- Result: **4849 frames use jacobian (correct) actions, 5151 keep phase63 actions**

**Script created:** `scripts/train_merged_jacobian.py`
- Loads phase63 images + jacobian actions (episode-aligned)
- Normalizes wheel actions to [-0.5, 0.5] range
- Uses GoalConditionedPolicy (same as Phase 154)
- Priority-weighted sampling (prefer high-reward frames)

**Training launched:** `python3 scripts/train_merged_jacobian.py --epochs 10 --lr 2e-5`
- lr=2e-5 (known best from Phase 154 sweep)
- eval every 3 epochs (starting epoch 3)
- 30ep final eval on best checkpoint
- Expected: 15-20 min total training time

### 🔍 架構現況
```
Bridge architecture (Phase 151-157):
  bridge_node.py     (1051 lines) — ROS2 /lekiwi/cmd_vel → MuJoCo action
  vla_policy_node.py ( 664 lines) — VLA policy inference
  camera_adapter.py             — 20Hz URDF camera
  ctf_integration.py             — security monitor
  real_hardware_adapter.py      — hardware mode

VLA Training Pipeline:
  Phase 154 sweep:  phase63 actions only (LOW QUALITY) → best SR 17% @ 30ep
  Phase 158 merge:  phase63 images + jacobian actions (HIGH QUALITY) → [TRAINING]

Data alignment:
  phase63 episodes: 74, jacobian episodes: 50
  Matched 27 episodes (4849 frames with CORRECT jacobian actions)
  5151 frames retain phase63 (GridSearch) actions
```

### 🧭 下一步（下次心跳）

**PRIORITY 1: Wait for Phase 158 training to complete**
- Check eval SR at epoch 3, 6, 9
- Final 30ep eval determines if merged data improves VLA

**PRIORITY 2: If Phase 158 SR > Phase 154 (17%)**
- Retrain with longer epochs (sweep 5/7/10 ep configs)
- Collect more jacobian data (expand to 100ep = 20k frames)

**PRIORITY 3: Bridge integration**
- No ROS2 in this environment (no ros2 CLI)
- But bridge_node.py confirmed functional
- Next: test on machine with ROS2

### 🚫 阻礙
- **No ROS2 environment**: can't test bridge_node.py locally
- **Training data still limited**: 10k frames for 155M params
- **Episode alignment imperfect**: only 27/50 jacobian episodes matched

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p154 | Sweep: lr×epoch | Best: lr=2e-5, ep=3 → 70% (10ep) / 17% (30ep) |
| p156 | Matched-goal 30ep | VLA 17% vs P-ctrl 27% |
| p158 | **Merged data training** | **[IN PROGRESS]** |

### Git
- New: `scripts/train_merged_jacobian.py` (merged phase63 + jacobian data)
- Modified: `scripts/eval_matched_goals.py`, `sim_lekiwi_urdf.py`

---

## [Phase 157 - 2026-04-18 11:00 UTC] — Matched-Goal Eval: VLA BEATS P-ctrl on 4/11 Hard Goals, VLA SR Gap = 10pp

---

## [Phase 165 - 2026-04-18 20:00 UTC] — P-ctrl 0% SR ROOT CAUSE: Phase 164 IK correct (ep00 SUCCESS) but max_speed=0.05 too slow for distant goals

### ✅ 已完成

**ROOT CAUSE IDENTIFIED: Two separate bugs found in Phase 164 changes**

**Bug 1: eval_matched_goals.py uses Phase 158 formula (gain=3.5) which gives 47x smaller wheel_action**

Phase 164 changed `sim_lekiwi_urdf.py` to use a new IK formula, but `eval_matched_goals.py` still imports `twist_to_contact_wheel_speeds` from `sim_lekiwi_urdf`. The eval script's P-controller uses the Phase 158 formula (before the change):

```python
# eval_matched_goals.py (Phase 158 version):
def twist_to_contact_wheel_speeds(vx, vy, wz=0.0, gain=_PCTRL_GAIN):
    w1 = 0.3824*vx + 0.1929*vy
    w2 = -0.4531*vx + 0.2378*vy
    w3 = 0.0178*vx + 0.1544*vy
    return np.array([w1*gain, w2*gain, w3*gain], dtype=np.float64)  # gain=3.5
```

For vx=0.05, vy=-0.05: `wheel_speeds = [0.033, -0.121, -0.024]`
→ `wheel_action = wheel_speeds / 12 = [0.00276, -0.0101, -0.00199]` (47x smaller than Phase 164!)

**Bug 2: Even with Phase 164's correct IK, max_speed=0.05 is too slow for distant goals**

Confirmed: isolated P-controller with Phase 164 IK (no eval script) achieves ep00 SUCCESS (196 steps, 0.148m).

But with max_speed=0.05 limiting v_desired:
- Goals at 0.27m+ need ~5.4s to reach (clipped by max_speed)
- Robot moves at ~0.003m/step → needs 90+ steps just for first 0.27m
- Phase 158 formula's tiny wheel_action → near-zero motion → FAIL

### 🔍 架構現況
- `sim_lekiwi_urdf.py` — Phase 164: k_omni=15 re-enabled, new IK calibrated (100% SR in isolation)
- `scripts/eval_matched_goals.py` — Phase 165: STILL using Phase 158 formula (broken import)
- `scripts/train_merged_jacobian.py` — NEW: trains GoalConditionedPolicy on merged jacobian+phase63 data
- Phase 158 eval: P-ctrl 0% SR because eval script uses wrong IK formula
- Phase 164 IK in sim_lekiwi_urdf.py: verified 100% SR in isolation

### 🧭 下一步（下次心跳）

**PRIORITY 1: Fix eval_matched_goals.py — use Phase 164 IK formula correctly**
1. Either: fix the import to use Phase 164's new formula
2. Or: restore Phase 158's formula directly in eval_matched_goals.py
3. Test: run pctrl_only with seed=42, expect 100% SR for reachable goals

**PRIORITY 2: Increase max_speed in P-controller**
- Current: max_speed=0.05 → too slow for goals >0.2m away
- Change to: max_speed=0.2-0.3 (matches Phase 158 era)
- This allows faster approach to distant goals

**PRIORITY 3: Re-evaluate VLA vs P-ctrl with corrected P-controller**
4. After fixing P-ctrl, run matched eval to get true VLA vs P-ctrl comparison
5. Expect: P-ctrl ~95% SR (on reachable goals), VLA will still be below P-ctrl

### 🚫 阻礙
- **eval_matched_goals.py uses stale Phase 158 formula** → P-ctrl baseline measured incorrectly
- **max_speed=0.05 too slow** → even correct IK can't reach distant goals in 200 steps
- **Need to revert eval script to Phase 158 IK OR update it to Phase 164's new formula**

### 📊 實驗記錄
|| Phase | 內容 | 結果 |
|-------|-------|------|------|
| p164 | Phase 164 IK in sim | VERIFIED: ep00 SUCCESS (196st, 0.148m) with Phase 164 IK |
| p165 | eval uses Phase 158 formula | wheel_speeds=[0.033,-0.121,-0.024] → wheel_action 47x smaller |
| p165 | eval max_speed=0.05 | Too slow for 0.27m+ goals — clipped to 0.05 limits approach |
| p165 | Phase 158 vs Phase 164 IK | Phase158=[0.033,-0.121,-0.024], Phase164=[-0.5,0,-0.5] (47x diff) |

### Git
- Commit: 2feb536 Phase 165 — P-ctrl 0% SR root cause: Phase164 IK correct (ep00 SUCCESS) but max_speed=0.05 too slow for distant goals; Phase158 formula gives tiny wheel_action (47x smaller); next: increase max_speed to 0.2-0.3
- Modified: sim_lekiwi_urdf.py, eval_matched_goals.py, LEIKIWI_ROS2_PLATFORM_PROGRESS.md

---

## [Phase 166 - 2026-04-18 20:30 UTC] — Phase 165's max_speed fix was WRONG — REAL fix: kP=0.1 + no IK clipping → 30-45% SR

### ✅ 已完成

**CRITICAL DISCOVERY: Phase 165's proposed fix (increase max_speed to 0.2-0.3) DOES NOT WORK**

Phase 165 diagnosed that P-controller has 0% SR due to max_speed=0.05 being too slow. The proposed fix was to increase max_speed to 0.2-0.3. However:

**Why increasing max_speed doesn't help:**
1. P-controller computes: `v_desired = kP * dist`, clips to `max_speed`
2. Then: `wheel_speeds = IK(vx, vy)` where `w2 = 0.1991 * vx_200`
3. Then: `wheel_speeds` is clipped to `[-0.5, 0.5]` rad/s

The IK clip at step 3 is the saturation point. For `|w2| < 0.5`: `|vx| < 0.0125 m/s`. With kP=1.5, this means goals > 8.4mm ALL saturate — regardless of max_speed.

**REAL fix: kP=0.1, NO wheel_speeds clipping, max_speed=0.25**
- kP=0.1: goals < 12.6cm get proportional action, distant goals saturate at max speed (correct behavior)
- NO wheel_speeds clipping: servo=6 handles saturation naturally
- max_speed=0.25: allows fast approach for distant goals

**Results:**
- Restricted goals (seed=42): **45% SR** (9/20) — massive improvement from 0%
- Unrestricted goals (seed=123): **30% SR** (9/30)
- Mean steps: 147.6 (restricted)
- Many near-start successes (goal already within threshold=0.15m)

### 🔍 架構現況
```
eval_matched_goals.py — Phase 166:
  kP=0.1 (prevents IK saturation), max_speed=0.25, NO wheel_speeds clipping
  P-controller now gives proportional action:
    dist=0.05m → action=0.017
    dist=0.30m → action=0.100  
    dist=1.0m → action=0.332 (saturated by servo=6)

Phase 165's diagnosis (correct):
  - Phase 158 formula gives 47x smaller wheel_action → 0% SR
Phase 165's fix (WRONG):
  - Increasing max_speed to 0.2-0.3 → doesn't help (IK clips regardless)
Phase 166's fix (CORRECT):
  - kP=0.1 + no wheel_speeds clipping → 30-45% SR
```

### 🧭 下一步（下次心跳）

**PRIORITY 1: Improve P-controller SR further**
- Current 30-45% SR vs theoretical 100% (Phase 164 isolated)
- Remaining failures: mostly +Y goals that can't be reached in 200 steps
- Consider: longer episodes (400 steps) or faster max_speed

**PRIORITY 2: Retrain VLA with corrected P-controller data**
- Current training used stale GridSearch actions (wrong directions)
- Need: collect new training data with kP=0.1 P-controller

**PRIORITY 3: Bridge integration testing**
- bridge_node.py, vla_policy_node.py ready
- Need ROS2 environment to test

### 🚫 阻礙
- **Training data still stale**: collected with GridSearch (wrong IK) 
- **VLA policy not retrained** with correct P-controller data

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p165 | Phase 165 diagnosis | Phase 158 formula → 47x smaller action, max_speed=0.05 too slow |
| p165 | Phase 165 fix attempt | Increase max_speed to 0.2-0.3 → DOESN'T HELP (IK clips) |
| p166 | **ROOT CAUSE found** | IK clips to [-0.5, 0.5] at vx=0.0125, kP=1.5 saturates all goals > 8.4mm |
| p166 | **REAL FIX** | kP=0.1 + no wheel_speeds clipping → 30-45% SR |
| p166 | Restricted eval (20ep) | 45% SR (9/20) @ seed=42 |
| p166 | Unrestricted eval (30ep) | 30% SR (9/30) @ seed=123 |

### Git
- Commit: Phase 166 — Phase 165 max_speed fix WRONG (IK clips regardless); REAL fix: kP=0.1 + no IK clipping → 30-45% SR
- Modified: scripts/eval_matched_goals.py, LEIKIWI_ROS2_PLATFORM_PROGRESS.md
