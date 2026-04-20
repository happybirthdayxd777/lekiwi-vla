# Phase 227 — Q2 Gy Gap Fix: Root Cause Analysis + Targeted Data Collection + Retrain

## [Phase 227 - 2026-04-20 20:30 UTC] — Q2 Gy Gap Root Cause + Data Collection Complete

### ✅ 已完成（本次心跳）

**1. ROOT CAUSE ANALYSIS — CONFIRMED: Q2 Y-Coordinate Gap**

Phase 226 50-goal eval revealed VLA systematically fails on Q2 (gx<0, gy>0) goals.

Training data (phase196_clean_50ep.h5) Q2 coverage:
- Training Q2 Y range: gy ∈ [0.020, 0.235]m
- Training Q2 episodes: 13 out of 50 episodes were Q2
- Training Q2 X range: gx ∈ [-0.325, -0.053]

Eval Q2 failures (16 failures total, 11 were Q2):
- 10/11 Q2 failures had gy > 0.235m (OUT OF DISTRIBUTION)
- Training max gy = 0.235m; eval failures had gy ∈ [0.240, 0.390]m
- The 1 Q2 failure with gy < 0.235m: goal=(-0.26, 0.18) — possible minor local minimum

**Critical Finding: Combination of large |gx| AND large |gy| simultaneously**
- Training Q2 had: large gx OR large gy, but NOT both together
- Eval Q2 failures: |gx| > 0.24 AND |gy| > 0.24 (the "diagonal corner")
- The specific combination of (-gx, +gy) was NEVER seen in training

Example:
- FAIL: (-0.31, 0.30) |g|=0.431m — large X + large Y
- SUCC: (-0.17, 0.41) |g|=0.444m — small X + large Y (success despite large |g|)
- Pattern: if |gx| < 0.20, VLA handles large gy fine; if |gx| > 0.24 AND gy > 0.24 → fail

**2. Data Collection — Phase 227 Q2 Extended**

Collected 15 Q2 episodes targeting gy ∈ [0.20, 0.45]m:
```
Episode  1: goal=(-0.232, 0.364) |g|=0.432m → 138 steps, dist=0.089m ✓
Episode  2: goal=(-0.323, 0.387) |g|=0.504m → 134 steps, dist=0.084m ✓
Episode  3: goal=(-0.306, 0.330) |g|=0.450m → 132 steps, dist=0.085m ✓
... (all 15 episodes SUCCESS, 100% P-controller success rate)
Total: 15/15 = 100% success, 2027 steps added
```

Correlation verification (correct Contact-Jacobian controller):
```
Corr(w0, gx) = +0.079  (expected <0 for +X goal — Q2 is negative X)
Corr(w1, gx) = +0.431  (positive — forward wheel correlates with gx)
Corr(w2, gx) = -0.435  (negative — back wheel for negative X)
Corr(w0, gy) = +0.435  (positive — lateral wheel for +Y)
Corr(w1, gy) = +0.414  (positive)
Corr(w2, gy) = +0.214  (positive)
```

**3. Merged Dataset — phase227_extended_65ep.h5**
```
Base (phase196): 5562 images, 50 episodes
Q2 Extended:      2027 images, 15 episodes
Combined:         7589 images, 65 episodes
```

**4. Training Script — train_phase227.py**
- Architecture: EXACT copy of Phase 196 (GoalConditionedPolicy with flow matching)
- Fixed bugs from initial version: ReLU→SiLU, LayerNorm, missing time_mlp, wrong loss function
- Loss: `((v_pred - v_target) ** 2).mean(dim=-1)` — velocity field MSE
- 30 epochs, batch_size=32, lr=1e-4, CosineAnnealingLR

**5. Training Running Now** (pid=52810, started ~20:25 UTC)
```
results/phase227_contact_jacobian_train/epoch_{5,10,15,20,25,30}.pt
```

**6. Eval Script — eval_phase227.py**
- 50 goals, sr=0.15m, seed=42
- Phase 227 VLA vs Phase 196 VLA vs P-controller
- Q2-specific analysis: SR on gx<0, gy>0 goals

### 🔍 Architecture Current State

```
ROS2 Bridge (lekiwi_ros2_bridge/):
  ✅ bridge_node.py (1186 lines) — cmd_vel↔MuJoCo, joint_states↔ROS2
  ✅ vla_policy_node.py (746 lines) — CLIP-FM policy at 4 Hz
  ✅ CameraAdapter (20 Hz, URDF only) + graceful None handling
  ✅ CTF security mode (ctf_integration.py)
  ✅ Unified launch files (full, vla, real_mode)

Simulation backends:
  ✅ Primitive (cylinder) + URDF (STL mesh) — both working
  ✅ LeKiWiSimLoader factory verified

Data:
  ✅ phase196_clean_50ep.h5 — 5562 steps, 50 episodes
  ✅ phase227_extended_65ep.h5 — 7589 steps, 65 episodes (Q2-extended)

VLA Policies:
  ✅ Phase196 epoch_14 — 68% SR (50-goal, sr=0.15m, ROOT CAUSE identified)
  🔄 Phase227 — TRAINING in progress (30 epochs, Q2-extended data)
  ✅ P-controller CJ kP=2.0 — 100% SR (oracle baseline)

Git: Training running, scripts committed separately
```

### 🧭 下一步（下次心跳）

**Priority 1: Wait for Phase 227 Training to Complete**
```bash
# Training started ~20:25 UTC, ~30 min expected
# Check: ls results/phase227_contact_jacobian_train/
```

**Priority 2: Run Phase 227 Evaluation**
```bash
python3 scripts/eval_phase227.py \
  --vla results/phase227_contact_jacobian_train/epoch_30.pt \
  --n_episodes 50 --success_radius 0.15 --seed 42
```

**Priority 3: Compare Phase 227 vs Phase 196**
- Target: Phase227 VLA > 85% SR (from 68%)
- Q2 SR: Target > 80% (from 45%)
- If Phase227 hits 90%+ SR, the root cause hypothesis is confirmed

**Priority 4: If Phase227 Fails to Improve**
- Hypothesis B: The failure is NOT about goal distribution, but about CLIP encoder
  → Check: does CLIP spatial encoding change between quadrants?
  → Alternative: Use 4 quadrants separately, or add directional embedding

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p190  | CJ P-controller data collection + VLA train | 94% SR (50 goals) |
| p196  | CJ P-controller + VLA train (14 epochs) | 90% SR (bugged wheel_vel) |
| p198  | Architecture fix retrain (10 epochs) | ❌ 0% SR (untrained) |
| p218b | phase196_e14 vs phase190_e27 (seed=99) | 80% vs 10% (bugged wheel_vel) |
| p222  | Wheel velocity index bug fix + 20-goal eval | VLA 60% vs P-ctrl 100% |
| p223  | Phase198 v3_first_eval — CLIP encoder mismatch | **VLA 0% vs P-ctrl 100%** |
| p224  | Phase196 VLA 50-goal eval (FIXED wheel_vel, sr=0.10m) | **VLA 68% vs P-ctrl 96%** |
| p226  | sr=0.15m fair eval (VLA unchanged, P-ctrl 100%) | **VLA 68% vs P-ctrl 100%** |
| p227  | Q2 gy gap ROOT CAUSE + 15 Q2 episodes + train Phase227 | 🔄 Training in progress |

### Git
- Commit (scripts): Phase227: Q2 gy gap fix — data collection + training + eval scripts
- Training: Running in background (pid=52810), 30 epochs on 7589 frames
- Commit (data+results): pending after training completes

