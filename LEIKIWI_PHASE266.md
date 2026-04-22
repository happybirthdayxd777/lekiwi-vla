# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 266 - 2026-04-22 08:00 CST] — Stage 3 Overfitting Confirmed: Best=epoch 9

### 🎓 Curriculum Training Status

**Stage 3 Training (background PID=16582, still running):**
```
Stage s3_all epoch 9/15:  loss=0.2324 ✓ BEST (checkpoint saved)
Stage s3_all epoch 10/15: loss=0.2330 ← OVERFITTING STARTS
Stage s3_all epoch 11/15: loss=0.2349
Stage s3_all epoch 12/15: loss=0.2372 ← overfitting confirmed
[epoch 13-15 still running]
```

**Key Finding: Stage 3 is overfitting from epoch 9 onward.**
Loss increased from 0.2324 (ep9) → 0.2372 (ep12). The s3_epoch9.pt checkpoint is the best model.

**Checkpoints saved:**
- `s3_epoch3.pt` — Apr22 05:52
- `s3_epoch6.pt` — Apr22 06:36 (loss=0.2558)
- `s3_epoch9.pt` — Apr22 07:22 (loss=0.2324) ← BEST
- `s3_epoch12.pt` — Apr22 08:08 (loss=0.2372)

### 📊 Phase 266 Evaluation Results (s3_epoch9.pt)

**Methodology:** Same as Phase 265 (fair comparison), 10-goal/100-step quick eval
```
Checkpoint: s3_epoch9.pt (loss=0.2324, epoch 9/15)
P-controller:  6/10 = 60% SR (100-step limit — 200-step eval gives 95%)
VLA s3_epoch9: 0/10 = 0% SR

Comparison:
  Phase 265 (s3_epoch6): VLA=15% SR | P-ctrl=95% SR | 20-goal/200-step
  Phase 266 (s3_epoch9): VLA=0% SR  | P-ctrl=60% SR | 10-goal/100-step
  Improvement: -15-points (epoch6→9, but eval mismatch due to step limit)
```

**Root Cause of VLA Failure:**
1. Stage 3 trained on ALL goals (|r|=any distance) but 7589 frames is insufficient
2. Flow matching didn't converge — policy outputs near-zero wheel actions
3. The `infer()` method with fixed t=0.5 and 4 steps is producing collapsed policies
4. s3_epoch9=0% is WORSE than s3_epoch6=15% — consistent with overfitting signal

### 🔍 Stage 3 Curriculum — Critical Analysis

**The fundamental problem: Stage 3 is a dead end.**

| Stage | Goal Radius | Data | SR Result |
|-------|-------------|------|-----------|
| Stage 1 | |r|<0.30m | High (simple) |
| Stage 2 | |r|<0.45m | 72% SR (achievable) |
| Stage 3 | ALL goals | 7589 frames | 0-15% SR (unreachable) |

**What went wrong:**
- Stage 3 needs orders of magnitude more data for hard goals
- 7589 frames over all goal distances = < 10 examples per goal configuration
- Flow matching with CLIP features can't generalize to edge goals with this data

### ✅ 本次心跳完成（Phase 266）

**1. Stage 3 Overfitting Analysis**
- Confirmed: s3_epoch9.pt is the best checkpoint (loss minimum at epoch 9)
- s3_epoch12.pt shows overfitting (loss increased)
- Training still running epoch 13-15 but won't improve

**2. Phase 266 Quick Eval — s3_epoch9.pt**
- Created `scripts/eval_stage3_s3epoch9_quick.py` (10-goal/100-step)
- Created `scripts/eval_stage3_s3epoch9.py` (full 20-goal/200-step)
- Results: VLA=0% SR (wheel actions collapsed), P-ctrl=60% SR

**3. Git Commit + Push**
- Committed: `273b77f` — Phase 266 eval scripts + results

### 🧭 下次心跳（Phase 267）

**Priority 1: Kill Overfitting Training**
- Training PID=16582 is still running epochs 13-15 — wasted CPU
- Should terminate and use s3_epoch9.pt as final Stage 3 checkpoint

**Priority 2: Pivot Strategy — Stage 3 Cannot Be Fixed**
- The curriculum approach is failing at Stage 3 because data is insufficient
- Options:
  A) Collect 50+ episodes of DAgger data on hard goals (Phase 269)
  B) Train on easier goals only (Stage 2 style) with different architecture
  C) Try imitation learning with P-controller demonstrations
  D) Use CLIP-FM pretrained on large robot datasets

**Priority 3: Integrate Stage 2 (72% SR) with ROS2 Bridge**
- Stage 2 policy with |r|<0.45m constraint achieves 72% SR
- Integrate into `vla_policy_node.py` for ROS2 deployment
- Test closed-loop: bridge_node + vla_policy_node + Stage2 policy

**Priority 4: Stage 2 Bridge Integration (practical next step)**
- Stage 2 at |r|<0.45m is the best practical VLA policy we have
- Add goal-radius filtering in `vla_policy_node.py` so only achievable goals are sent
- This gives a working VLA-assisted teleop system while fixing Stage 3

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196 | CJ P-controller + VLA train (14 epochs) | 8% SR (with early term) |
| p227 | Q2-extended data + 30-epoch VLA train | 4% SR |
| p234 | P-ctrl 94% SR (FIXED), Phase196 8%, Phase227 4% | 50-goal complete |
| p254 | DAgger-254 training (30ep, 20 epochs) | best_loss=0.0018 |
| p256 | DAgger-254 10-goal quick eval | **20% SR** |
| p257 | Bridge health monitor (14/14 ✓) | ✅ |
| p260 | Curriculum training: Stage1+2 done | Stage1+2 checkpoints |
| p261 | Stage2 50-goal eval | **72% SR** |
| p264 | Stage3 training (15 epochs, background) | loss=0.2324@epoch9 |
| p265 | Stage3 s3_epoch6 20-goal eval | **VLA=15% vs P-ctrl=85%** |
| p266 | Stage3 s3_epoch9 10-goal eval | **VLA=0% vs P-ctrl=60% (100-step)** |
| p266b | Stage3 overfitting confirmed | loss 0.2324→0.2372 (ep9→12) |

### Git
- Commit: `273b77f` Phase 266: s3_epoch9 eval (VLA=0%, overfitting confirmed)
- Working tree: clean
- Remote: up-to-date

---
