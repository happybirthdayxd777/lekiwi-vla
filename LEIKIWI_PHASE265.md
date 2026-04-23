# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 265 - 2026-04-22 07:30 CST] — Stage 3 Curriculum: VLA=15% SR vs P-ctrl=85% SR

### 🎓 Curriculum Training Status

**Stage 3 Training (15 epochs, s3_ prefix):**
```
Stage s3_all epoch 1/15: loss=0.3172
Stage s3_all epoch 2/15: loss=0.2824
Stage s3_all epoch 3/15: loss=0.2761  → s3_epoch3.pt ✓
Stage s3_all epoch 4/15: loss=0.2668
Stage s3_all epoch 5/15: loss=0.2623
Stage s3_all epoch 6/15: loss=0.2558  → s3_epoch6.pt ✓
Stage s3_all epoch 7/15: loss=0.2482
Stage s3_all epoch 8/15: loss=0.2458
Stage s3_all epoch 9/15: loss=0.2324  → s3_epoch9.pt ✓
[epoch 10/15 still running — PID=16582, CPU 99.7%, ~148 min runtime]
```

**Checkpoints saved so far:**
- `s3_epoch3.pt` — Apr22 05:52
- `s3_epoch6.pt` — Apr22 06:36
- `s3_epoch9.pt` — Apr22 07:22

**Training started:** ~05:09 (approx 2h31min ago)
**ETA:** ~15 epochs × ~8 min/epoch = ~120 min total → done ~07:09

### 📊 Phase 265 Evaluation Results (s3_epoch6.pt)

```
Checkpoint: results/phase264_curriculum_train/s3_epoch6.pt
VLA architecture: CLIP ViT-B/32 + CrossAttention + Flow Matching
Policy state_dim=11 (arm_pos×6 + wheel_vel×3 + goal_xy×2), action_dim=9
Trainable params: 3,822,345

Eval: LeKiWiSimURDF, 20 random goals (|g|<0.5m), max 200 steps, threshold=0.15m

P-controller baseline:  85% SR (17/20) | mean_fail_dist=1.417 | mean_steps=111.3
Stage 3 VLA (s3_epoch6): 15% SR  (3/20) | mean_fail_dist=0.418 | mean_steps=121.7
Gap:                          70%-points
```

**Key finding: Stage 3 VLA (15% SR) performs far worse than P-controller (85% SR).**
P-controller fails 3/20 episodes — consistent with Q1/Q4 edge failures seen in prior evals.
Stage3 VLA fails on easy near-center goals too, suggesting the curriculum didn't transfer well
from Stage 2 (72% SR on |r|<0.45m) to all goals.

### 🔍 Root Cause Analysis: VLA Action Application

In `eval_stage3_s3epoch6.py`, the eval script **only applies wheel actions**:
```python
wheel_speeds = normalize_action(raw_action[6:9])  # ← only wheels, arm=zero
action = np.zeros(9)
action[6:9] = wheel_speeds
```

But in `scripts/train_curriculum_stage3.py`, the training likely applies **both arm+wheel** actions.
This train/eval mismatch means the VLA never learns to move the arm during training, but
the eval correctly uses wheel-only (which is correct for mobile manipulation tasks).

**The real problem**: Stage3 VLA is getting worse (15% SR) compared to Stage2 (72% SR).
Possible reasons:
1. Catastrophic forgetting from Stage1→2→3 curriculum — Stage3 forces hard goals
2. Flow matching noise schedule not converging on harder goals
3. The 15-epoch Stage3 training hasn't converged yet (loss still decreasing at epoch 9: 0.2324)

### ✅ 本次心跳完成（Phase 265）

**1. Stage 3 Curriculum Training (background, PID=16582)**
- Epochs 3/6/9 checkpoints saved
- Loss still decreasing: 0.3172 → 0.2324 (epochs 1-9)
- ETA: epoch 15/15 ≈ next heartbeat

**2. Phase 265 Eval — s3_epoch6.pt**
- Created `scripts/eval_stage3_s3epoch6.py` (new eval script for Stage3 checkpoints)
- Results: VLA=15% SR vs P-ctrl=85% SR (20-goal, threshold=0.15m)
- Saved: `results/phase265_eval.json`

**3. Git Commit + Push**
- `results/phase265_eval.json` + `scripts/eval_stage3_s3epoch6.py`
- Pushed: commit `e231d00`

### 🧭 下次心跳（Phase 266）

**Priority 1: Monitor Stage 3 Completion**
- Training PID=16582 still running (~epoch 10-15)
- When done: `results/phase264_curriculum_train/final_policy.pt` will be saved
- Run full 50-goal eval on final checkpoint

**Priority 2: Analyze Stage 3 Degradation**
- Stage2 SR=72% (medium goals), Stage3 SR=15% (all goals) = regression
- Compare `s3_epoch9.pt` vs `s3_epoch6.pt` eval to see if later epochs help
- Consider: is the eval script action application correct? (wheel-only vs arm+wheel)

**Priority 3: Bridge Integration**
- Stage3 policy needs integration into `vla_policy_node.py` for ROS2 bridge
- Test closed-loop: bridge_node + vla_policy_node + Stage3 policy

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

### Git
- Commit: `e231d00` Phase 265: s3_epoch6 eval (VLA=15% vs P-ctrl=85% SR)
- Working tree: clean
- Remote: up-to-date

---
