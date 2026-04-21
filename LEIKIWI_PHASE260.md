# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 260 - 2026-04-22 01:30 CST] — Curriculum Training Stage 2 Done, Stage 3 Running

### 🎓 Curriculum Training Progress

**Stage 1 (|r| ≤ 0.25m, 5 epochs):** ✅ Completed → `stage1_r025.pt` (01:40)
**Stage 2 (|r| ≤ 0.45m, 10 epochs):** ✅ Completed → `stage2_r045.pt` (00:48)
**Stage 3 (ALL goals, 15 epochs):** 🔄 Running (started ~00:48, ETA ~15 min)

```
python3 scripts/train_curriculum.py \
  --epochs_1 5 --epochs_2 10 --epochs_3 15 \
  --batch_size 32 \
  --output results/phase260_curriculum_train
PID=2359 | Runtime: ~25 min | Stage 3 epoch 1/~15
```

**Base checkpoint:** `results/phase227_contact_jacobian_train/epoch_30.pt`
**Data:** `data/phase227_extended_65ep.h5` (7589 frames, 65 episodes)

### 🔍 Key Architecture Finding: URDF Wheel Joints

**lekiwi_modular URDF** (`lekiwi.urdf.resolved`) has 50 joints:
- **3 wheel joints** (continuous): `ST3215_Servo_Motor-v1-2_Revolute-60`, `ST3215_Servo_Motor-v1-1_Revolute-62`, `ST3215_Servo_Motor-v1_Revolute-64`
- **6 arm joints** (continuous revolute): shoulder, elbow, wrist pitch/roll
- **All others**: fixed structural joints

**omni_controller.py** publishes to `/lekiwi/wheel_{i}/cmd_vel` (Float64 × 3) — this is the exact interface that `bridge_node.py` mirrors.

**Bridge topic contract** (Phase 257 confirmed):
- Bridge subscribes: `/lekiwi/cmd_vel` ← from nav/teleop
- Bridge publishes: `/lekiwi/wheel_N/cmd_vel` ← mirrors real robot interface

**Next for URDF-Mujoco bridge:** Need to confirm which arm joints correspond to the 6 action dimensions in the VLA policy output.

### ✅ 本次心跳完成（Phase 260）

**1. Curriculum Training Stage 1 + 2 Complete**
- Stage 1 (5 epochs, easy goals |r|<0.25m) → `stage1_r025.pt`
- Stage 2 (10 epochs, medium goals |r|<0.45m) → `stage2_r045.pt`
- Stage 3 (15 epochs, all goals) is running

**2. Committed `scripts/train_curriculum.py`**
- 486 lines, multi-stage curriculum training
- Trains CLIP-FM VLA with 3 difficulty stages
- Freeze CLIP → train goal_mlp/state_net/cross_attn/flow_head

**3. Git Push**
- Working tree: clean (train_curriculum.py committed)
- Branch: main, up-to-date with origin

**4. Verified Bridge Topic Contract**
- Bridge mirrors `omni_controller.py` exactly (/wheel_N/cmd_vel)
- 50 URDF joints identified (3 wheel continuous + 6 arm + 41 fixed)

### 🧭 下次心跳（Phase 261）

**Priority 1: Wait for Stage 3 Completion**
- `final_policy.pt` and `best_policy.pt` will be saved when done
- Evaluate: `scripts/eval_dagger.py --policy phase260 --n_goals 50`

**Priority 2: VLA Closed-Loop Evaluation**
- Phase 260 curriculum policy needs 50-goal eval (sr=0.10m, early termination)
- Compare with Phase 227 (30 epochs, no curriculum) = 4% SR

**Priority 3: Git Commit + Push**
```bash
cd ~/hermes_research/lekiwi_vla && git add -A && git commit -m "Phase 260: curriculum train stage 3 running" && git push
```

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196 | CJ P-controller + VLA train (14 epochs) | 8% SR (with early term) |
| p227 | Q2-extended data + 30-epoch VLA train | 4% SR |
| p234 | P-ctrl 94% SR (FIXED), Phase196 8%, Phase227 4% | 50-goal complete |
| p254 | DAgger-254 training (30ep, 20 epochs) | best_loss=0.0018 |
| p256 | DAgger-254 10-goal quick eval | **20% SR** |
| p257 | Bridge health monitor (14/14 ✓) | ✅ |
| p260 | Curriculum training: Stage1+2 done, Stage3 running | PID=2359 |

### Git
- Commit: `fbb7068` Phase 256: DAgger-254 20% SR
- Phase 260 new files committed: `scripts/train_curriculum.py`
- Working tree: clean
