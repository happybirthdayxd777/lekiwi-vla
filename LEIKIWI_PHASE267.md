# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 267 - 2026-04-22 08:30 CST] — Stage 2 + Stage 3 Policy Loaders Added to ROS2 Bridge

### 🎓 Bridge Policy Integration — Phase 267

**Problem**: `vla_policy_node.py` had NO loader for the curriculum-trained `stage2` (72% SR) 
and `stage3` (0-15% SR) policies. These policies existed as checkpoints but couldn't be 
deployed via the ROS2 bridge.

**Solution**: Added `stage2` and `stage3` policy loaders to `vla_policy_node.py`:

| Added | Description |
|-------|-------------|
| `Stage2PolicyRunner` | Inference runner: same 4-step Euler as `Phase196PolicyRunner` |
| `_make_stage2_policy()` | Loads `GoalConditionedPolicy` from `scripts/train_curriculum.py` |
| `_make_stage2_wrapper()` | Wraps policy in `Stage2PolicyRunner` |
| `_make_stage3_policy()` | Loads `GoalConditionedPolicy` from `scripts/train_curriculum_stage3.py` |
| `_make_stage3_wrapper()` | Wraps policy in `Stage2PolicyRunner` (same infer interface) |

**Checkpoint defaults**:
- `stage2` → `results/phase260_curriculum_train/stage2_r045.pt` (loss=0.2938, epoch=s2_10)
- `stage3` → `results/phase264_curriculum_train/s3_epoch9.pt` (loss=0.2324, BEST)

**Both verified**: Policies load with 0 missing/unexpected keys.

**Launch file updates**:
- `vla.launch.py`: default changed `clip_fm` → `stage2`, docs updated
- `full.launch.py`: `stage2` and `stage3` added to policy list

**Usage**:
```bash
# Stage 2 — 72% SR on |r|<0.45m goals
ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=stage2

# Stage 3 — all goals (best checkpoint s3_epoch9.pt)
ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=stage3

# Full bridge + Stage2 VLA
ros2 launch lekiwi_ros2_bridge full.launch.py policy:=stage2
```

### 🔍 Architecture State: Phase 267

| Component | Status | File |
|-----------|--------|------|
| `bridge_node.py` | ✅ 1260 lines, primitive + URDF modes | `src/lekiwi_ros2_bridge/` |
| `vla_policy_node.py` | ✅ 726 lines, now with stage2/stage3 | `src/lekiwi_ros2_bridge/` |
| CTF Security Layer | ✅ Phase 239-243, C1-C8 challenges | `ctf_integration.py` |
| Camera Adapter | ✅ URDF mode 20Hz RGB | `camera_adapter.py` |
| Real Hardware Adapter | ✅ Real hardware interface | `real_hardware_adapter.py` |
| 5× Launch Files | ✅ bridge/vla/ctf/full/real_mode | `launch/` |
| **Stage2 loader** | ✅ **NEW Phase 267** | `_make_stage2_wrapper` |
| **Stage3 loader** | ✅ **NEW Phase 267** | `_make_stage3_wrapper` |

### ✅ 本次心跳完成（Phase 267）

**1. Stage2 + Stage3 Policy Loaders**
- Added `Stage2PolicyRunner` + `_make_stage2_policy/wrapper` to `vla_policy_node.py`
- Added `_make_stage3_policy/wrapper` to `vla_policy_node.py`
- Verified: both policies load cleanly (0 missing/unexpected keys)
- Stage2 default: `results/phase260_curriculum_train/stage2_r045.pt` (72% SR)
- Stage3 default: `results/phase264_curriculum_train/s3_epoch9.pt` (best loss=0.2324)

**2. Launch File Updates**
- `vla.launch.py`: default=`stage2`, docs updated
- `full.launch.py`: stage2/stage3 added to policy list

**3. Git Commit + Push**
- Committed: `512fd52` — Phase 267: stage2 + stage3 curriculum policy loaders

### 🧭 下次心跳（Phase 268）

**Priority 1: Quick Eval — Stage2 VLA via bridge_node**
- No ROS2 environment, but can test offline:
  - Load `Stage2PolicyRunner` with `stage2_r045.pt`
  - Run 10-goal eval in simulation mode
  - Verify goal-radius filtering works (only goals |r|<0.45m should succeed)

**Priority 2: Kill Overfitting Training**
- Training PID=16582 still running epochs 13-15 (overfitting — loss increased after epoch 9)
- Should terminate: `kill 16582`
- s3_epoch9.pt is the definitive best checkpoint for stage3

**Priority 3: Stage2 Goal-Radius Filtering in Bridge**
- `Stage2PolicyRunner` should check `|goal_xy|` before running inference
- If `|goal_xy| > 0.45`, fall back to P-controller or reject goal
- This prevents sending unachievable goals to stage2 policy

**Priority 4: DAgger Data Collection (for Stage 3 fix)**
- Stage 3 needs 50+ episodes on hard goals to improve beyond 15% SR
- Collect with P-controller as expert, DAgger correction loop
- New data → retrain Stage 3 with more coverage

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
| p267 | **Stage2+Stage3 loaders added to bridge** | ✅ |

### Git
- Commit: `512fd52` Phase 267: Add stage2 + stage3 curriculum policy loaders to vla_policy_node
- Working tree: clean
- Remote: up-to-date (pushed with rebase)

---
