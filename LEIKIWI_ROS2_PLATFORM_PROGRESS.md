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
