# Phase 282 — Stage3 s3_epoch9 50-Goal Evaluation + Policy Architecture Audit

**Date**: 2026-04-23 15:30 CST

## 本次心跳完成

### Phase 282: s3_epoch9 50-Goal Evaluation Launched

**Background Eval Running** (PID=53564, ~40min estimated):
```bash
scripts/eval_stage3_s3epoch9_50goal.py
- 50 goals, sr=0.10m, seed=42, max_steps=300
- P-controller baseline (gold standard)
- Stage3 VLA (s3_epoch9.pt from phase264_curriculum_train)
- Output: results/phase282_s3epoch9_50goal_eval.json
```

**Why 50 Goals?** Phase 281 used only 10 goals (20% SR with high variance). A definitive SR estimate requires 50 goals for ±10% CI at 80% power.

**Prior Results (for reference)**:
- s3_epoch6 (10-goal eval): SR unknown, needs full eval
- s3_epoch9 (10-goal eval, Phase 281): P-ctrl 100%, Stage3 VLA 20% SR (2/10)
- P-controller CJ kP=2.0 (urdf): 86% SR (100% on 10-goal)

---

### Architecture Audit: bridge_node.py + vla_policy_node.py

#### Topic Contract Map (Phase 281 confirmed)

```
bridge_node.py:
  ← /lekiwi/cmd_vel           Twist              — teleop input
  ← /lekiwi/vla_action        Float64MultiArray  — VLA policy output (9D native)
  ← /lekiwi/goal              Point              — goal position (x,y)
  ← /lekiwi/policy_mode       String             — policy selection
  → /lekiwi/joint_states      JointState         — arm×6 + wheel×3 positions/velocities
  → /lekiwi/odom              Odometry           — base odometry
  → /lekiwi/camera/image_raw  Image              — front camera 20Hz (URDF mode)
  → /lekiwi/wrist_camera/image_raw Image        — wrist camera 20Hz (URDF mode)
  → /lekiwi/wheel_N/cmd_vel   Float64×3          — per-wheel velocity

vla_policy_node.py:
  ← /lekiwi/joint_states      JointState         — reads current state
  ← /lekiwi/camera/image_raw  Image              — reads front camera
  ← /lekiwi/goal              Point              — reads goal
  → /lekiwi/vla_action        Float64MultiArray  — publishes 9D native action
```

#### Phase 278 normalize_action Fix 確認

**問題**：Native-unit policies (stage2, stage3, dagger) output in native units (arm torque Nm, wheel rad/s).
  normalize_action() was re-normalizing them → double-normalization.

**Fix** (vla_policy_node.py lines 944-948):
```python
_NATIVE_UNIT_POLICIES = frozenset(["stage2", "stage3", "dagger"])
if hasattr(self, '_policy_name') and self._policy_name in _NATIVE_UNIT_POLICIES:
    native_action = raw_action  # Skip normalize_action
else:
    native_action = normalize_action(raw_action)  # For [-1,1] policies
```

**確認**：bridge_node.py 的 _action_to_ctrl() (sim_lekiwi_urdf.py) 統一處理所有 action：
- action[:6] clip ±3.14 → arm torque normalized -1..1 → ±3.14 Nm
- action[6:9] clip ±0.5 → wheel torque normalized -1..1 → ±5.0 Nm (Phase 70)
- motor gear 10x → joint torque up to 50 Nm

#### Contact-Jacobian P-controller 確認

full.launch.py 預設 use_contact_jacobian=true：
- 使用 _CONTACT_JACOBIAN_PSEUDO_INV 而非舊 kinematick IK
- P-controller SR = 86% (eval baseline)
- VLA policy 在 direction agreement hybrid 中使用 P-controller 作為 fallback

#### DAgger 架構確認

scripts/collect_dagger.py：
- P-controller expert 提供 corrective actions
- VLA 前饋 30 steps，失敗後切换 P-controller
- 記錄 (image, state, expert_action) pairs
- 數據混合：DAgger data + Phase227 base data → retrain

---

### Policy Performance Summary (as of Phase 281)

| Policy | SR | n_goals | sr_threshold | Notes |
|--------|-----|---------|-------------|-------|
| P-controller CJ kP=2.0 | 86% | 50 | 0.10m | Gold standard (URDF sim) |
| P-controller CJ kP=2.0 | 100% | 10 | 0.10m | (Phase 281) |
| VLA-227 | 70% | 50 | 0.10m | Phase 227 contact-jacobian |
| Stage2 r<0.45m | 72% | 50 | 0.10m | Curriculum Stage2 |
| DAgger-254 | 50% | 50 | 0.10m | DAgger < P-controller |
| Stage3 s3_epoch9 | **EVAL** | **50** | **0.10m** | **RUNNING NOW** |
| Stage3 s3_epoch12 | 0-15% | 10 | 0.10m | Wheel actions collapsed |

---

### 下一步

- [ ] **Phase 283**: Collect Stage3 數據（100+ episodes with ALL goals, once SR is confirmed）
- [ ] **Phase 284**: If Stage3 SR > Stage2 (72%), retrain Stage3 with more data
- [ ] **Phase 285**: Stage2 整合進 ROS2 bridge（72% SR as production fallback）
- [ ] **Phase 286**: full.launch.py end-to-end test (needs ROS2 environment)

### 阻礙

- Stage3 s3_epoch9 wheel actions may be collapsed (0-20% SR in prior evals)
- No ROS2 environment for end-to-end testing
- DAgger needs 200+ episodes to beat P-controller

---

## Git

- Commit: scripts/eval_stage3_s3epoch9_50goal.py — 50-goal eval for s3_epoch9
