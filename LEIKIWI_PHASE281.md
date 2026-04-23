# Phase 281 — Architecture Code Audit + DAgger Strategy Analysis

**Date**: 2026-04-23 14:00 CST

## 本次心跳完成

### Phase 281: Code Audit — bridge_node.py + vla_policy_node.py Topic Contracts

**審計結論：架構無 further bugs，Phase 278 normalize_action fix 確認正確**

#### Topic Contract Map（已驗證）

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

### Phase 282: DAgger 數據增強策略分析

#### 根本問題：DAgger-254 = 50% SR < Phase227 = 70% SR

DAgger 失敗不是因為演算法，而是數據策略：

| Factor | DAgger-254 | Phase227 |
|--------|------------|----------|
| 訓練數據 | 50ep DAgger + 50ep base | 65ep base |
| 訓練epochs | 20 | 30 |
| 最佳loss | 0.0018 (epoch 18) | ~0.003 |
| SR (50 goals) | 50% | 70% |

**Root Cause**：DAgger 數據混合了來自同一個不完美的 P-controller專家數據。當 VLA 在某些狀態已經失敗時，專家糾正並不能提供真正多樣的視覺特徵。

#### Stage2 Curriculum 更有效

Stage2 (|r|<0.45m) = **72% SR**，而 DAgger (ALL goals) = **50% SR**。

原因：
1. 約束目標更容易學習 — 視覺特徵相似性高
2. 數據分佈更集中 — policy 更容易擬合
3. Stage3 (ALL goals) overfits — s3_epoch9 最好，epoch 12+ 開始過擬合

#### DAgger 改進策略（可選）

**Strategy A：更大的初始數據集**
- 收集 200+ episodes DAgger 數據（而非 50）
- 目標：多樣化涵蓋更多 goal 配置

**Strategy B：更好的專家**
- 目前專家 = P-controller (86% SR)
- 改進專家 = Stage2 policy + P-controller 混合
- OR 使用燒烤策略：多次專家rollout選擇最佳

**Strategy C：視覺數據增強**
- 當前：DAgger 只糾正 wheel actions，不改變視覺輸入
- 可選：模擬不同光照/紋理/目標大小，擴展視覺多樣性

**Strategy D：放棄 DAgger，專注 Stage2/Stage3**
- Stage2 已達 72% SR (|r|<0.45m)
- Stage3 需要更多數據（目前 7589 frames 不夠）
- 下一階段：收集 Stage3 數據（100+ episodes, ALL goals）

---

### 下一步

- [ ] Phase 283: 評估 s3_epoch9 在 URDF 上的 SR
- [ ] Phase 284: 收集 Stage3 數據（100+ episodes with ALL goals）
- [ ] Phase 285: Stage2 整合進 ROS2 bridge（72% SR 可作為 production fallback）

### 阻礙

- DAgger 需要大量額外數據（200+ episodes）才能超越 Phase227
- Stage3 需要更多數據才能泛化到邊緣目標

## Git

- No code changes — analysis only
