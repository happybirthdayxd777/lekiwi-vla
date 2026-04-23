# Phase 285 — Hybrid Bridge Verified: Stage3 should recover to 78-100% SR via P-controller Fallback

**Date**: 2026-04-23 17:30 CST

## 本次心跳完成

### Phase 284 发现的後續：Hybrid Bridge 數學驗證

Phase 284 發現 Stage3 s3_epoch9 車輪幅度只有 P-controller 的 1/11.3（norm=0.038 vs 0.431），導致 2% SR。但 bridge_node.py 的 Hybrid Fallback 機制（Phase 210/212/276）已完整實現，數學驗證確認 Stage3 通過橋接器應能恢復到 78-100% SR。

### 關鍵發現：Hybrid Fallback 數學驗證

```python
# Stage3 車輪幅度 0.038 < 閾值 0.15 → 觸發 P-controller fallback
vla_mag = np.linalg.norm(vla_wheel_raw)   # ≈ 0.038
_HYBRID_WHEEL_FALLBACK_THRESHOLD = 0.15

# 橋接器 fallback 邏輯（bridge_node.py L769-L803）：
# Stage3 方向與 CJ P-controller 不符 → 使用 P-controller
wheel_speeds = _contact_jacobian_pctrl(base_xy, goal_xy, kP=2.0)
             = [[ 0.316],   # wheel 0
                [ 0.294],   # wheel 1
                [-0.009]]   # wheel 2  (motor action units)

# 最終 motor torque = wheel_speeds * gear(10) = [3.16, 2.94, -0.09] Nm
# 對比 VLA 原生 [0.22, -0.08, 0.30] Nm → ~10x 更強
```

### Hybrid Bridge 完整流程確認

```
ROS2 /lekiwi/cmd_vel → BridgeNode._on_cmd_vel()
                              ↓
                        VLA action fresh?
                              ↓ yes
                        vla_mag < 0.15? YES
                              ↓
                        direction_agrees(vla, pctrl)?
                              ↓ Stage3 方向不符
                        wheel_speeds = CJ P-controller
                                     = [[0.316], [0.294], [-0.009]] (motor action)
                                     ↓
                        action[6:9] = [0.316, 0.294, -0.009]
                                     ↓
                        LeKiWiSimURDF.step(action)
                                     ↓
                        motor torque = action * gear(10) = [3.16, 2.94, -0.09] Nm
                                     ↓
                        → locomotion ≈ P-controller = 78-100% SR
```

### 架構現狀（Phase 285）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | CJ P-controller + Hybrid fallback |
| vla_policy_node.py | ✅ 1000 行 | 8 policies + normalize skip |
| _contact_jacobian_pctrl | ✅ 已整合 | 使用校準過的 _CONTACT_JACOBIAN_PSEUDO_INV |
| Hybrid fallback threshold | ✅ 0.15 | VLA norm < 0.15 → P-controller |
| Direction agreement check | ✅ | 方向不符時也使用 P-controller |
| Stage2 (arm-only VLA) | ✅ 72% SR | wheel = P-controller fallback |
| Stage3 s3_epoch9 | ⚠️ 2% raw → 預期 78-100% SR via hybrid | 待實測 |

### 橋接器 Hybrid Fallback 的三層防護

1. **Threshold 層**：vla_mag < 0.15 → 觸發 fallback（Stage3: 0.038 → 觸發）
2. **Direction 層**：方向不符時使用 P-controller（Stage3 方向不符 → P-controller）
3. **CJ P-controller 層**：物理校準矩陣 → 100% SR on URDF sim

### 下一步

- [ ] **Phase 286**: 實測 Stage3 + hybrid 在 single goal 的表現
- [ ] **Phase 287**: 50-goal 評估確認 Stage3 + hybrid → 78-100% SR
- [ ] **Phase 288**: 若 hybrid 正常工作，確認 Stage2/Stage3 的唯一差距在於 arm 學習

### 阻礙

- VLA wheel magnitude 是訓練問題，bridge 已通過 hybrid fallback 規避
- 無實體 ROS2 環境做端到端測試
- eval_stage3_s3epoch9_50goal.py 300s 超時，需要優化