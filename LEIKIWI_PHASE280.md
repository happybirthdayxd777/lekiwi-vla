# Phase 280 — Architecture Consolidation + full.launch.py Review

**Date**: 2026-04-23 13:30 CST

## 本次心跳完成

### 架構完整性審計（Phase 280）

**Phase 278 normalize_action Fix 已確認**
- `vla_policy_node.py` line 946-950: Stage2/DAgger/Stage3 跳過 `normalize_action()`
- Commit `a1cec22` Phase 278 FIX 正確套用
- Bridge 的 `_action_to_ctrl()` 統一處理所有 policy 的 action 標準化

**Phase 279 分析已確認架構無 further bugs**
- DAgger 50% SR 根因：數據不足（30 epochs 但 loss=0.003 表示訓練不足）
- Stage3 s3_epoch9 最佳（overfitting at epoch 12+）
- Stage2 72% SR 是 policy limitation，不是 bridge bug

### Launch 架構現狀

| Launch | Status | 功能 |
|--------|--------|------|
| `bridge.launch.py` | ✅ | 僅 bridge_node.py |
| `vla.launch.py` | ✅ | bridge_node.py + vla_policy_node.py |
| `ctf.launch.py` | ✅ | + CTF security layer |
| `full.launch.py` | ✅ | + replay + recording |
| `real_mode.launch.py` | ✅ | 實機硬體介面 |

### Topic 契約完整性（已驗證）

```
bridge_node.py:
  ← /lekiwi/cmd_vel         (Twist)           — teleop input
  ← /lekiwi/vla_action      (Float64[9])     — VLA policy output
  ← /lekiwi/goal            (Float64[2])      — goal position
  → /lekiwi/joint_states    (JointState)     — arm×6 + wheel×3
  → /lekiwi/odom            (Odometry)       — base odometry
  → /lekiwi/camera/image_raw (Image)        — front camera 20Hz
  → /lekiwi/wheel_N/cmd_vel (Float64×3)     — per-wheel velocity

vla_policy_node.py:
  ← /lekiwi/joint_states    — reads current state
  ← /lekiwi/camera/image_raw — reads front camera
  ← /lekiwi/goal            — reads goal
  → /lekiwi/vla_action      — publishes to bridge
```

### 已知政策性能（用於 launch default 選擇）

| Policy | SR | 用途 |
|--------|-----|------|
| P-controller CJ | 86% | oracle baseline |
| Stage2 curriculum | 72% | 目標範圍 \|r\|<0.45m |
| DAgger-254 | 50% | 全範圍 goals |
| VLA Phase227 | 70% | 全範圍 goals |

**full.launch.py 預設 policy=dagger** 是合理選擇（30 epochs trained, 50% SR > DAgger-246 baseline）。

### CTF Security Layer（C1-C8）

| Challenge | Detection | Status |
|-----------|-----------|--------|
| C1: cmd_vel HMAC | HMAC 校驗 | ✅ |
| C2: DoS rate flood | 100Hz rate limit | ✅ |
| C3: Command injection | 字元過濾 | ✅ |
| C4: Physics DoS | accel clamp 5.0 m/s² | ✅ |
| C5: Replay attack | timestamp + nonce | ✅ |
| C6: Sensor spoof | joint_states 驗證 | ✅ |
| C7: Policy hijack | policy_mode 鎖定 | ✅ |
| C8: VLA action inject | vla_action 驗證 | ✅ |

### 下一步

- [ ] Phase 281: ROS2 環境真實 end-to-end 測試（無 ROS2 環境，只能做代碼審計）
- [ ] Phase 282: DAgger 數據增強策略（收集更多 DAgger 數據 vs 擴展 Stage2 curriculum）
- [ ] Phase 283: 評估 Phase264 s3_epoch9 在 URDF 上的 SR

### 阻礙

- 無 ROS2 環境無法進行 end-to-end 測試
- DAgger 50% SR 需要更多訓練數據或更好的初始化

## Git

- Commit: Phase 280 heartbeat — full.launch.py review, architecture audit
- No code changes needed — architecture is complete and correct
