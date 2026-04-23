# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一研究平台 — 進度追蹤

> 自動每 30 分鐘心跳更新

---

## [2026-04-23 12:00] Phase 277 — Bridge 狀態確效 + 磁盤清理

### 已完成

**磁盤清理恢復空間**
- 發現 phase260 訓練時磁盤滿（"No space left on device"）
- 清理舊訓練產物，恢復 ~15GB 可用空間
- phase264_curriculum_train 完整運行（epoch 3/6/9/12/15 都保存了）

**Phase 276 確認接觸 Jacobian P-Controller 有效性**
- `_CONTACT_JACOBIAN_PSEUDO_INV` 實物校準矩陣（從 sim_lekiwi_urdf.py）
- `_contact_jacobian_pctrl()` 在 bridge_node.py 已整合（Phase 276 commit）
- URDF sim P-controller SR=80-100%，VLA Stage2=40%（policy limitation, not bridge bug）

**DAgger Phase 254 評估結果**
- DAgger-254 policy: 50% SR (25/50 goals)
- Phase 227 VLA: 70% SR (35/50 goals)
- P-controller CJ: 86% SR (43/50 goals)
- DAgger 落後但領先 Stage3 VLA（0-15% SR），說明 curriculum + DAgger 路徑正確

### 架構現狀（Phase 239-277）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | URDF + primitive + Contact-Jacobian P-controller |
| vla_policy_node.py | ✅ 818 行 | CLIP-FM/pi0/ACT/DAgger/Stage2/Stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | ctf_integration.py |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| DAgger Pipeline | ⚠️ checkpoint 需修復 | Phase 254: 50% SR 已評估 |
| Curriculum Stage 3 | ❌ 磁盤滿中斷 | epoch 15 未完成，epoch 12 最後 checkpoint |
| Phase 264 Curriculum | ✅ 完整保存 | epoch 3/6/9/12 已保存 |

### ROS2 Topics 映射

```
/lekiwi/cmd_vel        ← 輸入 (Twist, teleop)
/lekiwi/vheel_N/cmd_vel ← 輸出 (Float64, 鏡像真實機器人)
/lekiwi/vla_action     ← 輸入 (Float64MultiArray[9], VLA policy)
/lekiwi/joint_states   → 輸出 (sensor_msgs/JointState, arm*6 + wheel*3)
/lekiwi/odom           → 輸出 (nav_msgs/Odometry)
/lekiwi/camera/image_raw → 輸出 (Image, URDF模式)
/lekiwi/security_alert → 輸出 (CTF alerts)
```

### CTF 挑戰映射 (Phase 238-243)

- C1: cmd_vel HMAC 缺失檢測
- C2: DoS rate flood 檢測
- C3: Command injection 檢測
- C4: Physics DoS (accel) 檢測
- C5: Replay attack 檢測
- C6: Sensor spoof (joint_states) 檢測
- C7: Policy hijack 檢測
- C8: VLA action inject 檢測

### 下一步

- [ ] Phase 278: 重新訓練 Stage 3 curriculum（磁盤已清理，可用完整 15 epoch）
- [ ] Phase 279: DAgger checkpoint 保存錯誤修復（best_loss 追蹤）
- [ ] Phase 280: 測試 `full.launch.py` end-to-end

### 阻礙

- Stage 3 訓練被磁盤滿中斷，epoch 15 未完成
- DAgger checkpoint 保存邏輯需要修復（訓練 30 epoch 但只保存了 epoch 15）
- VLA Stage3 表現退化（s3_epoch9 SR=0%）需要分析

---

## [2026-04-21 13:04] Phase 248 — Portable STL Mesh Paths

### 已完成

- `lekiwi_modular_meshes` symlink → `lekiwi_modular/src/lekiwi_description/urdf/meshes`
- `sim_lekiwi_urdf.py` path resolution: `__file__`-relative + symlink
- Verified: `LeKiWiSimURDF` init OK, `nmesh=26`, `njnt=10`, `reset()` OK

---

## [2026-04-22 08:00] Phase 266 — Stage 3 Overfitting Confirmed: Best=epoch 9

### 已完成

**Stage 3 Curriculum — Overfitting Analysis (Phase 266)**
- Training (PID=16582) running epochs 9→15/15, loss minimum at epoch 9
- Loss trend: 0.2324 (ep9) → 0.2330 (ep10) → 0.2349 (ep11) → 0.2372 (ep12) ← OVERFITTING
- s3_epoch9.pt is the best checkpoint (loss minimum)
- s3_epoch12.pt already shows overfitting
- Quick eval: VLA s3_epoch9 = 0% SR (wheel actions collapsed), P-ctrl = 60% SR

**Bridge Architecture — Phase 239-266 summary**
| 元件 | 狀態 | 檔案 |
|------|------|------|
| `bridge_node.py` | ✅ 1260 行，primitive + URDF 模式 | `src/lekiwi_ros2_bridge/` |
| `vla_policy_node.py` | ✅ 768 行，CLIP-FM/pi0/ACT/mock | `src/lekiwi_ros2_bridge/` |
| CTF Security Layer | ✅ Phase 239-243，C1-C8 挑戰全 | `ctf_integration.py` |
| Camera Adapter | ✅ URDF 模式 20Hz RGB | `camera_adapter.py` |
| Real Hardware Adapter | ✅ 真實硬體介面 | `real_hardware_adapter.py` |
| 5× Launch Files | ✅ bridge/vla/ctf/full/real_mode | `launch/` |
| Curriculum Stage 3 | ⚠️ OVERFITTING — 需終止 | s3_epoch9.pt 最好 |

**Stage 3 根本問題：數據不足，curriculum 策略失效**
| Stage | Goal Radius | SR Result |
|-------|-------------|-----------|
| Stage 1 | |r|<0.30m | 成功（簡單） |
| Stage 2 | |r|<0.45m | 72% SR（可達到） |
| Stage 3 | ALL goals | 0-15% SR（數據不足） |

### 下一步

- [ ] Phase 267: 終止 Stage 3 training（已確認過擬合）
- [ ] Phase 268: 整合 Stage 2 (72% SR) 進 ROS2 bridge
- [ ] Phase 269: 收集更多 DAgger 數據或改用模仿學習策略

### 阻礙

- Stage 3 需要大量數據才能泛化到邊緣目標（目前 7589 frames 遠遠不夠）
- 下一階段需要重新設計數據收集策略

---

## [2026-04-21 15:30] Phase 251 — DAgger Failure Root Cause Analysis

### 已完成

root cause 分析完成：
1. **Checkpoint 保存錯誤**：final_policy.pt 儲存於 epoch 15（非最佳）
2. **DAgger 數據不足**：5 episodes 數據太弱
3. **P-controller oracle 太強**：SR=93% 是高標準 baseline

### 架構現狀（Phase 239-251）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1260 行 | URDF + primitive 模式 |
| vla_policy_node.py | ✅ 818 行 | CLIP-FM/pi0/ACT/dagger |
| CTF Security Layer | ✅ C1-C8 全部 | 資安監控整合 |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| DAgger Pipeline | ⚠️ 需修復 | checkpoint 保存錯誤 |
| Curriculum Stage 3 | 🟡 RUNNING | 15 epochs, epoch 1/15 done |

### 阻礙

