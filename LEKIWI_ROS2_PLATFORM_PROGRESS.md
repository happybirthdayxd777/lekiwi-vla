# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一研究平台 — 進度追蹤

> 自動每 30 分鐘心跳更新

---

## [2026-04-23 18:30] Phase 286 — Stage2 Integration Audit: 60% SR, CJ P-controller Fallback Verified

### 已完成

**Stage2 整合現狀全面審計**

Stage2 wheel magnitude 診斷（native rad/s）:
- Stage2 mean=0.189 rad/s (median=0.163), range=[0.069, 0.322]
- P-controller: ~0.389 rad/s | Stage3: ~0.032 rad/s
- **Stage2 是 Stage3 的 5.9x，但仍低於 P-controller**

Stage2 eval 結果（URDF sim, seed=42）:
- 10-goal: **70% SR** (7/10)
- 20-goal: **60% SR** (12/20)
- 50% steps 低於 0.15 threshold → P-controller 接管 locomotion

**Bridge hybrid fallback 對 Stage2 的影響：**
- ~50% steps vla_mag < 0.15 → P-controller takeover
- P-controller CJ: 78-86% SR (reliable loco fallback)
- Stage2 + hybrid: 預期 SR > 70%（取決於 hybrid 觸發頻率）

### 架構現狀（Phase 286）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | CJ P-controller + Hybrid fallback |
| vla_policy_node.py | ✅ 1000 行 | stage2 in _NATIVE_UNIT_POLICIES |
| Stage2 checkpoint | ✅ 可用 | s2_r045.pt (epoch=s2_10, loss=0.2938) |
| Stage2 wheel mag | ⚠️ 臨界 | 50% steps < 0.15 → P-controller 接管 |
| Stage2 SR | ✅ 60-70% | URDF sim eval |
| CJ P-controller | ✅ 78-86% SR | 可靠的 loco fallback |

### 下一步

- [ ] Phase 287: 50-goal Stage2 eval 完整評估
- [ ] Phase 288: Stage2 + bridge hybrid 端到端 SR
- [ ] Phase 289: Stage2 vs Stage3 + hybrid 對比

### 阻礙

- Stage2 wheel magnitude 仍低於 P-controller，hybrid fallback 必要
- 真實 ROS2 硬體環境仍不可用

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


---

## [2026-04-23 17:05] Phase 284 — VLA Wheel Magnitude: s3_epoch6 vs s3_epoch9 Confirmed Identical

### 已完成

**Phase 283 發現的後續：s3_epoch6 vs s3_epoch9 車輪動作幅度診斷**

Phase 283 假設：s3_epoch6（20-goal eval 15% SR）可能比 s3_epoch9（50-goal eval 2% SR）有更大的車輪動作幅度。快速 10-goal × 50-step 評估結果：

| Checkpoint | raw_wheel_mag | norm_wheel_mag | 10-goal SR |
|------------|---------------|----------------|------------|
| s3_epoch6  | 0.0629        | 0.0315 rad/s   | 0%         |
| s3_epoch9  | 0.0652        | 0.0326 rad/s   | 0%         |
| **P-controller** | —       | **0.389 rad/s** | ~80% SR   |

**關鍵發現：兩個 checkpoint 車輪幅度幾乎完全相同（~0.03 vs ~0.39, 相差 12x）**

- VLA raw wheel ∈ [-0.25, 0.25]，normalized 後 ∈ [-0.125, 0.125]
- P-controller 需要 ~0.39 rad/s，VLA 只有 ~0.03 rad/s → 12x 不足
- **車輪幅度問題是兩個 checkpoint 的共同問題，不是 epoch 9 特有的過擬合問題**

### 根本原因確認

| 層次 | 問題 | 修復方向 |
|------|------|---------|
| **訓練層** | VLA 訓練時 wheel loss 權重不足，網絡輸出趨近於零 | 重新設計 wheel loss weighting 或擴展數據 |
| **Eval 腳本** | `normalize_action()` 對兩個 checkpoint 都正確 | 不需修改 |
| **Bridge** | `_action_to_ctrl()` 對兩個 checkpoint 都正確 | 不需修改 |

### 架構現狀（Phase 277-284）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | Contact-Jacobian P-controller loco fallback |
| vla_policy_node.py | ✅ 818 行 | CLIP-FM/pi0/ACT/DAgger/Stage2/Stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | ctf_integration.py |
| s3_epoch6 / s3_epoch9 | ❌ 12x 車輪幅度不足 | VLA 訓練問題，非 bridge bug |
| P-controller CJ | ✅ 78-86% SR | 可靠的 loco fallback |

### 下一步

- [ ] Phase 285: 實現 Option C — VLA arm-only + P-controller wheel fallback
- [ ] Phase 286: 整合 Stage2 (arm + wheel P-controller) 進 ROS2 bridge
- [ ] Phase 287: 測試 full.launch.py end-to-end

### 阻礙

- VLA wheel locomotion 訓練需要完全重新設計
- 真實 ROS2 硬體環境仍不可用

---

## [2026-04-23 17:05] Phase 284 — VLA Wheel Magnitude: s3_epoch6 vs s3_epoch9 Confirmed Identical

### 已完成

**Phase 283 發現的後續：s3_epoch6 vs s3_epoch9 車輪動作幅度診斷**

Phase 283 假設：s3_epoch6（20-goal eval 15% SR）可能比 s3_epoch9（50-goal eval 2% SR）有更大的車輪動作幅度。快速 10-goal × 50-step 評估結果：

| Checkpoint | raw_wheel_mag | norm_wheel_mag | 10-goal SR |
|------------|---------------|----------------|------------|
| s3_epoch6  | 0.0629        | 0.0315 rad/s   | 0%         |
| s3_epoch9  | 0.0652        | 0.0326 rad/s   | 0%         |
| **P-controller** | —       | **0.389 rad/s** | ~80% SR   |

**關鍵發現：兩個 checkpoint 車輪幅度幾乎完全相同（~0.03 vs ~0.39, 相差 12x）**

### 根本原因確認

| 層次 | 問題 | 修復方向 |
|------|------|---------|
| **訓練層** | VLA 訓練時 wheel loss 權重不足 | 重新設計 wheel loss weighting |
| **Eval 腳本** | normalize_action() 對兩個 checkpoint 都正確 | 不需修改 |
| **Bridge** | _action_to_ctrl() 對兩個 checkpoint 都正確 | 不需修改 |

### 架構現狀（Phase 277-284）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | CJ P-controller loco fallback |
| vla_policy_node.py | ✅ 818 行 | CLIP-FM/pi0/ACT/DAgger/Stage2/Stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | ctf_integration.py |
| s3_epoch6 / s3_epoch9 | ❌ 12x 車輪幅度不足 | VLA 訓練問題 |
| P-controller CJ | ✅ 78-86% SR | 可靠的 loco fallback |

### 下一步

- [ ] Phase 285: 實現 VLA arm-only + P-controller wheel fallback
- [ ] Phase 286: 整合 Stage2 進 ROS2 bridge
- [ ] Phase 287: 測試 full.launch.py end-to-end

### 阻礙

- VLA wheel locomotion 訓練需要完全重新設計
- 真實 ROS2 硬體環境仍不可用
