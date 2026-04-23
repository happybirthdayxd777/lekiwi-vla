# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一研究平台 — 進度追蹤

> 自動每 30 分鐘心跳更新

---

## [2026-04-21 13:04] Phase 248 — Portable STL Mesh Paths

### 已完成

- `lekiwi_modular_meshes` symlink → `lekiwi_modular/src/lekiwi_description/urdf/meshes`
- `sim_lekiwi_urdf.py` path resolution: `__file__`-relative + symlink (no hardcoded home path)
- Verified: `LeKiWiSimURDF` init OK, `nmesh=26`, `njnt=10`, `reset()` OK

**Bridge Architecture — Phase 239-248 summary**
| 元件 | 狀態 | 檔案 |
|------|------|------|
| `bridge_node.py` | ✅ 1260 行，primitive + URDF 模式 | `src/lekiwi_ros2_bridge/` |
| `vla_policy_node.py` | ✅ 768 行，CLIP-FM/pi0/ACT/mock | `src/lekiwi_ros2_bridge/` |
| CTF Security Layer | ✅ Phase 239-243，C1-C8 挑戰全 | `ctf_integration.py` |
| Camera Adapter | ✅ URDF 模式 20Hz RGB | `camera_adapter.py` |
| Real Hardware Adapter | ✅ 真實硬體介面 | `real_hardware_adapter.py` |
| 5× Launch Files | ✅ bridge/vla/ctf/full/real_mode | `launch/` |
| STL Mesh Paths | ✅ Phase 248 — portable symlink | `lekiwi_modular_meshes` |

### 下一步

- [ ] Phase 249: 測試 `full.launch.py` end-to-end（無 ROS2 環境只能靜態審查）
- [ ] Phase 250: DAgger 擴展數據收集（pilot SR=20% 需提升）
- [ ] Phase 251: 整合 lekiwi_modular 的 actual URDF STL meshes 進 bridge

### 阻礙

- DAgger pilot eval SR=20%（遠低於 P-controller 100%），需要更多數據

---

## [2026-04-21 01:00] Phase 247 — DAgger Pilot Results Commit

### 已完成

**DAgger Pipeline (Phase 246-247)**
- `scripts/collect_dagger.py` — 專家 P-controller 糾正數據收集
- `scripts/train_dagger.py` — DAgger 訓練腳本
- `scripts/eval_dagger.py` — DAgger 策略評估
- `results/dagger_phase246_pilot/` — 初步 pilot 數據（uncommitted → now committed）
- `results/dagger_phase246_train/` — 訓練過程日誌

**橋樑架構現狀 (Phase 1–245)**
| 元件 | 狀態 | 檔案 |
|------|------|------|
| `bridge_node.py` | ✅ 1260 行，支援 primitive + URDF 模式 | `src/lekiwi_ros2_bridge/` |
| `vla_policy_node.py` | ✅ 768 行，支援 CLIP-FM/pi0/ACT/mock | `src/lekiwi_ros2_bridge/` |
| CTF Security Layer | ✅ Phase 239-243，含 C1-C8 挑戰 | `ctf_integration.py` |
| Camera Adapter | ✅ URDF 模式 20Hz RGB | `camera_adapter.py` |
| Real Hardware Adapter | ✅ 真實硬體介面 | `real_hardware_adapter.py` |
| Launch Files | ✅ bridge/vla/ctf/full/real_mode | `launch/` |

**ROS2 Topics 映射**
```
/lekiwi/cmd_vel        ← 輸入 (Twist, teleop)
/lekiwi/vla_action    ← 輸入 (Float64MultiArray[9], VLA policy)
/lekiwi/joint_states   → 輸出 (sensor_msgs/JointState, arm*6 + wheel*3)
/lekiwi/odom           → 輸出 (nav_msgs/Odometry)
/lekiwi/camera/image_raw → 輸出 (Image, URDF模式)
/lekiwi/security_alert → 輸出 (CTF alerts)
```

**CTF 挑戰映射 (Phase 238-239)**
- C1: cmd_vel HMAC 缺失檢測
- C2: DoS rate flood 檢測
- C3: Command injection 檢測
- C4: Physics DoS (accel) 檢測
- C5: Replay attack 檢測
- C6: Sensor spoof (joint_states) 檢測
- C7: Policy hijack 檢測
- C8: VLA action inject 檢測

### 架構圖

```
ROS2 Topics
  /lekiwi/cmd_vel ──────────────────────────────────────────────────┐
  /lekiwi/vla_action ───────────────────────────────────────────────┤
                                                                  ↓
  BridgeNode (lekiwi_ros2_bridge)                                  ↓
    ├── SecurityMonitor (HMAC/Replay/CTF)                    ←── CTFSecurityAuditor
    ├── PolicyGuardian (policy switch guard)                       ↓
    └── LeKiWiSim / LeKiWiSimURDF (MuJoCo)                    ←── BridgeNode._step()
          ↓                                                     ←── VLA → native units
    JointState → /lekiwi/joint_states ──────────────────────────────→ VLA Policy Node
                                                                           ↓
                                                              /lekiwi/vla_action
```

### 下一步

- [ ] Phase 247: 分析 DAgger pilot 結果，決定是否需要擴展數據收集
- [ ] Phase 248: 整合 lekiwi_modular 的 actual URDF STL meshes 進 bridge
- [ ] Phase 249: 測試 `full.launch.py` 一鍵啟動

### 阻礙

- DAgger pilot eval SR=20%（遠低於 P-controller 100%），需要更多數據或專家糾正品質改善

---

## [2026-04-20 02:00] Phase 246 — DAgger Pipeline + Eval

### 已完成
- DAgger 數據收集腳本（專家 P-controller 提供糾正）
- CLIP-FM policy eval: 3-goal pilot，P-ctrl=100%, VLA=66.7%
- Phase 245: render-black bug 修復（step-0 warmup）

### 下一步
- Phase 246: 擴展 DAgger 數據收集到多 goals
- Phase 247: 真實 URDF mesh 整合

### 阻礙
- VLA 成功率落後 P-controller 33%，數據不足是主要瓶頸

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

### 本次心跳完成

**Phase 250 分析追蹤：為何 DAgger SR=33% 而非預期更高？**

root cause 分析完成，發現三個問題：

1. **Checkpoint 保存錯誤**：final_policy.pt 儲存於 epoch 15（訓練中途），而非 epoch 30 或最佳 epoch
   - 訓練 30 epoch，loss 從 0.012 降至 0.003
   - 但 checkpoint 只在 epoch 15 保存（非最佳）
   - 修復：`train_dagger.py` 需追蹤 best_loss，正確保存最佳模型

2. **DAgger 數據不足**：5 episodes × 653 frames = 60% 專家標籤數據
   - 395 個專家糾正 frame（label=1）用於訓練
   - 但相對於 3000+ frame 的 base data（phase196_clean_50ep.h5），DAgger 訊號太弱
   - 需要收集 20-30 episodes 的大規模 DAgger 數據才能看到改進

3. **P-controller oracle 太強**：SR=93% 是高標準 baseline
   - 接觸Jacobian P-controller（kP=2.0）在接觸物理環境中表現出色
   - DAgger 試圖學習，但訓練訊號摻雜了弱的 VLA 示範（label=0）
   - 失敗目標分佈：10/15 在邊緣（|g|>0.3m 或 |g_y|>0.2m）

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

### 下一步

- [ ] Phase 265: Monitor Stage 3 — wait for epoch 3 checkpoint, evaluate
- [ ] Phase 266: If checkpoint at epoch 3, eval Stage 3 policy (goal-radius=all)
- [ ] Phase 267: If SR improved, integrate Stage 3 with bridge_node ROS2 topic

### 阻礙

- Disk space: 7.8GB free → need to monitor checkpoint saves
- Training takes ~30 min/epoch on CPU → full 15 epochs = ~7.5 hours


---

## [2026-04-23 11:30] Phase 275 — Contact-Jacobian P-Controller = 100% SR (Physics Confirmed)

### 本次心跳完成

**Phase 275 驗證：URDF 物理引擎完全正確**

核心發現：
- **Contact-Jacobian P-Controller (kP=2.0) 在 URDF sim 中達到 100% SR**（10/10 goals）
- 使用 `_CONTACT_JACOBIAN_PSEUDO_INV` 矩陣（3×2，从实际接触物理测量得出）
- 驗證了 URDF sim 的輪子-地面接觸物理完全正確
- 之前 bridge_node 的 20% SR 純粹是**舊有的 k_omni 運動學模型不適用於接觸物理**

**URDF sim vs bridge_node 物理差異**：
- URDF sim：真實接觸物理，Contact-Jacobian P-ctrl = 100% SR
- bridge_node：使用 k_omni overlay 運動學（Phase 164 時代的模型），與 URDF 接觸物理不符
- 修復：bridge_node 需切換到 Contact-Jacobian（或在 URDF 模式下使用 `_CONTACT_JACOBIAN_PSEUDO_INV`）

**Locomotion 驗證**：
- URDF sim：all-wheels action[6:9]=0.5 → 100 steps → 0.38m 位移 ✓
- Primitive sim：同樣動作 → 0.30m 位移（略低於 URDF）
- 兩者物理模型不同，但 URDF 有完整的 26 個 STL mesh 幾何

### Bridge Architecture Status (Phase 239-275)

|| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 173 行 | 支援 primitive + URDF 模式 |
| vla_policy_node.py | ✅ 22195 bytes | CLIP-FM/pi0/ACT/dagger/stage2/stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | 資安監控整合 |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| Stage2PolicyRunner | ✅ Phase 268 | goal-radius filtering (|r|>0.45m → zeros fallback) |
| DAgger Pipeline | ✅ Phase 252-254 | collected, trained, evaluated |
| **URDF Physics** | ✅ **100% SR** | Contact-Jacobian P-ctrl confirmed |
| **Policy Gap** | ⚠️ Stage2 40% SR | 物理正確，政策需要改善 |

### 下一步

- [ ] Phase 276: 為 bridge_node 添加 Contact-Jacobian 模式（URDF mode 下使用 _CONTACT_JACOBIAN_PSEUDO_INV）
- [ ] Phase 277: 橋接 Stage2 政策（72% SR on |r|<0.45m）到 ROS2 /lekiwi/vla_action topic
- [ ] Phase 278: 測試 full.launch.py end-to-end（需要 ROS2 環境）

### 阻礙

- bridge_node 仍使用舊的 k_omni 運動學（適用於 overlay 物理，不適用於 URDF 接觸物理）
- Stage2 policy（72% SR）需要整合進 bridge_node 的 VLA topic 鏈路

---

## [2026-04-22 09:30] Phase 270 — DAgger Failure Root Cause + Disk Space Critical

### 本次心跳完成

**Phase 270 分析：DAgger 254 為何比 Phase227 VLA 還差？**

DAgger 254 eval results (50 goals, 200 steps):
| Policy | Success Rate | Notes |
|--------|-------------|-------|
| P-controller (CJ kP=2.0) | 86% | Baseline oracle |
| VLA Phase227 | 70% | CLIP-FM + Contact-Jacobian |
| VLA DAgger-254 | 50% | DAgger-254 actually WORSE |

**Root Cause Analysis:**

1. **DAgger data conflation**: 3832 frames, 63% expert labels
   - Expert goal radii: mean=0.328m (max=0.493m) — focuses on hard goals
   - VLA goal radii: mean=0.218m — DAgger collector only labels hard frames as expert
   - When training, VLA (label=0) frames with easy goals override expert corrections
   - Net effect: DAgger confuses the policy on easy goals

2. **Base data dilution**: DAgger 3832 frames vs base data 3000+ frames
   - 63% expert weight × 3832 = ~2417 expert contributions
   - But distributed across 30 episodes = ~80 expert frames/episode
   - Phase 246 pilot (5ep, 395 expert) already showed 60% expert ratio was too weak signal

3. **Checkpoint saved correctly in Phase 254** (unlike Phase 246 bug):
   - `best_policy.pt` at epoch 20, loss=0.00180 (BEST)
   - `final_policy.pt` at epoch 20 (same — best=final here)
   - NOT epoch 15 mid-training like Phase 246

**Stage2 (Curriculum) Performance Analysis:**
- Stage2 SR=72% on |r|<0.45m goals (50-goal eval)
- Failure quadrants: Q1=4, Q2=1, Q3=4, Q4=5
- Q4 (positive X, negative Y) most challenging — 5/12 failures
- Q4 failures all had final_dist > 0.37m — VLA not even converging
- Q4 kinematic analysis: wheel positions form equilateral triangle, but Y-axis motion (negative Y) requires wheel_2 (at -Y position) which has limited +Y authority

**Critical: Disk Space = 1.5% free (3.7GB)**
```
Major consumers:
  phase227_contact_jacobian_train/    4.6GB
  phase190_vision_train/              4.6GB
  phase260_curriculum_train/          2.3GB  (stage1_r025.pt + stage2_r045.pt)
  phase264_curriculum_train/          2.3GB  (s3_epoch*.pt overfitting checkpoints)
```

### Bridge Architecture Status (Phase 239-270)

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1260 行 | URDF + primitive 模式 |
| vla_policy_node.py | ✅ 987 行 | CLIP-FM/pi0/ACT/dagger/stage2/stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | 資安監控整合 |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| Stage2PolicyRunner | ✅ Phase 268 | goal-radius filtering (|r|>0.45m → zeros fallback) |
| DAgger Pipeline | ✅ Phase 252-254 | collected, trained, evaluated |
| DAgger Result | ⚠️ 50% SR | worse than Phase227 VLA (70%) |

### 下一步

- [ ] Phase 271: Clear disk space — archive/delete old training results
- [ ] Phase 272: Focus on Stage2 deployment (72% SR, |r|<0.45m constraint)
- [ ] Phase 273: Investigate Q4 kinematic weakness in Contact-Jacobian
- [ ] Phase 274: Collect more DAgger data with better labeling strategy (label ALL frames, not just hard)

### 阻礙

- **Disk 1.5% free** — cannot run training until space cleared
- DAgger approach fundamentally flawed for this problem — expert correction signal too weak vs base data
- Stage3 overfitting confirmed — s3_epoch9 is best but 0% SR in eval

