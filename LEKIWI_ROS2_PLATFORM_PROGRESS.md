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
