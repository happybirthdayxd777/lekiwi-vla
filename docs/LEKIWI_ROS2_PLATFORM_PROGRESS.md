# LeKiWi ROS2-MuJoCo 研究平台進度

**Last Updated:** 2026-04-12 12:00 JST
**Status:** ✅ Kinematics Validation — primitive mode has simplified geometry, URDF mode correct

## 目標架構

```
┌─────────────────────────────────────────────────────────────────┐
│                    統一 LeKiWi 研究平台                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  ROS2 Layer  │───→│  ros2_bridge    │───→│  MuJoCo Sim   │  │
│  │  (lekiwi_    │    │  lekiwi_vla/    │    │  (sim_lekiwi) │  │
│  │   modular)    │    │  src/           │    │               │  │
│  └──────────────┘    └──────────────────┘    └───────────────┘  │
│         │                     ↑                       │          │
│         │                     │                       │          │
│         ▼                     │                       ▼          │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ CTF 安全模式  │    │  VLA Policy      │    │  URDF (real)  │  │
│  │ (robot_      │    │  CLIP-FM / ACT   │    │  STL meshes   │  │
│  │  security_   │    │                  │    │               │  │
│  │  workshop)   │    │                  │    │               │  │
│  └──────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 已發現的 ROS2 Topics

### lekiwi_controller
| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/lekiwi/cmd_vel` | `geometry_msgs/Twist` | ← input | 底盤速度命令 |
| `/lekiwi/wheel_0/cmd_vel` | `Float64` | → output | 輪1角速度 |
| `/lekiwi/wheel_1/cmd_vel` | `Float64` | → output | 輪2角速度 |
| `/lekiwi/wheel_2/cmd_vel` | `Float64` | → output | 輪3角速度 |
| `/lekiwi/odom` | `nav_msgs/Odometry` | → output | 里程計 |

### omni_teleop
| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | ← input | 搖桿速度命令 |

## URDF Joints（來自 lekiwi.urdf）

### 底盤（3個全向輪）
- `ST3215_Servo_Motor-v1_Revolute-64` — wheel 0 (前)
- `ST3215_Servo_Motor-v1-1_Revolute-62` — wheel 1 (左後)
- `ST3215_Servo_Motor-v1-2_Revolute-60` — wheel 2 (右後)

### 機械臂（6 DOF）
- `STS3215_03a-v1_Revolute-45` — arm_joint_1
- `STS3215_03a-v1-1_Revolute-49` — arm_joint_2
- `STS3215_03a-v1-2_Revolute-51` — arm_joint_3
- `STS3215_03a-v1-3_Revolute-53` — arm_joint_4
- `STS3215_03a_Wrist_Roll-v1_Revolute-55` — arm_joint_5
- `STS3215_03a-v1-4_Revolute-57` — arm_joint_6 (gripper?)

## 缺失的環節

1. **沒有 ROS2 → MuJoCo 橋樑** — 現有 `sim_lekiwi.py` 是獨立運行，沒有 ROS2 介面
2. **沒有 MuJoCo URDF 整合** — `lekiwi_modular` 的 URDF + STL meshes 沒有用於 MuJoCo
3. **沒有統一 launch** — 無法一鍵啟動「真實模式」vs「模擬模式」
4. **沒有圖像 bridge** — ROS2 image topic → MuJoCo simulated camera

## 開發計劃

### Phase 1: 基礎 Bridge
- [x] 分析 ROS2 topics 和 message types
- [x] `bridge_node.py` — 讀 `/lekiwi/cmd_vel` → 應用到 LeKiWiSim
- [x] 修正 WHEEL_JOINT_AXES（從錯誤的 [0,0,1] 改為正確的 [0.866, 0, 0.5]）
- [x] 修正 qpos index mapping（base=[0:3], arm=[7:13], wheel=[13:16]）
- [x] 修正 action 格式（[arm(6), wheel(3)] 不是 [wheel(3), arm(6)]）
- [x] 發布 `/lekiwi/joint_states` 回 ROS2
- [x] 發布 `/lekiwi/odom` 回 ROS2
- [ ] 整合 URDF 的 joint names（已定義橋接表）
- [ ] 統一 launch file

### Phase 2: 數據整合
- [ ] MuJoCo camera → ROS2 image topic
- [ ] lekiwi_modular URDF + STL → MuJoCo 模型
- [ ] 真實機械臂數據管道

### Phase 3: 統一 Launch
- [ ] `bridge.launch.py` — 一鍵啟動橋樑
- [ ] `full_simulation.launch.py` — 啟動所有（bridge + VLA + CTF）

### Phase 4: VLA 集成
- [ ] CLIP-FM policy 輸出到 ROS2 topic
- [ ] 從 ROS2 訂閱相機圖像用於 VLA 推理

### Phase 5: CTF 安全模式
- [ ] 異常指令監控
- [ ] 攻擊日誌記錄

---

## 進度日誌

### 2026-04-11
- 分析了 `lekiwi_modular` 的 ROS2 topics：
  - `/lekiwi/cmd_vel` (Twist) ← 底盤命令
  - `/lekiwi/wheel_{0,1,2}/cmd_vel` (Float64) ← 輪速命令
  - `/lekiwi/odom` (Odometry) ← 里程計
- 分析了 URDF joints：
  - 3個 omni-wheel 馬達（continuous revolute）
  - 6個機械臂關節（continuous revolute）
- 發現 bug：`omni_controller.py` 三輪 `joint_axes` 完全相同，已修復到 `omni_controller_fixed.py`

### 2026-04-11 19:30
- **Phase 5 CTF Security Mode: 完成！**
- 新增 `lekiwi_ros2_bridge/security_monitor.py`:
  - `SecurityMonitor` class — thread-safe入侵檢測
  - 5種異常檢測器：速度突刺、NaN/Inf、速率變化、Replay attack、政策篡改
  - CTF Challenge 7 (policy_hijack) 自動捕獲 flag: `ROBOT_CTF{policy_hijack_4c8e2a9f}`
  - 攻擊日誌寫入 `~/hermes_research/lekiwi_vla/security_log.jsonl`
- 更新 `bridge_node.py` 整合 SecurityMonitor:
  - `/lekiwi/cmd_vel` → 先過SecurityMonitor審查，異常直接阻擋
  - 新增 `/lekiwi/policy_input` 監控topic（CTF Challenge 7）
  - 阻擋計數 `_blocked_count` + throttled警告日誌
- 進度：Phase 1 ✅ 基礎 Bridge | Phase 2 🔄 URDF整合中 | Phase 5 ✅ CTF安全監控
- 下一步：統一 launch file + VLA 集成
