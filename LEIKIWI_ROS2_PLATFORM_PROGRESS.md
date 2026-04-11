# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一平台進度

## 目標架構
```
ROS2 topics/launch  →  ros2_lekiwi_bridge  →  lekiwi_vla MuJoCo sim  →  VLA policy  →  action back to ROS2
```

---

## [2026-04-14 03:30]

### 已完成
- **修復 VLA 閉環死鎖：`_vla_action_fresh` 計時器清除**
  - 問題：`_vla_action_fresh` 在 `_on_vla_action()` 設為 True 後**永不清除**，導致 VLA action 永久鎖死手臂（即使 VLA node 當機）
  - 修復：在 `_on_timer()` 末尾清除 `_vla_action_fresh = False`
  - 效果：每 50ms 週期結束後，若無新 VLA action，手臂自動釋放回 cmd_vel 控制
  - Docstring 更新以反映計時器驅動清除機制
  - Git pushed: `72172a4`

### 下一步
- Phase 5: Camera → VLA input pipeline — wrist camera image → VLA policy
- Phase 4: 統一 launch — `sim_type:=gazebo` 模式對接真實 Gazebo

### 阻礙
- なし（架構 Phase 1-3 完整）

---

## [2026-04-13 23:00]

### 已完成
- **Phase 3.5：ROS2 Topic 相容性對齊 — Odometry + TF + URDF joint names**
  - 發現 `omni_controller.py` 和 `omni_odometry.py` 的 joint_axes 為 `[0.866025, 0, 0.5]`（錯誤值），bridge_node.py 已有修正後的 `_JOINT_AXES`
  - 新增 `/lekiwi/odom` publisher (nav_msgs/Odometry) — 從 MuJoCo wheel velocities 積分，完整鏡射 `omni_odometry.py` 的運動學
  - 新增 TF broadcaster: `odom → base_link` transform @ 20Hz
  - 新增 `/lekiwi/joint_states_urdf` publisher — 使用真實 URDF joint names:
    - Wheel: `ST3215_Servo_Motor-v1_Revolute-64` (w0), `ST3215_Servo_Motor-v1-1_Revolute-62` (w1), `ST3215_Servo_Motor-v1-2_Revolute-60` (w2)
    - Arm: 5 joints (ST3215_Servo_Motor-v1-1_Revolute-49 ~ STS3215_03a-v1-4_Revolute-57)
  - `bridge_node.py` 從 353 行增至 454 行

### URDF 關鍵發現
- Gazebo plugin joint list: `ST3215_Servo_Motor-v1_Revolute-64, -v1-1_Revolute-62, -v1-2_Revolute-60` (3 wheel joints)
- Gazebo joint_state_publisher: 只包含 3 個 wheel joints，沒有 arm joints
- Arm joints 全是 `continuous` 類型，axis = X 軸 (翻譯後: `[1,0,0]` 或旋轉後的方向)
- Wrist Roll joint axis: `[0, 0.4226, -0.9063]` — 非標準 axis
- Gripper joint axis: `[-2.74e-31, -0.9063, -0.4226]` — 幾乎是 Z 軸翻轉

### 下一步
- Phase 5: Camera → VLA input pipeline — wrist camera image → VLA policy
- Phase 4: 統一 launch — `sim_type:=gazebo` 模式對接真實 Gazebo

### 阻礙
- Gripper joint axis 非標準：需特殊處理才能在 bridge 中正確映射
- Arm joints 不在 Gazebo plugin joint_state_publisher 中：需另外處理

---

## [2026-04-12 22:52]
### 已完成
- **Phase 5.3：Gripper 幾何改善 — 新增被動爪（fixed jaw）**
  - 將 gripper body 重構：從單一 body → gripper_base_fixed（被動爪）+ gripper（主動爪）
  - `gripper_base_fixed`：包含被動爪固定板（gripper_horn mesh）+ 舵機本體（servo_gripper）
  - `gripper`：滑動關節 j5（range 0-0.04m）+ 主動爪（moving_jaw mesh）
  - 被動爪模擬 gripper action 的"闭合时另一侧"（real gripper 有一侧固定）
  - MuJoCo 解析成功：bodies=14, meshes=23, joints=10, geoms=24
  - 物理测试通过，front + wrist camera 渲染正常
- **資安修正：security_monitor.py 日誌路徑**
  - 從 hardcoded `/root/hermes_research/...` → `~/hermes_research/...`
  - 新增 `import os` 到 security_monitor.py imports

### 下一步
- Phase 5: VLA input pipeline — Camera image → VLA policy input
  - 需要確認 VLA policy node 的 image 输入 (已有 `/lekiwi/wrist_camera/image_raw`)
- Phase 4: 真實模式 launch — `sim_type:=gazebo` + 硬體驅動

### 阻礙
- Phase 3 camera→VLA pipeline 尚未整合（VLA policy node 只用 joint_states）
- 需要 LeRobot 格式的 checkpoint 才能跑真實 policy

---

## [2026-04-11 21:30]
### 已完成
- **Phase 5 第一階段：STL arm meshes 替換 cylinders**
  - 17 mesh geoms + 23 total meshes（arm joints 全部使用真實 URDF STL）
  - Servo bodies: STS3215_03a-v1 系列（每個 6506 triangles）
  - Arm links: arm_square (7418), arm_mirror (14272), arm_clip (5366)
  - Wrist: wrist_pitch (16850), wrist_horn (10868), wrist_servo (STS3215-v1-3)
  - Gripper: servo_gripper + moving_jaw (10000 triangles)
  - Camera: wrist_cam_mount + wrist_cam_body
  - 接觸穩定性：100 次隨機手臂動作後僅 2 個 contacts
  - 新增 meshes：`wrist_servo`, `horn_fixed`, `gripper_horn`
- 修正 omni wheel 運動學（上一個 heartbeat）
- 新增 `vla.launch.py`（上一個 heartbeat）

### 下一步
- Phase 5 第二階段：Wrist camera → MuJoCo camera sensor（已有 mount mesh）
- Phase 4: 統一 launch — 一鍵啟動「真實模式」或「模擬模式」

### 阻礙
- 手臂幾何重疊問題：已通過合理初始化角度解決
- STL mm 單位問題：已通過 `scale="0.001"` 解決
- Omni wheel 超大 mesh：已用 cylinder primitives 代替

---

## [2026-04-11 18:00]
### 已完成
- `sim_lekiwi.py` — 獨立 MuJoCo 模擬（cylinder primitives, 快速穩定）
- `sim_lekiwi_urdf.py` — STL mesh MuJoCo 模擬（真實幾何）
- `bridge_node.py` — ROS2 ↔ MuJoCo 雙向 bridge
  - 讀取 `/lekiwi/cmd_vel` → 轉換為 MuJoCo action
  - 發布 `/lekiwi/joint_states` 回 ROS2
  - 發布 `/lekiwi/camera/image_raw` (URDF mode, 20Hz)
  - 支援 `sim_type` 參數：`primitive` | `urdf`
- `bridge.launch.py` — 統一 launch file
  - `ros2 launch lekiwi_ros2_bridge bridge.launch.py sim_type:=urdf`

### 關鍵技術發現
- lekiwi_modular URDF 的 STL 是 **mm 單位**，MuJoCo 需要 `scale="0.001 0.001 0.001"`
- Omni wheel STL 有 314k triangles，超出 MuJoCo 200k 上限 → 用 cylinder primitives 代替
- Arm 幾何重疊會導致接觸爆炸 → 需合理初始化角度 + 接觸參數

---

## Phase 1: 理解現有代碼

### 已完成
- [x] `lekiwi_modular/src/lekiwi_controller/` — omni_controller, omni_odometry
- [x] `lekiwi_modular/src/lekiwi_description/` — URDF + Gazebo launch
- [x] `lekiwi_vla/sim_lekiwi.py` — 獨立 MuJoCo 模擬（無 ROS2）
- [x] `robot-security-workshop/vulnerable_robot.py` — UDP 控制介面

---

## Phase 3-7: 未來工作

### Phase 3: VLA 整合 ✅ (closed-loop done)
- [x] `/lekiwi/vla_action` → bridge 訂閱，閉環完成
  - `_on_vla_action`: 接收 Float64MultiArray (arm*6 + wheel*3 native units)，clamp 後寫入 `_last_action`
  - `_vla_action_fresh` flag: VLA 寫入時設 True，timer tick 末尾清除
  - `_on_cmd_vel`: VLA 活躍時保留手臂 portion，只override車輪
  - 迴路: `joint_states → VLA policy → /lekiwi/vla_action → bridge → MuJoCo → joint_states`
- [ ] Camera image → VLA input pipeline
  - `/lekiwi/camera/image_raw` (front) + `/lekiwi/wrist_camera/image_raw` (wrist) @ 20Hz

### Phase 4: 統一 launch
- [ ] 一鍵啟動「真實模式」或「模擬模式」
- [ ] 整合 VLA policy 進 launch

### Phase 5: URDF 深化
- [x] 所有 arm joints 改用真實 STL ✅（Phase 5.1 完成）
- [x] Wrist camera → MuJoCo camera sensor ✅（Phase 5.2 完成）
  - `/lekiwi/wrist_camera/image_raw` @ 20Hz, 80° FOV, follows arm_j4
- [ ] Gripper geometry refinement（真實 gripper STL vs current）

### Phase 6: 資安模式
- [ ] 監控異常指令、記錄攻擊痕跡

### Phase 7: 硬體對接
- [ ] 對接 lekiwi_modular 的真實 URDF 用於 Gazebo
- [ ] ROS2 ↔ 硬體 CAN bus 整合

---

## Git 歷史
| 時間 | Commit | 內容 |
|------|--------|------|
| 2026-04-13 23:00 | `3f5a20b` | Odometry + TF + URDF joint names compatibility |
| 2026-04-12 10:30 | `16efbd8` | Wrist camera: MuJoCo sensor + render_wrist() + wrist_camera/image_raw |
| 2026-04-11 18:00 | `f5c3a91` | Camera bridge 整合 |
| 2026-04-11 20:00 | `8606aae` | VLA closed-loop: /lekiwi/vla_action → bridge subscribe |
| 2026-04-11 19:00 | `11fa758` | URDF STL mesh 整合 |

---

## 阻礙
- 手臂幾何重疊問題：已通過合理初始化角度解決
- STL mm 單位問題：已通過 `scale="0.001"` 解決
- Omni wheel 超大 mesh：已用 cylinder primitives 代替
