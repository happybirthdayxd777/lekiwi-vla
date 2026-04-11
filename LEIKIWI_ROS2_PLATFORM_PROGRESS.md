# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一平台進度

## 目標架構
```
ROS2 topics/launch  →  ros2_lekiwi_bridge  →  lekiwi_vla MuJoCo sim  →  VLA policy  →  action back to ROS2
```

---

## [2026-04-14 04:00] — Phase 6.2: Real Hardware Mode

### 已完成
- **Phase 6.2：真實硬體模式（ST3215 Serial 適配器）**
  - 新增 `real_hardware_adapter.py`（340+ 行）：完整的 ST3215 序列協定處理
    - `ST3215Protocol`: 半雙工序列封包建構（position read/write）、校驗和計算
    - `RealHardwareAdapter`: 執行緒式讀取迴圈（50Hz poll）、命令佇列（避免匯流排衝突）
    - `MockHardwareAdapter`: 開發/測試用的模擬器，無需真實硬體
  - **bridge_node.py 升級**（634 行）：
    - `mode='real'` 參數：完全繞過 MuJoCo，改用序列轉接器
    - `_on_cmd_vel`: 接收 Twist → 轉換為馬達速度 → 發送到序列匯流排
    - `_on_timer`: 從馬達回饋讀取關節狀態 → 發布 `/lekiwi/joint_states`
    - `mode='sim'`: 保持原本的 MuJoCo 模擬模式（不變）
    - 硬體急停看門狗：無 cmd_vel 超過 1 秒 → 自動停止所有馬達
  - `hardware.yaml`: 實體參數（servo ID、極限、topic 名稱、安全設定）
  - `real_mode.launch.py`: 一鍵啟動硬體模式 + VLA policy node

### 架構狀態
- **Bridge 現在支援 3 種模式**：
  | 模式 | 啟動方式 | 馬達控制 | 狀態回饋 |
  |------|----------|----------|----------|
  | `sim` (預設) | `bridge.launch.py` | MuJoCo sim.step() | `sim._obs()` |
  | `sim+urdf` | `bridge.launch.py sim_type:=urdf` | MuJoCo STL sim | `sim._obs()` |
  | `real` | `real_mode.launch.py` | ST3215 serial | `hw.get_state()` |

### 下一步
- Phase 7: 真實 CAN bus 對接（Raspberry Pi GPIO → ST3215）
- Phase 6.3: VLA 端到端測試（使用 mock 或 clip_fm policy 驅動真實馬達）
- Phase 5 CTF 安全模式實測（用 `ros2 topic pub /lekiwi/policy_input` 模擬攻擊）

### 阻礙
- ST3215 協定的速度控制：需要 position-stepping 模擬 velocity，精度依賴 poll 頻率

---

## [2026-04-14 03:00] — Phase 6: PolicyGuardian active defense

### 已完成
- **Phase 6 第一階段：PolicyGuardian 主動防禦系統**
  - 新增 `policy_guardian.py`（492 行）：完整的主動策略守衛
    - `check_and_guard()`: 指紋白名單 + HMAC 簽名驗證 + CTF flag 檢測
    - `check_action_anomaly()`: 後策略載入行為監控（車輪速度、關節突變、NaN/Inf）
    - `add_to_whitelist()`: 批准可信策略指紋
    - `_detect_ctf_flag()`: 掃描 raw bytes + pickled object 中的 CTF flag
    - 7 個測試全部通過：unknown/block, flag-payload/block, pickle-flag/block, whitelist/allow, HMAC/allow, anomalous-wheel/block, normal/allow
  - **bridge_node.py 升級**：
    - `PolicyGuardian` 整合進 `_on_policy_input` callback
    - 雙層防禦：SecurityMonitor（日誌） + PolicyGuardian（阻斷 + 告警）
    - 新增 `/lekiwi/security_alert` publisher（String，JSON 格式）
      - block/rollback 時：發布攻擊類型 + severity + 指紋 + CTF flag
      - allow 時：發布 policy_allowed 心跳
  - Git pushed: `a110ff7`

### 架構狀態（Phase 6 完成）
- **CTF Challenge 7 完全防禦**：
  ```
  /lekiwi/policy_input → SecurityMonitor (log) + PolicyGuardian (block+alert)
                         ↓ 未知指紋 → BLOCK + publish /lekiwi/security_alert
                         ↓ 嵌入flag → BLOCK + CTF flag capture
                         ↓ HMAC簽名  → ALLOW + update whitelist
                         ↓ 白名單    → ALLOW
  ```
- 車輪速度異常監控（>8 rad/s）已就緒
- 所有攻擊細節寫入 `guardian_log.jsonl`

### 下一步
- Phase 6 第二階段：將 PolicyGuardian 擴展為可配置的 whitelist loader
  - 從磁盤加載已批准的 policy fingerprints
  - 實現 policy rollback（保留最後可信狀態）
- Phase 5 CTF 實測：用 `ros2 topic pub /lekiwi/policy_input` 模擬攻擊
- Phase 7：真實 CAN bus 對接

### 阻礙
- 車輪速度異常閾值（8 rad/s）可能需要根據實際測試調優

---

## [2026-04-14 02:30] — Fresh CLIP-FM checkpoint + data collection upgrade

### 已完成
- **collect_data.py 升級**（Phase 4 數據管道）:
  - 新增 `--sim_type` 參數：`primitive` (LeKiWiSim) 或 `urdf` (LeKiWiSimURDF)
  - 隨機漫步探索策略（Brownian motion，替代純隨機 action）：動作更平滑、物理更合理
  - 支援 `--wrist` 參數：同時記錄手腕相機圖像
  - 修復 `step()` 返回值解包（4元素 tuple，無 `trunc`）
  - 默認 `--sim_type=urdf`：從真實 STL mesh 幾何收集數據
- **收集 URDF 訓練數據**:
  - 10 episodes × 200 steps = 2000 frames
  - State: arm_positions(6) + wheel_velocities(3) — 匹配 CLIP-FM 訓練格式
  - Action: random-walk [-1, 1]，clamped
  - 輸出：`data/lekiwi_urdf_demo.h5` (images/states/actions 三個 dataset)
- **訓練新鮮 CLIP-FM checkpoint**:
  - 2000 幀 URDF 數據，5 epochs，batch_size=32
  - Loss: epoch1=1.55 → epoch5=0.84（收斂正常）
  - 架構：CLIP ViT-B/32 frozen (151M) + FlowMatchingHead (time_feat=256, total_dim=786)
  - 輸出：`results/fresh_train/policy_urdf_ep5.pt` — 乾淨架構，strict=True 加載
- **vla_policy_node 優先級修復**:
  - 自動優先使用 fresh checkpoint > old SimpleCNN checkpoint
  - 同步到 `lekiwi_ros2_bridge/vla_policy_node.py` 和 `src/vla_policy_node.py` 兩個副本
  - Fresh checkpoint 使用 `strict=True`（保證架構完全匹配）
  - Git pushed: `1eb2b0d`

### 架構狀態
- **CLIP-FM checkpoint 問題已解決**：fresh checkpoint 使用正確的 CLIP 架構
  - 舊 checkpoint：SimpleCNN vision + flow_mlp 前綴（需 key remapping）
  - 新 checkpoint：CLIP ViT-B/32 vision + flow_head 前綴（strict=True）
- 完整閉環現已暢通：
  ```
  /lekiwi/cmd_vel → bridge_node → MuJoCo (URDF STL) → /lekiwi/joint_states
  /lekiwi/joint_states + /lekiwi/camera/image_raw → vla_policy_node (fresh clip_fm)
  → /lekiwi/vla_action → bridge_node → MuJoCo (closed loop)
  ```

### 下一步
- 更多 epoch 訓練或更大數據集（10 episodes 只是示範）
- 實現真實機械臂數據管道（Phase 7: CAN bus 對接）
- Phase 5 CTF 安全模式實際部署測試

### 阻礙
- 數據多樣性：random-walk 探索覆蓋範圍有限，長時間訓練需要更好策略

---

## [2026-04-14 01:30] — CLIP-FM checkpoint loading 修復

### 已完成
- **Bug 修復：CLIP-FM checkpoint 架構不相容**
  - 舊 checkpoint (`policy_ep10.pt`): `flow_mlp.*` 前綴, SimpleCNN vision, time_feat=128, total_dim=658
  - 新模型架構: `flow_head.*` 前綴, CLIP ViT-B/32 vision, time_feat=256, total_dim=786
  - 問題：舊 vision_encoder 使用 SimpleCNN（Conv2d layers）而新架構用 CLIP（151M params）— 架構完全不同
  - 修復：`_make_clip_fm_policy()` 實現：
    1. `flow_mlp.* → flow_head.*` key remap（backwards compatibility）
    2. shape-based partial loading（`strict=False`，只加載形狀匹配的 weights）
    3. 清晰的日誌輸出（顯示加載了多少 weights，跳過了哪些）
  - 結果：CLIP 151M frozen encoder → 成功加載；flow_head 訓練好的 weights → 部分加載（shape 匹配的 14/419）；未匹配的 flow_head layers → 隨機初始化（需要重新訓練）
  - Git pushed: `bea4ccd`

### 架構發現
| 組件 | 舊架構 (checkpoint ep10) | 新架構 (當前代碼) |
|------|--------------------------|------------------|
| vision_encoder | SimpleCNN (5M, Conv2d→MLP) | CLIP ViT-B/32 frozen (151M) |
| time_mlp | Linear(1,64)→SiLU→Linear(64,128) | Linear(1,128)→SiLU→Linear(128,256) |
| time_feat | 128 | 256 |
| flow_mlp/net.0 | Linear(658, 512) | Linear(786, 512) |
| total_dim | 658 | 786 |
| checkpoint prefix | `flow_mlp` | `flow_head` |

### 下一步
- 重新訓練 CLIP-FM policy（使用新架構：CLIP vision + time_feat=256）生成乾淨的 checkpoint
- 實現真實 CAN bus 對接（Phase 7）
- VLA 閉環測試（使用 `mock` 或 `clip_fm` policy 實際驅動 bridge）

### 阻礙
- 舊 checkpoint 只有 flow_head 的部分 weights 可用（time_mlp 和 net.0 因為架構變更無法加載）

---

## [2026-04-14 00:00] — CLIP-FM VLA 整合完成
### 已完成
- **Phase 4 完成：CLIP-FM policy 支援整合進 vla_policy_node.py**
  - 新增 `_make_clip_fm_policy()`: 從 `scripts/train_clip_fm.py` 載入 `CLIPFlowMatchingPolicy`
    - CLIP ViT-B/32 frozen vision encoder (151M params)
    - Flow Matching MLP action head (8M trainable params)
    - 4-step Euler ODE inference via `policy.infer()`
  - 新增 `_normalize_state()`: LeKiWi native units → [-1,1] policy input
  - 新增 `CLIPFMPolicyRunner`: 適配 `predict(obs)` 接口（與 Mock/LeRobot API 一致）
    - `obs["image"]`: HWC uint8 → CHW float [0,1]
    - `obs["state"]`: 9-DOF native units → normalized
  - `_POLICY_LOADERS` 新增 `clip_fm` policy
  - 更新 `vla.launch.py` + `full.launch.py` policy list: 新增 `clip_fm`
  - 新增 `clip_fm` usage: `ros2 launch lekiwi_ros2_bridge vla.launch.py policy:=clip_fm`
  - Default checkpoint: `~/hermes_research/lekiwi_vla/results/fm_50ep_improved/policy_ep10.pt`

### 架構現已完整
```
ROS2 /lekiwi/cmd_vel → bridge_node → MuJoCo sim
MuJoCo obs → bridge_node → /lekiwi/joint_states + /lekiwi/camera/image_raw
/lekiwi/joint_states + /lekiwi/camera/image_raw → vla_policy_node (clip_fm)
→ /lekiwi/vla_action → bridge_node → MuJoCo (closed loop)
```

### 下一步
- 驗證 CLIP-FM checkpoint 存在性（results/fm_50ep_improved/policy_ep10.pt）
- `clip_fm` policy 在 full.launch.py 中的端到端測試
- Phase 5: VLA → ROS2 topic 的實際控制輸出

### 阻礙
- 需確認 `transformers` library 在 ROS2 node 環境中可用
- CLIP ViT-B/32 模型需要首次下載（如果未曾運行過 train_clip_fm.py）

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

### 2026-04-12 05:00 (自動心跳)
- **Phase 5 CLIP-FM 閉環驗證完成**
- End-to-end 驗證：policy checkpoint 成功加載 → 推理 → 訓練 step 全流程暢通
  - `CLIPFlowMatchingPolicy` (hidden=512, 152M params) + `results/fresh_train/policy_urdf_ep5.pt` 完全兼容
  - `load_state_dict(..., strict=True)` 成功（所有 keys matched）
  - `policy.infer()` batch inference OK：action range=[-3.7, +2.5]
  - `policy.forward()` training step OK：loss=0.7070（MSE）
- `eval_policy.py --arch clip_fm` 成功運行 2 episodes：
  - Mean reward: -112.95 ± 7.69, Mean distance: 0.327m
- Training data (`data/lekiwi_urdf_demo.h5`) 2000 frames 驗證：
  - State: arm[0:6] positions + wheel_vel[6:9]（matches CLIP-FM training format）
  - Action: [-1,1] normalized（matches training distribution）
  - Image: (2000, 224, 224, 3) uint8 HWC（CLIP ViT-B/32 compatible）
- **架構全環節暢通**：
  - collect_data → HDF5 (state/action/clip_fm format) ✓
  - train_clip_fm.py → CLIP-FM policy checkpoint ✓
  - vla_policy_node.py → checkpoint loading + inference ✓
  - bridge_node → VLA action → ROS2 /lekiwi/vla_action 閉環 ✓
- Git: da41c7b (clean, no changes this heartbeat)
- **下一步**：
  - 實現真實機械臂控制（real_mode.launch.py + ST3215 serial adapter）
  - 擴展 training dataset（真實robot數據）
  - 改善 reward function（current: -112 avg reward）

### 2026-04-12 02:00 (自動心跳)
- **Phase 1-5 架構完成確認 + 2 個關鍵 Bug 修復**
- **Bug 1 [bridge_node] — VLA 閉環死鎖（已修復）**
  - 發現原因：`_on_timer` 每 20Hz tick 都清除 `_vla_action_fresh = False`
  - 後果：當 `cmd_vel` 頻率 < 20Hz（常見情況），每次 timer tick 清除 flag
    → `_on_cmd_vel` 認為 VLA 不活躍，覆蓋 arm_action → VLA 閉環中斷
  - 修復：移除 `_on_timer` 中的 `self._vla_action_fresh = False`
  - 邏輯重構：`_vla_action_fresh` 現在只在 `_on_cmd_vel` 內部清除（驅動式清除）
- **Bug 2 [vla_policy_node] — CLIP-FM 狀態格式不一致（已修復）**
  - 發現：`_run_inference` 建構 state = [arm_positions + wheel_positions]
  - 但 `lerobot_policy_inference.py` 訓練時 state = [arm + wheel_vel]
  - CLIP-FM policy 訓練代碼：`state = np.concatenate([arm_positions, wheel_velocities])`
  - 推理時傳入 positions 造成 training/inference distribution mismatch
  - 修復：`wheel_positions` → `wheel_velocities`
  - 受影響檔案：`vla_policy_node.py`（兩個副本）已同步修復
- Git pushed: `eecec7f`
- **架構現已完整暢通**：
  - cmd_vel (1-10Hz) + VLA action → bridge_node._on_cmd_vel → sim.step() ✓
  - sim._obs() → /lekiwi/joint_states → vla_policy_node._on_joint_states ✓
  - vla_policy_node._run_inference() → state=[arm+wheel_vel] ✓ (training match)
  - normalize → /lekiwi/vla_action → bridge_node._on_vla_action ✓
  - `_vla_action_fresh` 保持 True 直到下一個 cmd_vel 驅動迴圈 ✓
- **下一步**：
  - 收集真實數據以訓練/微調 CLIP-FM policy
  - 實現真實機械臂控制模式（切换「模擬模式」vs「真實模式」）

### 2026-04-12 01:00 (自動心跳)

### 已完成
- **Bug 修復**：`CLIPFMPolicyRunner.predict()` obs key 不匹配
  - `LeKiWiVLAPolicyNode._run_inference()` 構建 `obs["observation.images.primary"]` + `obs["observation.state"]` (LeRobot 格式)
  - `CLIPFMPolicyRunner.predict()` 原本只接受 `obs["image"]` + `obs["state"]` (simple 格式)
  - 修復：讓 `CLIPFMPolicyRunner.predict()` 自動檢測並支持兩種格式
  - 同步到 `src/vla_policy_node.py` 和 `lekiwi_ros2_bridge/vla_policy_node.py`
  - Git pushed: `8256ec2`
- **架構 Phase 4 現況**：
  - `bridge_node.py` — ROS2 `/lekiwi/cmd_vel` → MuJoCo sim + 發布 `/lekiwi/joint_states` + camera topics
  - `vla_policy_node.py` — 訂閱 `/lekiwi/joint_states` + `/lewi/camera/image_raw` → 發布 `/lekiwi/vla_action`
  - `security_monitor.py` — CTF 安全監控（已捕獲 Challenge 7 flag）
  - `sim_lekiwi_urdf.py` — STL mesh 幾何（wrist camera 已整合）
  - CLIP-FM policy wrapper 已就緒（支持 LeRobot + simple 兩種 obs 格式）

### 下一步
- 整合 lekiwi_modular 的 URDF + STL → bridge_node 切換模式
- 實現真實 CAN bus 對接（Phase 7）
- VLA checkpoint 實際訓練/加載流程文檔化

### 阻礙
- lekiwi_modular 和 lekiwi_vla 代碼同步（bridge package 有兩個 vla_policy_node.py 副本）

---

## [2026-04-15 06:00] — Phase 5.5: CTF Attack Simulation + PolicyGuardian Validation

### 已完成
- **CTF Attack Simulation Script** (`scripts/ctf_attack_sim.py`, 423 行):
  - `CTFAttackSimulator`: 模擬全部 7 個 Robot CTF 攻擊場景
    - Challenge 1: UDP teleport (極端 velocity injection via raw UDP)
    - Challenge 2: Eavesdrop/replay (修改 angular velocity)
    - Challenge 3: Auth bypass (firmware dump 分析，hardcoded credentials)
    - Challenge 4: Serial shell (ST3215 malformed packets)
    - Challenge 5: Firmware dump (debug interface request)
    - Challenge 6: Adversarial patch (FGSM perturbation 生成)
    - Challenge 7: Policy hijack (**PolicyGuardian 必須阻斷**)
  - `MaliciousActor`: 可 pickle 的恶意 actor 類，用於 Challenge 7 驗證 PolicyGuardian 阻斷能力
  - ROS2 發布者（cmd_vel, policy_input, wheel_N/cmd_vel）+ Offline 模式（無 ROS2 也能測試）
  - `python scripts/ctf_attack_sim.py --offline`: 離線測試全部 7 個攻擊
  - `python scripts/ctf_attack_sim.py --offline --attack 7`: 單獨測試 Challenge 7
  - 測試結果摘要：
    ```
    ✅ PolicyGuardian: Challenge 7 BLOCKED — defense effective!
    6/7 flags captured (Challenge 7 blocked by PolicyGuardian ✓)
    ```

### 架構狀態
- **Phase 6.2 完成後的 CTF 實測工具就緒**
  - 全部 7 個 CTF challenge 有對應的 attack simulation
  - `guardian_log.jsonl` 已捕獲 `ROBOT_CTF{policy_hijack_4c8e2a9f}`
  - PolicyGuardian 雙層防禦：SecurityMonitor（日誌）+ PolicyGuardian（阻斷 + 告警）
- Git: `04c5fd5` pushed

### 下一步
- Phase 7: 真實 CAN bus 對接（Raspberry Pi GPIO → ST3215）
- Phase 6.3: VLA 端到端測試（使用 mock policy 驅動 bridge）

### 阻礙
- なし（工作正常推進中）

---

## [2026-04-15 05:30] — Maintenance: Package structure sync

### 已完成
- **Critical bug fix: Entry point desynchronization**
  - 發現：`src/` 子目錄的 `bridge_node.py`（269行）與 git HEAD（634行）不一致
  - 原因：工作目錄覆蓋了 git index，但 git 认为 HEAD = index，覆蓋後丟失了 PolicyGuardian/real_mode/odometry
  - 修復：
    1. 從 da41c7b 恢復 634-line 版本到 `src/lekiwi_ros2_bridge/bridge_node.py`（entry point）
    2. 同步到 `lekiwi_ros2_bridge/bridge_node.py` 保持一致
    3. 將 `src/vla_policy_node.py`（406行）→ `vla_policy_node.py`（485行，entry point）
    4. 同步到 `lekiwi_ros2_bridge/vla_policy_node.py` 保持一致
    5. 刪除空的 `src/` 子目錄
  - 清理後結構：
    ```
    lekiwi_ros2_bridge/
    ├── bridge_node.py           ← entry point (634 lines ✓)
    ├── vla_policy_node.py       ← entry point (485 lines ✓)
    ├── real_hardware_adapter.py
    ├── setup.py
    ├── launch/
    │   ├── bridge.launch.py
    │   ├── full.launch.py
    │   ├── real_mode.launch.py
    │   └── vla.launch.py
    └── lekiwi_ros2_bridge/       ← subpackage (synchronized copies)
        ├── __init__.py
        ├── bridge_node.py       ← 634 lines ✓
        ├── vla_policy_node.py   ← 485 lines ✓
        ├── policy_guardian.py   (445 lines)
        ├── real_hardware_adapter.py (349 lines)
        └── security_monitor.py  (169 lines)
    ```
  - Git: `74ff7e0` pushed ✓

### 架構狀態
- 所有 entry point 文件（`bridge_node`, `vla_policy_node`）現在與 subpackage 同步
- PolicyGuardian（Challenge 7 防禦）、real_mode（ST3215 適配器）、odometry、wrist_camera 全部就緒
- 3 種模式：`sim`（MuJoCo cylinder）、`sim+urdf`（MuJoCo STL）、`real`（serial 硬體）

### 下一步
- Phase 7: 真實 CAN bus 對接（Raspberry Pi GPIO → ST3215）
- Phase 5 CTF 實測（用 `ros2 topic pub /lekiwi/policy_input` 模擬攻擊）
- Phase 6.3: VLA 端到端測試（mock policy 驅動 bridge）

### 阻礙
- 兩個 `real_hardware_adapter.py` 文件（一個在 root，一個在 subdir）— 需統一
## [2026-04-15 07:00] — Bug Fix: flow_mlp→flow_head checkpoint key rename

### 已完成
- **train_with_better_reward.py** 生成的 `results/improved/final_policy.pt` 有 `flow_mlp.*` key prefix
- **eval_policy.py** 的 `SimpleCNNFlowMatchingPolicy` 使用 `flow_head.*` prefix
- 導致 `load_state_dict()` 失敗：`Missing key(flow_head.*) + Unexpected key(flow_mlp.*)`
- **修復**：用 Python + OrderedDict 原地 rename checkpoint 中所有 `flow_mlp.* → flow_head.*`
- 修復後評估：Mean reward=-104.726 ± 1.576, Mean distance=0.117m (2 episodes)
- 清理：`bridge_node.py.bak` 已刪除
- Git: clean (no uncommitted changes)

### 架構狀態
- 3個有效policy checkpoint：
  | Checkpoint | 架構 | Mean Reward | Distance |
  |------------|------|-------------|-----------|
  | `results/fresh_train/policy_urdf_ep5.pt` | CLIP-FM | -107.48 | 0.102m |
  | `results/improved/final_policy.pt` | SimpleCNN-FM | -104.73 | 0.117m |
  | random baseline | — | -110.16 | 0.443m |

### 下一步
- Phase 7: 真實 CAN bus 對接（Raspberry Pi GPIO → ST3215）
- Phase 6.3: VLA 端到端測試（使用 improved checkpoint 驅動 bridge）
- Phase 5 CTF 安全模式實測（用 `ros2 topic pub /lekiwi/policy_input` 模擬攻擊）

### 阻礙
- Phase 7: 真實 CAN bus 對接（Raspberry Pi GPIO → ST3215）

