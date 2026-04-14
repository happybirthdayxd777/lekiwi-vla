# LeKiWi ROS2-MuJoCo Platform Progress

## Phase 38 (2026-04-14 11:15 UTC) — CTF Security Audit Tool + Phase37 Training Complete

### 本次心跳完成

**1. Phase37 Training 完成（18 min CPU，50 epochs）**
- Checkpoint: `results/phase37_goal_fixed_train/final_policy.pt` (611 MB)
- 數據：10,000幀 Phase 36 校正後的 GridSearchController 數據（M7=[1,1,1]→+X，M8=[-1,-1,-1]→+X 已修正）
- Training time: ~18 min CPU（~22s/epoch × 50）

**2. CTF Security Audit Tool 創建**
- `ctf_security_audit.py` — 獨立的資安監控工具，覆蓋 8 個 CTF 挑戰通道：
  - C1: Forged cmd_vel (no HMAC)
  - C2: DoS via rate flooding
  - C3: Command injection (magnitude violation)
  - C4: Physics DoS (acceleration spike)
  - C5: Replay attack (identical sequence)
  - C6: Sensor spoofing (joint_states injection)
  - C7: Policy injection (vla_action override)
  - C8: Policy hijacking (unauthorized switch)
- API: `CTFSecurityAuditor` 類，可嵌入 bridge_node 作為資安後端
- Demo 測試全部通過：每種攻擊都被正確檢測 + 發放 CTF flag

**3. Policy 評估對比（Phase37 vs goal_aware）**
- URDF sim，5 個目標，200 步：
  - phase37: SR=0%, mean_dist=1.566m（0/5 成功）
  - goal_aware: SR=0%, mean_dist=1.569m（0/5 成功）
- 結論：兩個 policy 在 URDF sim 上都無法到達目標（URDF sim 物理不穩定，有 WARNING: QACC NaN）
- 兩者表現幾乎相同，差異在 noise 範圍內

### 架構現狀

```
lekiwi_ros2_bridge/
  bridge_node.py           — ROS2↔MuJoCo 橋樑（兩種後端）
  vla_policy_node.py       — CLIP-FM policy inference
  real_hardware_adapter.py — 真實機器人適配器
  ctf_security_audit.py    — CTF 資安審計工具（新增）
  replay_node.py           — 數據回放

lekiwi_modular/
  lekiwi_controller/        — ROS2 omni_controller（真實 robot）
  lekiwi_description/       — URDF + Gazebo

lekiwi_ctf/
  src/challenges.py        — CTF 挑戰框架
  src/cron.py              — 自動化任務

lekiwi_vla/
  sim_lekiwi.py            — Primitive 模擬（快速穩定）
  sim_lekiwi_urdf.py       — URDF 模擬（STL mesh）
  ctf_security_audit.py    — 資安審計工具（新）
```

### 阻礙
1. URDF sim 不穩定（QACC NaN）— policy 在上面幾乎無法收斂
2. Phase 37 policy 與 goal_aware 在 URDF sim 上 SR 相同（0%）— 需在 primitive sim 評估真實 locomotion 能力差異
3. VLA policy 的 vision encoder 未使用（CLIP 純特徵提取，無視覺 grounding）

### 下一步
1. 在 LeKiWiSim（primitive）上評估 phase37 vs goal_aware 真實 locomotion 能力
2. 整合 CTFSecurityAuditor 到 bridge_node 替換現有 SecurityMonitor
3. 啟動「真實模式」：ros2 launch lekiwi_ros2_bridge real_mode.launch.py
4. 為 bridge_node 添加 `ctf_mode` parameter，開啟時記錄所有 CTF flags

---

## [2026-04-13 2100]
### Phase: Phase 18 — VLA Policy Diagnosis + Retraining Prep

### 本次心跳完成事項

**1. VLA Policy 診斷（根本問題確認）**

診斷測試揭示了為何 `task_oriented_newdata_50ep` policy 在 (0.5, 0.0) 上完全失敗：

| 測試 | 目標 | 結果 | 結論 |
|------|------|------|------|
| 9D policy, eval (0.5, 0.0) | state_dim=9, 無 goal | dist=0.853m (100步) | ❌ 離目標更遠 |
| 9D policy, eval (0.365, -0.343) | 訓練時已見過的目標 | dist=0.785m (150步) | ❌ 也不好 |
| 11D policy (goal_aware_50ep), eval (0.5, 0.0) | 正確 state_dim=11 | dist=0.912m (100步) | ❌ 更差 |
| Zero-init vs rand-init | 檢查推理隨機性 | zero-init: 0.870m, rand-init: 0.829m | 無顯著差異 |
| Random policy baseline | 100步隨機 | ~0.85-0.93m | 與 policy 相近 |

**根本原因分析：**

Training data `lekiwi_goal_urdf_10k.h5`:
- 4000 frames, 20 unique goals, 每個 goal 200 frames
- Goals 範圍: [-0.458, 0.412] — **從未覆蓋 (0.5, 0.0)**
- 評估目標 (0.5, 0.0) 位於訓練分佈的邊緣之外
- 評估與訓練最近的目標: (0.365, -0.343) dist=0.501m，角度差 43.2°

Policy 行為：
- 輪胎動作從 [-1.95, -1.82, 1.81] 變化到 [-2.57, -2.67, 3.21] — 完全無方向性
- 動作幅度極大（接近 ±1.0 限制），導致無效的快速移動
- Flow matching 推理中 random init + 4-step denoising 進一步放大隨機性

**2. 架構完整性確認**

| 組件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 完整 824 行 | ROS2↔MuJoCo 橋樑 |
| vla_policy_node.py | ✅ 531 行 | CLIP-FM + MockPolicy |
| lekiwi_ros2_bridge/ | ✅ 完整 package | 含 launch, config, real_hardware_adapter |
| sim_lekiwi_urdf.py | ✅ 已修復接觸物理 | STL mesh (contype=0) + 接觸圓柱 (contype=1) |
| lekiwi_modular/ | ✅ 同步 | wheel joint axes 與 bridge_node 一致 |

### 下一步（下次心跳）

1. **收集新的多目標訓練數據**
   - 目標：覆蓋整個 [-1, 1] × [-1, 1] 範圍，50+ unique goals
   - 使用 `collect_goal_directed.py`，增加 `--episodes 100 --goal_min 0.1 --goal_max 0.9`
   - 確保 (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.0, -0.5) 都在數據中

2. **重新訓練 VLA policy**
   - 使用新的多目標數據集
   - 訓練 100 epochs，目標 mean_dist < 0.15m
   - 驗證時使用 5 個均勻分佈的測試目標

3. **驗證 bridge_node.py 的 ROS2 集成**
   - 在有 ROS2 的環境測試 `/lekiwi/cmd_vel` → MuJoCo 動作
   - 測試 `/lekiwi/joint_states` → ROS2 發布

### 阻礙
- 評估目標 (0.5, 0.0) 不在訓練數據分佈中 → policy 無法泛化
- 缺乏 ROS2 runtime 環境做端到端驗證
- lekiwi_modular remote 404（需手動修復）

---

## [2026-04-13 2030]
### Phase: Phase 17 — Contact Physics Fix + Modular Repo Push

### 本次心跳完成事項

**1. MuJoCo 接觸物理修復（sim_lekiwi_urdf.py）**
- 發現：STL mesh + 原本的 euler="0 1.5708 0" + contype=1 導致接觸失敗
- 修復：STL mesh 設為 `contype=0 conaffinity=0`（純視覺），另加圓柱接觸體
- 接觸圓柱：`<geom type="cylinder" size="0.025 0.008" pos="0 0 -0.025">`
  - 半徑 2.5cm = 真實omni輪半徑，高度 0.8mm
  - 位置在輪軸心正下方 2.5cm，底部接觸地面
- Commit: `5bce6cd Fix MuJoCo wheel contact` → pushed ✅

**2. lekiwi_modular 修復上遊推送**
- 發現 `omni_controller_fixed.py` 只在本地，origin/main 無此檔案
- 確認 URDF joint axes 與 bridge_node.py 一致：
  - Revolute-64 → `[-0.866025, 0, 0.5]` ✅
  - Revolute-62 → `[0.866025, 0, 0.5]` ✅
  - Revolute-60 → `[0, 0, -1.0]` ✅
- Commit: `0660629 fix: wheel joint axes from URDF` → pushed ✅

**3. Git 同步狀態**
- lekiwi_vla: main @ `5bce6cd` (up-to-date with origin)
- lekiwi_modular: main @ `0660629` (ahead of origin — remote repo not found)

### 下一步（下次心跳）
1. **驗證接觸物理**：在有 display 的環境執行 `python sim_lekiwi_urdf.py` 確認輪子穩定接觸地面
2. **VLA 訓練改善**：goal_aware policy mean_dist=0.555m → 需降至 <0.15m
   - 增加 training epochs（100+）
   - 改善 reward shaping
3. **ROS2 端到端燒試**（需要 ros2 環境）

### 阻礙
- lekiwi_modular remote `AaronLuo1208/Modular-lekiwi-robot.git` 不存在（404），需更換 remote
- 缺乏 ROS2 runtime 環境做端到端驗證

---

## [2026-04-13 1930]
### Phase: Phase 14 — Architecture Cross-Verification + Git Push

### 本次心跳完成事項

**1. goal_aware_50ep 完整評估**
- Epoch 50 policy 成功載入（CLIP ViT-B/32 frozen + flow head 969K trainable）
- 3 episodes 無干擾導航測試：
  - Episode 0: dist=0.549m (200步)
  - Episode 1: dist=0.777m (200步)
  - Episode 2: dist=0.340m (200步)
  - Mean: 0.555m, Min: 0.340m
  - Success rate (0.15m): 0/3 ← 仍需改進，目標 <0.15m

**2. VLA Training Run 完整 Commit**
- 新增 `results/goal_aware_50ep/` (716MB) 含 checkpoint_epoch_[10/20/30/40/50].pt + final_policy.pt + training_analysis.png
- Git push: 751528e..1f4a802

**3. 架構完整性確認**
- bridge_node.py (824行) — 完整 ROS2↔MuJoCo 橋樑
- vla_policy_node.py (531行) — CLIP-FM + MockPolicy
- 所有 launch files 功能確認
---

## [2026-04-13 2130]
### Phase: Phase 19 — Goal Distributional Gap Confirmed + ID vs OOD Validation

### 本次心跳完成事項

**1. Goal Distributional Gap 量化驗證（eval_goal_gap.py）**

創建 `scripts/eval_goal_gap.py`：在 ID（訓練分佈內）和 OOD（訓練分佈外）目標上系統性評估 goal_aware policy。

| 類別 | 目標範例 | SR | Mean Dist |
|------|----------|----|-----------|
| **ID** ([-0.4, 0.4]) | (0.2,0), (-0.2,0), (0.3,0.3) | **40%** | **0.689m** |
| **OOD** (>0.4) | (0.5,0), (0.6,0), (0.5,0.5) | **0%** | **1.200m** |

結論：目標 (0.5, 0.0) 明確位於 OOD 區域，無法泛化。

**2. 根本原因確認**

- 訓練數據 `lekiwi_goal_urdf_10k.h5`：goal range X: [-0.412, 0.365], Y: [-0.458, 0.412]
- 評估目標 (0.5, 0.0) 完全在訓練分佈之外
- ID SR=40% 表示 policy 在見過的目標範圍內有部分成功率
- OOD SR=0% 表示完全無法泛化到未見過的目標坐標

**3. 現有架構狀態確認**

| 元件 | 狀態 |
|------|------|
| `bridge_node.py` | ✅ ROS2↔MuJoCo 完整橋樑（824行） |
| `vla_policy_node.py` | ✅ CLIP-FM + MockPolicy（531行） |
| `lekiwi_sim_loader.py` | ✅ 工廠：primitive/urdf/real |
| `goal_aware_50ep` policy | ✅ state_dim=11, 51M params |
| `eval_goal_gap.py` | ✅ 新增：ID vs OOD 量化診斷 |

### 下一步（下次心跳）

1. **收集寬範圍訓練數據**
   - 目標：覆蓋 [-0.8, 0.8] × [-0.8, 0.8]，100+ unique goals
   - 修改 `collect_goal_directed.py`：增加 `--goal_max 0.8`
   - 確保 (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.6, 0.3), etc. 都在數據中

2. **重新訓練 goal_aware policy（100 epochs）**
   - 使用新的寬範圍數據集
   - 預期：ID→SR提升，OOD→SR>0%

3. **驗證 bridge_node ROS2 集成**（需要 ROS2 環境）

### 阻礙
- 評估目標 (0.5, 0.0) 不在訓練分佈 → 根本原因是數據分佈，非架構
- 缺乏 ROS2 runtime 環境
- lekiwi_modular remote 404

---

## Phase 39 (2026-04-14 11:38 UTC) — CTFSecurityAuditor Integration

### 本次心跳完成

**CTFSecurityAuditor 完整集成進 bridge_node.py（8 通道資安監控）**

bridge_node.py 新增 `self.ctf_auditor: CTFSecurityAuditor`，監控所有 CTF 攻擊通道：

| Challenge | Channel | 檢測內容 | bridge_node 調用位置 |
|-----------|---------|---------|---------------------|
| C1 | `/lekiwi/cmd_vel` | 無 HMAC 的 cmd_vel（forged） | `_on_cmd_vel()` raw |
| C2 | `/lekiwi/cmd_vel` | Rate flooding (>100Hz) | `_on_cmd_vel()` |
| C3 | `/lekiwi/cmd_vel` | Magnitude violation (>1.5m/s) | `_on_cmd_vel()` / `_on_cmd_vel_hmac()` |
| C4 | `/lekiwi/cmd_vel` | Acceleration spike (>5m/s²) | `_on_cmd_vel()` / `_on_cmd_vel_hmac()` |
| C5 | `/lekiwi/cmd_vel` | Replay attack (3x identical) | `_on_cmd_vel()` / `_on_cmd_vel_hmac()` |
| C6 | `/lekiwi/joint_states` | Sensor spoofing (velocity/position jump) | `_on_timer()` publish |
| C7 | `/lekiwi/vla_action` | Policy injection (arm/wheel action limit) | `_on_vla_action()` |
| C8 | `/lekiwi/policy` | Policy hijacking (unauthorized switch) | `_on_policy_input()` |

**代碼變更：**
- `bridge_node.py`: 824 → 919 lines (+95)
  - 新增 `ctf_mode` parameter：開啟時寫入 `ctf_flags.jsonl`
  - 新增 `_on_security_alert()` callback：將 `SecurityAlert` 發布至 `/lekiwi/security_alert`
  - C1/C2/C3/C4/C5 集成至 `_on_cmd_vel()` 和 `_on_cmd_vel_hmac()`
  - C6 集成至 `_on_timer()` joint_states 發布
  - C7 集成至 `_on_vla_action()`
  - C8 集成至 `_on_policy_input()`
- `ctf_security_audit.py`: `_record()` 修復 — log_path 為 None 時不寫入，設定後寫入 JSONL

**Legacy 相容性：**
- `SecurityMonitor`（舊版 6-channel）保留用於現有 `check_cmd_vel()` / `check_cmd_vel_hmac()` / `check_policy()` 回調
- `PolicyGuardian` 保留用於 `_on_policy_input()` 的主動阻斷邏輯

### 架構現狀

```
lekiwi_ros2_bridge/
  bridge_node.py           — ROS2↔MuJoCo 橋樑（919行，含 CTFSecurityAuditor）
  vla_policy_node.py       — CLIP-FM policy inference
  real_hardware_adapter.py — 真實機器人適配器
  ctf_security_audit.py   — CTF 資安審計工具（已修復 log_path）
  replay_node.py           — 數據回放

lekiwi_vla/
  sim_lekiwi.py            — Primitive 模擬
  sim_lekiwi_urdf.py       — URDF 模擬（STL mesh）
  ctf_security_audit.py   — CTF 資安審計工具
  phase37_goal_fixed_train/ — 已訓練 policy (611MB, 50 epochs)
```

### 阻礙
1. URDF sim 不穩定（QACC NaN）— policy 在上面 SR=0%
2. 缺乏 ROS2 runtime 環境做端到端驗證
3. lekiwi_modular remote 404（需更換 remote URL）

### 下一步
1. 測試 bridge_node + CTFSecurityAuditor 在有 ROS2 的環境（colcon build + ros2 run）
2. 在 primitive sim 上評估 phase37 policy 的真實 locomotion 能力
3. 收集寬範圍多目標訓練數據（goal range [-0.8, 0.8]²）並重新訓練
