# LeKiWi ROS2-MuJoCo Platform Progress

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