# Hermes Research — 跨專案研究地圖

## 專案總覽

```
~/hermes_research/
│
├── lekiwi_vla/              ⭐ 核心 VLA 研究
│   ├── sim_lekiwi.py        — MuJoCo 模擬（9 DOF）
│   ├── lerobot_policy_inference.py — LeRobot 工廠 (ACT/Diffusion/FM/GR00T)
│   ├── client/hardware_interface.py — LeRobot LeKiwi 硬體包裝
│   ├── server/vla_server.py  — VLA API 服務器
│   ├── scripts/
│   │   ├── collect_data.py              — 模擬數據 → HDF5 ✅
│   │   ├── train_flow_matching_real.py  — Flow Matching 訓練 ✅
│   │   ├── eval_policy.py               — 策略評估 (random/FM) ✅
│   │   ├── convert_docking_data.py       — 真實 JSON → HDF5 ✅
│   │   ├── train_flow_matching_lekiwi.py — 獨立 FM (無 LeRobot) ✅
│   │   ├── record_lekiwi.py             — LeRobot 數據錄製 (需真實硬體)
│   │   └── infer_groot.py               — GR00T-N1.5 推論
│   └── docs/VLA_COMPARISON.md
│
├── lekiwi_modular/          🔧 真實硬體 ROS2 專案
│   ├── src/lekiwi_description/urdf/  — URDF + STL (3D列印)
│   ├── src/lekiwi_controller/   — omni_controller (有 bug, 已修復)
│   ├── src/lekiwi_servo_control/ — C++ 馬達控制
│   ├── src/scripts/
│   │   ├── analyze_docking.py  — 55+ 軌跡分析 ✅
│   │   ├── analyze_imu.py      — IMU 噪聲分析 ✅
│   │   └── log_*/             — 真實機器人數據 (13486 frames)
│   └── RESEARCH_ANALYSIS.md
│
├── go2-vla/                 🐕 四足步態控制
│   ├── gait_controller.py     — FK/IK + trot/walk/stand
│   └── go2_vla_node.py       — ROS2 節點
│
├── robot-security-workshop/  🔒 安全研究
│   ├── adversarial_toolkit/   — FGSM/PGD/CW + ROS Message 篡改
│   ├── vulnerable_robot/      — CTF 靶機 (UDP/ROS/Serial)
│   └── ctf-platform/         — Flask 計分系統
│
├── unifolm-vla/             🏋️ Unitree VLA 分析
│   └── RESEARCH_ANALYSIS.md  — Flow Matching + DiT + Qwen2.5-VL
│
└── unitree_rl_gym/          🏃 Isaac Gym RL 訓練
    └── RESEARCH_ANALYSIS.md  — PPO / Go2 / G1 配置
```

---

## 硬體對應表

| 設備 | 軟體位置 | 控制方式 | 狀態 |
|------|----------|----------|------|
| **LeKiwi** (輪式機械臂, 9 DOF) | lekiwi_modular/ | ROS2 | 🔴 Bug修復中 |
| LeKiwi 模擬 | lekiwi_vla/sim_lekiwi.py | MuJoCo | ✅ 可用 |
| SO-101 Leader Arm | lekiwi_vla/scripts/record_lekiwi.py | USB `/dev/tty.usbmodem585A0077581` | 🟡 待測 |
| **LeRobot SO-101** | ~/lerobot/ | LeRobot | ✅ 本地安裝 |
| **Unitree Go2** | go2-vla/ | ROS2 | ✅ 可用 |
| Unitree G1 | unitree_rl_gym/ | Isaac Gym RL | 🔵 僅分析 |
| CTF 靶機 | robot-security-workshop/vulnerable_robot/ | UDP/TCP | ✅ 可用 |

---

## 數據流向圖

```
真實機器人數據
┌─────────────────────────────────────────────┐
│ lekiwi_modular/src/scripts/log_*/           │
│ 55+ 軌跡 (single_L, dual_L, straight...)   │
│ JSON: {t, phase, pitch1, pitch_smooth...}  │
└──────────────────┬──────────────────────────┘
                   │ convert_docking_data.py
                   ▼
┌─────────────────────────────────────────────┐
│ data/docking_real.h5                        │
│ 13486 frames, states(7D), actions(9D)      │
│ ⚠️ 無圖像（只有姿態感測器）                  │
└──────────────────┬──────────────────────────┘
                   │ train_flow_matching_real.py
                   ▼
┌─────────────────────────────────────────────┐
│ results/fm_*/final_policy.pt                │
│ Flow Matching Policy (8M params)            │
│ 4-step Euler inference                      │
└──────────────────┬──────────────────────────┘
                   │ eval_policy.py
                   ▼
┌─────────────────────────────────────────────┐
│ Simulation Benchmark Results                 │
│ Random: -106.3 | FM(5ep): -106.6           │
└─────────────────────────────────────────────┘

        +++++++++++++++++++++++++++

模擬數據（完整，可訓練 VLA）
┌─────────────────────────────────────────────┐
│ python3 scripts/collect_data.py             │
│ MuJoCo sim → HDF5 (224x224 images)         │
└──────────────────┬──────────────────────────┘
                   │ train_flow_matching_real.py
                   ▼
┌─────────────────────────────────────────────┐
│ Trained FM Policy on Sim Data               │
│ → 需要 50+ epochs 才有有意義的 policy        │
└─────────────────────────────────────────────┘

        +++++++++++++++++++++++++++

真實硬體錄製（完整，有圖像）
┌─────────────────────────────────────────────┐
│ python3 scripts/record_lekiwi.py            │
│ LeRobot + SO-101 → HuggingFace Dataset     │
│ Robot IP: 172.18.134.136                   │
│ Arm: /dev/tty.usbmodem585A0077581          │
└─────────────────────────────────────────────┘
```

---

## 關鍵研究問題

### Q1: VLA 視覺編碼器 — 用哪個？
| 選項 | 參數 | 訓練需求 | 備註 |
|------|------|----------|------|
| **Simple CNN (現在用的)** | ~5M | 從頭訓練 | 免費但效果有限 |
| **CLIP (ViT-B/32)** | ~151M | 凍住 | 免費，已整合在 LeRobot |
| **DINOv2** | ~86M | 凍住或微調 | 高視覺質量 |
| **Qwen2.5-VL (GR00T基礎)** | ~7B | 凍住 | 太重，Mac 無法運行 |
| **SmolVLA** | ~1B | 全量訓練 |邊緣設備可行 |

### Q2: 動作預測方法
| 方法 | 推理步數 | 訓練穩定性 | 實際表現 |
|------|----------|------------|----------|
| **ACT** | 1 | 中等 | 需要大量數據 |
| **Flow Matching (現在用的)** | **4** | ✅ 高 | 需要 50+ epochs |
| **DDPM** | 50-100 | 低 | 太慢 |
| **GR00T-N1.5** | 4 | ✅ 高 | 需要 8GB VRAM |

### Q3: 數據瓶頸
- **真實 LeKiwi 數據**: 13486 frames（無圖像，只有姿態感測器）
- **模擬數據**: 可無限生成，但 domain gap 大
- **LeRobot 錄製**: 需要 SO-101 arm + 真實 LeKiwi 硬體
- **建議**: 用模擬數據預訓練 → 真實數據微調

---

## 演算法對比（今天基準測試）

```
隨機策略 (baseline):   reward = -106.3 ± 1.3
Flow Matching (5 ep):  reward = -106.6 ± 5.6  ← 需更多訓練

結論: 5 epochs 太少，需要 50+ epochs + 真實數據
```

---

## 立即可做的事

### 高優先順序
1. **訓練 50 epochs FM policy** — 最有機會看到有意義的 policy 改善
2. **整合 CLIP 視覺編碼器** — 替換 SimpleCNN，提升視覺特徵質量
3. **嘗試 ACT 基線** — LeRobot 已有 ACT 实现，直接用

### 中優先順序
4. **真實機器人錄製** — 聯結 SO-101 arm，錄製 100+ episodes
5. **修復 lekiwi_modular git remote** — 讓你能 commit/push
6. **benchmark ACT vs FM** — 對比兩種方法

### 低優先順序（興趣驅動）
7. **安全 CTF 框架** — 部署 vulnerable_robot
8. **GR00T-N1.5 測試** — 需要 transformers 安裝
9. **Go2 RL 訓練** — Isaac Gym 配置

---

## 定時任務 (Cron Jobs)

| 任務 | Cron ID | 下次執行 |
|------|---------|----------|
| 每4小時心跳 | `3a6e462b162a` | 16:00 |
| 每日研究日誌 | `dc77e516eecf` | 21:00 |

---

## 已知問題

| 問題 | 嚴重性 | 解決方案 |
|------|--------|----------|
| lekiwi_modular git remote 無法推送 | 🔴 | 在 GitHub 建立新 repo 後 `git remote set-url` |
| Flow Matching 5 epochs 不夠 | 🟡 | 訓練 50+ epochs |
| 真實數據無圖像 | 🟡 | 用模擬數據或重新錄製 |
| GR00T-N1.5 需 transformers dev | 🟡 | `pip install git+https://github.com/huggingface/transformers` |
| omni_controller.py bug | ✅ 已修復 | 用 `omni_controller_fixed.py` |
