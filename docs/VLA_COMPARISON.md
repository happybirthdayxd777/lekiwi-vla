# VLA 架構完整比較報告
**研究日期**: 2026-04-11 | **委託**: Aaron Luo

---

## 1. 完整 VLA 矩陣

| VLA | 機構 | Action 生成 | 參數量 | 推理步數 | 開源? | LeRobot 支援 |
|-----|------|------------|--------|---------|-------|-------------|
| **Pi0** | Physical Intelligence | Diffusion | 7B | 10+ | ❌ 閉源 | ✅ (PI0Policy) |
| **Pi0-fast** | Physical Intelligence | Diffusion | ~7B | 5 | ❌ 閉源 | ✅ (PI0FastPolicy) |
| **GR00T-N1.5** | NVIDIA | Flow Matching | 3B | 4 | ✅ (HuggingFace) | ✅ (GrootPolicy) |
| **UnifoLM-VLA-0** | Unitree | Flow Matching + DiT | ~7B | 4 | ✅ (訓練+推理) | ❌ (獨立框架) |
| **Multi-Task DiT** | LeRobot/HF | Flow Matching / Diffusion | ~100M | 4-100 | ✅ | ✅ (內建) |
| **OpenVLA** | UC Berkeley | Regression | 7B | 1 | ✅ | ⚠️ (需自己整合) |
| **ACT** | MIT/Stanford | Action Chunking | ~40M | 1 | ✅ | ✅ (內建) |
| **SmolVLA** | LeRobot | Regression | ~1B | 1 | ✅ | ✅ (內建) |
| **Diffusion Policy** | Columbia/NYU | Diffusion | ~80M | 50-100 | ✅ | ✅ (內建) |

---

## 2. 核心技術深度解析

### 2.1 Action Generation 方法對比

```
┌─────────────────────────────────────────────────────────────┐
│                    ACTION GENERATION                        │
├──────────────┬──────────────┬──────────────┬──────────────┤
│  Regression  │ Action       │ Diffusion    │ Flow         │
│  (OpenVLA,  │ Chunking     │ (DDPM/DDIM)  │ Matching     │
│   SmolVLA)  │ (ACT)        │              │ (GR00T,     │
│              │              │              │  UnifoLM,   │
│              │              │              │  MultiTaskDiT)│
├──────────────┼──────────────┼──────────────┼──────────────┤
│  1-step     │ 1-step       │ 50-100 steps │ 4-10 steps  │
│  output     │ chunk output  │ iterative    │ linear ODE  │
│  (fastest)  │ (fast)       │ (slow)       │ (fast)      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│  No temporal│ Temporal      │ Smooth action│ Smooth +    │
│  consistency│ consistency   │ sequences    │ 快速收斂   │
│  between    │ via chunk    │              │             │
│  predictions │              │              │             │
├──────────────┼──────────────┼──────────────┼──────────────┤
│  OpenVLA:   │ ACT: 8-16    │ Diffusion:    │ GR00T: 4    │
│  7B params  │ action chunk  │ 50-100 steps │ steps!     │
│  (too large)│ size varies   │ (too slow    │ ⚡ Best     │
│             │              │  for real)    │ speed/qual  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 2.2 Flow Matching 詳細原理

Flow Matching 是一種 **Continuous Normalizing Flow** 訓練目標，比傳統 DDPM 收斂更快：

```
DDPM 訓練:
  x_t = sqrt(1-β_t) * x_{t-1} + β_t * ε
  預測: ε_θ(x_t, t) → ε
  推理: 100+ denoising steps

Flow Matching 訓練:
  x_t = (1-t) * x_0 + t * ε    (線性插值)
  預測: v_θ(x_t, t) = x_0 - ε  (velocity)
  推理: 4-10 Euler steps (線性ODE積分)

  關鍵技巧:
  - repeated_diffusion_steps (同一action重複k次) → sample efficiency ↑↑
  - Beta timestep sampling → 訓練更穩定
  - Euler/RK4 ODE solver → 推理速度保證
```

### 2.3 多模態融合架構對比

```
┌────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION                           │
├────────────────┬─────────────────┬────────────────────────────┤
│   Late Fusion  │  Cross-Attention│  Unified Transformer      │
│  (ACT, SmolVLA)│  (GR00T, UnifoLM)│  (OpenVLA, Pi0)          │
├────────────────┼─────────────────┼────────────────────────────┤
│ ViT → LLM     │ ViT + LLM →     │  ViT + LLM → Action       │
│ text concat   │ Cross-attn DiT   │  unified forward pass     │
│ → Action      │ → Action chunk   │  (single backbone)        │
├────────────────┼─────────────────┼────────────────────────────┤
│ 簡單, 成熟     │ 最靈活, 支持     │  最新架構, 需要           │
│ 但信息壓縮在   │ heterogeneous    │  大量訓練數據             │
│ 最後一個token  │ modalities      │                            │
└────────────────┴─────────────────┴────────────────────────────┘

GR00T/UnifoLM 的 Cross-Attention 架構:
  vision_features ──────────────────→ Cross-Attention ─→ action
  text_features   ──→ LLM ─→ text_emb ─→ Cross-Attention ↗
  proprio_features ──────────────────→ Concatenate ↗
  
  優點: 各 modality 独立编码，互不干扰
  缺点: DiT 层数多，计算量大
```

---

## 3. 對你項目的具體建議

### 3.1 LeKiwi (你的核心項目)

```
當前 pipeline: MuJoCo sim → LeRobot (ACT/Diffusion/Smolvla)
升級建議:

  階段1 (立即可做):
  ├─ 新增 Multi-Task DiT (Flow Matching) 作為替換
  │   - 推理: 4 steps (vs ACT 1-step, 但更平滑)
  │   - 訓練: 穩定，收斂快
  │   - 你的 LeKiwi sim 已經整合好了
  │
  階段2 (下個月):
  ├─ 新增 GR00T-N1.5 (nvidia/GR00T-N1.5-3B)
  │   - Flow Matching + cross-attention DiT
  │   - NVIDIA 已在 HuggingFace 開源
  │   - 3B 參數，你的 Mac GPU 可跑 (需要約 8GB VRAM)
  │
  階段3 (研究):
  └─ 對比實驗: ACT vs Flow Matching (Multi-Task DiT) vs GR00T
      - 同一 LeKiwi dataset
      - 測: 任務完成率, 推理延遲, 訓練時間
```

### 3.2 SO-101 (機械臂項目)

```
建議技術棧:
  - GR00T-N1.5: 因為已經支援人形機械臂 G1 (23 DOF)
  - Multi-Task DiT: 你自己的數據，微調
  - 不要用 OpenVLA (7B太大)

優先實驗:
  1. 用 Unitree 開源的 G1 數據集測 GR00T (無需你的硬體)
  2. 你的 SO-101 數據錄製 → 轉換為 LeRobot 格式
  3. 對比: GR00T (zero-shot) vs SO-101 fine-tuned
```

### 3.3 Go2 (四足步行)

```
重要提醒: VLA 不適合 locomotion！
  
  Go2 現有方案:
  ├─ 你的 gait_controller.py (trot/walk/stand) ✅ 正確
  ├─ Unitree RL gym (Isaac Gym) → 訓練 RL gait
  └─ 別用 VLA 替換 gait controller ❌

  VLA 的用途在 Go2:
  - "走到物體旁邊" (vision-guided navigation)
  - "避開障礙物" (vision-based obstacle avoidance)
  - 不是用 VLA 生成關節角度，而是高層指令
```

---

## 4. 數據集與預訓練模型

### 4.1 免費可用的數據集

```
Unitree 開源 (HuggingFace):
├─ G1_Stack_Block         (抓取放置)
├─ G1_Bag_Insert          (插入)
├─ G1_Erase_Board         (擦拭)
├─ G1_Clean_Table         (清理)
├─ G1_Pack_PencilBox      (包裝)
├─ G1_Pour_Medicine       (傾倒)
├─ G1_Pack_PingPong       (包裝)
├─ G1_Prepare_Fruit       (水果準備)
├─ G1_Organize_Tools      (工具整理)
├─ G1_Fold_Towel          (摺毛巾)
├─ G1_Wipe_Table          (擦桌子)
└─ G1_DualRobot_Clean_Table (雙機清潔)

OXE (Open X-Embodiment):
├─ 100+ 機器人數據集
├─ Bridge Dataset
├─ RT-1/RT-2 data
└─ 可用於預訓練 OpenVLA/SmolVLA

你的 LeKiwi 數據:
└─ LeRobot 格式 → 直接用 lerobot CLI 錄製
```

### 4.2 預訓練模型可獲得性

```
GR00T-N1.5-3B (NVIDIA):
  ✅ HuggingFace: nvidia/GR00T-N1.5-3B
  ✅ LeRobot 內建支援 (GrootPolicy)
  ⚡ 3B 參數, 8GB VRAM, 4-step inference

UnifoLM-VLA:
  ✅ HuggingFace: unitreerobotics/Unifolm-VLA-Base
  ✅ 訓練代碼全開源
  ⚠️ 需要自己訓練最終模型

SmolVLA:
  ✅ LeRobot 內建
  ⚡ ~1B 參數, 你的 Mac 可跑 (需要 MPS/CUDA)

Pi0:
  ❌ 完全閉源
  ⚠️ 需要申請 或等待 OpenPI 開源
```

---

## 5. 實施路線圖

### 立即 (這週)
```
1. 測試 Multi-Task DiT (Flow Matching) 在 LeKiwi sim 上
   - 你已有的 lerobot_policy_inference.py
   - 新增 --policy multi_task_dit

2. 下載 GR00T-N1.5-3B 測試
   - python3 scripts/eval_policy.py --policy groot --groot-model nvidia/GR00T-N1.5-3B

3. 錄製第一個 LeKiwi dataset
   - python3 scripts/record_lekiwi.py --task "walk to marker"
```

### 短期 (這個月)
```
1. 完成 ACT vs Flow Matching vs GR00T 對比實驗
   - 同一數據集
   - 記錄: 任務完成率, 訓練時間, 推理延遲, 記憶體

2. 將 Unitree G1 數據轉換為 LeRobot 格式
   - 測試 SO-101 的 domain adaptation

3. 整合 SO-101 錄製腳本
   - 基於 examples/so100_to_so100_EE/record.py
```

### 中期 (這季)
```
1. 在真實硬體上驗證最佳 policy
2. 研究對抗魯棒性 (結合 adversarial_toolkit)
3. 發布你的 benchmark 結果
```

---

## 6. 總結

```
最佳性價比選擇:
  LeRobot 項目 → Multi-Task DiT (Flow Matching) ✅ 內建, 4-step
  有 NVIDIA GPU  → GR00T-N1.5-3B                  ✅ HuggingFace
  需要快速部署   → SmolVLA / ACT                   ✅ 1-step
  有 Pi0 access  → Pi0 (目前最強，但閉源)

千萬不要:
  ❌ 在 Go2 locomotion 上用 VLA
  ❌ 用 OpenVLA 7B 在邊緣設備
  ❌ 用 Diffusion (50-100 steps) 在實時場景
```

---

## 附錄: 相關資源

```
代碼位置:
  LeRobot: ~/lerobot/src/lerobot/
  LeKiwi VLA: ~/lekiwi_vla/
  UnifoLM-VLA: ~/unifolm-vla/
  Go2 VLA: ~/go2-vla/

關鍵文件:
  Multi-Task DiT 配置: lerobot/src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py
  GR00T 配置: lerobot/src/lerobot/policies/groot/configuration_groot.py
  Flow Matching Head: lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py
  UnifoLM Config: unifolm-vla/src/unifolm_vla/config/training/unifolm_vla_train.yaml
```