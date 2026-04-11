# LeKiwi VLA Research — Quick Start Guide

## 已完成的工作

### 1. 數據收集 + 訓練 + 評估 完整流程

```bash
# 收集數據 (5 episodes × 100 steps, 224x224 image)
cd ~/hermes_research/lekiwi_vla
python3 scripts/collect_data.py --episodes 5 --steps 100 --output /tmp/lekiwi_demo_224.h5

# 訓練 Flow Matching (MPS, 5 epochs)
python3 scripts/train_flow_matching_real.py \
  --data /tmp/lekiwi_demo_224.h5 \
  --epochs 5 \
  --device mps \
  --batch-size 16 \
  --output /tmp/fm_real

# 評估 (trained vs random)
python3 scripts/eval_policy.py --policy random --episodes 5
python3 scripts/eval_policy.py --policy flow_matching --checkpoint /tmp/fm_real/final_policy.pt --episodes 5

# 推送到 GitHub
cd ~/hermes_research/lekiwi_vla && git add -A && git commit -m "message" && git push
```

### 2. Baseline 結果 (5 episodes each)

| Policy | Mean Reward | Notes |
|--------|-------------|-------|
| **Random** | -106.3 ± 1.3 | Uniform [-1,1] |
| **Flow Matching** | -106.6 ± 5.6 | 5 epochs, 500 frames |

Flow Matching with random policy are comparable because:
- Only 5 epochs of training on 500 synthetic random frames
- Real improvement requires 50+ epochs on real teleop data

### 3. 腳本說明

| Script | Purpose |
|--------|---------|
| `collect_data.py` | 從 MuJoCo sim 錄製數據 → HDF5 |
| `train_flow_matching.py` | 用 LeRobot Multi-Task DiT 訓練 |
| `train_flow_matching_lekiwi.py` | 獨立 Flow Matching (無 LeRobot dependency) |
| `train_flow_matching_real.py` | 用真實 HDF5 數據訓練 |
| `eval_policy.py` | 評估任意 policy (random / flow_matching) |
| `infer_groot.py` | GR00T-N1.5 推論 (需 transformers dev) |

## 下一步

### 數據不足？用真實 Docking 數據
```bash
# 轉換 real robot data → HDF5
python3 scripts/convert_docking_data.py \
  --input ~/hermes_research/lekiwi_modular/src/scripts/log_single_L/ \
  --output data/docking_single_L.h5
```

### 訓練更長時間
```bash
python3 scripts/train_flow_matching_real.py \
  --data /tmp/lekiwi_demo_224.h5 \
  --epochs 100 \
  --device mps \
  --output results/fm_100ep
```

### 嘗試 GR00T-N1.5 (需更多 RAM)
```bash
# 升級 transformers 並下載模型
pip install git+https://github.com/huggingface/transformers
python3 scripts/infer_groot.py --task "move arm to target" --steps 30
```

## 架構圖

```
lekiwi_vla/
├── sim_lekiwi.py              # MuJoCo 模擬 (LeKiwiSim)
├── scripts/
│   ├── collect_data.py        # 數據收集 → HDF5
│   ├── train_flow_matching_real.py  # 訓練 (HDF5 → policy)
│   └── eval_policy.py         # 評估 (sim → metrics)
└── lerobot_policy_inference.py # LeRobot 工廠 (GR00T/ACT/Diffusion)
```

## 關鍵發現

1. **Flow Matching = 4-step inference** (vs Diffusion 的 50-100 steps)
2. **8M 參數**，MPS 可訓練，8GB VRAM 即可跑 GR00T-N1.5
3. **lekiwi_modular 的 URDF** 可直接用於 lekiwi_vla 的 MuJoCo 模型
4. **omni_controller.py bug**：三輪 joint_axes 完全相同 → 已修復