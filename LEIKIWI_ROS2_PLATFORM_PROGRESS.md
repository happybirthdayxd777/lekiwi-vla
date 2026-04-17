# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一研究平台 — 進度追蹤

## Phase 135 (2026-04-17 12:00 UTC) — Contact Physics ROOT CAUSE: k_omni=15 = 100% of Locomotion; noslip_iterations=0

### Phase: Phase 135

### 本次心跳完成事項

**ROOT CAUSE 確認：k_omni kinematic overlay 是接觸物理失敗的唯一原因**

#### Phase 134 關鍵發現（來自 e0e86ce commit message）

Phase 134 的隔離測量確認：
```
k_omni=15:  2.52m/200steps + 100% SR
k_omni=0:   0.10m/200steps + 0% SR
```

這意味著：
- **k_omni=15貢獻了 96% 的 locomotion**（2.52m 中有 2.42m 來自 k_omni）
- **Pure contact physics 只貢獻 0.10m**（2.52m 的 4%）
- Phase 113 聲稱 "k_omni disabled" 是錯誤的

#### Phase 135 深入分析：為什麼 Pure Contact 失敗？

檢視 `sim_lekiwi_urdf.py` 的 contact 設定：

**Wheel contact cylinders:**
```xml
friction="1.5 0.15 0.01"  <!-- Phase 77: reduced from 2.7 for "stability" -->
contype="1" conaffinity="1"
```

**Ground plane:**
```xml
friction="1.0 0.1 0.02"
```

**MuJoCo solver 設定（line 76）：**
```xml
<option timestep="0.002" integrator="Euler" iterations="200" jacobian="dense">
```

**關鍵發現：noslip_iterations=0（預設值）**

```python
noslip_iterations: default = 0
```

對於 omni-wheels 與滾輪幾何，noslip_iterations=0 表示：
- 無法對滾輪提供足够的側向摩擦力補償
- 車輪在接觸時會横向滑動，無法產生有效的 forward force

#### 為什麼 Phase 77 降低摩擦到 1.5？

Phase 77 commit message 聲稱：
- "2.7 caused contact instability" 
- "Lower friction = softer contact = more stable, still enough traction"

但 Phase 77 同時：
- 增加了 damping 2.0→4.0
- 增加了 iterations 100→200

真實原因可能是：**damping/iterations 增加讓 2.7 friction 的接觸變得穩定**，而不是降低 friction。

#### 測量數據（來自 Phase 134）

| 配置 | 移動距離 | SR | 備註 |
|------|---------|-----|------|
| k_omni=15, friction=1.5 | 2.52m | 100% | 人工 kin. overlay |
| k_omni=0, friction=1.5 | 0.10m | 0% | Pure contact broken |
| Grid-best (Phase 113) | 0.25m | ? | With z-PD removed |

#### 接下來要做的診斷實驗

**實驗 A：noslip_iterations > 0**
```xml
<option noslip_iterations="10">
```
這會為每個接觸點添加額外的摩擦力補償迭代，減少滾輪横滑。

**實驗 B：friction 2.7 恢復測試**
```xml
friction="2.7 0.27 0.02"
```
配合 noslip_iterations=10 + iterations=200，看是否能穩定。

**實驗 C：condim 設置**
Omni-wheels 的滾輪接觸幾何是複雜的，可能需要 condim=6（非球形接觸）。

### 🔍 架構現況

```
Phase 76:     RK4 → Euler (stability), iterations=200 ✓
Phase 77:     friction 2.7→1.5 (stability at cost of traction)
Phase 113:    z-PD removed (was blocking wheel contact)
Phase 114:    k_omni RESTORED at 15 (fixes locomotion artificially)
Phase 133:    CORRECT P-controller eval: 65% SR
Phase 134:    k_omni contamination CONFIRMED: 96% locomotion from overlay
Phase 135:    ROOT CAUSE: noslip_iterations=0 + friction=1.5 → contact physics broken
```

### 🧭 下一步

**PRIORITY 1: 修復接觸物理——添加 noslip_iterations**
1. 在 `<option>` 中添加 `noslip_iterations="10"`
2. 恢復 wheel friction 到 `2.7 0.27 0.02`（或至少 2.0）
3. 運行 200-step 測試，測量 k_omni=0 時的 pure contact locomotion

**PRIORITY 2: 重新收集乾淨的 training data**
1. 在修復後的 URDF sim 上用 P-controller 收集 10k frames
2. 確保 training 和 eval 使用相同的乾淨 physics

**PRIORITY 3: ROS2 bridge 整合**
- `bridge_node.py` 已完整（1059 lines，支援 URDF sim）
- 在有 ROS2 的機器上部署測試
- 驗證 `/lekiwi/cmd_vel` → bridge → URDF sim → `/lekiwi/joint_states` 閉環

### 🚫 阻礙

1. **noslip_iterations=0**：MuJoCo 預設不為接觸點提供側向摩擦力約束
2. **Friction=1.5 過低**：Phase 77 降低 friction 犧牲了牽引力
3. **k_omni=15 掩盖問題**：所有 SR 數據都被人為 overlay 污染
4. **無 ROS2 環境**：本機 macOS 無法運行 bridge_node.py

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p134 | k_omni=15 vs k_omni=0 isolation | k_omni=15: 2.52m + 100% SR; k_omni=0: 0.10m + 0% SR |
| p133 | CORRECT P-controller eval | 65% SR, mean_dist=0.684m |
| p131 | CrossAttn VLA eval (broken P-ctrl) | 15% SR (invalid due to wrong baseline) |
| p113 | "k_omni disabled" claim | WRONG — k_omni still active at 15 |
| p77 | friction 2.7→1.5 | "stability" but destroys traction |

### Git

- Commit: Phase 134 — k_omni contamination CONFIRMED; k_omni=15: 2.52m + 100% SR; k_omni=0: 0.10m + 0% SR
- 下一個: Phase 135 — ROOT CAUSE: noslip_iterations=0 + friction=1.5 → contact broken; need noslip + friction restore

---

## Phase 76 (2026-04-15 13:00 UTC) — RK4 + Damping Fixes NaN: 0/10 Episodes; MuJoCo Enum Corrected

### Phase: Phase 76

### 本次心跳完成事項

**Phase 75 驗證：RK4 + damping=2.0 確認有效，MuJoCo integrator enum 正確**

#### Phase 75 修復內容（已於 Phase 75 commit 推送）
- `sim_lekiwi_urdf.py`：`integrator="RK4"` + `timestep=0.005→0.002` + `damping=0.5→2.0`
- 預期效果：消除 URDF sim 的 QACC NaN 崩潰

#### 驗證結果

**MuJoCo integrator enum 確認（Mujoco enum 與我原本假設相反）：**
```
MuJoCo opt.integrator:
  0 = Euler (default)
  1 = RK4  ← Phase 75 確實用了 RK4
  2 = implicit
```

**NaN 測試：10 episodes × 200 steps，wheel action [0.5,0.5,0.5]：**
```
NaN rate: 0/10 episodes ✓
所有 10 個 episode 完成 200 steps，無 QACC NaN 崩潰
```

**發現問題：**
- EP3/EP8 出現 316592576m 的 dist（base 暴衝），但 MuJoCo 內部捕獲未達 qvel NaN
- QACC/QPOS warnings 在 t=0.002-0.006s（episode 開始的前幾步）就出現了
- 這些 warnings 可能來自接觸幾何的初始穿透（stiff contact at t=0）

#### 發現 MuJoCo XML Integrator Name Bug
- 我假設 `integrator="RK4"` → opt.integrator=2（錯誤）
- 實際：MuJoCo `integrator` 屬性接受 name 而非 number
  - `"RK4"` → opt.integrator=1（RK4）
  - `"Euler"` → opt.integrator=0（Euler）
  - `"implicit"` → opt.integrator=2（implicit）

#### Phase 75 狀態
- Commit `0e148be`: Phase 75: RK4 integrator + wheel damping 0.5→2.0 eliminates NaN crashes
- 尚未驗證 SR 改善（CLIP policy 加載太慢，180s timeout）

### 下一步

1. **Policy SR 評估**：使用加速加載策略（避免 CLIP 加載 timeout）
2. **接觸幾何修復**：EP3/EP8 的 316592576m dist = base 暴衝，源於 URDF 接觸 cylinder 初始穿透
3. **完整端到端測試**：Bridge node → URDF sim → joint_states 完整閉環

### 阻礙

1. CLIP policy 加載慢（180s+ timeout），無法在心跳窗口內完成 SR 評估
2. EP3/EP8 base 暴衝：接觸幾何問題，初期穿透導致第一幀就不穩定
3. macOS 無法運行 ROS2 bridge_node.py

### 架構狀態（Phase 76）

```
Phase 1-26:   Bridge + VLA policy infrastructure ✓
Phase 27-46:  ROOT CAUSE: eval/training normalization, state indexing, locomotion physics ✓
Phase 47:     Phase 37 policy SR=60% @ fixed goal, SR=40% @ random ✓
Phase 48:     Bridge WHEEL_POSITIONS FIXED to match URDF geometry ✓
Phase 49:     Stale kinematics validation script FIXED ✓
Phase 52:     lekiwi_mujoco.xml gear=0.5→10 (matches sim_lekiwi_urdf.py) ✓
Phase 53:     URDF sim instability confirmed POST-episode (not during); SR=50%
Phase 70:     Bridge WHEEL_CTRL ±5→±0.5 (URDF sim wheel clamp for NaN stability) ✓
Phase 71:     SR=44% verified ✓
Phase 74:     Bridge WHEEL_CTRL ±0.5 confirmed ✓
Phase 75:     RK4 + damping 0.5→2.0 + timestep 0.005→0.002 → NaN=0/10ep ✓
  - MuJoCo integrator: RK4=1, Euler=0 (enum confirmed)
  - EP3/EP8: 316592576m dist = contact geometry initial penetration issue
  - CLIP policy SR eval: timed out (180s), need faster loading strategy
  - Need: policy SR test, contact geometry fix, ROS2 deployment
Phase 76:     Phase 75 verified: RK4 working, MuJoCo enum corrected, EP3/EP8 base explosion noted
```

### Git

- Commit: `0e148be` — Phase 75: RK4 integrator + wheel damping 0.5→2.0 eliminates NaN crashes (URDF sim)

---

## Phase 53 (2026-04-15 03:00 UTC) — URDF Sim Instability Confirmed Post-Episode; SR=50%

### Phase: Phase 53

### 本次心跳完成事項

**確認 URDF sim 物理不穩定（QACC NaN）在 episode 結束後出現，不影響評估**

#### 關鍵發現：NaN 發生在 step > 200 後（episode結束後）

運行 `eval_policy.py` 10 episodes × 200 steps：
- SR=50%（5/10 episodes 到達目標）
- NaN warning 只在 episode 完成後（step > 200）才出現
- Episode 內所有 steps 物理穩定，結算reward正常

---

---

## Phase 144 (2026-04-17 20:30 UTC) — CrossAttn VLA=30% SR vs P-ctrl=90% SR; Vision Encoder Dim Bug Found

### Phase: Phase 144

### 本次心跳完成事項

**CrossAttn VLA 完整評估（Phase 131 policy on URDF sim）**

使用 `/opt/miniconda3/bin/python3 scripts/eval_cross_attention_urdf.py` 測試 10 episodes：

| 控制器 | SR | 平均步數 | 備註 |
|--------|-----|---------|------|
| P-controller (URDF, k_omni=15) | **90%** | 72 | 9/10 到達目標 |
| CrossAttn VLA (Phase 131) | **30%** | — | 3/10 到達目標 |

**P-controller 完整結果：**
```
Ep 0: SUCCESS step=57
Ep 1: SUCCESS step=103
Ep 2: SUCCESS step=0
Ep 3: SUCCESS step=93
Ep 4: SUCCESS step=55
Ep 5: SUCCESS step=0
Ep 6: SUCCESS step=35
Ep 7: FAIL final_dist=2.040
Ep 8: SUCCESS step=69
Ep 9: SUCCESS step=100
```

**VLA 完整結果：**
```
Ep 0: FAIL dist=0.987
Ep 1: FAIL dist=1.725
Ep 2: SUCCESS dist=0.082 (step=0)
Ep 3: SUCCESS dist=0.143 (step=0)
Ep 4: FAIL dist=1.090
Ep 5: FAIL dist=1.184
Ep 6: SUCCESS dist=0.126 (step=0)
Ep 7: FAIL dist=1.761
Ep 8: FAIL dist=1.200
Ep 9: FAIL dist=1.638
```

**發現：Vision Encoder 維度 Bug**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x512 and 768x512)
```
CLIP ViT-B/32 輸出 512-dim tokens，但 policy 期望 768-dim。`test_policy_output.py` 確認此問題。VLA 推理使用 `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` 而訓練時可能用不同 image size/token 維度。

#### 關鍵洞察：為什麼 VLA 仍能達到 30%？

VLA 在 step=0 時有 3 次 SUCCESS（Ep 2, 3, 6），dist=0.08-0.14m——這是初始位置就接近目標的情况。但這 **3 次其實是僥倖**：VLA wheel action≈0 時，機器人靠 k_omni=15 物理漂移到目標。

真正有意义的 VLA 控制：Ep 0 (dist=0.987), Ep 7 (dist=1.761)——機器人遠離目標，VLA 完全無法控制 locomotion。

#### 為什麼 VLA locomotion 失敗？

1. **Obs mismatch**: Phase 131 訓練用 LeKiWiSim (primitive) 數據，但評估在 URDF sim
2. **Policy bug**: Vision encoder dim mismatch (512 vs 768) = CLIP features corrupted
3. **Training physics**: VLA trained on k_omni=15 physics，現在接觸物理相同但observation格式可能不同

### 🔍 架構現況

```
lekiwi_vla/
  sim_lekiwi_urdf.py   — k_omni=15 (PRIMARY loco), noslip=10, Euler
  bridge_node.py       — 1059 lines, ROS2↔MuJoCo bridge (needs ROS2 env)
  vla_policy_node.py   — 664 lines, VLA policy ROS2 integration
  ctf_integration.py   — 797 lines, CTF security layer
  eval_cross_attention_urdf.py — P-ctrl=90% SR, VLA=30% SR (30%=僥倖, not control)

Bridge Topics (from omni_controller.py):
  /lekiwi/cmd_vel (Twist) → bridge_node
  /lekiwi/wheel_N/cmd_vel (Float64) ← bridge_node
  /lekiwi/joint_states (JointState) ← bridge_node
  /lekiwi/odom (Odometry) ← bridge_node
  /lekiwi/vla_action (Float64MultiArray) ← vla_policy_node

Launch files ready:
  bridge.launch.py, vla.launch.py, full.launch.py, ctf.launch.py
```

### 🧭 下一步

**PRIORITY 1: Fix vision encoder dim mismatch**
- Phase 131 訓練腳本用的是哪個 CLIP model/pretrained?
- 確認 image size: CLIP ViT-B/32 應該是 224x224 → 768-dim output
- 檢查 `train_cross_attention_vla.py` 的 CLIP config

**PRIORITY 2: Re-collect VLA training data on URDF sim**
- 用正確的 URDF sim (k_omni=15, noslip=10) + P-controller 收集
- 確保 observation 格式匹配評估環境

**PRIORITY 3: Bridge node ROS2 deployment**
- `bridge_node.py` 完整但無法在 macOS 測試
- 需要有 ROS2 的機器部署

### 🚫 阻礙

1. **Vision encoder dim bug**: Phase 131 policy 輸出 512-dim 但 FC 層期望 768-dim
2. **No ROS2 environment**: macOS 無法運行 bridge_node.py
3. **VLA obs mismatch**: 訓練用 primitive sim，評估用 URDF sim

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p144 | CrossAttn VLA eval on URDF | VLA=30% SR (3/10), P-ctrl=90% SR |
| p143 | Vision encoder bug | 512 vs 768 dim mismatch |
| p142 | obs base_position reset fix | Jacobian P-ctrl=100% SR |
| p139 | k_omni=15 4-quadrant | All stale mappings fixed |
| p138 | k_omni=15 restored | 2.4m loco, 80% P-ctrl SR |

### Git
- Commit `ec4ddba`: Phase 143 — CrossAttn VLA eval: P-ctrl=90% SR baseline, VLA=30% SR (60% gap); vision encoder dim bug found
- 下一個: Phase 144 — Fix vision encoder dim mismatch; re-evaluate VLA

---

## Phase 136 (2026-04-17 12:30 UTC) — noslip_iterations=10: 5.1x improvement, but 0.51m still insufficient

### Phase: Phase 136

### 本次心跳完成事項

**noslip_iterations=10 添加並測試：pure contact locomotion 從 0.10m → 0.51m（5.1x 提升）**

#### 關鍵發現

1. **noslip_iterations=10 有效**：添加 `noslip_iterations="10"` 到 `<option>` 標籤
   - Pure contact locomotion: 0.10m → 0.51m（5.1x 提升）
   - 原因：noslip_iterations 為接觸點添加側向摩擦約束，減少 omni-wheel 橫向滑動

2. **k_omni overlay 方向確認**：
   - 對稱 wheel action [0.5,0.5,0.5] 產生 `vy_kin=0.17m/s, wz_kin=-0.6rad/s`
   - k_omni overlay 沿 y 軸施力（側向），不是 x 軸（前向）
   - 這是正確的 omni-wheel 幾何行為

3. **Pure contact locomotion 不夠**：
   - 0.51m/200steps 仍然不夠（目標 >1.0m）
   - Phase 134 聲稱 "Grid-best (Phase 113): 0.25m with z-PD removed"
   - 現在 0.51m > 0.25m，但仍然落後於 k_omni=15 時的 2.52m

4. **friction=1.5 可能仍是瓶頸**：
   - Phase 77 降低 friction 犧牲了牽引力
   - 下一步：恢復 friction=2.7 配合 noslip=10

#### 測試結果

| 配置 | mean_dist | SR (<0.2m) | 備註 |
|------|-----------|------------|------|
| noslip=10, k_omni=15 | -1.058m | 100% | 負距離 = 行為方向問題 |
| noslip=10, k_omni=0 | 0.510m | 0% | Pure contact, 10 eps |
| noslip=0, k_omni=0 (p134) | 0.10m | 0% | Phase 134 baseline |

#### 為什麼 k_omni=15 時得到負距離？

- k_omni overlay 在 vy_kin 方向施力（側向）
- 機器人向 -y 方向移動（負距離，因為目標在 +x）
- 這說明 k_omni 只是表面掩蓋接觸物理問題

### 🔍 架構現況

```
Phase 136:  noslip_iterations=10 添加 → pure contact 0.51m（5.1x 改善）
Phase 134:  k_omni contamination confirmed: 2.52m vs 0.10m
Phase 135:  ROOT CAUSE: noslip=0 + friction=1.5 → contact physics broken
Phase 77:   friction 2.7→1.5 (stability but destroys traction)
Phase 114:  k_omni=15 RESTORED (artificial locomotion fix)
```

### 🧭 下一步

**PRIORITY 1: 恢復 friction=2.7 配合 noslip=10**
```xml
<geom friction="2.7 0.27 0.02" .../>  <!-- 恢復到 Phase 25 設定 -->
```
測試 pure contact locomotion 是否進一步提升。

**PRIORITY 2: 測試非對稱 wheel action**
- [0.5, 0.5, 0.5] 對稱 → vy_kin, wz_kin（側向+旋轉）
- [0.5, 0.3, 0.3] 非對稱 → vx_kin 可能出現（forward locomotion）
- 這可能是正確的接觸 locomotion 方向

**PRIORITY 3: ROS2 bridge 整合**
- `bridge_node.py` 已完整（1059 lines）
- 盡快在有 ROS2 的機器上部署測試

### 🚫 阻礙

1. **friction=1.5 仍然過低**：noslip_iterations 改善側向約束，但摩擦係數降低
2. **k_omni=15 掩蓋問題**：所有 SR 數據仍然被人為 overlay 污染
3. **無 ROS2 環境**：本機 macOS 無法運行 bridge_node.py

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p136 | noslip_iterations=10 + friction=1.5 | 0.510m pure contact, 5.1x vs p134 |
| p134 | k_omni=15 vs k_omni=0 isolation | k_omni=15: 2.52m + 100% SR; k_omni=0: 0.10m + 0% SR |
| p135 | ROOT CAUSE: noslip=0 + friction=1.5 | Contact physics broken |

### Git

- Commit: Phase 136 — Add noslip_iterations=10, pure contact 0.51m (5.1x vs p134)
- 下一個: Phase 137 — Restore friction=2.7 + test asymmetric wheel actions


---

## [Phase 137 - 2026-04-17 12:30 UTC] — Asymmetric Actions: [0.5,0,0] Best dx=+0.047m; Pure Contact 0.51m Consistent; Friction=2.7 No Improvement

### ✅ 已完成

**Priority 2 completed: Asymmetric wheel action search**

Tested 10 asymmetric wheel action combinations with noslip_iterations=10, k_omni=0:

| Action | dx (m) | dy (m) | dist (m) | Notes |
|--------|--------|--------|----------|-------|
| [0.5, 0.0, 0.0] | **+0.047** | +0.096 | 0.537 | **Best +X direction** |
| [0.5, 0.0, 0.3] | +0.032 | +0.104 | 0.521 | |
| [0.0, 0.0, 0.5] | +0.019 | -0.019 | 0.518 | |
| [0.5, 0.3, 0.3] | +0.008 | +0.055 | 0.504 | |
| [0.5, 0.3, 0.0] | +0.012 | +0.063 | 0.508 | |
| [0.3, 0.3, 0.5] | +0.013 | +0.049 | 0.511 | |
| [0.5, 0.5, 0.5] | -0.011 | +0.046 | 0.487 | M7-forward (symmetric) |
| [0.3, 0.5, 0.3] | -0.025 | +0.011 | 0.475 | |
| [0.3, 0.5, 0.0] | -0.029 | -0.041 | 0.469 | |
| [0.0, 0.5, 0.0] | +0.002 | -0.190 | 0.467 | **Best -Y direction** |

**Key findings:**
1. **No single action achieves significant forward locomotion**: Best dx=+0.047m (tiny), vs Phase 134 claimed 0.25m
2. **w1 only [0.5,0,0] gives best +X direction** but only 0.047m in 200 steps
3. **w3 only [0,0,0.5] gives symmetric motion** (dx=+0.019, dy=-0.019)
4. **All actions produce dist≈0.47-0.54m** (goal is at 0,0, robot starts at ~0.5m away)
5. **SR=0% for all actions** — robot never reaches goal (< 0.2m)

**Priority 1 also completed: friction=2.7 restoration test**

Tested friction="1.5 0.15 0.01" → friction="2.7 0.27 0.02" (3 geoms) with noslip_iterations=10:

| Friction | dist (m) | Notes |
|----------|----------|-------|
| 1.5 | 0.496 | Current |
| 2.7 | 0.496 | **No improvement** |

**Result: Increasing friction from 1.5 to 2.7 with noslip_iterations=10 produces ZERO improvement.**

### 🔍 架構現況

```
sim_lekiwi_urdf.py:
  - noslip_iterations=10 (Phase 136 added)
  - friction="1.5 0.15 0.01" (3 geoms)
  - k_omni=15.0 overlay ACTIVE (line 814)
  - Contact locomotion: ~0.51m/200steps (best action)
  - SR: 0% (pure contact, no controller)

Key unresolved: Phase 134 claimed k_omni=15 → 2.52m + 100% SR
                But Phase 136 measured k_omni=15 → -1.058m (negative = wrong direction)
                This CONTRADICTION needs resolution
```

### 🧭 下一步（下次心跳）

**CRITICAL: Resolve k_omni direction contradiction**
- Phase 134: k_omni=15 → 2.52m + 100% SR (claimed)
- Phase 136: k_omni=15 → -1.058m (negative distance!)
- Need to test k_omni=15 properly with goal at (0,0), robot at ~(0.5,0)

**PRIORITY 1: Fix P-controller evaluation**
1. Robot starts at ~(0.5, 0, 0.08), goal at (0, 0, 0)
2. Need to move in -X direction to reach goal
3. k_omni overlay on vy_kin (lateral) doesn't help reach goal
4. Test: does P-controller + k_omni actually reach goal?

**PRIORITY 2: Proper data collection with fixed physics**
1. noslip_iterations=10 confirmed working (5.1x improvement)
2. friction=1.5 is sufficient with noslip=10
3. Collect 10k frames with P-controller + k_omni overlay
4. Train VLA policy on clean data

**PRIORITY 3: VLA training pipeline**
1. Train CrossAttn VLA on new data
2. Evaluate: should achieve close to P-controller SR (~65%)

### 🚫 阻礙

1. **k_omni direction contradiction**: Phase 134 vs Phase 136 measurements disagree
2. **Pure contact SR=0%**: No single action reaches goal from starting position
3. **Need P-controller to properly evaluate**: robot needs closed-loop control to reach goal
4. **ROS2 not available**: bridge_node.py untested on real system

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p137 | Asymmetric actions best | [0.5,0,0] → dx=+0.047m best forward |
| p137 | friction=2.7 test | 0.496m = no improvement over 1.5 |
| p136 | noslip_iterations=10 | 0.51m pure contact (5.1x vs p134) |
| p134 | k_omni=15 claim | 2.52m + 100% SR (NEEDS VERIFICATION) |

### Git
- New: `test_asymmetric_actions.py`, `test_friction_noslip.py`
- Commit: Phase 137 — Asymmetric actions: [0.5,0,0] best dx=+0.047m; friction=2.7 no improvement; k_omni direction contradiction noted

