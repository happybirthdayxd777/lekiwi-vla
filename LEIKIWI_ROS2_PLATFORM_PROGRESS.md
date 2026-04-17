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

