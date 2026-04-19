# LeKiWi ROS2-MuJoCo Platform Progress — Phase 191

## [Phase 191 - 2026-04-19 20:00 UTC] — VLA eval confirms phase189 data CORRUPTED

### ✅ 已完成

**Phase 191 Eval: Phase 190 VLA policy vs P-controller baseline**

- Created `scripts/eval_phase191_fast.py` — FAST eval (no render, 5 goals, 170s)
- Policy: `results/phase190_vision_train/epoch_14.pt` (trained on phase189 data)
- Architecture: GoalConditionedPolicy (11D state, CLIP ViT-B/32, 4-step flow matching)

**Results:**
```
VLA:    SR=0% (0/5), avg_dist=0.6415
P-ctrl: SR=20% (1/5), avg_dist=0.1315
```

**Root cause CONFIRMED:** phase189 data has ALL wheel speeds saturating to ±0.5 due to `*200` scaling:
- P-controller with kP=0.5 → vx=0.15 → vx_200=30 → w1=5.268 → clipped to 0.5
- VLA trained on these saturated wheel targets → outputs ±0.5 (normalized ±1) for ALL goals
- Actual required wheel speeds for small goals are 0.03-0.08 rad/s
- VLA's constant ±0.5 makes robot barely move → 0% SR

**Evidence from eval:**
- Goal (0.3, -0.3): VLA dist=0.5742, P-ctrl dist=0.2317 — VLA barely moved
- Goal (-0.3, -0.3): VLA dist=1.2467, P-ctrl dist=0.1147 — VLA moved in WRONG direction
- Only goal (0.4, 0.1): P-ctrl reached in 58 steps, VLA stopped at 0.17 (barely moved)

### 🔍 架構現況
```
Phase 189 corrupted data flow:
  P-controller → ALL wheel speeds saturate at ±0.5
  → VLA trains on ±0.5 wheel targets (normalized as ±1)
  → VLA infers: ANY goal → output ±1 wheel command
  → In eval: robot barely moves → 0% SR

Phase 191 findings:
  - VLA is WORSE than random baseline (0% vs 20% SR)
  - P-controller (with correct Phase 164 formula) still poor on URDF model
  - BUT: the P-controller itself is using the *200 formula on a model that may not have k_omni=15
```

### 🧭 下一步（下次心跳）

**PRIORITY 1: Re-collect CLEAN data (Phase 192)**
- Fix the `twist_to_contact_wheel_speeds` to REMOVE `*200`
- Use corrected formula: w = [-0.0124*vx + 0.188*vy, 0.199*vx + 0.199*vy, -0.199*vx + 0.187*vy]
- Collect 10k frames with proper wheel speed variation
- MUST have: Corr(w0,gy) > 0.9, Corr(w1,gx) > 0.6

**PRIORITY 2: Evaluate P-controller on clean sim**
- Before training VLA, verify P-controller works on LeKiWiSim (not URDF)
- Test: goal (0.3,0.3), goal (-0.3,-0.3), goal (0.3,-0.3)

**PRIORITY 3: Train Phase 192 VLA on clean data**

### 🚫 阻礙
- **phase189 data: ALL saturated** → VLA learns wrong behavior
- **P-controller on URDF model: 20% SR only** → the URDF model may not have k_omni=15 overlay
- **Eval script with real rendering: times out** → need faster per-episode limit

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p189 | Data: 10000 images, *200 bug | CORRUPTED: all wheel speeds saturate ±0.5 |
| p190 | Training: 15 epochs on phase189 | loss=0.40, VLA SR=0% on 5-goal eval |
| p190 | VLA output | Almost no motion toward goal (dist=0.64-1.25) |
| p191 | P-ctrl baseline | SR=20% on URDF sim (needs k_omni=15) |
| p192 | **NEXT** | Re-collect clean data WITHOUT *200 |

### Git
- New: `scripts/eval_phase191_fast.py` — fast VLA eval (no render, 170s, 5 goals)
- Modified: `scripts/eval_phase190.py` (incomplete — full render version)
- Pending: Phase 191 findings — VLA SR=0% vs P-ctrl SR=20% due to phase189 data corruption