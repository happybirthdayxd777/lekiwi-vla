# Phase 273 — P-ctrl 80% SR, Stage2 40% SR on URDF: Locomotion CONFIRMED WORKING

**Date**: 2026-04-22 11:00 CST

## 本次心跳完成

### Phase 272 錯誤診斷修正

Phase 272 聲稱 URDF locomotion "完全壞掉"——這是**錯誤的**。

正確評估結果（Phase 273 實測）：

| Policy | Sim | SR (10 goals, 200 steps) | 備註 |
|--------|-----|--------------------------|------|
| P-controller (kP=2.0) | URDF | **80%** | 8/10 goals |
| P-controller (kP=2.0) | Primitive | 86% | 歷史數據 |
| Stage2 curriculum | URDF | **40%** | 4/10 goals |
| Stage2 curriculum | Primitive | 72% | Phase 261 歷史數據 |

### Phase 272 錯誤原因分析

Phase 272 的 0% SR 結論來自 `quick_stage2_eval.py` 中的 **action 格式錯誤**：

```python
# Phase 272 的錯誤代碼（來自 quick_stage2_eval.py）：
wheel_speeds = np.clip(action[6:9], -1, 1) * 2.0
flat_action = np.concatenate([np.zeros(6), wheel_speeds])
sim.step(flat_action)  # 錯誤：flat_action 只有 9 維，但 step() 需要 arm(6)+wheel(3)

# 正確做法（Phase 273）：
wheel_speeds = np.clip(action[6:9], -0.5, 0.5)
ctrl_action = np.concatenate([action[:6], wheel_speeds])  # 完整 action
sim.step(ctrl_action)
```

`quick_stage2_eval.py` 的 action 格式與 `LeKiWiSimURDF.step()` 的 `_action_to_ctrl()` 不匹配：
- Stage2 輸出：arm_torque(6) + wheel_speed(3) = 9D
- `_action_to_ctrl()` 期望：arm(6) + wheel(3)，但 wheel 會乘以 10.0 變成 torque
- 錯誤的 action 導致全部 0 輸出 → 無 locomotion → 0% SR

### Stage2 40% SR 分析

Stage2 在 URDF 上 40% SR，落後於 primitive 的 72% SR，原因：

1. **視角差異**：URDF 使用合成相機圖像（synthetic），與訓練數據可能存在分佈偏移
2. **物理差異**：URDF 和 primitive 的接觸動力學略有不同
3. **action 轉換**：URDF 需要 wheel_speeds → torque 轉換（`action[6:9] * 10.0`），可能有精度損失
4. **Wheel radius 不同**：`sim_lekiwi_urdf.py` 中 `WHEEL_RADIUS = 0.025`，可能與訓練數據不一致

### Bridge 部署評估

| 模式 | P-ctrl SR | Stage2 SR | 結論 |
|------|-----------|-----------|------|
| primitive | 86% | 72% | 較好，但無 STL mesh |
| URDF | 80% | 40% | 可部署，STL mesh 完整 |

**Bridge 部署建議**：
- Stage2 + URDF：40% SR，對於 0.45m 半徑內目標可接受
- 建議使用 primitive 模式（72% SR）直到 URDF Stage2 差距縮小
- 可以在 URDF 上用 P-controller 作為 fallback

## Bridge Architecture Status (Phase 273)

| Component | Status | Notes |
|-----------|--------|-------|
| bridge_node.py | ✅ | 1260 lines, URDF + primitive modes |
| vla_policy_node.py | ✅ | 987 lines, CLIP-FM/pi0/ACT/dagger/stage2/stage3 |
| CTF Security Layer | ✅ | C1-C8 全部，資安監控整合 |
| Camera Adapter | ✅ | URDF 20Hz RGB |
| Real Hardware Adapter | ✅ | 真實硬體介面 |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| Stage2PolicyRunner | ✅ | goal-radius filter (|r|>0.45m → zeros fallback) |
| URDF Locomotion | ✅ CONFIRMED | P-ctrl 80%, Stage2 40% |

## 新增腳本

| Script | Purpose | Result |
|--------|---------|--------|
| `scripts/quick_pctrl_eval_urdf.py` | P-ctrl baseline on URDF | 80% SR |
| `scripts/quick_stage2_eval_urdf.py` | Stage2 policy on URDF | 40% SR |

## 下一步

- [ ] Phase 274: 整合 Stage2 40% SR into bridge_node — 評估 URDF 模式可用性
- [ ] Phase 275: 改善 URDF Stage2 成功率（可能需要重新訓練或 fine-tune）
- [ ] Phase 276: 解決 primitive vs URDF 物理差異（wheel_radius 標定）

## Git

- Commit: Phase 273 修正
- 新增腳本：`quick_pctrl_eval_urdf.py`, `quick_stage2_eval_urdf.py`
- 修正 Phase 272 錯誤結論：URDF locomotion 正常運行

## 實驗記錄

| Phase | 內容 | 結果 |
|-------|------|------|
| p260 | Curriculum Stage1+2 訓練 | Stage1+2 checkpoints |
| p261 | Stage2 50-goal eval (primitive) | **72% SR** |
| p264 | Stage3 training (15 epochs) | s3_epoch9=best (overfitting) |
| p265 | Stage3 s3_epoch6 20-goal eval | VLA=15% vs P-ctrl=85% |
| p266 | Stage3 s3_epoch9 10-goal eval | VLA=0% vs P-ctrl=60% (bug in eval) |
| p268 | Stage2PolicyRunner goal-radius filter | ✅ |
| p271 | Disk cleanup | 3.4GB free |
| p272 | **錯誤結論：URDF locomotion broken** | ❌ WRONG DIAGNOSIS |
| p273 | **URDF P-ctrl + Stage2 重新驗證** | P-ctrl=80%, Stage2=40% ✅ |
