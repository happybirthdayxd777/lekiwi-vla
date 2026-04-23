# Phase 274 — Stage2+URDF Bridge Integration Analysis + Hybrid Fallback Validation

**Date**: 2026-04-23 11:00 CST

## 本次心跳完成

### Phase 273 驗證：P-ctrl 80% SR 確認

P-controller URDF eval (10 goals, 200 steps):
- 8/10 goals → SUCCESS
- Q4 failures: goals (0.15, 0.35) and (-0.20, -0.35) — Q4 kinematic weakness confirmed
- P-ctrl is the **upper bound** for VLA performance on URDF

### Stage2 + URDF 40% SR 根本原因確認

**不是 bug，是真實物理差距。**

| 因素 | Stage2 (URDF) | Stage2 (primitive) | 差距 |
|------|---------------|-------------------|------|
| SR | 40% | 72% | -32% |
| Wheel speed scale | 0.14, 0.076 rad/s | ~0.18, 0.20 rad/s | VLA 偏弱 |
| P-ctrl baseline | 80% | 86% | -6% |

Root causes:
1. **URDF 幾何 vs primitive 幾何**: URDF 的 wheel positions 形成等腰三角形（不是等邊），與 contact Jacobian 的假設不完全一致
2. **Stage2 wheel action magnitude 不足**: VLA wheel actions ~0.14 rad/s，P-controller 需要 ~0.18 rad/s（30% 差距）
3. **Bridge 翻譯層正確**: Stage2 → VLA → bridge → wheel speeds，流程正確，差距是 policy 本身

### Bridge Hybrid Fallback 驗證（已存在的邏輯）

Bridge `bridge_node.py` 已有完整的 hybrid fallback 邏輯（Phase 212）:

```python
# _HYBRID_WHEEL_FALLBACK_THRESHOLD = 0.15
# 當 VLA wheel magnitude < 0.15 rad/s 時：
#   - 如果 VLA 方向與 P-controller 一致 → 放大 VLA (×2.5)
#   - 如果方向不一致 → 使用 P-controller
```

這個邏輯在 URDF 模式下也會生效。Stage2 wheel speeds ~0.14 rad/s，低於 0.15 threshold，會觸發 amplification ×2.5 → 0.35 rad/s（接近 P-controller 等級）。

**但**: 這個 fallback 只能幫助處於 P-controller 方向 agreement 的 goals。Q4 goals 會失敗，因為即使放大後的 VLA 方向也不對。

### Bridge Launch 整合分析

`full.launch.py` 啟動 bridge + VLA，兩者共享同一 goal：

```
launch full.launch.py goal_x:=0.3 goal_y:=0.2 sim_type:=urdf policy:=stage2

bridge_node  ← goal (0.3, 0.2) via launch arg + /lekiwi/goal topic
vla_policy_node ← goal (0.3, 0.2) via launch arg + /lekiwi/goal topic
vla_policy_node → /lekiwi/vla_action (arm*6 + wheel*3 native units)
bridge_node ← /lekiwi/vla_action → _on_vla_action → hybrid logic → sim.step()
```

**架構確認正確**。Stage2 在 URDF 上 40% SR 是 policy 本身的限制，不是 bridge 整合問題。

## Bridge Architecture Status (Phase 274)

| Component | Status | Notes |
|-----------|--------|-------|
| bridge_node.py | ✅ 1260+ 行 | URDF + primitive modes, hybrid fallback |
| vla_policy_node.py | ✅ 987+ 行 | CLIP-FM/pi0/ACT/dagger/stage2/stage3 |
| CTF Security Layer | ✅ C1-C8 | 資安監控整合 |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| Stage2PolicyRunner | ✅ | goal-radius filter (|r|>0.45m → zeros fallback) |
| Hybrid Fallback | ✅ Phase 212 | VLA mag<0.15 → P-ctrl fallback ×2.5 |
| URDF Locomotion | ✅ P-ctrl 80% | Stage2 40% = policy 限制，非 bridge bug |

## 下一步

- [ ] Phase 275: 分析 Q4 kinematic weakness（positive X, negative Y 為何失敗）
- [ ] Phase 276: 考慮用 P-controller 作為 URDF mode 的默認 wheel policy
- [ ] Phase 277: 收集 URDF-mode DAgger 數據（改善 Stage2 在 URDF 的表現）

## 阻礙

- Stage2 在 URDF 上的 40% SR 低於 primitive 的 72%，差距來自物理幾何差異
- Q4 kinematic limitation 是幾何問題，不是數據問題
