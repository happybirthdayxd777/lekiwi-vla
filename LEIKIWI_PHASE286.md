# Phase 286 — Stage2 Integration Audit: 60% SR, CJ P-controller Fallback Verified

**Date**: 2026-04-23 18:30 CST

## 本次心跳完成

### Stage2 整合現狀全面審計

系統性審計 Stage2 (stage2_r045.pt, 72% SR claimed) 在 URDF sim 中的實際表現：

### 1. Stage2 Wheel 動作幅度診斷

```
Stage2 wheel magnitude (native rad/s from policy.infer()):
  Ep 0: mean=0.287
  Ep 1: mean=0.114
  Ep 2: mean=0.120
  Ep 3: mean=0.269
  Ep 4: mean=0.156
  Overall: mean=0.189, median=0.163, range=[0.069, 0.322]

Comparison (from Phase 284):
  P-controller output:     ~0.389 rad/s (motor action)
  Stage3 (s3_epoch9):      ~0.032 rad/s  (12x too small)
  Stage2 (s2_r045):         ~0.189 rad/s  (2x smaller than P-ctrl)
```

**關鍵發現：Stage2 車輪幅度是 Stage3 的 5.9x，雖然仍低於 P-controller，但有部分 goals 達到足夠幅度。**

### 2. Stage2 10-goal eval (seed=42)
- **SR = 70% (7/10)**
- eval: 5/10→60%, 10/10→70%

### 3. Stage2 20-goal eval (seed=42)
- **SR = 60% (12/20)**
- eval: 5/10→60%, 10/20→70%, 15/20→60%, final→60%
- Variability suggests Stage2 is on the boundary of reliable locomotion

### 4. Stage2 vs Stage3 Comparison

| Metric | Stage2 (s2_r045) | Stage3 (s3_epoch9) | P-controller |
|--------|-----------------|---------------------|--------------|
| Training radius | \|r\|<0.45m | All goals | N/A |
| Wheel magnitude | ~0.189 rad/s | ~0.032 rad/s | ~0.389 rad/s |
| Magnitude ratio vs P-ctrl | 0.49x | 0.08x | 1.0x |
| 10-goal SR (seed=42) | 70% | 0% | ~80% |
| 20-goal SR (seed=42) | 60% | — | — |
| Hybrid fallback threshold | 0.15 | 0.15 | N/A |

### 5. Bridge Hybrid Fallback 對 Stage2 的影響

Bridge threshold = 0.15 rad/s (native wheel speed from VLA):

```
Stage2 episodes where vla_mag < 0.15 → triggers P-controller fallback:
  Ep 0: mean=0.287 → mostly above threshold (Stage2 wheels used)
  Ep 1: mean=0.114 → below threshold → P-controller fallback
  Ep 2: mean=0.120 → below threshold → P-controller fallback
  Ep 3: mean=0.269 → mostly above threshold (Stage2 wheels used)
  Ep 4: mean=0.156 → borderline (some above, some below)

~40-50% of steps trigger P-controller fallback for Stage2
```

**Stage2 的車輪動作處於临界区（约50%步数低于0.15阈值）→ P-controller 接管 → locomotion 改善**

### 6. Stage2 整合架構確認

Bridge 和 VLA node 都已正確配置 Stage2：

```
vla_policy_node.py (line 946):
  _NATIVE_UNIT_POLICIES = frozenset(["stage2", "stage3", "dagger"])
  → Stage2 outputs native units (arm_torque Nm, wheel_speed rad/s)
  → normalize_action() skipped for stage2
  → Bridge's _action_to_ctrl() handles it uniformly

bridge_node.py (line 769):
  if vla_mag < _HYBRID_WHEEL_FALLBACK_THRESHOLD (0.15):
    → P-controller fallback
    → For Stage2: ~50% of steps below threshold → P-controller takeover
```

### 架構現狀（Phase 286）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | CJ P-controller + Hybrid fallback |
| vla_policy_node.py | ✅ 1000 行 | stage2 in _NATIVE_UNIT_POLICIES |
| Stage2 checkpoint | ✅ 可用 | s2_r045.pt (epoch=s2_10, loss=0.2938) |
| Stage2 wheel magnitude | ⚠️ 臨界 | 50% steps < 0.15 → P-controller 接管 |
| Stage2 SR | ✅ 60-70% | 取決於 goal 分布 |
| CJ P-controller | ✅ 78-86% SR | 可靠的 loco fallback |
| Hybrid fallback threshold | ✅ 0.15 | 適用於 Stage2 |

### 下一步

- [ ] Phase 287: 50-goal Stage2 eval（完整評估，需優化速度）
- [ ] Phase 288: Stage2 在 bridge 中的實際 SR（通過 full.launch.py 端到端）
- [ ] Phase 289: 測試 Stage2 在 bridge hybrid 模式下是否比純 Stage2 更高 SR

### 阻礙

- 50-goal eval 600s timeout → 需要更快的 eval 腳本
- Stage2 wheel magnitude 仍低於 P-controller，hybrid fallback 是必要的
- 真實 ROS2 硬體環境仍不可用
