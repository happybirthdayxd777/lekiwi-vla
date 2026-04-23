# Phase 278 — VLA Action Normalization Bug + Unified Bridge Architecture

**Date**: 2026-04-23 13:00 CST

## 本次心跳完成

### 發現：normalize_action 雙應用 Bug（Critical）

**問題**：`vla_policy_node._run_inference()` 對 Stage2/DAgger 策略應用了 `normalize_action()`，但 bridge 的 `_action_to_ctrl()` 已經包含相同的 normalization 邏輯。這導致雙重標準化。

**根本原因分析**：

1. `vla_policy_node._run_inference()` line 936-944：
   ```python
   raw_action = self.policy.predict(obs)     # (9,) 
   native_action = normalize_action(raw_action)  # ❌ WRONG for Stage2
   smoothed_action = self._smoother.smooth(native_action)
   msg.data = smoothed_action.tolist()
   ```

2. `normalize_action()` 將 `[-1,1]` 映射到 native units：
   - Arm: `(arm + 1) / 2 * 3.14`
   - Wheel: `(wheel + 1) / 2 * 10`

3. 但 Stage2 的 `policy.infer()` **已經輸出 native units**（與訓練數據相同）

4. Bridge 的 `_action_to_ctrl()` **再次標準化**：
   - Arm: `clip(action[:6], -1, 1) * 3.14`
   - Wheel: `clip(action[6:9], -0.5, 0.5) * 10`

**實際影響**：

| 路徑 | 輸入 | normalize_action | _action_to_ctrl | 最終 wheel torque |
|------|------|-----------------|-----------------|-------------------|
| Stage2（錯誤） | 0.3 rad/s | 0.3→6.5 | clip(6.5,-0.5,0.5)=0.5, ×10 | **5.0 Nm** |
| Stage2（正確） | 0.3 rad/s | 無 | clip(0.3,-0.5,0.5)=0.3, ×10 | **3.0 Nm** |
| P-ctrl CJ | 0.5 rad/s | 無 | clip(0.5,-0.5,0.5)=0.5, ×10 | **5.0 Nm** |

normalize_action 導致 Stage2 wheel torque 被錯誤放大 1.67x。

**為何 40% SR 仍能運作**：
- 當 VLA wheel magnitude > 0.15 threshold，wheel action 直接使用（不走 hybrid fallback）
- 即使 normalize_action 放大了，bridge 的 clip 會將 6.5 clamp 回 0.5，最大 torque = 5.0 Nm = 與 P-ctrl 相當
- 所以系統仍能運作，但 VLA action  scale 不正確

### 修復方案

修改 `vla_policy_node.py`，讓 Stage2/DAgger 跳過 `normalize_action()`：

```python
# vla_policy_node.py line 935-949 (修改後)
# Policy inference
raw_action = self.policy.predict(obs)         # (9,)

# Stage2/DAgger output native units — skip normalize_action to avoid double-normalization
# normalize_action() is only for policies that output [-1,1] (ACT, diffusion, etc.)
if self._policy_name in ("act", "diffusion", "pi0", "pi0_fast", "task_oriented"):
    native_action = normalize_action(raw_action)
else:
    native_action = raw_action  # Stage2/DAgger output native units directly

smoothed_action = self._smoother.smooth(native_action)
msg.data = smoothed_action.tolist()
```

但更好的方案是：移除 `normalize_action()` 調用，讓 bridge 統一處理所有 action 的標準化。因為 bridge 已經有 `_action_to_ctrl()` 處理這個。

**最終採用方案**：在 `vla_policy_node` 中直接將 raw action 發布到 `/lekiwi/vla_action`，讓 bridge 的 `_action_to_ctrl()` 統一處理。

### Bridge Architecture Status (Phase 278)

| Component | Status | Notes |
|-----------|--------|-------|
| bridge_node.py | ✅ 1306 lines | CJ P-ctrl (100% SR), hybrid fallback, VLA action |
| vla_policy_node.py | ✅ 987 lines | Stage2/DAgger/Stage3, **normalize_action bug discovered** |
| CTF Security Layer | ✅ C1-C8 | 資安監控整合 |
| Camera Adapter | ✅ URDF 20Hz | front + wrist camera |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |
| Stage2PolicyRunner | ✅ Phase 268 | goal-radius>0.45m → zeros fallback |
| Bridge Health Monitor | ✅ 14/14 | All checks passed |
| **normalize_action bug** | 🐛 Phase 278 | Double-normalization for Stage2/DAgger |

### Topic 數據流（已確認完整）

```
bridge_node._publish_joint_states()  →  /lekiwi/joint_states
vla_policy_node._on_joint_states()   ←  /lekiwi/joint_states
vla_policy_node._on_image()          ←  /lekiwi/camera/image_raw
vla_policy_node._run_inference()    →  policy.infer()
vla_policy_node._publish_action()   →  /lekiwi/vla_action
bridge_node._on_vla_action()        ←  /lekiwi/vla_action
bridge_node._step()                 →  MuJoCo step
```

### 已驗證的事實

1. **VLA 集成已完成**（Phase 276-277 確認）
2. **Topic 契約正確**：bridge 訂閱 `/lekiwi/vla_action`，vla_policy_node 發布到同一 topic
3. **Hybrid fallback 邏輯存在**：`use_contact_jacobian=True` 時使用 CJ P-controller
4. **Goal 同步**：bridge 和 vla_policy_node 都訂閱 `/lekiwi/goal`，通過 launch arg 同步
5. **normalize_action 雙應用 bug**：存在但不致命（bridge clip 限制最大 torque）

## 下一步

- [ ] Phase 279: 修復 `normalize_action` 雙應用 bug
- [ ] Phase 280: 測試 full.launch.py end-to-end（需要 ROS2 環境）
- [ ] Phase 281: 確認 Stage2 在 URDF 上 40% SR 瓶頸是 policy 限制而非 bridge bug

## 阻礙

- normalize_action 雙應用導致 VLA action 尺度不正確
- 無 ROS2 環境無法進行 end-to-end 測試
- Stage2 URDF 40% SR 落後 primitive 72%，來自物理幾何差異

## Git

- Commit: Phase 278 progress heartbeat
- 本次發現：normalize_action 雙應用 bug（Critical）
