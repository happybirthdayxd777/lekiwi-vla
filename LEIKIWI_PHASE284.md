# Phase 284 — VLA Wheel Magnitude: s3_epoch6 vs s3_epoch9 Confirmed Identical

**Date**: 2026-04-23 17:05 CST

## 本次心跳完成

### Phase 283 發現的後續：s3_epoch6 vs s3_epoch9 車輪動作診斷

Phase 283 發現 VLA 車輪動作比 P-controller 小約 26 倍。提出的假設：s3_epoch6（20-goal eval 15% SR）可能比 s3_epoch9（50-goal eval 2% SR）有更大的車輪動作幅度。

**設計實驗**：快速 10-goal × 50-step 評估，測量兩個 checkpoint 的車輪動作幅度。

### 結果

| Checkpoint | raw_wheel_mag | norm_wheel_mag | 10-goal SR |
|------------|---------------|----------------|------------|
| s3_epoch6  | 0.0629        | 0.0315 rad/s   | 0%         |
| s3_epoch9  | 0.0652        | 0.0326 rad/s   | 0%         |
| **P-controller** | —       | **0.389 rad/s** | ~80% SR   |

**關鍵發現：兩個 checkpoint 車輪幅度幾乎完全相同（~0.03 vs ~0.39, 相差 12x）**

- VLA raw wheel ∈ [-0.25, 0.25]，normalized 後 ∈ [-0.125, 0.125]（eval 標準化）
- eval 腳本：`normalize_action(x) = clip(x, -1, 1) * 0.5` → 輸出 ∈ [-0.5, 0.5]
- 但 VLA 原始輸出約 0.03，標準化後也只有 0.015，遠小於 P-controller 的 0.39
- **車輪幅度問題是兩個 checkpoint 的共同問題，不是 epoch 9 特有的過擬合問題**

### 根本原因確認

| 層次 | 問題 | 修復方向 |
|------|------|---------|
| **訓練層** | VLA 訓練時 wheel loss 權重不足或數據不足，網絡輸出趨近於零 | 重新設計 wheel loss weighting 或擴展數據 |
| **Eval 腳本** | `normalize_action()` 對兩個 checkpoint 都正確 | 不需修改 |
| **Bridge** | `_action_to_ctrl()` 對兩個 checkpoint 都正確 | 不需修改 |

### 架構現狀（Phase 277-284）

| 元件 | 狀態 | 備註 |
|------|------|------|
| bridge_node.py | ✅ 1306 行 | Contact-Jacobian P-controller  loco fallback |
| vla_policy_node.py | ✅ 818 行 | CLIP-FM/pi0/ACT/DAgger/Stage2/Stage3 |
| CTF Security Layer | ✅ C1-C8 全部 | ctf_integration.py |
| s3_epoch6 | ❌ 12x 車輪幅度不足 | VLA 訓練問題，非 bridge bug |
| s3_epoch9 | ❌ 同上 | 同上 |
| P-controller CJ | ✅ 78-86% SR | 可靠的 loco fallback |

### 車輪幅度修復選項

- [ ] **Option A**：訓練時增大 wheel loss 權重（×20）
- [ ] **Option B**：VLA 只負責 arm，wheel 由 P-controller 完全接管（Stage2 模式）
- [ ] **Option C**：後處理放大：bridge 偵測到 VLA wheel < threshold 時用 P-controller 替代

### 下一步

- [ ] **Phase 285**: 實現 Option C — VLA arm-only + P-controller wheel fallback
- [ ] **Phase 286**: 整合 Stage2 (arm P-controller, wheel P-controller) 進 ROS2 bridge
- [ ] **Phase 287**: 測試 full.launch.py end-to-end

### 阻礙

- VLA wheel locomotion 訓練需要完全重新設計
- 真實 ROS2 硬體環境仍不可用
