# LeKiWi ROS2 ↔ MuJoCo ↔ VLA 統一研究平台 — 進度追蹤

> 自動每 30 分鐘心跳更新

---

## [Phase 268 - 2026-04-22 09:00 CST] — Stage2PolicyRunner Goal-Radius Filtering

### 🎯 本次心跳完成（Phase 268）

**Problem**: `Stage2PolicyRunner` 沒有goal-radius過濾。Stage2訓練數據是|r|<0.45m goals，
但bridge在運行時可能會把所有goals（包括|r|>0.45m）發給Stage2 policy，導致它對訓練分佈外的goals做出錯誤預測。

**Solution**: 在 `Stage2PolicyRunner.__call__()` 添加goal-radius檢查：
```python
goal_norm = state[9:11]   # [-1, 1] normalized goal
goal_xy_m = goal_norm * 0.4   # un-normalize to meters
goal_radius = np.linalg.norm(goal_xy_m)
if goal_radius > 0.45:
    return np.zeros(9, dtype=np.float32)  # → bridge falls back to P-controller
```
- 當goal在半徑外時，返回zeros action
- Bridge的hybrid邏輯（_HYBRID_WHEEL_FALLBACK_THRESHOLD）會檢測到vla_wheel_raw=0，使用P-controller作為fallback
- 完全不需要bridge代碼改動，安全、被動防護

**Git**: Commit `0ed5551` — Phase 268: Stage2PolicyRunner goal-radius filtering

### 🔍 Architecture State: Phase 268

| Component | Status | File |
|-----------|--------|------|
| `bridge_node.py` | ✅ 1260 lines, primitive + URDF modes | `src/lekiwi_ros2_bridge/` |
| `vla_policy_node.py` | ✅ 735 lines, now with stage2/stage3 + **goal-radius filter** | `src/lekiwi_ros2_bridge/` |
| CTF Security Layer | ✅ Phase 239-243, C1-C8 challenges | `ctf_integration.py` |
| Camera Adapter | ✅ URDF mode 20Hz RGB | `camera_adapter.py` |
| Real Hardware Adapter | ✅ Real hardware interface | `real_hardware_adapter.py` |
| 5× Launch Files | ✅ bridge/vla/ctf/full/real_mode | `launch/` |
| Stage2PolicyRunner | ✅ goal-radius filter Phase 268 | `vla_policy_node.py` |

### 🧭 下次心跳（Phase 269）

**Priority 1: Kill Overfitting Training**
- Phase 264 curriculum training可能還在跑（PID可能16582但已不存在）
- s3_epoch9.pt 是 definitive best checkpoint (loss=0.2324, epoch 9/15)
- Training PID已消失，可能已自然結束

**Priority 2: DAgger Stage3 Data Collection**
- Stage3（0-15% SR）需要50+ episodes在hard goals上
- 用P-controller作為expert，DAgger correction loop收集新數據
- 新數據 → retrain Stage3 with better coverage

**Priority 3: Stage2 10-Goal Quick Eval via Bridge**
- 隔離測試Stage2PolicyRunner goal-radius filtering
- 5個|r|<0.45m goals（Stage2擅長） + 5個|r|>0.45m goals（Stage2應返回zeros）

**Priority 4: Integrate lekiwi_modular actual URDF STL meshes**
- lekiwi_modular_meshes symlink已建立
- 把lekiwi_modular的STL meshes接入bridge的LeKiWiSimURDF

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196 | CJ P-controller + VLA train (14 epochs) | 8% SR (with early term) |
| p227 | Q2-extended data + 30-epoch VLA train | 4% SR |
| p234 | P-ctrl 94% SR (FIXED), Phase196 8%, Phase227 4% | 50-goal complete |
| p254 | DAgger-254 training (30ep, 20 epochs) | best_loss=0.0018 |
| p256 | DAgger-254 10-goal quick eval | **20% SR** |
| p257 | Bridge health monitor (14/14 ✓) | ✅ |
| p260 | Curriculum training: Stage1+2 done | Stage1+2 checkpoints |
| p261 | Stage2 50-goal eval | **72% SR** |
| p264 | Stage3 training (15 epochs, background) | loss=0.2324@epoch9 |
| p265 | Stage3 s3_epoch6 20-goal eval | **VLA=15% vs P-ctrl=85%** |
| p266 | Stage3 s3_epoch9 10-goal eval | **VLA=0% vs P-ctrl=60% (100-step)** |
| p266b | Stage3 overfitting confirmed | loss 0.2324→0.2372 (ep9→12) |
| p267 | **Stage2+Stage3 loaders added to bridge** | ✅ |
| p268 | **Stage2PolicyRunner goal-radius filtering** | ✅ |

### Git
- Commit: `0ed5551` Phase 268: Stage2PolicyRunner goal-radius filtering
- Working tree: clean
- Remote: up-to-date (pushed with rebase)
