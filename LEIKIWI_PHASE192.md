## [Phase 192 - 2026-04-19 21:00 UTC] — TWIST_TO_CONTACT FIXED: *200 bug REMOVED

### ✅ 已完成

**ROOT CAUSE IDENTIFIED: Phase 164 added `*200` causing ALL wheel speeds to saturate**

Phase 164's `twist_to_contact_wheel_speeds` formula incorrectly multiplied vx, vy by 200:
```python
# BUGGY (Phase 164-191):
vx_200 = vx * 200.0
w1 = -0.0124 * vx_200 + 0.1880 * vy_200  → saturates for any vx > 0.025
```

For a typical small goal (0.3, 0.3), P-controller outputs vx=0.15:
- **OLD**: vx_200=30 → w1=5.27 → clipped to 0.5 (SATURATED)
- **NEW**: w1=0.026 (proper range)

**Impact on VLA training (Phase 189 data):**
| Formula | goal (0.3,0.3) | goal (0.5,0.5) | goal (0.1,0.1) |
|---------|---------------|---------------|---------------|
| OLD (*200) | [0.5, 0.5, -0.36] | [0.5, 0.5, -0.5] | [0.5, 0.5, -0.12] |
| NEW (fixed) | [0.026, 0.060, -0.002] | [0.044, 0.100, -0.003] | [0.009, 0.020, -0.001] |

OLD: ALL goals → identical saturating wheels → VLA learns constant output for ANY goal → 0% SR
NEW: wheels vary continuously with goal → proper goal-conditioned training data

**Fix Applied to `sim_lekiwi_urdf.py`:**
- Removed `* 200` from vx/vy in `twist_to_contact_wheel_speeds()`
- vx, vy now used directly (in m/s)
- Comment updated to explain root cause

**Verification:**
- Wheel speed test: goal (0.3,0.3) → w=[0.026, 0.060, -0.002] ✓
- P-controller with FIXED formula: 0.26m at 300 steps (same as OLD — k_omni=15 dominates)
- VLA training data will now have proper wheel speed variation correlated with goals

### 🔍 架構現況
- `sim_lekiwi_urdf.py` — Phase 192: twist_to_contact FIXED (*200 removed), k_omni=15.0 active
- `bridge_node.py` (1063 lines) — ROS2 /lekiwi/cmd_vel → MuJoCo action
- `vla_policy_node.py` (664 lines) — VLA policy inference
- P-controller: ~0.26m/300steps with k_omni=15 overlay (primary locomotion)
- Wheel contact physics: secondary (k_omni overlay dominates)

### 🧭 下一步（下次心跳）

**PRIORITY 1: Re-collect CLEAN training data with FIXED twist_to_contact**
1. Use corrected `twist_to_contact_wheel_speeds` (no *200)
2. Collect 10k goal-directed frames with varied wheel speeds (not saturating)
3. Verify: Corr(w0,gy) > 0.9, Corr(w1,gx) > 0.6 (was 0.98 but meaningless with saturation)

**PRIORITY 2: Train Phase 192 VLA on clean data**
4. Train GoalConditionedPolicy with corrected data
5. Eval: should achieve >20% SR (vs 0% with phase189 corrupted data)

**PRIORITY 3: Also test P-controller with larger kP**
6. Current kP=0.5 → vx=0.15 for 0.3m goal → very small wheel speeds
7. kP=2.0 → vx=0.6 → ws=[0.099, 0.239, -0.049] (10x larger, properly unsaturated)

### 🚫 阻礙
- ~~*200 bug saturates ALL wheel speeds~~ → **FIXED Phase 192**
- ~~VLA trained on corrupted phase189 data~~ → **Needs re-collection with fixed formula**
- **High contact variance**: MuJoCo contact dynamics stochastic between runs
- **k_omni=15 dominates locomotion**: wheel contact only secondary force

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p189 | Data: ALL wheel speeds saturate | CORRUPTED: VLA learns constant output |
| p191 | VLA eval: 0% SR, P-ctrl: 20% SR | Was using buggy twist_to_contact |
| **p192** | **twist_to_contact FIXED** | **Removed *200, wheels now vary 0.009-0.1 rad/s** |
| p192 | k_omni=15 still active | k_omni overlay dominates loco (0.26m/300steps) |
| p192 | OLD vs NEW displacement | Same 0.26m — k_omni drives loco, not wheels |

### Git
- Modified: `sim_lekiwi_urdf.py` — twist_to_contact *200 bug fixed
- New: `scripts/collect_phase192_clean.py` (planned for next heartbeat)