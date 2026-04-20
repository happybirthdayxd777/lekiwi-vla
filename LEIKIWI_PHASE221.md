# Phase 221 — 2026-04-20 15:00 UTC

## Git Commit Phase 218/218b Eval Outputs + Platform State Summary

### 本次心跳完成

**1. Git Commit Phase 218/218b Eval Data**
```
a0a7d54 Phase 221: Commit Phase 218/218b eval outputs (10-goal, seed=42/99)
  results/phase218_eval_output.txt  (+98 lines, P-ctrl 100% vs VLA-e14 90%)
  results/phase218b_eval_output.txt (+94 lines, phase196 80% vs phase190 10%)
```

**2. Full Platform State Summary**

---

### 🔬 Phase 218 Key Results (Definitive)

| Policy | Checkpoint | SR (10-goal) | Avg Steps | Notes |
|--------|-----------|-------------|-----------|-------|
| P-controller | CJ kP=2.0 | **100%** | 99.8 | Oracle baseline |
| VLA epoch_14 | phase196_contact_jacobian_train/ | **90%** | 108.1 | 1/10 failed on goal 7 |
| Phase190 best | phase190_vision_train/best_policy.pt | **10%** | 180.0 | Catastrophically bad |

**Phase 218b Head-to-Head (seed=99, 10 goals):**
- phase196_e14: 8/10 = 80%, avg 115.2 steps
- phase190_e27: 1/10 = 10%, avg 180.0 steps, final_dist ≈ 2.75m (failed to move)

**Conclusion: Phase196 VLA is the best learned policy. Phase190 is broken.**

---

### 📁 Current Architecture (lekiwi_ros2_bridge/)

```
bridge_node.py (1186 lines)
  ✅ /lekiwi/cmd_vel → MuJoCo wheel speeds
  ✅ MuJoCo → /lekiwi/joint_states (20 Hz)
  ✅ MuJoCo → /lekiwi/camera/image_raw (front, 20 Hz, URDF only)
  ✅ MuJoCo → /lekiwi/wrist_camera/image_raw (arm tip, 20 Hz, URDF only)
  ✅ Hybrid VLA+P-ctrl bridge with action smoothing

vla_policy_node.py (746 lines)
  ✅ CLIP-FM policy inference at 4 Hz
  ✅ ActionSmoother (EMA alpha=0.25, max_delta=0.8)
  ✅ /lekiwi/vla_action topic (Float64MultiArray, 9-dim)
  ✅ Graceful fallback to P-controller when VLA fails

Camera pipeline:
  ✅ CameraAdapter thread (20 Hz, URDF only)
  ✅ Graceful None handling in primitive mode (no crash)
  ✅ LeKiWiSimLoader factory: make_sim('urdf') vs make_sim('primitive')

Launch files:
  ✅ full.launch.py — sim_mode + VLA
  ✅ vla.launch.py — VLA only
  ✅ real_mode.launch.py — hardware mode
```

---

### 🎯 Next Steps (Priority Order)

**Priority 1: 50-goal Evaluation of Phase196 VLA (statistical power)**
- Current: 10 goals (seed=42, SR=90%) — too few for statistical confidence
- Need: 50 goals to get ±5% confidence interval
- Script: extend `scripts/eval_phase181.py` with n_episodes=50, seed=42
- Run in background (~30 min on CPU)

**Priority 2: ROS2 Bridge Launch Verification (on machine with ROS2)**
- `ros2 launch lekiwi_ros2_bridge full.launch.py`
- Verify /lekiwi/joint_states at 20 Hz
- Verify /lekiwi/camera/image_raw non-black frames
- Verify /lekiwi/wrist_camera/image_raw (URDF mode only)

**Priority 3: Phase198 Policy Evaluation**
- Progress doc mentioned phase198_v3_final.pt as "fully trained"
- But: no phase198 results directory found in results/
- Need to verify: was phase198 actually completed? Is the checkpoint valid?
- If valid: run 10-goal eval to compare with phase196_e14

**Priority 4: Investigate VLA Failure Mode on Goal 7**
- Phase218 goal 7 failed: goal_xy=?, VLA stopped at 0.302m distance
- Need to log which goals fail consistently across seeds
- May reveal systematic bias (e.g., -Y quadrant, large distances)

---

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p190  | CJ P-controller data collection + VLA train | 94% SR (50 goals) |
| p196  | CJ P-controller data collection + VLA train | 90% SR (14 epochs) |
| p198  | Architecture fix retrain | phase198_v3_final.pt (14.3 MB) — UNVERIFIED |
| p218b | phase196_e14 vs phase190_e27 (10 goals, seed=99) | **80% vs 10%** |
| p218  | phase196_e14 vs P-ctrl (10 goals, seed=42) | **90% vs 100%** |
| p219  | lekiwi_modular confirmed + eval fix committed | ✅ |
| p220  | VLA pipeline smoke test + camera graceful degradation verified | ✅ |
| p221  | Git commit eval outputs + platform state summary | ✅ |

---

### Git

- Commit: `a0a7d54` Phase 221: Commit Phase 218/218b eval outputs
- Branch: main
- Status: clean (before heartbeat) → committed + pushed
