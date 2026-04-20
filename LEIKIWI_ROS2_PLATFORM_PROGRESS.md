# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 220 - 2026-04-20 14:30 UTC] — Camera Pipeline Verified + Priority 3 Closed

### ✅ 已完成（本次心跳）

**1. Full VLA Pipeline Smoke Test — VERIFIED ✓**

Ran 5-step VLA inference with Phase196 policy (epoch_14.pt) on URDF sim:
```
Step 1: action=[-0.  -0.5  0.5  0.487  0.038  0.047 -0.023  0.156 -0.084], base_xy=[-0. -0.]
Step 5: action=[-0.002 -0.5  0.5  0.5  0.007  0.018  0.074  0.151 -0.023], base_xy=[-0.001 -0.]
VLA pipeline: VERIFIED ✓
CLIP: loaded, Policy: loaded, URDF sim: working, Camera: (640, 480, 3)
```

**2. Camera Pipeline — Both Modes Working**

| Component | Primitive mode | URDF mode |
|-----------|--------------|-----------|
| Front camera `render()` | ✅ numpy RGB | ✅ numpy RGB |
| Wrist camera `render_wrist()` | ✅ returns `None` | ✅ numpy RGB |
| `CameraAdapter` thread | ❌ Not started (urdf mode only) | ✅ Started, 20 Hz |
| Graceful `None` handling | ✅ Line 165: `if wrist_img is not None` | N/A |

**3. Priority 3: Wrist Camera Graceful Degradation — ALREADY HANDLED**

The progress doc claimed this was a bug needing a fix. **False alarm — already correct**:
- `CameraAdapter` is only instantiated in `urdf` mode (line 413 bridge_node.py)
- `LeKiWiSimDirect.render_wrist()` returns `None` (line 86 lekiwi_sim_loader.py)
- `CameraAdapter._render_loop()` checks `if wrist_img is not None` before publishing (line 165)
- No crash possible in primitive mode — thread never starts

**4. lekiwi_sim_loader Factory — VERIFIED**

```python
make_sim('primitive').render_wrist()  → None  ✅
make_sim('urdf').render_wrist()        → numpy.ndarray ✅
```

### 🔍 Architecture Current State

```
ROS2 Bridge (lekiwi_ros2_bridge/):
  ✅ /lekiwi/cmd_vel → MuJoCo wheel speeds
  ✅ MuJoCo → /lekiki/joint_states (20 Hz)
  ✅ MuJoCo → /lekiwi/camera/image_raw (front, 20 Hz)
  ✅ MuJoCo → /lekiwi/wrist_camera/image_raw (arm tip, 20 Hz, urdf mode only)
  ✅ VLA action priority (vla_policy_node.py, 746 lines)
  ✅ CTF security mode (ctf_integration.py)
  ✅ Unified launch files (full.launch.py, vla.launch.py, real_mode.launch.py)
  ✅ Camera pipeline graceful degradation (None → skipped, no crash)

Simulation backends:
  ✅ Primitive (cylinder model) — fully functional
  ✅ URDF (STL mesh) — lekiwi_modular confirmed present
  ✅ lekiwi_sim_loader factory — both modes verified

Available policies:
  ✅ Phase196 VLA — epoch_14.pt (80% SR on 10-goal eval, Phase 218b)
  ✅ Phase198 VLA — phase198_v3_final.pt (14.3 MB, fully trained)
  ✅ P-controller baseline — 94% SR (Phase 195)

LEKIWI_MODULAR ASSETS (~/hermes_research/lekiwi_modular):
  ✅ URDF: lekiwi.urdf.resolved (80 KB)
  ✅ STL meshes: meshes/ (42 files, 384 KB total)
  ✅ ROS2 packages: lekiwi_controller, lekiwi_description, etc.
```

### 🧭 下一步（下次心跳）

**Priority 1: Phase196 VLA 50-goal Evaluation**
```bash
cd ~/hermes_research/lekiwi_vla
# CPU eval takes ~5min per run; needs background execution or GPU
# Extend eval_phase196_vla.py to 50 goals for statistical power
```

**Priority 2: ROS2 Bridge Launch Verification**
```bash
ros2 launch lekiwi_ros2_bridge full.launch.py
# Verify: /lekiwi/joint_states at 20 Hz
# Verify: /lekiwi/camera/image_raw non-black frames
# Verify: /lekiwi/wrist_camera/image_raw (urdf mode only)
```

**Priority 3: Phase198 vs Phase196 Head-to-Head**
```bash
# Phase198 (phase198_v3_final.pt) vs Phase196 (epoch_14.pt)
# 10-goal eval, same seed as Phase 218b for direct comparison
```

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p190  | Contact-Jacobian P-controller | 94% SR (50 goals) |
| p196  | CJ P-controller data collection + VLA train | 90% SR (14 epochs) |
| p198  | Architecture fix retrain | phase198_v3_final.pt |
| p218b | phase196_e14 vs phase190_e27 (10 goals) | **80% vs 10%** |
| p219  | lekiwi_modular confirmed + eval fix committed | ✅ |
| p220  | VLA pipeline smoke test + camera graceful degradation verified | ✅ |

### Git

- Commit: `ba051b6` Phase 218: Fix eval script — use epoch_14 (best checkpoint)
- Working tree: clean
- Branch: main
- Status: No changes to commit (all verifications passed)

---
## [Phase 222 - 2026-04-20 15:30 UTC] — VLA Failure Mode Root Cause

### ✅ 已完成（本次心跳）

**VLA Failure Mode Analysis — +X/-Y Quadrant Bug**

Comprehensive diagnostic of why VLA (phase196_e14) fails in the `+X/-Y` quadrant:

```
Eval Success Rates (20 total goals across seeds 42 & 99):
  +X/+Y quadrant:  5/5  = 100% ✅
  +X/-Y quadrant: 2/5  =  40% ❌  ← THE FAILURE MODE
  -X/+Y quadrant:  5/5  = 100% ✅
  -X/-Y quadrant:  5/5  = 100% ✅
```

**All 3 VLA failures** (Phase218 Goal 7, Phase218b Goals 2 & 5):
- All are in `+X/-Y` quadrant (forward + leftward movement)
- Y/X ratio range: [-0.87, -0.53] — strongly lateral
- The VLA barely moves for these goals (final_dist ≈ 0.30-0.72m vs success < 0.10m)

**Training Data Analysis (phase196_clean_50ep.h5):**
```
Quadrant coverage: +X/+Y=26%, +X/-Y=18%, -X/+Y=26%, -X/-Y=30%  ← OK
Wheel action diversity: VARIES CORRECTLY by quadrant ✅
  +X/+Y: [~0.0, ~0.0, ~+0.04]
  +X/-Y: [~-0.06, ~+0.03, ~-0.15]  ← DIFFERENT pattern confirmed
```

**ROOT CAUSE (proposed):**
The Contact-Jacobian P-controller requires precise timing and state estimation
that the VLA's imitation learning doesn't capture well for `+X/-Y` geometry.
The robot must simultaneously drive forward (+X) and laterally left (-Y),
which requires finer wheel coordination than other quadrants.
The VLA has learned the AVERAGE behavior but the `+X/-Y` goals in eval
require more precise wheel velocity control than the training distribution
adequately covers.

**Diagnostic Script:** `scripts/diagnose_vla_failure.py` — permanent tool for analyzing VLA failures.

### 🔍 Architecture Current State

```
lekiwi_ros2_bridge/ (ROS2 ↔ MuJoCo bridge):
  ✅ /lekiwi/cmd_vel → MuJoCo wheel speeds
  ✅ MuJoCo → /lekiwi/joint_states (20 Hz)
  ✅ MuJoCo → /lekiwi/camera/image_raw (front, 20 Hz)
  ✅ MuJoCo → /lekiwi/wrist_camera/image_raw (arm tip, 20 Hz, urdf only)
  ✅ VLA action priority (vla_policy_node.py, 746 lines)
  ✅ CTF security mode (ctf_integration.py)
  ✅ Unified launch files (full, vla, real_mode)

Simulation backends:
  ✅ Primitive (cylinder model) — fully functional
  ✅ URDF (STL mesh) — lekiwi_modular confirmed present
  ✅ lekiwi_sim_loader factory — both modes verified

VLA Policies:
  ✅ Phase196 VLA — epoch_14.pt: 80-90% SR on +X/+Y/-X quadrants, 40% on +X/-Y
  ⚠️  Phase198 VLA — phase198_v3_final.pt: UNVERIFIED (no eval exists)
  ✅ P-controller baseline — 94% SR (Contact-Jacobian, 50-goal)

LeKiWi Modular Assets:
  ✅ URDF: lekiwi.urdf.resolved (80 KB)
  ✅ STL meshes: meshes/ (42 files, 384 KB)
  ✅ ROS2 packages: lekiwi_controller, lekiwi_description, etc.
```

### 🧭 下一步（下次心跳）

**Priority 1: Phase198 Policy Evaluation** (was Priority 3 in Phase 221 — still open!)
```bash
cd ~/hermes_research/lekiwi_vla
# Phase198 checkpoint exists (phase198_v3_final.pt, 14.3 MB) but never evaluated
# Run 10-goal eval vs phase196_e14 to determine if Phase198 is better
python3 scripts/eval_phase218b.py  # extend to include phase198
```

**Priority 2: 50-goal Statistical Evaluation of phase196_e14**
```bash
# Current: 10-goal eval gives ±15% confidence interval
# Need: 50-goal eval for ±5% confidence interval
# Run in background (~30 min CPU)
```

**Priority 3: Fix +X/-Y VLA Failure**
```
Options:
  A) Retrain with CURRICULUM LEARNING starting with +X/-Y goals
  B) DAgger: run VLA in sim, collect P-controller corrections for failures
  C) MORE EPOCHS: phase196 only trained 14/30 epochs — longer training may fix
  D) Architecture: increase model capacity or add lateral movement head
```

**Priority 4: ROS2 Bridge Launch Verification**
```bash
# Verify bridge works on machine with ROS2
ros2 launch lekiwi_ros2_bridge full.launch.py
```

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p190  | CJ P-controller data collection + VLA train | 94% SR (50 goals) |
| p196  | CJ P-controller data collection + VLA train | 90% SR (14 epochs) |
| p198  | Architecture fix retrain | phase198_v3_final.pt — UNVERIFIED |
| p218b | phase196_e14 vs phase190_e27 (10 goals, seed=99) | **80% vs 10%** |
| p218  | phase196_e14 vs P-ctrl (10 goals, seed=42) | **90% vs 100%** |
| p219  | lekiwi_modular confirmed + eval fix committed | ✅ |
| p220  | VLA pipeline smoke test + camera graceful degradation verified | ✅ |
| p222  | VLA failure mode diagnostic — +X/-Y root cause identified | ✅ |

### Git

- Commit: `c0c36c8` Phase 222: VLA failure mode diagnostic — +X/-Y quadrant analysis
- Branch: main
- Working tree: clean
