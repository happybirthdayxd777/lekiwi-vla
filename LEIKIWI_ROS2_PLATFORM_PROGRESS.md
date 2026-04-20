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
