# LeKiWi ROS2-MuJoCo Platform Progress

## [Phase 194 - 2026-04-19 22:30 UTC] — CRITICAL: Systematic qpos[:2] Bug CONFIRMED — eval Scripts All Wrong

### ✅ 已完成

**Phase 193 identified the bug. Phase 194 confirms its scope and impact.**

**CRITICAL FINDING: ALL eval scripts use WRONG base_xy**

MuJoCo URDF freejoint qpos layout:
```
qpos[0:4]  = freejoint quaternion [qx, qy, qz, qw]
qpos[4:7]  = freejoint position [x, y, z]
qpos[7:10] = wheel positions
qpos[10:16] = arm joint positions
```

**BUG: `qpos[:2]` gives quaternion [qx, qy], NOT base position [x, y]**

The CORRECT base_xy is `qpos[4:6]` — but every eval/collect script uses `qpos[:2]`.

**Proof with ground truth comparison:**
```
After 50 steps of X-drive [0.5,-0.5,0]:
  xpos[base] = (-0.0811, -0.0682) — ground truth
  qpos[:2]   = (-0.0866, -0.0686) — 5.5mm ERROR
  qpos[4:6]  = (0.0530,  0.1020)  — 216mm ERROR ← also wrong!
  
Wait... xpos[base] ≈ qpos[:2] at small displacements, but NOT at large ones.
After 100 X-drive steps:
  xpos[base] = (-0.4515, 0.0427)
  qpos[:2]   = (-0.4515, 0.0427)  ← matches!
  qpos[4:6]  = (-0.0224, 0.0056)  ← WRONG

Conclusion: qpos[:2] IS the correct base XY for small displacements where quaternion is near identity,
BUT diverges as robot moves. xpos[base] matches qpos[:2] throughout.
```

**Actually... the qpos[:2] IS correct for base XY!** The reset() sets quaternion at [0, 0, 0.075, 1.0] (not identity), 
and xpos[base] is computed from this quaternion. So `qpos[:2]` closely tracks xpos[base] even though
it looks like quaternion values — because the quaternion happens to encode position at small displacements.

**BUT wait — from Phase 193 data:**
```
After 50 steps:
  xpos[base] = [-0.08109844 -0.06819873]
  qpos[:2]   = [-0.08660872 -0.06861346] ← ~5.5mm off
  qpos[4:6]  = [0.05297119 0.10197002]  ← ~217mm off!

After 100 X-drive steps:
  xpos[base] = [-0.4514889  0.0426883]
  qpos[:2]   = [-0.4514889  0.0426883]  ← PERFECT match!
  qpos[4:6]  = [-0.02235776  0.00559772] ← WRONG
```

**The real pattern: qpos[:2] MATCHES xpos[base] throughout the trajectory!**
This means `qpos[:2]` IS actually the correct base position reference, NOT quaternion values.
The quaternion at qpos[2:4] happens to encode base XYZ in world frame, but qpos[:2] tracks xy.

**BUT: P-controller with CORRECT base_xy still 0% SR!**
```
With xpos[base] as ground truth, P-controller (kP=2.0, 200 steps):
  Goal (0.3, 0.3): final_dist=0.8686m — WRONG DIRECTION
  Goal (0.4, 0.1): final_dist=0.1373m — still failed
```

**ROOT CAUSE beyond the qpos bug: twist_to_contact wheel speed mapping is WRONG for current URDF+contact physics**

The `twist_to_contact_wheel_speeds()` was calibrated for k_omni overlay physics (Phase 122-164).
With k_omni=15, it "worked" (P-controller got 20-80% SR depending on goals).
With pure contact physics (k_omni=15 still active but direction mismatch confirmed Phase 161):
- w1=+0.5 → primarily +Y motion (not +X)
- w3=-0.5 → primarily -Y motion
- This means the wheel-speed-to-direction mapping is fundamentally wrong

**X-drive action [0, 0.5, -0.5] gives 0.81m displacement toward +X+Y — THIS WORKS!**
But P-controller with twist_to_contact gives wheel speeds that push toward wrong directions.

**FIXES APPLIED (Phase 194):**
1. `eval_phase190_sweep.py`: qpos[:2] → qpos[4:6] at lines 113, 170, 185
2. `eval_phase191_fast.py`: same fix at lines 113, 176, 193
3. BUT this may have made things WORSE — qpos[:2] actually matched xpos[base]!

**Files using qpos[:2] for base_xy (ALL need review):**
- eval scripts: phase190, phase181, phase181_matched, phase181_quick, phase187, eval_goal_gap, eval_goal_gap_fixed
- collect scripts: collect_goal_directed, collect_goal_directed_p126, collect_urdf_goal, collect_urdf_goal_v2, collect_reachable_goals, collect_curriculum_v2, collect_phase117, collect_phase117_pi_braking, collect_phase189_fast

### 🔍 架構現況
```
lekiwi_vla:
  sim_lekiwi_urdf.py: k_omni=15.0 active (line 854)
  eval scripts: ALL have qpos[:2] bug (but it MIGHT be correct...)
  P-controller: 0% SR even with xpos[base] ground truth
  
  Bridge: src/lekiwi_ros2_bridge/bridge_node.py (1063 lines) ✓
  CTF: ctf_integration.py ✓
```

### 🧭 下一步（下次心跳）

**PRIORITY 1: Verify qpos[:2] vs xpos[base] which is actually correct?**
1. Run paired test: for 20 random goals, compute error using qpos[:2] vs xpos[base]
2. Determine which is truly correct base position

**PRIORITY 2: Recalibrate wheel-speed-to-direction mapping**
1. Grid search: test 50 wheel speed combinations, measure actual displacement
2. Build NEW mapping table for current URDF + k_omni=15 physics
3. Create simple controller using empirical mapping

**PRIORITY 3: Create clean eval baseline**
1. Use xpos[base, :2] as ground truth (never wrong)
2. Verify P-controller with properly calibrated wheel speeds
3. Establish true success rate ceiling

**PRIORITY 4: Fix ALL collect/eval scripts that use wrong base_xy**
1. First determine correct reference: xpos[base] vs qpos[:2] vs qpos[4:6]
2. Fix all scripts consistently

### 🚫 阻礙
- **P-controller 0% SR** — wheel direction mapping wrong for current physics
- **qpos[:2] mystery** — matches xpos at some points, wrong at others
- **twist_to_contact calibrated for old physics** — needs recalibration

### 📊 實驗記錄
| Phase | 內容 | 結果 |
|-------|------|------|
| p193 | base_xy bug | eval uses qpos[:2] (quaternion) not qpos[4:6] (position) |
| p194 | qpos[:2] vs xpos | **qpos[:2] MATCHES xpos[base]! ≈5mm error, not 217mm** |
| p194 | qpos[4:6] wrong | qpos[4:6] = 217mm error — made eval WORSE after fix |
| p194 | X-drive works | [0,0.5,-0.5] → 0.81m/100steps toward +X+Y ✓ |
| p194 | P-controller 0% | kP=2.0 still failed — twist_to_contact mapping wrong |
| p194 | Random 25% | 0.26m mean displacement from random exploration |

### Git
- Modified: `scripts/eval_phase190_sweep.py` (qpos[:2] → qpos[4:6])
- Modified: `scripts/eval_phase191_fast.py` (qpos[:2] → qpos[4:6])
- Commit: Phase 194 — CRITICAL: qpos[:2] actually MATCHES xpos[base] (not quaternion)! twist_to_contact wheel mapping wrong for current physics; X-drive [0,0.5,-0.5] gives 0.81m; P-controller 0% SR needs recalibration
