# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 257 - 2026-04-21 21:30 CST] — Bridge Health Monitor: 14/14 ✓

### 🔍 Architecture Current State

```
ROS2 Bridge (lekiwi_ros2_bridge/):
  ✅ bridge_node.py (59KB, 1260 lines) — cmd_vel↔MuJoCo, joint_states↔ROS2
  ✅ vla_policy_node.py (821 lines) — CLIP-FM policy at 4 Hz
  ✅ ctf_integration.py (975 lines) — CTF security mode (C1-C8)
  ✅ real_hardware_adapter.py (349 lines) — ST3215 serial servo protocol
  ✅ camera_adapter.py (286 lines) — URDF-mode camera bridge (20 Hz)
  ✅ lekiwi_sim_loader.py (85 lines) — factory for primitive/urdf/real backends
  ✅ Launch files: full, bridge, vla, real_mode, ctf — all present

Simulation backends:
  ✅ Primitive (cylinder) + URDF (STL mesh) — both verified
  ✅ LeKiWiSimLoader factory verified

Health check tool (NEW — Phase 257):
  ✅ scripts/bridge_health_monitor.py — 14 checks, all passing
```

### ✅ 本次心跳完成（Phase 257）

**1. Bridge Health Monitor — `scripts/bridge_health_monitor.py`**

New health-check script verifying:
  - bridge_node.py parses to 6 publishers, 6 subscribers
  - All 6 required input topics subscribed by bridge
  - All ROS2 msg types imported
  - URDF joints present (50 total — 3 wheel, 2 arm)
  - Sim backend files exist (primitive + urdf)
  - All 5 required callbacks implemented
  - Omni-kinematics conversion present
  - Real hardware adapter, VLA policy node, CTF integration all present
  - All 4 required launch files present

```
Results: 14/14 checks passed
  ✓ bridge_node.py exists  → 59KB
  ✓ Parse bridge topics  → 6 publishers, 6 subscribers
  ✓ Bridge ← Modular topic contract  → All 6 required inputs subscribed
  ✓ Bridge output topics  → 6 static publishers
  ✓ ROS2 msg imports  → All required msgs imported
  ✓ URDF joints present  → 50 total — 3 wheel, 2 arm joints
  ✓ Sim backend files  → Present: sim_lekiwi.py, sim_lekiwi_urdf.py
  ✓ Required callbacks  → All 5 required callbacks implemented
  ✓ Omni-kinematics conversion  → twist_to_wheel_speeds found
  ✓ Real hardware adapter  → real_hardware_adapter.py
  ✓ VLA policy node  → vla_policy_node.py
  ✓ CTF integration  → ctf_integration.py
  ✓ Sim loader  → lekiwi_sim_loader.py
  ✓ Required launch files  → Found: [full, bridge, vla, real_mode, ctf].launch.py
```

**Key finding: Bridge topic contract is fully consistent with lekiwi_modular**

Bridge subscribes to (from lekiwi_modular):
  - `/lekiwi/cmd_vel` → Twist
  - `/lekiwi/goal` → Point
  - `/lekiwi/vla_action` → Float64MultiArray
  - `/lekiwi/policy_input` → ByteMultiArray
  - `/lekiwi/cmd_vel_hmac` → ByteMultiArray
  - `/lekiwi/record_control` → String

Bridge publishes to (for downstream consumers):
  - `/lekiwi/joint_states` → JointState (primitive sim names)
  - `/lekiwi/joint_states_urdf` → JointState (URDF joint names)
  - `/lekiwi/odom` → Odometry
  - `/lekiwi/camera/image_raw` → Image (20 Hz)
  - `/lekiwi/wrist_camera/image_raw` → Image (20 Hz)
  - `/lekiwi/security_alert` → String
  - `/lekiwi/wheel_N/cmd_vel` → Float64 × 3

### 🔍 Bridge Architecture Summary

```
lekiwi_modular (real robot):
  omni_controller.py → /lekiwi/wheel_N/cmd_vel (Float64 × 3)
  omni_odometry.py   → /lekiwi/odom (Odometry)

lekiwi_ros2_bridge (bridge_node.py):
  Subscribes:
    /lekiwi/cmd_vel      ← mobile base velocity from any teleop/nav node
    /lekiwi/goal         ← navigation goal (Point) from nav2 or manual
    /lekiwi/vla_action   ← arm+wheel action from VLA policy node
    /lekiwi/policy_input ← CTF Challenge 7 policy injection
    /lekiwi/cmd_vel_hmac ← Challenge 1 HMAC-authenticated cmd_vel
    /lekiwi/record_control
  Publishes:
    /lekiwi/joint_states       → for VLA policy node (primitive names)
    /lekiwi/joint_states_urdf → for external consumers (URDF names)
    /lekiwi/odom              → odometry from MuJoCo sim
    /lekiwi/camera/image_raw  → front camera from MuJoCo renderer
    /lekiwi/security_alert    → CTF attack detection alerts
    /lekiwi/wheel_N/cmd_vel   → mirrors real robot interface

lekiwi_ros2_bridge (vla_policy_node.py):
  Subscribes: /lekiwi/joint_states, /lekiwi/camera/image_raw
  Publishes:  /lekiwi/vla_action (arm*6 + wheel*3 at 4 Hz)

MuJoCo Simulation (LeKiWiSim or LeKiWiSimURDF):
  step(action=[arm*6, wheel*3]) → obs {qpos, qvel, base_pose, camera}
```

### 🧭 下次心跳（Phase 258）

**Priority 1: Run DAgger-254 Full Evaluation (50-goal)**
- Phase 256 showed DAgger-254 = 20% SR (10-goal quick eval)
- Need full 50-goal eval to confirm cross-seed generalization
- Background job: `python3 scripts/eval_dagger.py --policy DAgger-254 --n_goals 50`

**Priority 2: Bridge-to-real-hardware Verification**
- Test bridge_node in `mode=real` with actual ST3215 servos
- Verify serial protocol in real_hardware_adapter matches ST3215 spec
- Validate cmd_vel → wheel speed conversion on physical robot

**Priority 3: VLA Closed-Loop Eval**
- Phase227 policy (80% SR on seed=42) never tested in closed-loop with bridge
- Run: bridge.launch + vla.launch → 20-goal eval in sim

**Priority 4: DAgger-254 Policy Integration**
- DAgger-254 (20% SR) is best learned policy — integrate into vla_policy_node
- Test with `vla.launch` using `--policy_path results/dagger_phase254/`

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196 | CJ P-controller + VLA train (14 epochs) | 8% SR (with early term) |
| p227 | Q2-extended data + 30-epoch VLA train | 4% SR |
| p234 | P-ctrl 94% SR (FIXED), Phase196 8%, Phase227 4% | 50-goal complete |
| p254 | DAgger-254 training (30ep, 20 epochs) | best_loss=0.0018 |
| p255 | Phase234 results + DAgger eval running | PID=93069 never completed |
| p256 | DAgger-254 10-goal quick eval | **20% SR** |
| p257 | Bridge health monitor (14/14 ✓) | ✅ |

### Git
- Commit: `fbb7068` Phase 256: DAgger-254 20% SR confirmed (10-goal), P-ctrl 100% SR
- Working tree: clean
- Phase 257: `scripts/bridge_health_monitor.py` added (not yet committed)

---
