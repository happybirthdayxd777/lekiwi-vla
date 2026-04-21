# LeKiWi ROS2-MuJoCo Platform Progress

---
## [Phase 256 - 2026-04-21 20:30 CST] — DAgger-254 Quick Eval: 20% SR, Phase255 Uncommitted

### 🔍 Current Architecture Status

```
ROS2 Bridge (lekiwi_ros2_bridge/):
  ✅ bridge_node.py (1186 lines) — cmd_vel↔MuJoCo, joint_states↔ROS2
  ✅ vla_policy_node.py (746 lines) — CLIP-FM policy at 4 Hz
  ✅ ctf_integration.py (797 lines) — CTF security mode
  ✅ Launch files: full, vla, bridge, real_mode, ctf — all present

Simulation backends:
  ✅ Primitive (cylinder) + URDF (STL mesh) — both working
  ✅ LeKiWiSimLoader factory verified

VLA Policies:
  ✅ Phase196 epoch_14.pt — 8% SR (with early termination)
  ✅ Phase227 epoch_30.pt — 4% SR (worse despite Q2-extended)
  ✅ DAgger-254 best_policy.pt — 20% SR (better than both VLAs)
  ✅ P-controller CJ kP=2.0 — 100% SR oracle baseline (10-goal quick test)

Training Data:
  ✅ phase178_symmetrized.h5 — 3312 clean frames
  ✅ phase196_clean_50ep.h5 — 50 episodes
  ✅ phase227_extended_65ep.h5 — 65 episodes Q2-extended
  ✅ dagger_phase254_30ep.h5 — 30 episodes DAgger data

Pending:
  ⏳ Phase255 commit not pushed: "Phase234 results + DAgger-254 eval running"
```

### ✅ 本次心跳完成

**1. DAgger-254 Quick Eval (10-goal, sr=0.10m)**

| Policy | SR | Mean Final Dist | Notes |
|--------|-----|-----------------|-------|
| P-controller CJ kP=2.0 | **100%** | 0.098m | Oracle baseline |
| DAgger-254 | **20%** | 2.377m | Better than Phase227 VLA (4%) |

**Key finding: DAgger improves over pure VLA training but still far below P-controller**

**2. Phase255 Commit Verified**
- `f970b41 Phase 255: Phase234 results (VLA collapse), DAgger-254 eval running`
- Branch is clean, up to date with origin
- Phase255 claimed DAgger-254 eval was running (PID=93069) — eval never completed but is now confirmed at 20% SR

**3. Git Status Confirmed**
```
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

### 🔍 DAgger-254 Analysis

DAgger-254 (20% SR) outperforms:
- Phase196 VLA: 8% SR
- Phase227 VLA: 4% SR

But remains far below P-controller (100% SR). Root cause:
- DAgger was trained on 30ep with 3832 DAgger frames + 5562 base frames
- Best loss: 0.0018 at epoch 20
- Still learns aggressive overshoot behavior inherited from VLA base

### 🧭 下次心跳（Phase 257）

**Priority 1: Architecture Decision**
- DAgger-254 = 20% SR is the best learned policy so far
- But P-controller = 100% SR — the gap is enormous
- The P-controller IS the practical policy for this platform

**Priority 2: ROS2 Bridge Verification**
- Verify bridge_node.py connects to real ROS2 topics
- Test `/lekiwi/cmd_vel` → MuJoCo action pipeline

**Priority 3: Commit + Push**
```bash
cd ~/hermes_research/lekiwi_vla && git add -A && git commit -m "Phase 256: DAgger-254 20% SR confirmed, P-ctrl 100%" && git push origin main
```

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196 | CJ P-controller + VLA train (14 epochs) | 8% SR (with early term) |
| p227 | Q2-extended data + 30-epoch VLA train | 4% SR |
| p234 | P-ctrl 94% SR (FIXED), Phase196 8%, Phase227 4% | 50-goal complete |
| p254 | DAgger-254 training (30ep, 20 epochs) | best_loss=0.0018 |
| p255 | Phase234 results + DAgger eval running | PID=93069 never completed |
| p256 | DAgger-254 10-goal quick eval | **20% SR** |

### Git
- Commit: `f970b41` Phase 255: Phase234 results (VLA collapse), DAgger-254 eval running
- Branch: main
- Status: clean, up to date with origin

---