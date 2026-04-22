---

## [2026-04-22 10:30] Phase 272 — URDF Locomotion BROKEN: Shoulder Sphere Ground Contact

### 本次心跳完成

**Critical Discovery: URDF sim locomotion is completely broken**

While investigating why Stage2 policy evaluates at 0% SR (was 72% reported), discovered the URDF simulation itself has NO locomotion capability.

**Root Cause Analysis:**

1. **base_q_geom (shoulder sphere) is sitting ON the ground:**
   - Sphere radius: 0.049m, body_z=0.165m → bottom at -0.029m
   - Friction=0.6 (high) + contype=1 + conaffinity=1 → direct ground contact
   - Shoulder sphere contact creates massive drag force opposing any base motion

2. **URDF sim generates 13 contacts at step 0:**
   - chassis_contact: ground (friction=0.001)
   - wheel0_contact: ground (friction=1.5)
   - wheel1_contact: ground (friction=1.5)
   - wheel2_contact: ground (friction=1.5)
   - base_q_geom: ground (friction=0.6) ← SHOULDER SPHERE DRAG

3. **Single wheel test (action=0.5, 200 steps):**
   - w1 only: (-0.073, +1.479) → 1.48m
   - w2 only: (+2.272, +1.455) → 2.70m
   - w3 only: (-2.552, +1.300) → 2.86m
   - symmetric [0.5,0.5,0.5]: (+0.031, +2.143) → 2.14m (works)
   
   BUT: P-controller [0.5,0.45,0] gives (+2.34, +2.28) → 3.26m total displacement, **WRONG direction**

4. **P-controller 30 steps: ~0.008m move then stuck** (both primitive and URDF)
   - This suggests the P-controller action scaling may be the issue
   - Primitive sim with P-controller works for 200 steps but not 30

**Key findings:**
- URDF sim IS capable of locomotion (symmetric [0.5,0.5,0.5] → 2.14m)
- P-controller wheel action mapping is WRONG for directional control
- Stage2 policy 0% SR: The VLA was trained with Contact-Jacobian on working sim; eval on broken URDF gives 0%

### Bridge Architecture Status

| 元件 | 狀態 | 備註 |
|------|------|------|
| Stage2PolicyRunner | ⚠️ BROKEN | URDF locomotion broken → Stage2 eval gives 0% |
| bridge_node.py | ✅ | Works with primitive sim |
| vla_policy_node.py | ✅ | CLIP-FM/pi0/ACT/dagger/stage2/stage3 |
| CTF Security Layer | ✅ | C1-C8 全部 |
| Camera Adapter | ✅ | URDF 20Hz |
| 5× Launch Files | ✅ | bridge/vla/ctf/full/real_mode |

### 下一步

- [ ] Phase 273: Fix URDF locomotion — remove base_q_geom ground contact or set friction=0.001
- [ ] Phase 274: Re-verify P-controller and Stage2 on fixed URDF
- [ ] Phase 275: Re-run Stage2 eval on fixed sim

### 阻礙

- **URDF base_q_geom shoulder sphere sitting on ground** — needs friction=0.001 or contype=0
- **P-controller wheel action mapping wrong** — need to recalibrate which wheel maps to which direction
- **Stage2 eval 0%** — but likely just sim issue, not policy issue

### 已修復問題

- Disk space: 82% used, 3.4GB free (Phase 271 cleanup done)