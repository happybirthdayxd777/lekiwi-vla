# Phase 255 — DAgger-254 Eval Running + Phase234 Results Analysis

## [2026-04-21 19:00 CST] — Phase234 Full Eval Results + DAgger-254 50-Goal Eval Running

### 🔴 Phase234 Final Results: VLA Collapse Confirmed

**Phase234 eval (50 goals, 200 steps, sr=0.10m, early termination) — COMPLETED:**

| Policy | SR | Mean Final Dist | Status |
|--------|-----|----------------|--------|
| P-controller CJ kP=2.0 | **94.0%** | 0.154m | ✅ Oracle baseline |
| VLA Phase196 (e14) | **8.0%** | 0.555m | 🔴 Catastrophic |
| VLA Phase227 (e30) | **4.0%** | 0.693m | 🔴 Worse than Phase196 |

**Q2 (gx<0, gy>0) quadrant — VLA complete failure:**
```
Phase196 Q2: 0/15 = 0.0%  (goals all within training range)
Phase227 Q2: 0/15 = 0.0%  (Q2-extended data didn't help)
```

**Root Cause: VLA cannot stabilize at goal**
- VLA converges transiently but exits success radius → early termination fires
- P-controller: reaches goal, decelerates, stays within radius → success maintained
- Phase234 confirmed: all VLA failures are CATASTROPHIC overshoot (0.5m to 1.3m away)

### ✅ 已完成

**1. Phase234 Full Eval — DONE (72.1 min)**
- 50-goal eval with early termination
- P-controller = 94% (was 0% in Phase233 due to missing early termination bug)
- Phase196 VLA = 8% (vs 68% in Phase224 WITHOUT early termination)
- Phase227 VLA = 4% (WORSE than Phase196 despite Q2-extended data)

**2. DAgger-254 Training — COMPLETE (already in results/dagger_phase254_train/)**
- 30 episodes DAgger data: 3832 frames (63% expert, 37% VLA)
- Base data: phase196_clean_50ep.h5 (5562 frames)
- 20 epochs training, best loss = 0.0018
- best_policy.pt saved at epoch 20 (best loss)

**3. Phase255 Commit: eval_dagger.py DAgger-254 support**
```
00bde5e Phase255: Add DAgger-254 eval support in eval_dagger.py
55467dd Phase 252: DAgger pipeline v2 — best checkpoint + 50ep data + Phase227 best base
```

**4. Phase254 Training Results Committed**
```
af4ab4b Phase 254: DAgger-254 training complete (30ep, 20 epochs, best_loss=0.0018)
```

**5. DAgger-254 50-Goal Eval — STARTED (background, ~70 min)**
```
PID=93069: /opt/miniconda3/bin/python3 scripts/eval_dagger.py --n_goals 50 --seed 42
```
- Evaluates: P-controller, Phase227 VLA, DAgger-254
- Results → results/dagger_phase252_eval/eval_results.json

### 🔍 VLA Failure Architecture Analysis

```
Training data (phase196_clean_50ep.h5):
  - 5562 steps from 50 episodes
  - Goal distance range: 0.038m to 0.371m
  - P-controller reached goals efficiently, then stayed

What VLA learned:
  - Converge toward goal (reaches transiently)
  - NO deceleration/stabilization behavior
  - Overshoots catastrophically when near goal

Why Phase227 made it WORSE:
  - Q2-extended data had MORE oscillation patterns
  - VLA learned to overshoot more aggressively
  - Phase227 SR = 4% < Phase196 SR = 8%
```

### 🧭 下次心跳（Phase 256）

**Priority 1: Wait for DAgger-254 Eval Results**
- Check results/dagger_phase252_eval/eval_results.json (PID=93069)
- Expected: ~60-70 min from eval start
- Compare DAgger-254 SR vs Phase227 vs P-controller

**Priority 2: DAgger Failure Analysis**
- DAgger-246: 33% SR (15-goal, Phase250 eval)
- DAgger-254: ??? (50-goal, seed=42)
- If DAgger-254 < 50% SR → DAgger approach needs more data or architectural fix

**Priority 3: Git Push**
```bash
cd ~/hermes_research/lekiwi_vla && git push origin main
```

### 📊 Experiment Record

| Phase | Content | Result |
|-------|---------|--------|
| p196  | CJ P-controller + VLA train (14 epochs) | 68% SR (Phase224, 50-goal, no early term) |
| p227  | Q2-extended data + VLA train (30 epochs) | 4% SR (Phase234, 50-goal, sr=0.10m) |
| p234  | P-ctrl 94%, Phase196 8%, Phase227 4% | Complete (72 min) |
| p246  | DAgger pilot (5ep, 15-goal eval) | 33% SR (Phase250) |
| p252  | DAgger-252 (50ep, 20 epochs) | best_policy.pt saved |
| p254  | DAgger-254 (30ep, 20 epochs, best_loss=0.0018) | Best checkpoint saved |
| p255  | DAgger-254 eval running (50 goals, PID=93069) | Running ~70 min |

### Git

- Branch: main
- Last commit: `af4ab4b` Phase 254: DAgger-254 training complete
- Pending push: `00bde5e` Phase255: Add DAgger-254 eval support
- DAgger eval running: PID=93069

