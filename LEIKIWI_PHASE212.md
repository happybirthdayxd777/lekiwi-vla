# Phase 212 — 2026-04-20 09:30 UTC

## VLA Wheel Amplitude Issue — Deep Analysis + Fix

### Problem Statement

VLA wheel actions are 3-6x smaller than P-controller optimal, causing hybrid bridge
to frequently fallback to P-controller. The VLA learns correct wheel direction but
underestimates magnitude.

### Root Cause Analysis

**Action Flow:**
```
Joint states (wheel_vel in rad/s ~[-0.5, 0.5])
  → VLA policy predict() → raw_action in [-1, 1]
  → normalize_action() → native_action in [-5, 5] for wheels (LEKIWI_WHEEL_LIMITS)
  → ActionSmoother (EMA, alpha=0.25, max_delta=0.8)
  → Published to /lekiwi/vla_action (now in [-5, 5] rad/s)
  → bridge_node._on_vla_action() clips to WHEEL_CTRL [-0.5, 0.5]
  → MuJoCo step(action)
```

**Key Finding: ActionSmoother has 0.8 rad/s max_delta at alpha=0.25**

The EMA smoothing formula: `smoothed += 0.25 * delta_clamped`

If policy outputs wheel_action = [0.5, 0.5, 0.5] rad/s (large forward command),
the first step change is clamped to 0.8 rad/s, then smoothed:
- Step 0: smoothed = [0, 0, 0] → delta=[0.5, 0.5, 0.5] → clamped=[0.5, 0.5, 0.5]
  smoothed = [0,0,0] + 0.25*[0.5,0.5,0.5] = [0.125, 0.125, 0.125]
- Step 1: smoothed=[0.125,0.125,0.125] → delta=[0.375,0.375,0.375]
  smoothed = [0.125,0.125,0.125] + 0.25*[0.375,0.375,0.375] = [0.218, 0.218, 0.218]

After 10 warmup steps, smoothed ≈ 0.5 * (1 - 0.75^10) / (1 - 0.75) ≈ 0.44 rad/s

But when VLA action magnitude is ~0.1 rad/s (typical for conservative policy),
it gets amplified by blending: VLA_mag=0.1 → blend=1-(0.1/0.15)=0.33 → P-controller 67% + VLA 33%.

**The VLA wheel actions are small (magnitude ~0.1-0.2) and then partially blended with
P-controller, resulting in only 33-50% of the intended wheel motion reaching the robot.**

### Bridge Hybrid Architecture (Current State)

```
bridge_node._on_cmd_vel():
  cmd_vel → twist_to_wheel_speeds() → wheel_speeds (no scale factor)
  
  if _vla_action_fresh:
    vla_wheel_raw = _last_action[6:9]  ← from VLA policy
    vla_mag = ||vla_wheel_raw||
    
    if vla_mag < 0.15:
      # VLA too conservative → blend with P-controller
      blend = 1.0 - vla_mag/0.15
      pctrl_ws = J_c_pinv @ (kP_FALLBACK * error)
      wheel_speeds = blend * pctrl_ws + (1-blend) * vla_wheel_raw
    else:
      wheel_speeds = vla_wheel_raw
  else:
    wheel_speeds = twist_to_wheel_speeds(vx*0.4, vy*0.4, wz)
```

### Fix Strategy

**Option A: Amplify VLA wheel actions before blending**
Add wheel_amplification_factor parameter (default 2.0-3.0) in bridge_node._on_vla_action
or vla_policy_node. Amplify wheel actions by 2-3x before blend.

**Option B: Increase WHEEL_CTRL limits**
Change WHEEL_CTRL_MIN/MAX from [-0.5, 0.5] to [-1.5, 1.5]. The URDF sim handles
±1.5 rad/s without NaN (Phase 74 found ±0.5 necessary but that may be outdated).

**Option C: Skip blending when VLA direction is correct**
Instead of magnitude-based fallback, use direction agreement: if VLA wheel direction
agrees with P-controller direction, use VLA fully (no blend).

### Recommended: Option C — Direction-Agreement Hybrid

```python
# In bridge_node._on_cmd_vel, replace magnitude-based blending:
def _wheel_direction_agrees(vla_wheels, pctrl_ws, threshold=0.5):
    """Check if VLA and P-controller wheel directions agree."""
    if np.linalg.norm(pctrl_ws) < 0.01:
        return False  # No P-controller reference
    vla_dir = vla_wheels / np.linalg.norm(vla_wheels)
    pctrl_dir = pctrl_ws / np.linalg.norm(pctrl_ws)
    return np.dot(vla_dir, pctrl_dir) > threshold  # cos(angle) > threshold

# Then:
if vla_mag < _HYBRID_WHEEL_FALLBACK_THRESHOLD:
    if _wheel_direction_agrees(vla_wheel_raw, pctrl_ws, threshold=0.5):
        # VLA direction correct → amplify and use VLA
        wheel_speeds = vla_wheel_raw * 2.5  # amplify by 2.5x
    else:
        # VLA direction wrong → use P-controller
        wheel_speeds = pctrl_ws
else:
    wheel_speeds = vla_wheel_raw  # VLA confident
```

### Next Steps

1. Implement Option C — direction-agreement hybrid (bridge_node.py)
2. Run validate_hybrid_bridge.py after fix
3. Test on actual ROS2 launch to verify closed-loop behavior

### Git
- Working dir: ~/hermes_research/lekiwi_vla
- Phase 212 commit: VLA direction-agreement hybrid bridge