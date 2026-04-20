#!/usr/bin/env python3
"""
Diagnostic: Systematic VLA failure mode analysis
Identifies whether +X/-Y is specifically problematic and why.
"""
import h5py, numpy as np, json, os, sys

os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

print("=" * 70)
print("LEKIWI VLA FAILURE MODE DIAGNOSTIC — Phase 222")
print("=" * 70)

# ── 1. Load training data ──────────────────────────────────────────────────────
f = h5py.File('data/phase196_clean_50ep.h5', 'r')
goals = f['goals'][:]
episode_starts = f['episode_starts'][:]
actions = f['actions'][:]
states = f['states'][:]
n_steps = len(goals)
n_eps_raw = len(episode_starts)
n_actual_eps = n_eps_raw - 1

# Episode boundaries
ep_lengths = []
for i in range(n_actual_eps):
    length = int(episode_starts[i+1]) - int(episode_starts[i])
    ep_lengths.append(length)
ep_lengths = np.array(ep_lengths)

# Unique goal per episode
ep_goals = np.array([goals[int(episode_starts[i])] for i in range(n_actual_eps)])
egx, egy = ep_goals[:,0], ep_goals[:,1]

# Episode-level quadrant mask
egx, egy = ep_goals[:,0], ep_goals[:,1]
quadrant_ep = np.where(
    (egx >= 0) & (egy >= 0), 0,
    np.where((egx >= 0) & (egy < 0), 1,
    np.where((egx < 0) & (egy >= 0), 2, 3))
)
quad_names = ['+X/+Y', '+X/-Y', '-X/+Y', '-X/-Y']

print(f"\nDataset: {n_steps} steps, {n_actual_eps} episodes")
print(f"Training epochs: 14 (phase196_contact_jacobian_train/epoch_14.pt)")
print()

# ── 2. Quadrant coverage in training ──────────────────────────────────────────
print("Training episode coverage per quadrant:")
for qi, qname in enumerate(quad_names):
    n_eps = (quadrant_ep == qi).sum()
    print(f"  {qname}: {n_eps} episodes ({n_eps/n_actual_eps*100:.0f}%)")

# ── 3. Wheel action diversity analysis ────────────────────────────────────────
print(f"\n" + "-" * 70)
print("WHEEL ACTION DIVERSITY (P-controller training data)")
print("-" * 70)
print(f"\nActions shape: {actions.shape} (9-dim: 3 arm + 6 wheel)")
# Wheel actions (cols 3:9)
for qi, qname in enumerate(quad_names):
    mask = quadrant_ep == qi
    # Find step indices for this quadrant by checking which episode they belong to
    step_quadrant = np.zeros(n_steps, dtype=int)
    for i in range(n_actual_eps):
        start = int(episode_starts[i])
        end = int(episode_starts[i+1]) if i+1 < n_eps_raw else n_steps
        step_quadrant[start:end] = quadrant_ep[i]
    mask = step_quadrant == qi
    wa = actions[mask][:, 3:9]
    print(f"  {qname}:")
    print(f"    mean = {wa.mean(axis=0).round(4)}")
    print(f"    std  = {wa.std(axis=0).round(4)}")
    print(f"    min  = {wa.min(axis=0).round(4)}")
    print(f"    max  = {wa.max(axis=0).round(4)}")

# ── 4. Key finding: ALL wheel actions are nearly constant! ─────────────────────
print(f"\n" + "-" * 70)
print("KEY FINDING: P-controller training data has near-constant wheel velocity")
print("-" * 70)
print("""
The P-controller data was collected with FIXED open-loop wheel velocity [0.5, 0, 0].
The P-controller's error feedback is in the ARM joints, not the wheel velocity!
This means the VLA has almost NO wheel velocity diversity to learn from.

When the VLA outputs wheels, it is mostly learning WHEN to stop/turn
(not how to vary wheel speed during movement).
""")

# ── 5. Evaluate the +X/-Y hypothesis ──────────────────────────────────────────
print(f"-" * 70)
print("FAILED GOAL GEOMETRY ANALYSIS")
print("-" * 70)

failed = [
    ("p218 G7",  0.274, -0.144, 0.310),
    ("p218b G2", 0.269, -0.234, 0.357),
    ("p218b G5", 0.393, -0.247, 0.464),
]

# Find similar training goals in +X/-Y
plusX_minusY_ep_goals = ep_goals[(egx >= 0) & (egy < 0)]
print(f"\nTraining +X/-Y goals (n={len(plusX_minusY_ep_goals)}):")
for g in plusX_minusY_ep_goals:
    d = np.linalg.norm(g)
    ratio = g[1] / g[0]  # Y/X ratio
    print(f"  ({g[0]:+.3f}, {g[1]:+.3f}) dist={d:.3f}m Y/X={ratio:.3f}")

print(f"\nFailed eval goals:")
for name, gx_, gy_, dist_ in failed:
    ratio = gy_ / gx_
    # Find closest training goal
    dists = np.linalg.norm(plusX_minusY_ep_goals - [gx_, gy_], axis=1)
    closest_idx = dists.argmin()
    closest = plusX_minusY_ep_goals[closest_idx]
    closest_dist = dists[closest_idx]
    print(f"  {name}: ({gx_:+.3f}, {gy_:+.3f}) Y/X={ratio:.3f}")
    print(f"    Closest training goal: ({closest[0]:+.3f}, {closest[1]:+.3f}) dist={closest_dist:.3f}m")

# ── 6. Y/X ratio analysis ─────────────────────────────────────────────────────
print(f"\n" + "-" * 70)
print("Y/X RATIO — THE TRUE FAILURE DIMENSION")
print("-" * 70)

print(f"\nFailed goals Y/X ratios:")
for name, gx_, gy_, dist_ in failed:
    ratio = gy_ / gx_
    print(f"  {name}: Y/X={ratio:.3f} (need to move leftward {(abs(ratio)*100):.0f}% as much as forward)")

print(f"\nTraining +X/-Y goals Y/X ratios:")
ratios = plusX_minusY_ep_goals[:, 1] / plusX_minusY_ep_goals[:, 0]
for i, (g, r) in enumerate(zip(plusX_minusY_ep_goals, ratios)):
    print(f"  ep{i}: ({g[0]:+.3f}, {g[1]:+.3f}) Y/X={r:.3f}")

print(f"\nFailed goals Y/X range: [{min(gy_/gx_ for _,gx_,gy_,__ in failed):.3f}, {max(gy_/gx_ for _,gx_,gy_,__ in failed):.3f}]")
print(f"Training +X/-Y Y/X range: [{ratios.min():.3f}, {ratios.max():.3f}]")
