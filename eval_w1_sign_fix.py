#!/usr/bin/env python3
"""
Phase 176: Test w1 sign correction hypothesis.
Phase 175 found: VLA ALWAYS outputs positive w1, but -Y goals need NEGATIVE w1.
This script tests: does manually flipping w1 sign for goal_y < 0 fix +X-Y SR?

Architecture: state_dim=9, action_dim=9, hidden=512 (matches checkpoint).
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')

# Find best available policy
import glob, os

# Priority: phase158 (most recent), then any with high SR
POLICY_PATHS = [
    'results/phase158_merged_jacobian_lr2e-05_ep10_20260419_0004/best_policy.pt',
    'results/phase158_merged_jacobian_lr2e-05_ep10_20260418_1915/best_policy.pt',
    'results/phase154_sweep_lr2e-05_ep10_20260418_0754/best_policy.pt',
]

policy_path = None
for p in POLICY_PATHS:
    if os.path.exists(p):
        policy_path = p
        break

if policy_path is None:
    # Fallback: find any best_policy.pt
    candidates = glob.glob('results/**/best_policy.pt', recursive=True)
    if candidates:
        policy_path = sorted(candidates)[-1]

print(f'Using policy: {policy_path}')

# Load and inspect
ckpt = torch.load(policy_path, map_location='cpu', weights_only=False)
loaded = ckpt.get('policy_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

# Detect state_dim from flow_head
flow_key = 'flow_head.net.0.weight'
if flow_key in loaded:
    dim = loaded[flow_key].shape[1]  # 512 + state_dim + action_dim + timestep_emb
    # timestep_emb is always 256 from code
    # dim = 512 + state_dim + 9 + 256 = 777 + state_dim
    # For dim=786: state_dim = 9
    # For dim=788: state_dim = 11
    state_dim = dim - 777
    action_dim = loaded[flow_key].shape[0]
    print(f'flow_head input dim: {dim}, inferred state_dim={state_dim}')
else:
    state_dim = 9
    action_dim = 9

from scripts.train_task_oriented import CLIPFlowMatchingPolicy
policy = CLIPFlowMatchingPolicy(state_dim=state_dim, action_dim=action_dim, hidden=512, device='cpu')
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print(f'Policy loaded: state_dim={state_dim}, action_dim={action_dim}')

# ── Test Goals ──────────────────────────────────────────────────────────────
# +X-Y quadrant: the PROBLEM cases (VLA w1 wrong sign)
test_goals = [
    (0.3, -0.2),   # +X-Y ep00 (was FAIL)
    (0.5, -0.3),   # +X-Y ep07 (was FAIL)
    (0.4, -0.2),   # +X-Y ep08 (was FAIL)
    (0.5, -0.1),   # +X-Y ep10 (was FAIL)
    (0.2, -0.3),   # +X-Y ep12 (was FAIL)
    # +X+Y controls (should work either way)
    (0.3, 0.2),    # +X+Y ep01 (was SUCC)
    (0.5, 0.3),    # +X+Y ep03 (was SUCC)
    # -X+Y and -X-Y
    (-0.3, 0.3),   # -X+Y ep01
    (-0.3, -0.3),  # -X-Y ep02
]

from sim_lekiwi_urdf import LeKiWiSimURDF
sim = LeKiWiSimURDF()
MAX_STEPS = 100
THRESHOLD = 0.1

def run_vla_episode(goal_pos, use_w1_flip=False):
    """Run one episode with VLA policy. Returns (success, steps, final_dist)."""
    sim.reset(target=goal_pos, seed=None)
    arrived_count = 0
    
    for step in range(MAX_STEPS):
        img = sim.render()
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_bchw = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = F.interpolate(img_bchw, size=(224, 224), mode='bilinear', align_corners=False).float()
        
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_v = obs["wheel_velocities"]
        
        # state_dim may be 9 or 11
        if state_dim == 9:
            state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        else:
            state9d = np.concatenate([arm_pos, wheel_v, list(goal_pos)]).astype(np.float32)
        state_t = torch.from_numpy(state9d).float().unsqueeze(0)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        
        # Phase 176: Apply w1 sign correction for goal_y < 0
        if use_w1_flip and goal_pos[1] < 0:
            action_np = action_np.copy()
            action_np[6] = -action_np[6]  # Flip w1 (wheel0) sign
        
        sim.step(action_np)
        
        base_pos = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_pos - np.array(goal_pos))
        
        if dist < THRESHOLD:
            arrived_count += 1
            if arrived_count >= 3:
                return True, step + 1, dist
        else:
            arrived_count = 0
    
    final_dist = float(np.linalg.norm(sim.data.qpos[:2] - np.array(goal_pos)))
    return False, MAX_STEPS, final_dist

# ── VLA WITHOUT w1 flip ───────────────────────────────────────────────────────
print('\n=== VLA WITHOUT w1 sign correction ===')
no_flip_results = []
for g in test_goals:
    succ, steps, dist = run_vla_episode(g, use_w1_flip=False)
    no_flip_results.append((g, succ, steps, dist))
    status = "SUCC" if succ else "FAIL"
    print(f'  VLA goal={g}: {status}, steps={steps}, dist={dist:.3f}m')

# ── VLA WITH w1 flip ─────────────────────────────────────────────────────────
print('\n=== VLA WITH w1 sign correction (goal_y < 0) ===')
flip_results = []
for g in test_goals:
    succ, steps, dist = run_vla_episode(g, use_w1_flip=True)
    flip_results.append((g, succ, steps, dist))
    status = "SUCC" if succ else "FAIL"
    print(f'  VLA+flip goal={g}: {status}, steps={steps}, dist={dist:.3f}m')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n=== SUMMARY ===')
print(f'{"Goal":<15} {"VLA":<8} {"VLA+flip":<8} {"Delta":<10}')
print('-' * 45)

xy_problem = [(0.3,-0.2), (0.5,-0.3), (0.4,-0.2), (0.5,-0.1), (0.2,-0.3)]
for i, g in enumerate(test_goals):
    vs = "SUCC" if no_flip_results[i][1] else "FAIL"
    vf = "SUCC" if flip_results[i][1] else "FAIL"
    delta = "FIXED" if (flip_results[i][1] and not no_flip_results[i][1]) else ("IMPROVED" if flip_results[i][2] < no_flip_results[i][2] else "")
    print(f'{str(g):<15} {vs:<8} {vf:<8} {delta:<10}')

xy_no = sum(1 for g,s,*_ in no_flip_results if g in xy_problem and s)
xy_flip = sum(1 for g,s,*_ in flip_results if g in xy_problem and s)
xy_dist_no = np.mean([d for g,s,_,d in no_flip_results if g in xy_problem])
xy_dist_flip = np.mean([d for g,s,_,d in flip_results if g in xy_problem])

print(f'\n+X-Y SR: VLA={xy_no}/5, VLA+flip={xy_flip}/5')
print(f'+X-Y mean dist: VLA={xy_dist_no:.3f}m, VLA+flip={xy_dist_flip:.3f}m')

all_no = sum(1 for _,s,*_ in no_flip_results if s)
all_flip = sum(1 for _,s,*_ in flip_results if s)
print(f'\nOverall SR: VLA={all_no}/{len(test_goals)}, VLA+flip={all_flip}/{len(test_goals)}')
