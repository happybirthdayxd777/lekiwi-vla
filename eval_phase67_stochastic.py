#!/usr/bin/env python3
"""Phase 67: Stochastic eval of phase63 policy — test true diversity.
Test: Does policy achieve different outcomes across episodes with random seeds?
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF

# Load phase63 policy with state_dim=9
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load('/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase63_reachable_train/final_policy.pt',
                  map_location='cpu', weights_only=False)
loaded = ckpt.get('policy_state_dict', ckpt)
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print('Policy loaded OK (phase63_reachable_train, state_dim=9)')

# ── LeKiWiSimURDF with stochastic reset ──────────────────────────────────────
sim = LeKiWiSimURDF()

# +X hemisphere goals only (reachable)
goals = [(0.5, 0.0), (0.3, 0.2), (0.4, 0.3)]  # Quick: 3 goals

NUM_SEEDS = 3   # 3 seeds per goal — quick diversity check
MAX_STEPS = 60   # Quick: 60 steps enough to see locomotion
THRESHOLD = 0.1

results = []
all_dists_by_goal = {}

for g in goals:
    goal_pos = np.array(g)
    goal_results = []
    final_dists = []
    
    for seed_idx in range(NUM_SEEDS):
        seed = seed_idx * 1000 + 42  # e.g., 42, 1042, 2042, 3042, 4042
        
        # Reset with unique seed → different initial perturbation each episode
        sim.reset(target=goal_pos, seed=seed)
        
        arrived = False
        steps_at_goal = 0
        
        for step in range(MAX_STEPS):
            img = sim.render()
            # Handle numpy array directly (640, 480, 3) uint8, resize to 224x224 for CLIP
            img_np = np.array(img, dtype=np.float32) / 255.0
            # Resize HWC (640,480,3) → NCHW (1,3,224,224), keep batch dim for policy
            img_bchw = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,640,480)
            img_t = F.interpolate(img_bchw, size=(224, 224), mode='bilinear', align_corners=False).float()
            
            # State: arm_pos(6) + wheel_vel(3) = 9D
            obs = sim._obs()
            arm_pos = obs["arm_positions"]
            wheel_v = obs["wheel_velocities"]
            state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
            state_t = torch.from_numpy(state9d).float().unsqueeze(0)
            
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
            
            sim.step(action_np)
            
            base_pos = sim.data.qpos[:2].copy()
            dist = np.linalg.norm(base_pos - goal_pos)
            
            if dist < THRESHOLD:
                steps_at_goal += 1
                if steps_at_goal >= 3:
                    arrived = True
                    break
            else:
                steps_at_goal = 0
        
        final_dist = float(np.linalg.norm(sim.data.qpos[:2] - goal_pos))
        goal_results.append((seed, arrived, step+1, final_dist))
        final_dists.append(round(final_dist, 3))
        print(f'  goal={g} seed={seed}: {"SUCCESS" if arrived else "FAIL"}, steps={step+1}, dist={final_dist:.3f}m')
    
    results.append((g, goal_results))
    all_dists_by_goal[g] = final_dists
    
    # Diversity check for this goal
    unique_dists = len(set(final_dists))
    if unique_dists > 1:
        print(f'    → DIVERSITY: {unique_dists} distinct outcomes for goal={g}')
    else:
        print(f'    → SAME OUTPUT for goal={g} (all dists={final_dists})')

# ── Summary ─────────────────────────────────────────────────────────────────
print('\n=== Phase 67 Stochastic Eval Summary ===')
for g, gresults in results:
    success_count = sum(r[1] for r in gresults)
    mean_steps = float(np.mean([r[2] for r in gresults]))
    mean_dist = float(np.mean([r[3] for r in gresults]))
    sr = success_count / NUM_SEEDS * 100
    print(f'  goal={g}: SR={sr:.0f}% ({success_count}/{NUM_SEEDS}), mean_steps={mean_steps:.0f}, mean_dist={mean_dist:.3f}m')

overall_sr = float(np.mean([sum(r[1] for r in gresults)/NUM_SEEDS for _, gresults in results])) * 100
print(f'\n  Overall SR: {overall_sr:.1f}%')

# Final diversity check
total_diverse = 0
for g, gresults in results:
    final_dists = [round(r[3], 3) for r in gresults]
    if len(set(final_dists)) > 1:
        total_diverse += 1

if total_diverse == len(goals):
    print(f'  ✓ DIVERSITY CONFIRMED: All {len(goals)} goals show different outcomes across seeds')
elif total_diverse > 0:
    print(f'  ⚠ PARTIAL DIVERSITY: {total_diverse}/{len(goals)} goals show seed-dependent outcomes')
    for g, gresults in results:
        final_dists = [round(r[3], 3) for r in gresults]
        if len(set(final_dists)) <= 1:
            print(f'    SAME OUTPUT: goal={g}, dists={final_dists}')
else:
    print(f'  ✗ DETERMINISTIC STILL: All goals produce identical outputs across seeds')
    for g, gresults in results:
        final_dists = [round(r[3], 3) for r in gresults]
        print(f'    dists={final_dists}')
