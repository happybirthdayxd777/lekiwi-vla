#!/usr/bin/env python3
"""
Phase 176 FAST: Test w1 sign correction — no rendering, just state-based inference.
Runs in ~30s instead of ~300s.
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF

# Load Phase 158 policy
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load(
    '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase158_merged_jacobian_lr2e-05_ep10_20260419_0004/best_policy.pt',
    map_location='cpu', weights_only=False
)
loaded = ckpt.get('policy_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print('Policy loaded OK')

# Pre-compute a dummy image (224x224x3) — same for all steps
# We'll use a black image since we only care about wheel action sign
dummy_img = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

# Key test: only +X-Y goals (the problem cases)
test_goals = [
    (0.3, -0.2), (0.5, -0.3), (0.4, -0.2),  # +X-Y (was 8% SR)
]

sim = LeKiWiSimURDF()
MAX_STEPS = 50
THRESHOLD = 0.1

def run_episode(goal_pos, use_w1_flip=False):
    sim.reset(target=goal_pos, seed=None)
    arrived_count = 0
    final_w1 = None
    w1_values = []
    
    for step in range(MAX_STEPS):
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_v = obs["wheel_velocities"]
        state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        state_t = torch.from_numpy(state9d).float().unsqueeze(0)
        
        with torch.no_grad():
            action = policy.infer(dummy_img, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        
        # Apply w1 sign flip for goal_y < 0
        if use_w1_flip and goal_pos[1] < 0:
            action_np = action_np.copy()
            action_np[6] = -action_np[6]
        
        w1_values.append(action_np[6])
        if step == MAX_STEPS - 1:
            final_w1 = action_np[6]
        
        sim.step(action_np)
        
        base_pos = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_pos - np.array(goal_pos))
        
        if dist < THRESHOLD:
            arrived_count += 1
            if arrived_count >= 3:
                return True, step + 1, dist, final_w1, w1_values
        else:
            arrived_count = 0
    
    final_dist = float(np.linalg.norm(sim.data.qpos[:2] - np.array(goal_pos)))
    return False, MAX_STEPS, final_dist, final_w1, w1_values

# Run both conditions for each goal
print('\n=== Testing +X-Y goals (3 episodes each) ===')
for g in test_goals:
    print(f'\nGoal: {g}')
    for use_flip in [False, True]:
        label = "VLA+flip" if use_flip else "VLA"
        s, st, d, w1, w1s = run_episode(g, use_w1_flip=use_flip)
        w1_mean = np.mean(w1s)
        w1_pos = sum(1 for w in w1s if w > 0)
        w1_neg = sum(1 for w in w1s if w < 0)
        print(f'  {label}: {"SUCC" if s else "FAIL"} steps={st} dist={d:.3f}m '
              f'w1_final={w1:+.3f} w1_mean={w1_mean:+.3f} '
              f'+w1={w1_pos} -w1={w1_neg}')
