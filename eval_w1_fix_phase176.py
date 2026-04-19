#!/usr/bin/env python3
"""
Phase 176: Test w1 sign correction hypothesis on Phase 158 policy.
Phase 175 found: VLA ALWAYS outputs positive w1, but -Y goals need NEGATIVE w1.
Policy: Phase 158 merged_jacobian (state_dim=9, action_dim=9, hidden=512)
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF

# Load Phase 158 policy (state_dim=9)
print("Loading Phase 158 policy...")
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load(
    '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase158_merged_jacobian_lr2e-05_ep10_20260419_0004/best_policy.pt',
    map_location='cpu', weights_only=False
)
loaded = ckpt.get('policy_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print('Policy loaded OK')

# Test goals — same as Phase 175 diagnostic
test_goals = [
    (0.3, -0.2), (0.5, -0.3), (0.4, -0.2), (0.5, -0.1), (0.2, -0.3),  # +X-Y (problematic)
    (0.3, 0.2),  (0.5, 0.3),                                        # +X+Y (control)
    (-0.3, 0.3), (-0.3, -0.3),                                      # -X+Y, -X-Y
]

sim = LeKiWiSimURDF()
MAX_STEPS = 100
THRESHOLD = 0.1

def run_episode(goal_pos, use_w1_flip=False):
    """Returns (success, steps, final_dist, final_w1)."""
    sim.reset(target=goal_pos, seed=None)
    arrived_count = 0
    final_w1 = None
    
    for step in range(MAX_STEPS):
        img = sim.render()
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_bchw = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_t = F.interpolate(img_bchw, size=(224, 224), mode='bilinear', align_corners=False).float()
        
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_v = obs["wheel_velocities"]
        # state_dim=9: arm_pos(6) + wheel_vel(3)
        state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        state_t = torch.from_numpy(state9d).float().unsqueeze(0)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        
        # Apply w1 sign flip for goal_y < 0
        if use_w1_flip and goal_pos[1] < 0:
            action_np = action_np.copy()
            action_np[6] = -action_np[6]
        
        if step == MAX_STEPS - 1:
            final_w1 = action_np[6]
        
        sim.step(action_np)
        
        base_pos = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_pos - np.array(goal_pos))
        
        if dist < THRESHOLD:
            arrived_count += 1
            if arrived_count >= 3:
                return True, step + 1, dist, final_w1
        else:
            arrived_count = 0
    
    final_dist = float(np.linalg.norm(sim.data.qpos[:2] - np.array(goal_pos)))
    return False, MAX_STEPS, final_dist, final_w1

# ── WITHOUT w1 flip ───────────────────────────────────────────────────────────
print('\n=== VLA WITHOUT w1 flip ===')
no_flip = []
for g in test_goals:
    s, st, d, w1 = run_episode(g, use_w1_flip=False)
    no_flip.append((g, s, st, d, w1))
    print(f'  {g}: {"SUCC" if s else "FAIL"} steps={st} dist={d:.3f}m w1={w1:+.3f}')

# ── WITH w1 flip ─────────────────────────────────────────────────────────────
print('\n=== VLA WITH w1 flip (goal_y < 0) ===')
flip = []
for g in test_goals:
    s, st, d, w1 = run_episode(g, use_w1_flip=True)
    flip.append((g, s, st, d, w1))
    print(f'  {g}: {"SUCC" if s else "FAIL"} steps={st} dist={d:.3f}m w1={w1:+.3f}')

# ── Summary ───────────────────────────────────────────────────────────────────
xy = [(0.3,-0.2),(0.5,-0.3),(0.4,-0.2),(0.5,-0.1),(0.2,-0.3)]
print('\n=== SUMMARY ===')
print(f'{"Goal":<15} {"NoFlip":<12} {"Flip":<12} {"Verdict"}')
print('-'*55)
for i, g in enumerate(test_goals):
    nf = f'SUCC/{no_flip[i][2]}' if no_flip[i][1] else f'FAIL d={no_flip[i][3]:.3f}'
    fl = f'SUCC/{flip[i][2]}' if flip[i][1] else f'FAIL d={flip[i][3]:.3f}'
    v = ''
    if g in xy and not no_flip[i][1] and flip[i][1]:
        v = '✓ FLIP FIXES'
    elif g in xy and no_flip[i][1] and not flip[i][1]:
        v = '✗ FLIP BREAKS'
    print(f'{str(g):<15} {nf:<12} {fl:<12} {v}')

xy_no = sum(1 for g,s,*_ in zip(xy, [r[1] for r in no_flip]) if s)
xy_fl = sum(1 for g,s,*_ in zip(xy, [r[1] for r in flip]) if s)
print(f'\n+X-Y SR: noflip={xy_no}/5 flip={xy_fl}/5')
all_no = sum(1 for _,s,*_ in no_flip)
all_fl = sum(1 for _,s,*_ in flip)
print(f'Overall: noflip={all_no}/9 flip={all_fl}/9')
