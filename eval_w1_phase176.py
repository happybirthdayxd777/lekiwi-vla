#!/usr/bin/env python3
"""
Phase 176: Complete w1 sign analysis — all quadrants, 100 steps.
Key finding from quick run: VLA outputs NEGATIVE w1 for -Y goals (opposite of Phase 175 claim).
Now test all quadrants and compare displacement.
"""
import sys, torch, numpy as np
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF

policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load(
    '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase158_merged_jacobian_lr2e-05_ep10_20260419_0004/best_policy.pt',
    map_location='cpu', weights_only=False
)
loaded = ckpt.get('policy_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print('Policy loaded OK')

dummy_img = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

# All 4 quadrants
all_goals = [
    (0.3, -0.2), (0.5, -0.3), (0.4, -0.2),  # +X-Y
    (0.3,  0.2), (0.5,  0.3),                 # +X+Y
    (-0.3, 0.3),                              # -X+Y
    (-0.3,-0.3),                              # -X-Y
]

sim = LeKiWiSimURDF()
MAX_STEPS = 100
THRESHOLD = 0.1

def run_episode(goal_pos, use_w1_flip=False, verbose=False):
    sim.reset(target=goal_pos, seed=None)
    arrived_count = 0
    trajectory = []
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
        
        if use_w1_flip and goal_pos[1] < 0:
            action_np = action_np.copy()
            action_np[6] = -action_np[6]
        
        w1_values.append(action_np[6])
        sim.step(action_np)
        
        base_pos = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_pos - np.array(goal_pos))
        trajectory.append((step, base_pos[0], base_pos[1], dist, action_np[6], action_np[7], action_np[8]))
        
        if dist < THRESHOLD:
            arrived_count += 1
            if arrived_count >= 3:
                if verbose:
                    print(f'    Arrived at step {step}')
                return True, step + 1, dist, w1_values
        else:
            arrived_count = 0
    
    final_dist = float(np.linalg.norm(sim.data.qpos[:2] - np.array(goal_pos)))
    return False, MAX_STEPS, final_dist, w1_values

print('\n=== Phase 158 Policy: ALL QUADRANTS ===')
results = {}
for g in all_goals:
    print(f'\nGoal: {g}')
    for use_flip in [False, True]:
        label = "VLA+flip" if use_flip else "VLA    "
        s, st, d, w1s = run_episode(g, use_w1_flip=use_flip)
        w1_mean = np.mean(w1s)
        w1_pos = sum(1 for w in w1s if w > 0)
        w1_neg = sum(1 for w in w1s if w < 0)
        
        # Show first 5 and last 5 w1 values
        first5 = [f'{w:+.2f}' for w in w1s[:5]]
        last5 = [f'{w:+.2f}' for w in w1s[-5:]]
        print(f'  {label}: {"SUCC" if s else "FAIL"} steps={st:3d} dist={d:.3f}m '
              f'w1_mean={w1_mean:+.3f} +w1={w1_pos} -w1={w1_neg}')
        print(f'    first5: {first5}')
        print(f'    last5:  {last5}')
        results[(g, use_flip)] = (s, st, d, w1_mean, w1s)
    
    # P-controller comparison
    sim.reset(target=g, seed=None)
    arrived_count = 0
    for step in range(MAX_STEPS):
        obs = sim._obs()
        base_xy = obs["base_position"][:2]
        dist = np.linalg.norm(base_xy - np.array(g))
        kp = 2.0
        err = np.array(g) - base_xy
        vx = np.clip(err[0] * kp, -0.5, 0.5)
        vy = np.clip(err[1] * kp, -0.5, 0.5)
        R = 0.05
        w1 = (2*vx - 1.732*vy) / (3*R) * 0.1
        w2 = (2*vx + 1.732*vy) / (3*R) * 0.1
        w3 = (-2*vx) / (3*R) * 0.1
        action = np.array([0,0,0,0,0,0, w1, w2, w3])
        sim.step(action)
        if dist < THRESHOLD:
            arrived_count += 1
            if arrived_count >= 3:
                break
        else:
            arrived_count = 0
    p_succ = arrived_count >= 3
    p_dist = float(np.linalg.norm(sim.data.qpos[:2] - np.array(g)))
    print(f'  P-ctrl : {"SUCC" if p_succ else "FAIL"} steps={step+1} dist={p_dist:.3f}m')

print('\n=== DISPLACEMENT ANALYSIS ===')
print(f'{"Goal":<12} {"VLA dist":<12} {"VLA+flip dist":<14} {"VLA w1_mean":<12} {"flip w1_mean":<13}')
print('-'*65)
for g in all_goals:
    r0 = results.get((g, False), (None,)*5)
    r1 = results.get((g, True), (None,)*5)
    vla_d = f'{r0[2]:.3f}m' if r0[0] is not None else 'N/A'
    fl_d = f'{r1[2]:.3f}m' if r1[0] is not None else 'N/A'
    vla_w1 = f'{r0[3]:+.3f}' if r0[3] is not None else 'N/A'
    fl_w1 = f'{r1[3]:+.3f}' if r1[3] is not None else 'N/A'
    print(f'{str(g):<12} {vla_d:<12} {fl_d:<14} {vla_w1:<12} {fl_w1:<13}')
