#!/usr/bin/env python3
"""Phase 176: w1 behavior across all 4 quadrants -- quick diagnostic."""
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

dummy_img = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
sim = LeKiWiSimURDF()
MAX_STEPS = 80

def run(goal):
    sim.reset(target=goal, seed=None)
    w1s, w2s, w3s, dxs, dys = [], [], [], [], []
    for _ in range(MAX_STEPS):
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_v = obs["wheel_velocities"]
        state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        state_t = torch.from_numpy(state9d).float().unsqueeze(0)
        with torch.no_grad():
            action = policy.infer(dummy_img, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        sim.step(action_np)
        base = sim.data.qpos[:2].copy()
        w1s.append(action_np[6]); w2s.append(action_np[7]); w3s.append(action_np[8])
        dxs.append(base[0]); dys.append(base[1])
    final_d = np.linalg.norm(sim.data.qpos[:2] - np.array(goal))
    return {
        'goal': goal, 'final_dist': final_d,
        'w1_mean': np.mean(w1s), 'w2_mean': np.mean(w2s), 'w3_mean': np.mean(w3s),
        'w1_pos': sum(1 for w in w1s if w > 0), 'w1_neg': sum(1 for w in w1s if w < 0),
        'dx_range': max(dxs)-min(dxs), 'dy_range': max(dys)-min(dys),
        'final_x': dxs[-1], 'final_y': dys[-1],
    }

goals = [
    ('+X-Y', (0.3, -0.2)),
    ('+X+Y', (0.3, 0.2)),
    ('-X+Y', (-0.3, 0.3)),
    ('-X-Y', (-0.3, -0.3)),
]

print('\n=== Phase 158 VLA: w1 by quadrant ===')
for name, g in goals:
    r = run(g)
    print(name + ': dist=' + str(round(r['final_dist'],3)) + ' w1_mean=' + str(round(r['w1_mean'],3)) +
          ' w1_pos/neg=' + str(r['w1_pos']) + '/' + str(r['w1_neg']) +
          ' w2_mean=' + str(round(r['w2_mean'],3)) + ' w3_mean=' + str(round(r['w3_mean'],3)) +
          ' dx=' + str(round(r['dx_range'],3)) + ' dy=' + str(round(r['dy_range'],3)))

print('\n=== P-controller baseline ===')
for name, g in goals:
    sim.reset(target=g, seed=None)
    arrived = 0
    for step in range(MAX_STEPS):
        obs = sim._obs()
        base_xy = obs["base_position"][:2]
        dist = np.linalg.norm(base_xy - np.array(g))
        kp = 2.0
        err = np.array(g) - base_xy
        vx = np.clip(err[0]*kp, -0.5, 0.5)
        vy = np.clip(err[1]*kp, -0.5, 0.5)
        R = 0.05
        w1 = (2*vx - 1.732*vy)/(3*R)*0.1
        w2 = (2*vx + 1.732*vy)/(3*R)*0.1
        w3 = (-2*vx)/(3*R)*0.1
        sim.step(np.array([0,0,0,0,0,0,w1,w2,w3]))
        if dist < 0.1:
            arrived += 1
            if arrived >= 3:
                break
        else:
            arrived = 0
    final_d = np.linalg.norm(sim.data.qpos[:2] - np.array(g))
    print(name + ': ' + ('SUCC' if arrived >= 3 else 'FAIL') + ' dist=' + str(round(final_d,3)) + 'm')
