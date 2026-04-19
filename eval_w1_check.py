#!/usr/bin/env python3
"""Phase 176: Quick 1-episode diagnostic."""
import sys, torch, numpy as np
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF
import time

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
sim = LeKiWiSimURDF()
goal = (0.3, -0.2)

t0 = time.time()
sim.reset(target=goal, seed=None)
print(f'reset took {time.time()-t0:.2f}s')

t0 = time.time()
for step in range(100):
    obs = sim._obs()
    arm_pos = obs["arm_positions"]
    wheel_v = obs["wheel_velocities"]
    state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
    state_t = torch.from_numpy(state9d).float().unsqueeze(0)
    with torch.no_grad():
        action = policy.infer(dummy_img, state_t, num_steps=4)
    action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
    sim.step(action_np)
print(f'100 steps took {time.time()-t0:.2f}s')
base_pos = sim.data.qpos[:2].copy()
dist = np.linalg.norm(base_pos - np.array(goal))
print(f'Final dist: {dist:.3f}m, w1 final: {action_np[6]:+.3f}')
