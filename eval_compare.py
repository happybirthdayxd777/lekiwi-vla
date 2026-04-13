#!/usr/bin/env python3
"""Compare goal_aware_50ep policy on primitive vs URDF sim."""
import sys, numpy as np, torch
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from scripts.improve_reward import TaskEvaluator
from sim_lekiwi import LeKiwiSim
from sim_lekiwi_urdf import LeKiWiSimURDF

policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load('/Users/i_am_ai/hermes_research/lekiwi_vla/results/goal_aware_50ep/final_policy.pt', map_location='cpu', weights_only=False)
loaded = ckpt.get('policy_state_dict', ckpt)
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print(f'Policy loaded OK')

goals = [(0.3, 0.2), (0.5, 0.0), (-0.3, 0.3), (0.2, -0.4), (0.4, 0.4)]

# LeKiwiSim primitive
sim_prim = LeKiwiSim()
eval_prim = TaskEvaluator(sim_prim, policy=policy, device='cpu')
print('\n=== LeKiwiSim (primitive) ===')
prim_results = []
for g in goals:
    s, st, d = eval_prim.reach_target(target=g, start=(0.0, 0.0), threshold=0.15, max_steps=200)
    prim_results.append((g, s, d))
    print(f'  goal={g}: success={s}, dist={d:.3f}m')

# LeKiwiSimURDF
sim_urdf = LeKiWiSimURDF()
eval_urdf = TaskEvaluator(sim_urdf, policy=policy, device='cpu')
print('\n=== LeKiWiSimURDF (STL mesh) ===')
urdf_results = []
for g in goals:
    s, st, d = eval_urdf.reach_target(target=g, start=(0.0, 0.0), threshold=0.15, max_steps=200)
    urdf_results.append((g, s, d))
    print(f'  goal={g}: success={s}, dist={d:.3f}m')

prim_sr = sum(s for _,s,_ in prim_results) / len(prim_results)
urdf_sr = sum(s for _,s,_ in urdf_results) / len(urdf_results)
prim_md = np.mean([d for _,_,d in prim_results])
urdf_md = np.mean([d for _,_,d in urdf_results])
print(f'\nPrimitive SR: {prim_sr*100:.0f}%, Mean dist: {prim_md:.3f}m')
print(f'URDF SR: {urdf_sr*100:.0f}%, Mean dist: {urdf_md:.3f}m')