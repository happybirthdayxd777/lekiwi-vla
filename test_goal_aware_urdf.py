#!/usr/bin/env python3
"""
Phase 104: Evaluate goal_aware_50ep policy on URDF sim
======================================================
CRITICAL FINDING: There exists a goal-aware 11D policy trained in Phase 16
(results/goal_aware_50ep/) that was never evaluated on URDF sim!

The policy was trained on lekiwi_goal_urdf_10k.h5 which has:
  - states: (N, 9) = arm_pos(6) + wheel_vel(3)
  - goal_positions: (N, 2) = goal x, y
  - Policy builds 11D state = [arm_pos(6) + wheel_vel(3) + goal_xy(2)]

This is DIFFERENT from the Phase 63 policy (state_dim=9, no goal) which got SR=0%.

Test with URDF sim (sim_lekiwi_urdf.py) which has proper k_omni locomotion.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.expanduser('~/hermes_research/lekiwi_vla'))
os.chdir(os.path.expanduser('~/hermes_research/lekiwi_vla'))

from PIL import Image
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF

print("=" * 60)
print("Phase 104: Evaluate goal_aware_50ep on URDF sim")
print("=" * 60)

# Load goal-aware 11D policy
policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
ckpt_path = 'results/goal_aware_50ep/final_policy.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
loaded = ckpt.get('policy_state_dict', ckpt)
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print(f'Policy loaded: {ckpt_path}')
print(f'State dim: 11 (goal-aware: arm_pos6 + wheel_vel3 + goal_xy2)')

# Use URDF sim (has k_omni locomotion unlike LeKiWiSim)
sim = LeKiWiSimURDF()
print(f'URDF sim: k_omni locomotion enabled')

# Test goals - both +X and -X hemisphere
goals = [
    (0.5, 0.0), (0.3, 0.2), (0.4, -0.3), (0.2, 0.4),
    (-0.3, 0.2), (-0.4, -0.3), (-0.2, 0.1), (0.0, 0.5),
]
threshold = 0.1
max_steps = 200

results = []
for idx, g in enumerate(goals):
    sim.reset(target=g, seed=idx*10 + 42)  # deterministic seed for reproducibility
    goal_pos = np.array(g)
    
    arrived = False
    steps_at_goal = 0
    
    for step in range(max_steps):
        # Get observation
        obs = sim._obs()
        img = sim.render()
        
        # Resize image for CLIP (always use PIL to avoid numpy.resize vs PIL.resize ambiguity)
        img_arr = np.array(img)
        img_pil = Image.fromarray(img_arr.astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Build 11D state: arm_pos(6) + wheel_vel(3) + goal_xy(2)
        arm_pos = obs['arm_positions']
        wheel_v = obs['wheel_velocities']
        goal_norm = np.clip(goal_pos / 1.0, -1.0, 1.0)  # normalize to [-1, 1]
        state11d = np.concatenate([arm_pos, wheel_v, goal_norm]).astype(np.float32)
        state_t = torch.from_numpy(state11d).float().unsqueeze(0)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        
        obs = sim.step(action_np)

        base_body_id = sim.model.body('base').id
        base_pos = sim.data.xpos[base_body_id][:2].copy()
        dist = np.linalg.norm(base_pos - goal_pos)
        
        if dist < threshold:
            steps_at_goal += 1
            if steps_at_goal >= 3:
                arrived = True
                break
        else:
            steps_at_goal = 0
    
    final_dist = dist
    results.append((g, arrived, step+1, final_dist))
    status = "SUCCESS" if arrived else "FAIL"
    print(f'  {status} goal={g}: steps={step+1}, dist={final_dist:.3f}m')

sr = sum(r[1] for r in results) / len(results)
md = np.mean([r[3] for r in results])
print(f'\ngoal_aware_50ep (11D state, URDF sim, various goals):')
print(f'  SR: {sr*100:.0f}% ({sum(r[1] for r in results)}/{len(results)})')
print(f'  Mean dist: {md:.3f}m')