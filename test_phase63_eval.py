#!/usr/bin/env python3
"""Phase 64: Evaluate phase63 policy — quick smoke test."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi import LeKiwiSim
from PIL import Image

# Load phase63 policy with state_dim=9
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load('results/phase63_reachable_train/final_policy.pt', map_location='cpu', weights_only=False)
loaded = ckpt.get('policy_state_dict', ckpt)
policy.load_state_dict(loaded, strict=False)
policy.to('cpu').eval()
print('Policy loaded OK (state_dim=9)')

# Use primitive sim (faster)
sim = LeKiwiSim()
sim.reset()

# +X hemisphere goals only (reachable)
goals = [(0.3, 0.2), (0.5, 0.0), (0.4, 0.3), (0.2, -0.2)]

results = []
for g in goals:
    sim.reset()
    goal_pos = np.array(g)
    if hasattr(sim, 'set_target'):
        sim.set_target(goal_pos)
    
    arrived = False
    steps_at_goal = 0
    threshold = 0.15
    max_steps = 100
    
    for step in range(max_steps):
        img = sim.render()
        # Handle both PIL Image and np.ndarray
        if hasattr(img, 'resize'):
            img_pil = img.resize((224, 224), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
        else:
            img_arr = np.array(img)
            img_pil = Image.fromarray(img_arr.astype(np.uint8)).resize((224, 224), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # LeKiwiSim state: qpos[0:6]=arm, qpos[6:9]=wheel, qvel[6:9]=wheel_vel
        arm_pos = sim.data.qpos[0:6]
        wheel_v = sim.data.qvel[6:9]
        state9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        state_t = torch.from_numpy(state9d).float().unsqueeze(0)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        
        sim.step(action_np)
        base_pos = sim.data.qpos[6:8].copy()
        dist = np.linalg.norm(base_pos - goal_pos)
        
        if dist < threshold:
            steps_at_goal += 1
            if steps_at_goal >= 3:
                arrived = True
                break
        else:
            steps_at_goal = 0
    
    final_dist = np.linalg.norm(sim.data.qpos[6:8] - goal_pos)
    results.append((g, arrived, step+1, final_dist))
    status = "SUCCESS" if arrived else "FAIL"
    print(f'  {status} goal={g}: steps={step+1}, dist={final_dist:.3f}m')

sr = sum(r[1] for r in results) / len(results)
md = np.mean([r[3] for r in results])
print(f'\nPhase63 (9D state, primitive sim, +X goals):')
print(f'  SR: {sr*100:.0f}% ({sum(r[1] for r in results)}/{len(results)})')
print(f'  Mean dist: {md:.3f}m')
