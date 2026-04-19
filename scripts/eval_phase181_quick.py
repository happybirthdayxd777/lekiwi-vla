#!/usr/bin/env python3
"""
Phase 181 Quick Eval: 5 episodes, 100 steps max — faster evaluation.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

DEVICE = 'cpu'
STATE_DIM = 9
ACTION_DIM = 9
HIDDEN = 512
MAX_STEPS = 100  # Reduced from 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 5   # Reduced from 20
SEED = 42

CKPT_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_vision_train/best_policy.pt'

def load_policy():
    print(f"[Policy] Loading from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    
    config = ckpt.get('policy_config', {
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'hidden': HIDDEN
    })
    
    policy = CLIPFlowMatchingPolicy(
        state_dim=config.get('state_dim', STATE_DIM),
        action_dim=config.get('action_dim', ACTION_DIM),
        hidden=config.get('hidden', HIDDEN),
        device=DEVICE
    )
    
    if 'flow_head_state_dict' in ckpt:
        policy.flow_head.load_state_dict(ckpt['flow_head_state_dict'])
    elif 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'])
    else:
        raise ValueError(f"Unknown checkpoint keys: {list(ckpt.keys())}")
    
    print(f"[Policy] Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.6f}")
    return policy

def evaluate_policy(policy, n_episodes=5, seed=42):
    """Evaluate VLA on random goals."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = LeKiWiSimURDF()
    
    results = {
        'total': 0, 'success': 0,
        'steps': [], 'final_dists': [],
        'details': []
    }
    
    for ep in range(n_episodes):
        goal = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])
        
        sim.reset()
        for _ in range(15):
            sim.step([0]*9)
        
        obs = sim._obs()
        base_xy = sim.data.qpos[:2].copy()
        
        success = False
        steps = 0
        dists = []
        
        for step in range(MAX_STEPS):
            obs = sim._obs()
            state = np.concatenate([
                obs['arm_positions'],
                obs['wheel_velocities']
            ]).astype(np.float32)
            
            state[0:6] = np.clip(state[0:6] / 2.0, -1, 1)
            state[6:9] = np.clip(state[6:9] / 0.5, -1, 1)
            
            img = sim.render().astype(np.float32) / 255.0
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            img_t = F.interpolate(img_t, size=(224, 224), mode='bilinear', align_corners=False)
            state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().detach().numpy()[0]
            action_np = np.clip(action_np, -0.5, 0.5)
            
            sim.step(action_np)
            
            base_xy = sim.data.qpos[:2].copy()
            dist = np.linalg.norm(base_xy - goal)
            dists.append(dist)
            
            if dist < GOAL_THRESHOLD:
                success = True
                steps = step + 1
                break
        
        if not success:
            steps = MAX_STEPS
        
        final_dist = dists[-1] if dists else 999
        
        results['total'] += 1
        results['success'] += 1 if success else 0
        results['steps'].append(steps)
        results['final_dists'].append(final_dist)
        
        results['details'].append({
            'ep': ep, 'goal': goal.tolist(),
            'success': success, 'steps': steps, 'final_dist': final_dist
        })
        
        print(f"  Ep {ep}: goal=[{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={final_dist:.3f})")
    
    sr_total = results['success'] / results['total'] * 100
    mean_steps = np.mean(results['steps'])
    mean_dist = np.mean(results['final_dists'])
    
    print(f"\n=== Phase 181 Quick Eval ===")
    print(f"Policy: {CKPT_PATH}")
    print(f"Overall SR: {results['success']}/{results['total']} = {sr_total:.1f}%")
    print(f"Mean steps: {mean_steps:.1f}, Mean final dist: {mean_dist:.3f}m")
    
    return results

if __name__ == '__main__':
    print(f"Phase 181 Quick Eval: {N_EPISODES} ep, {MAX_STEPS} steps max")
    
    policy = load_policy()
    policy.eval()
    
    t0 = time.time()
    results = evaluate_policy(policy, n_episodes=N_EPISODES, seed=SEED)
    elapsed = time.time() - t0
    
    out_path = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_quick_eval.json'
    with open(out_path, 'w') as f:
        json.dump({
            'phase': 181,
            'policy_path': CKPT_PATH,
            'episodes': N_EPISODES,
            'max_steps': MAX_STEPS,
            'sr_total': results['success'] / results['total'],
            'mean_steps': float(np.mean(results['steps'])),
            'mean_final_dist': float(np.mean(results['final_dists'])),
            'details': results['details'],
            'elapsed': elapsed
        }, f, indent=2)
    
    print(f"\nResults saved: {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")
