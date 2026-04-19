#!/usr/bin/env python3
"""
Phase 181: Evaluate trained VLA on URDF sim with quadrant-based success rate.

Loads best_policy.pt from phase181_vision_train and evaluates on 20 random goals.
Reports per-quadrant SR to detect +Y/-Y bias.
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
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 20
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

def evaluate_policy(policy, n_episodes=20, seed=42):
    """Evaluate VLA on random goals, per-quadrant SR."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    sim = LeKiWiSimURDF()
    
    results = {
        'total': 0, 'success': 0,
        'quadrants': {
            '+X+Y': {'n': 0, 'success': 0},
            '+X-Y': {'n': 0, 'success': 0},
            '-X+Y': {'n': 0, 'success': 0},
            '-X-Y': {'n': 0, 'success': 0},
        },
        'steps': [],
        'final_dists': [],
        'details': []
    }
    
    for ep in range(n_episodes):
        # Random goal in [-0.5, 0.5] x [-0.5, 0.5]
        goal = np.array([
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5)
        ])
        
        # Determine quadrant (handle edge cases where goal is on axis boundary)
        if goal[0] >= 0 and goal[1] > 0:
            quad = '+X+Y'
        elif goal[0] >= 0 and goal[1] <= 0:
            quad = '+X-Y'
        elif goal[0] < 0 and goal[1] >= 0:
            quad = '-X+Y'
        else:
            quad = '-X-Y'
        
        sim.reset()
        # Warmup
        for _ in range(15):
            sim.step([0]*9)
        
        obs = sim._obs()
        base_xy = sim.data.qpos[:2].copy()
        
        success = False
        steps = 0
        dists = []
        
        for step in range(MAX_STEPS):
            obs = sim._obs()
            # State: arm_pos(6) + wheel_vel(3)
            state = np.concatenate([
                obs['arm_positions'],
                obs['wheel_velocities']
            ]).astype(np.float32)
            
            # Normalize state
            state[0:6] = np.clip(state[0:6] / 2.0, -1, 1)
            state[6:9] = np.clip(state[6:9] / 0.5, -1, 1)
            
            # Render image, resize to CLIP expected 224x224
            img = sim.render().astype(np.float32) / 255.0
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            img_t = F.interpolate(img_t, size=(224, 224), mode='bilinear', align_corners=False)
            state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
            
            # VLA inference
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().detach().numpy()[0]
            
            # Clip to valid range
            action_np = np.clip(action_np, -0.5, 0.5)
            
            # Apply to sim
            sim.step(action_np)
            
            # Check goal
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
        results['quadrants'][quad]['n'] += 1
        results['quadrants'][quad]['success'] += 1 if success else 0
        
        results['details'].append({
            'ep': ep, 'goal': goal.tolist(),
            'quadrant': quad, 'success': success,
            'steps': steps, 'final_dist': final_dist
        })
        
        print(f"  Ep {ep}: quad={quad:8s} goal=[{goal[0]:+.3f},{goal[1]:+.3f}] "
              f"→ {'SUCC' if success else 'FAIL'} ({steps} steps, dist={final_dist:.3f})")
    
    # Summary
    sr_total = results['success'] / results['total'] * 100
    mean_steps = np.mean(results['steps'])
    mean_dist = np.mean(results['final_dists'])
    
    print(f"\n=== Phase 181 VLA Evaluation ===")
    print(f"Policy: {CKPT_PATH}")
    print(f"Episodes: {results['total']}")
    print(f"Overall SR: {results['success']}/{results['total']} = {sr_total:.1f}%")
    print(f"Mean steps: {mean_steps:.1f}, Mean final dist: {mean_dist:.3f}m")
    print(f"\nPer-Quadrant SR:")
    for quad, data in results['quadrants'].items():
        if data['n'] > 0:
            sr = data['success'] / data['n'] * 100
            print(f"  {quad:8s}: {data['success']}/{data['n']} = {sr:.1f}%")
        else:
            print(f"  {quad:8s}: 0/0 = N/A")
    
    return results

def main():
    print(f"Phase 181: Vision VLA Evaluation")
    print(f"Device: {DEVICE}")
    print(f"Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")
    
    # Load policy
    policy = load_policy()
    policy.eval()
    
    # Evaluate
    t0 = time.time()
    results = evaluate_policy(policy, n_episodes=N_EPISODES, seed=SEED)
    elapsed = time.time() - t0
    
    # Save results
    out_path = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_eval.json'
    with open(out_path, 'w') as f:
        json.dump({
            'phase': 181,
            'policy_path': CKPT_PATH,
            'episodes': N_EPISODES,
            'max_steps': MAX_STEPS,
            'sr_total': results['success'] / results['total'],
            'mean_steps': float(np.mean(results['steps'])),
            'mean_final_dist': float(np.mean(results['final_dists'])),
            'quadrants': results['quadrants'],
            'details': results['details'],
            'elapsed': elapsed
        }, f, indent=2)
    
    print(f"\nResults saved: {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")

if __name__ == '__main__':
    main()
