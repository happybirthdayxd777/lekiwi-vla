#!/usr/bin/env python3
"""
Phase 182: Matched eval — P-controller AND VLA on IDENTICAL goals.
Runs 20 episodes, same goals for both controllers.
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
CKPT_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_vision_train/best_policy.pt'
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 20
SEED = 42

def twist_to_contact_wheel_speeds(vx, vy, wz, k_omni=15.0):
    """Jacobian P-controller: desired twist → wheel velocities."""
    R = 0.041  # wheel radius
    vx_kin =  0.0000*vx + 0.3824*vy - 0.4531*(-wz)
    vy_kin =  0.1544*vx + 0.1929*vy + 0.2378*(-wz)
    wz_kin =  0.0
    w1 = (-0.866*vx_kin + 0.5*vy_kin + 1.0*wz_kin) / R
    w2 = ( 0.866*vx_kin + 0.5*vy_kin + 1.0*wz_kin) / R
    w3 = ( 0.0     *vx_kin + 0.0*vy_kin - 1.0*wz_kin) / R
    # k_omni overlay
    w1 += -0.866*k_omni*vx + 0.5*k_omni*vy
    w2 +=  0.866*k_omni*vx + 0.5*k_omni*vy
    w3 +=  0.0*k_omni*vx + 0.0*k_omni*vy
    return [w1, w2, w3]

def p_controller_action(sim, goal_xy, kP=0.5):
    """P-controller: go straight toward goal at constant speed."""
    base_xy = sim.data.qpos[:2].copy()
    dx = goal_xy[0] - base_xy[0]
    dy = goal_xy[1] - base_xy[1]
    dist = np.linalg.norm([dx, dy])
    if dist < 0.005:
        return [0.0, 0.0, 0.0]
    vx = kP * dx
    vy = kP * dy
    wz = 0.0
    return twist_to_contact_wheel_speeds(vx, vy, wz)

def load_policy():
    print(f"[Policy] Loading from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 9, 'action_dim': 9, 'hidden': 512})
    policy = CLIPFlowMatchingPolicy(
        state_dim=config.get('state_dim', 9), action_dim=config.get('action_dim', 9),
        hidden=config.get('hidden', 512), device=DEVICE)
    if 'flow_head_state_dict' in ckpt:
        policy.flow_head.load_state_dict(ckpt['flow_head_state_dict'])
    print(f"[Policy] Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.6f}")
    return policy

def run_episode(sim, policy, goal, use_vla=True, max_steps=200):
    """Run one episode. Returns (success, steps, final_dist, actions)."""
    sim.reset()
    for _ in range(15): sim.step([0]*9)
    
    ep_actions = []
    for step in range(max_steps):
        obs = sim._obs()
        state = np.concatenate([obs['arm_positions'], obs['wheel_velocities']]).astype(np.float32)
        state[0:6] = np.clip(state[0:6] / 2.0, -1, 1)
        state[6:9] = np.clip(state[6:9] / 0.5, -1, 1)
        img = sim.render().astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
        img_t = F.interpolate(img_t, size=(224,224), mode='bilinear', align_corners=False)
        state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        
        if use_vla:
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().detach().numpy()[0]
            action_np = np.clip(action_np, -0.5, 0.5)
        else:
            ctrl = p_controller_action(sim, goal)
            action_np = np.array(list([0]*6) + ctrl)
        
        ep_actions.append(action_np)
        sim.step(action_np)
        
        base_xy = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_xy - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist, ep_actions
    
    return False, max_steps, dist, ep_actions

def main():
    print(f"Phase 182: Matched P-ctrl vs VLA eval")
    print(f"Device: {DEVICE}, Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")
    
    # Generate shared goals
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    goals = []
    for _ in range(N_EPISODES):
        g = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        goals.append(g)
    
    # Quadrant analysis
    quads = {'+X+Y': 0, '+X-Y': 0, '-X+Y': 0, '-X-Y': 0}
    for g in goals:
        if g[0] >= 0 and g[1] > 0: quads['+X+Y'] += 1
        elif g[0] >= 0 and g[1] <= 0: quads['+X-Y'] += 1
        elif g[0] < 0 and g[1] >= 0: quads['-X+Y'] += 1
        else: quads['-X-Y'] += 1
    print(f"Goal quadrants: {quads}")
    
    # P-controller baseline
    print(f"\n=== P-Controller Baseline ===")
    sim_p = LeKiWiSimURDF()
    p_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist, _ = run_episode(sim_p, None, goal, use_vla=False, max_steps=MAX_STEPS)
        p_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  P-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={fdist:.3f})")
    
    p_sr = sum(r['success'] for r in p_results) / N_EPISODES * 100
    p_steps = np.mean([r['steps'] for r in p_results])
    print(f"P-controller: {sum(r['success'] for r in p_results)}/{N_EPISODES} = {p_sr:.1f}% SR, mean_steps={p_steps:.1f}")
    
    # VLA eval
    print(f"\n=== VLA Policy ===")
    policy = load_policy()
    policy.eval()
    sim_v = LeKiWiSimURDF()
    v_results = []
    all_vla_actions = []
    for ep, goal in enumerate(goals):
        success, steps, fdist, actions = run_episode(sim_v, policy, goal, use_vla=True, max_steps=MAX_STEPS)
        v_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        all_vla_actions.extend(actions)
        print(f"  V-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={fdist:.3f})")
    
    v_sr = sum(r['success'] for r in v_results) / N_EPISODES * 100
    v_steps = np.mean([r['steps'] for r in v_results])
    print(f"VLA: {sum(r['success'] for r in v_results)}/{N_EPISODES} = {v_sr:.1f}% SR, mean_steps={v_steps:.1f}")
    
    # Action diversity
    all_vla_actions = np.array(all_vla_actions)
    wheel_actions = all_vla_actions[:, 6:9]
    print(f"\nVLA action diversity (wheel cols 6-9):")
    print(f"  Mean: {wheel_actions.mean(axis=0)}")
    print(f"  Std:  {wheel_actions.std(axis=0)}")
    print(f"  Range: {wheel_actions.max(axis=0) - wheel_actions.min(axis=0)}")
    
    # Save
    results = {
        'phase': 182,
        'p_controller': {'sr': p_sr, 'mean_steps': float(p_steps), 'details': p_results},
        'vla': {'sr': v_sr, 'mean_steps': float(v_steps), 'details': v_results},
        'goals': [g.tolist() for g in goals],
        'action_diversity': {
            'wheel_mean': wheel_actions.mean(axis=0).tolist(),
            'wheel_std': wheel_actions.std(axis=0).tolist(),
            'wheel_range': (wheel_actions.max(axis=0) - wheel_actions.min(axis=0)).tolist(),
            'total_steps': len(all_vla_actions),
            'unique_actions': len(np.unique(np.round(all_vla_actions, 2), axis=0))
        }
    }
    out_path = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase182_matched_eval.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nSUMMARY: P-ctrl={p_sr:.1f}% SR, VLA={v_sr:.1f}% SR")

if __name__ == '__main__':
    main()
