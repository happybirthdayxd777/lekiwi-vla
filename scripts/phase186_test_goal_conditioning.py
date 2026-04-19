#!/usr/bin/env python3
"""
Phase 186: Test if goal conditioning fixes VLA 0% SR.

Hypothesis: Phase 181 policy was trained with state_dim=9 (no goal_xy).
But the training DATA (phase181_symmetrized_10k.h5) has goals stored separately.
If we augment the 9D state with goal_xy at eval time (to make 11D),
does the policy suddenly become goal-aware?

Method: Take the trained 9D policy, load with state_dim=11 (matching train_task_oriented.py
default), but the flow_head was trained with 9D input. When we pass 11D state,
the flow_head's state_net (Linear(11,256)) will receive 11 values but its first
weight row expects 9. This corrupts the input.

CORRECT FIX: Create a new model with state_dim=11, copy the flow_head weights
(except the first Linear layer), and fine-tune with goal-conditioned data.

But for a QUICK TEST: we can check if the 9D policy loaded as state_dim=11
produces different actions for different goals (using dummy CLIP features).

Actually the real test: Run eval with goal_augmented state vs original 9D state.
If there's ANY difference, it means the flow_head's learned representation
is somewhat goal-aware through correlation (even though it never explicitly
received goal info during training).
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

DEVICE = 'cpu'
CKPT_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase181_vision_train/best_policy.pt'
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 10
SEED = 42


def twist_to_contact_wheel_speeds(vx, vy, wz, k_omni=15.0):
    R = 0.0508
    vx_kin =  0.0000*vx + 0.3824*vy - 0.4531*(-wz)
    vy_kin =  0.1544*vx + 0.1929*vy + 0.2378*(-wz)
    wz_kin =  0.0
    w1 = (-0.866*vx_kin + 0.5*vy_kin + 1.0*wz_kin) / R
    w2 = ( 0.866*vx_kin + 0.5*vy_kin + 1.0*wz_kin) / R
    w3 = ( 0.0     *vx_kin + 0.0*vy_kin - 1.0*wz_kin) / R
    w1 += -0.866*k_omni*vx + 0.5*k_omni*vy
    w2 +=  0.866*k_omni*vx + 0.5*k_omni*vy
    w3 +=  0.0*k_omni*vx + 0.0*k_omni*vy
    return [w1, w2, w3]


def p_controller_action(sim, goal_xy, kP=0.5):
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


def load_policy(state_dim=9):
    print(f"[Policy] Loading from {CKPT_PATH} as state_dim={state_dim}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    policy = CLIPFlowMatchingPolicy(
        state_dim=state_dim, action_dim=9, hidden=512, device=DEVICE)
    if 'flow_head_state_dict' in ckpt:
        policy.flow_head.load_state_dict(ckpt['flow_head_state_dict'])
    print(f"[Policy] Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.6f}")
    return policy, ckpt


def run_episode_pctrl(sim, goal, max_steps=200):
    """P-controller baseline."""
    sim.reset()
    for _ in range(15): sim.step([0]*9)
    
    for step in range(max_steps):
        ctrl = p_controller_action(sim, goal)
        action_np = np.array(list([0]*6) + ctrl)
        sim.step(action_np)
        
        base_xy = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_xy - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    
    return False, max_steps, dist


def run_episode_vla_9d(sim, policy, goal, max_steps=200):
    """VLA with 9D state (NO goal conditioning) — original Phase 181 eval."""
    sim.reset()
    for _ in range(15): sim.step([0]*9)
    
    for step in range(max_steps):
        obs = sim._obs()
        state = np.concatenate([obs['arm_positions'], obs['wheel_velocities']]).astype(np.float32)
        state[0:6] = np.clip(state[0:6] / 2.0, -1, 1)
        state[6:9] = np.clip(state[6:9] / 0.5, -1, 1)
        img = sim.render().astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
        img_t = F.interpolate(img_t, size=(224,224), mode='bilinear', align_corners=False)
        state_t = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = action.cpu().detach().numpy()[0]
        action_np = np.clip(action_np, -0.5, 0.5)
        
        sim.step(action_np)
        
        base_xy = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_xy - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    
    return False, max_steps, dist


def run_episode_vla_11d(sim, policy, goal, max_steps=200):
    """VLA with 11D state (goal appended to end of 9D state) — TEST."""
    sim.reset()
    for _ in range(15): sim.step([0]*9)
    
    for step in range(max_steps):
        obs = sim._obs()
        state9 = np.concatenate([obs['arm_positions'], obs['wheel_velocities']]).astype(np.float32)
        state9_clipped = np.concatenate([
            np.clip(state9[:6] / 2.0, -1, 1),
            np.clip(state9[6:9] / 0.5, -1, 1)
        ])
        # Augment with goal: 11D = 9D + goal_xy (normalized to [-1,1] assuming ~0.5m range)
        goal_norm = np.clip(goal / 0.5, -1, 1).astype(np.float32)
        state11 = np.concatenate([state9_clipped, goal_norm])
        
        img = sim.render().astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE)
        img_t = F.interpolate(img_t, size=(224,224), mode='bilinear', align_corners=False)
        state_t = torch.from_numpy(state11).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = action.cpu().detach().numpy()[0]
        action_np = np.clip(action_np, -0.5, 0.5)
        
        sim.step(action_np)
        
        base_xy = sim.data.qpos[:2].copy()
        dist = np.linalg.norm(base_xy - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    
    return False, max_steps, dist


def main():
    print(f"Phase 186: Test if goal conditioning fixes 0% SR")
    print(f"Device: {DEVICE}, Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")
    print(f"CKPT: {CKPT_PATH}")
    
    # Generate shared goals
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    goals = []
    for _ in range(N_EPISODES):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.2, 0.5)
        goals.append([r*np.cos(angle), r*np.sin(angle)])
    goals = np.array(goals)
    
    print(f"\nGoals: {goals.tolist()}")
    
    # Load 9D policy (original)
    policy_9d, ckpt = load_policy(state_dim=9)
    
    # Load 11D policy (goal-augmented test)
    # Note: We load the SAME checkpoint but with state_dim=11
    # This means state_net receives 11 values but was trained with 9
    # The first 9 values are meaningful, the last 2 are garbage from random init
    # This is a BROKEN test — we need a PROPERLY trained 11D policy
    policy_11d, _ = load_policy(state_dim=11)
    
    sim_p = LeKiWiSimURDF()
    
    # Run P-controller baseline
    print(f"\n=== P-Controller Baseline ===")
    p_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_pctrl(sim_p, goal, max_steps=MAX_STEPS)
        p_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  P-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={fdist:.3f})")
    
    p_sr = sum(r['success'] for r in p_results) / len(p_results) * 100
    print(f"P-controller: {sum(r['success'] for r in p_results)}/{len(p_results)} = {p_sr:.1f}% SR, mean_steps={np.mean([r['steps'] for r in p_results]):.1f}")
    
    # Run VLA 9D (original)
    print(f"\n=== VLA (9D, NO goal conditioning) ===")
    vla_9d_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_vla_9d(sim_p, policy_9d, goal, max_steps=MAX_STEPS)
        vla_9d_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  VLA9d-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={fdist:.3f})")
    
    vla_9d_sr = sum(r['success'] for r in vla_9d_results) / len(vla_9d_results) * 100
    print(f"VLA 9D: {sum(r['success'] for r in vla_9d_results)}/{len(vla_9d_results)} = {vla_9d_sr:.1f}% SR")
    
    # Run VLA 11D (goal-augmented, BROKEN test — just to see if ANY difference)
    print(f"\n=== VLA (11D, goal-augmented, EXPECTED BROKEN) ===")
    print("NOTE: 11D model was NOT trained with goals — this tests if random goal input causes any difference")
    vla_11d_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_vla_11d(sim_p, policy_11d, goal, max_steps=MAX_STEPS)
        vla_11d_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  VLA11d-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps, dist={fdist:.3f})")
    
    vla_11d_sr = sum(r['success'] for r in vla_11d_results) / len(vla_11d_results) * 100
    print(f"VLA 11D: {sum(r['success'] for r in vla_11d_results)}/{len(vla_11d_results)} = {vla_11d_sr:.1f}% SR")
    
    # Save results
    results = {
        'phase': 186,
        'p_controller': {'sr': p_sr, 'results': p_results},
        'vla_9d': {'sr': vla_9d_sr, 'results': vla_9d_results},
        'vla_11d': {'sr': vla_11d_sr, 'results': vla_11d_results, 'note': 'BROKEN — policy not trained with 11D'},
        'goals': goals.tolist(),
        'ckpt_state_dim': ckpt.get('policy_config', {}).get('state_dim', 9),
    }
    
    out_path = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase186_goal_conditioning_test.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    print(f"\n=== SUMMARY ===")
    print(f"P-controller:   {p_sr:.1f}% SR")
    print(f"VLA (9D):       {vla_9d_sr:.1f}% SR  ← Phase 181 original")
    print(f"VLA (11D):      {vla_11d_sr:.1f}% SR  ← BROKEN test (not trained with goals)")
    print(f"\nConclusion: Need to TRAIN with state_dim=11 to fix goal conditioning")


if __name__ == '__main__':
    main()
