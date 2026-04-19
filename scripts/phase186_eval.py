#!/usr/bin/env python3
"""
Phase 186: Evaluate goal-conditioned VLA vs P-controller.
Same 10 goals, same seed as Phase 182 eval for fair comparison.
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

DEVICE = 'cpu'
CKPT_PATH = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase186_goal_conditioned_train/best_policy.pt'
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 20
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


def load_policy():
    print(f"[Policy] Loading {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    print(f"  Config: state_dim={config.get('state_dim')}, action_dim={config.get('action_dim')}")
    policy = CLIPFlowMatchingPolicy(
        state_dim=config.get('state_dim', 11),
        action_dim=config.get('action_dim', 9),
        hidden=config.get('hidden', 512), device=DEVICE)
    policy.flow_head.load_state_dict(ckpt['flow_head_state_dict'])
    policy.eval()
    print(f"  Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.6f}")
    return policy


def run_episode_pctrl(sim, goal, max_steps=200):
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


def run_episode_vla(sim, policy, goal, max_steps=200):
    """VLA with 11D goal-conditioned state."""
    sim.reset()
    for _ in range(15): sim.step([0]*9)
    for step in range(max_steps):
        obs = sim._obs()
        # Build 9D clipped state
        state9 = np.concatenate([obs['arm_positions'], obs['wheel_velocities']]).astype(np.float32)
        state9_clipped = np.concatenate([
            np.clip(state9[:6] / 2.0, -1, 1),
            np.clip(state9[6:9] / 0.5, -1, 1)
        ])
        # Append goal to make 11D
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
    print(f"Phase 186: Evaluate goal-conditioned VLA (state_dim=11)")
    print(f"Device: {DEVICE}, Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    goals = []
    for _ in range(N_EPISODES):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.2, 0.5)
        goals.append([r*np.cos(angle), r*np.sin(angle)])
    goals = np.array(goals)

    print(f"\nGoals: {goals.tolist()}")
    policy = load_policy()
    sim = LeKiWiSimURDF()

    # P-controller baseline
    print(f"\n=== P-Controller ===")
    p_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_pctrl(sim, goal, MAX_STEPS)
        p_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  P-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps)")
    p_sr = sum(r['success'] for r in p_results) / len(p_results) * 100
    print(f"P-controller: {sum(r['success'] for r in p_results)}/{len(p_results)} = {p_sr:.1f}% SR")

    # VLA
    print(f"\n=== VLA (state_dim=11, goal-conditioned) ===")
    vla_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_vla(sim, policy, goal, MAX_STEPS)
        vla_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(fdist)})
        print(f"  VLA-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps)")
    vla_sr = sum(r['success'] for r in vla_results) / len(vla_results) * 100
    print(f"VLA (11D): {sum(r['success'] for r in vla_results)}/{len(vla_results)} = {vla_sr:.1f}% SR")

    results = {
        'phase': 186,
        'p_controller': {'sr': p_sr, 'results': p_results},
        'vla': {'sr': vla_sr, 'results': vla_results},
        'goals': goals.tolist(),
        'ckpt': CKPT_PATH,
    }
    out_path = '/Users/i_am_ai/hermes_research/lekiwi_vla/results/phase186_eval.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: P={p_sr:.1f}% vs VLA={vla_sr:.1f}%")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
