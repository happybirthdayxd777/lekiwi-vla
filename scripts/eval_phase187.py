#!/usr/bin/env python3
"""
Phase 187 eval: Matched P-ctrl vs VLA on identical 20-goal set.
Uses same goals as Phase 182/186 for fair comparison.

VLA: phase187_goal_conditioned_train/best_policy.pt
State: 11D (arm_pos+wheel_vel+goal_norm) — MATCHES training
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

DEVICE = 'mps'
CKPT_PATH = 'results/phase187_goal_conditioned_train/best_policy.pt'
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 20
SEED = 42

# ── P-controller (same as phase186_eval.py) ──────────────────────────────────

def p_controller_action(sim, goal_xy, kP=0.5):
    base_xy = sim.data.qpos[:2].copy()
    dx, dy = goal_xy[0] - base_xy[0], goal_xy[1] - base_xy[1]
    dist = np.linalg.norm([dx, dy])
    if dist < 0.005:
        return [0.0, 0.0, 0.0]
    vx, vy = kP * dx, kP * dy
    return twist_to_contact_wheel_speeds(vx, vy, 0.0)


def run_episode_pctrl(sim, goal, max_steps=200):
    sim.reset()
    for _ in range(15):
        sim.step([0]*9)
    for step in range(max_steps):
        ctrl = p_controller_action(sim, goal)
        action_np = np.array(list([0]*6) + ctrl)
        sim.step(action_np)
        dist = np.linalg.norm(sim.data.qpos[:2] - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    return False, max_steps, dist


# ── VLA inference ─────────────────────────────────────────────────────────────

def load_vla():
    from scripts.train_phase187 import GoalConditionedPolicy
    print(f"[VLA] Loading {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    policy = GoalConditionedPolicy(state_dim=config.get('state_dim', 11),
                                   action_dim=config.get('action_dim', 9),
                                   hidden=config.get('hidden', 512), device=DEVICE)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    print(f"  Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}")
    return policy


def run_episode_vla(sim, policy, goal, max_steps=200):
    goal_norm = np.clip(goal / 0.5, -1, 1).astype(np.float32)
    sim.reset()
    for _ in range(15):
        sim.step([0]*9)
    for step in range(max_steps):
        obs = sim._obs()
        arm = obs['arm_positions']
        wheel_v = obs['wheel_velocities']
        state9 = np.concatenate([arm, wheel_v]).astype(np.float32)
        state9_clipped = np.concatenate([
            np.clip(state9[:6] / 2.0, -1, 1),
            np.clip(state9[6:9] / 0.5, -1, 1)
        ])
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

        dist = np.linalg.norm(sim.data.qpos[:2] - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    return False, max_steps, dist


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Phase 187 Eval: VLA vs P-controller ({N_EPISODES}ep, {MAX_STEPS}steps)")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    goals = []
    for _ in range(N_EPISODES):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.2, 0.5)
        goals.append([r*np.cos(angle), r*np.sin(angle)])
    goals = np.array(goals)

    sim = LeKiWiSimURDF()

    # P-controller baseline
    print(f"\n=== P-Controller ===")
    p_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_pctrl(sim, goal, MAX_STEPS)
        p_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success,
                         'steps': steps, 'final_dist': float(fdist)})
        print(f"  P-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps)")

    p_sr = sum(r['success'] for r in p_results) / len(p_results) * 100
    print(f"P-controller: {sum(r['success'] for r in p_results)}/{len(p_results)} = {p_sr:.1f}% SR")

    # VLA
    print(f"\n=== VLA (Phase 187) ===")
    policy = load_vla()
    vla_results = []
    for ep, goal in enumerate(goals):
        success, steps, fdist = run_episode_vla(sim, policy, goal, MAX_STEPS)
        vla_results.append({'ep': ep, 'goal': goal.tolist(), 'success': success,
                            'steps': steps, 'final_dist': float(fdist)})
        print(f"  VLA-ep {ep}: [{goal[0]:+.3f},{goal[1]:+.3f}] → {'SUCC' if success else 'FAIL'} ({steps} steps)")

    vla_sr = sum(r['success'] for r in vla_results) / len(vla_results) * 100
    print(f"VLA (Phase 187): {sum(r['success'] for r in vla_results)}/{len(vla_results)} = {vla_sr:.1f}% SR")

    results = {
        'phase': 187,
        'p_controller': {'sr': p_sr, 'results': p_results},
        'vla': {'sr': vla_sr, 'results': vla_results},
        'goals': goals.tolist(),
        'ckpt': CKPT_PATH,
    }
    out_path = 'results/phase187_eval.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: P={p_sr:.1f}% vs VLA={vla_sr:.1f}%")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
