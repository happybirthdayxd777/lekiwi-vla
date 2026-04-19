#!/usr/bin/env python3
"""
Phase 191: Multi-Epoch Sweep Eval of Phase 190 Policy Checkpoints
====================================================================
Evaluates all available epoch checkpoints (4, 9, 14, 19) on 5 goals each.
Compares VLA success rate vs P-controller across training progress.

Usage: python3 scripts/eval_phase190_sweep.py
"""
import sys, os, json, glob
import numpy as np
import torch
import time

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

DEVICE = 'cpu'
MAX_STEPS = 100
GOAL_THRESHOLD = 0.1
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DUMMY_IMG = torch.zeros(1, 3, 224, 224, dtype=torch.float32)


# ── Policy (same arch as train_phase190.py) ───────────────────────────────────

class CLIPVisionEncoder(torch.nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[CLIP] Loading ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            out = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            return out.last_hidden_state


class GoalConditionedPolicy(torch.nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.encoder = CLIPVisionEncoder(device=device)
        self.goal_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )
        self.state_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )
        self.goal_q_proj = torch.nn.Linear(128, 768)
        self.cross_attn = torch.nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = torch.nn.LayerNorm(768)
        self.flow_head = torch.nn.Sequential(
            torch.nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, action_dim)
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 256), torch.nn.SiLU()
        )

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)
        goal_emb = self.goal_mlp(state[:, -2:])
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)
        state_feat = self.state_net(state)
        t_emb = self.time_mlp(timestep)
        cls_token = clip_tokens[:, 0:1, :]
        x = torch.cat([cls_token, cross_out, state_feat.unsqueeze(1),
                       t_emb.unsqueeze(1), noisy_action.unsqueeze(1)], dim=-1)
        x = x.squeeze(1)
        return self.flow_head(x)

    def infer(self, images, state, num_steps=4):
        self.eval()
        with torch.no_grad():
            x = torch.zeros_like(state[:, :self.action_dim])
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((images.shape[0], 1), i * dt, device=state.device)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ── P-controller ─────────────────────────────────────────────────────────────

def p_controller_action(sim, goal_xy, kP=0.5):
    base_xy = sim.data.qpos[:2].copy()
    dx, dy = goal_xy[0] - base_xy[0], goal_xy[1] - base_xy[1]
    dist = np.linalg.norm([dx, dy])
    if dist < 0.005:
        return [0.0, 0.0, 0.0]
    vx, vy = kP * dx, kP * dy
    return twist_to_contact_wheel_speeds(vx, vy, 0.0)


# ── Load policy checkpoint ─────────────────────────────────────────────────────

def load_vla(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    policy = GoalConditionedPolicy(
        state_dim=config.get('state_dim', 11),
        action_dim=config.get('action_dim', 9),
        hidden=config.get('hidden', 512),
        device=DEVICE
    )
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    return policy, ckpt.get('epoch', '?'), ckpt.get('loss', 0.0)


# ── Test goals ───────────────────────────────────────────────────────────────

TEST_GOALS = [
    (0.3, 0.3),
    (0.3, -0.3),
    (-0.3, 0.3),
    (-0.3, -0.3),
    (0.4, 0.1),
]


def run_episode_vla(sim, policy, goal, max_steps=100):
    goal_arr = np.array(goal, dtype=np.float32)
    goal_norm = np.clip(goal_arr / 0.5, -1, 1).astype(np.float32)
    sim.reset(target=goal, seed=None)
    for _ in range(15):
        sim.step([0]*9)

    for step in range(max_steps):
        obs = sim._obs()
        arm = obs['arm_positions']
        wheel_v = obs['wheel_velocities']
        state11 = np.concatenate([
            np.clip(arm / 2.0, -1, 1),
            np.clip(wheel_v / 0.5, -1, 1),
            goal_norm
        ]).astype(np.float32)
        state_t = torch.from_numpy(state11).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = policy.infer(DUMMY_IMG, state_t, num_steps=4)
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        sim.step(action_np)
        dist = np.linalg.norm(sim.data.qpos[:2] - goal_arr)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    return False, max_steps, dist


def run_episode_pctrl(sim, goal, max_steps=100):
    goal_arr = np.array(goal, dtype=np.float32)
    sim.reset(target=goal, seed=None)
    for _ in range(15):
        sim.step([0]*9)
    for step in range(max_steps):
        ctrl = p_controller_action(sim, goal_arr)
        action_np = np.array([0.0]*6 + list(ctrl))
        sim.step(action_np)
        dist = np.linalg.norm(sim.data.qpos[:2] - goal_arr)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    return False, max_steps, dist


# ── Main sweep ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 191: Multi-Epoch Sweep — Phase 190 VLA Training Progress")
    print("=" * 70)

    train_dir = 'results/phase190_vision_train'
    ckpt_pattern = os.path.join(train_dir, 'epoch_*.pt')
    ckpts = sorted(glob.glob(ckpt_pattern), key=lambda p: int(p.split('epoch_')[1].split('.')[0]))

    if not ckpts:
        print(f"ERROR: No checkpoints found in {train_dir}")
        return

    print(f"Found checkpoints: {[os.path.basename(c) for c in ckpts]}")
    print(f"Test goals: {TEST_GOALS}")
    print(f"Max steps per episode: {MAX_STEPS}, Goal threshold: {GOAL_THRESHOLD}m")
    print()

    # Run P-controller once (stable baseline)
    print("[P-CTRL] Baseline evaluation (5 goals × 1 run)...")
    sim_pctrl = LeKiWiSimURDF()
    pctrl_results = {'success': [], 'steps': [], 'final_dist': []}
    t0 = time.time()
    for i, goal in enumerate(TEST_GOALS):
        ok, steps, dist = run_episode_pctrl(sim_pctrl, goal, MAX_STEPS)
        pctrl_results['success'].append(ok)
        pctrl_results['steps'].append(steps)
        pctrl_results['final_dist'].append(float(dist))
        print(f"  Goal {i+1} {goal}: {'✓' if ok else '✗'} steps={steps}, dist={dist:.4f}")
    pctrl_time = time.time() - t0
    pctrl_sr = np.mean(pctrl_results['success']) * 100
    print(f"  P-ctrl SR: {pctrl_sr:.0f}% ({sum(pctrl_results['success'])}/{len(TEST_GOALS)})")
    print(f"  P-ctrl time: {pctrl_time:.1f}s")
    print()

    # Sweep each epoch checkpoint
    all_results = {}
    summary_rows = []

    for ckpt_path in ckpts:
        epoch_name = os.path.basename(ckpt_path).replace('.pt', '')
        print(f"[{epoch_name}] Loading and evaluating...")

        policy, epoch_num, loss = load_vla(ckpt_path)
        sim = LeKiWiSimURDF()

        vla_results = {'success': [], 'steps': [], 'final_dist': []}

        t0 = time.time()
        for i, goal in enumerate(TEST_GOALS):
            ok, steps, dist = run_episode_vla(sim, policy, goal, MAX_STEPS)
            vla_results['success'].append(ok)
            vla_results['steps'].append(steps)
            vla_results['final_dist'].append(float(dist))
            print(f"  Goal {i+1} {goal}: {'✓' if ok else '✗'} steps={steps}, dist={dist:.4f}")

        elapsed = time.time() - t0
        vla_sr = np.mean(vla_results['success']) * 100
        avg_steps_ok = np.mean([s for s, ok in zip(vla_results['steps'], vla_results['success']) if ok] or [0])
        avg_dist = np.mean(vla_results['final_dist'])

        print(f"  => VLA SR={vla_sr:.0f}% ({sum(vla_results['success'])}/{len(TEST_GOALS)}), "
              f"avg_steps={avg_steps_ok:.1f}, avg_dist={avg_dist:.4f}, time={elapsed:.1f}s")
        print()

        all_results[epoch_name] = {
            'vla': vla_results,
            'pctrl': pctrl_results,
            'epoch': epoch_num,
            'loss': loss,
        }

        summary_rows.append({
            'checkpoint': epoch_name,
            'epoch': epoch_num,
            'loss': loss,
            'vla_sr': vla_sr,
            'vla_success': sum(vla_results['success']),
            'vla_total': len(TEST_GOALS),
            'vla_avg_steps': avg_steps_ok,
            'vla_avg_dist': avg_dist,
            'pctrl_sr': pctrl_sr,
            'eval_time': elapsed,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Checkpoint':<20} {'Epoch':>5} {'Loss':>8} {'VLA SR':>7} {'VLA ✓':>6} "
          f"{'AvgStep':>7} {'AvgDist':>8} {'P-ctrl SR':>9} {'Time':>6}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['checkpoint']:<20} {row['epoch']:>5} {row['loss']:>8.4f} "
              f"{row['vla_sr']:>7.0f}% {row['vla_success']:>5}/{row['vla_total']} "
              f"{row['vla_avg_steps']:>7.1f} {row['vla_avg_dist']:>8.4f} "
              f"{row['pctrl_sr']:>8.0f}% {row['eval_time']:>6.1f}s")

    # Save results
    out_path = f'results/phase191_sweep_{int(time.time())}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'summary': summary_rows,
            'pctrl_baseline': pctrl_results,
            'all_results': {k: {
                'vla': v,
                'pctrl': pctrl_results,
                'epoch': all_results[k]['epoch'],
                'loss': all_results[k]['loss'],
            } for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Find best epoch
    best = max(summary_rows, key=lambda r: r['vla_sr'])
    print(f"\n[BEST] {best['checkpoint']}: VLA SR={best['vla_sr']:.0f}% "
          f"vs P-ctrl={pctrl_sr:.0f}% (delta={best['vla_sr']-pctrl_sr:+.0f}%)")


if __name__ == '__main__':
    main()
