#!/usr/bin/env python3
"""
Phase 176: VLA Wheel Magnitude Root Cause Analysis
==================================================
CRITICAL FINDING from Phase 175:

VLA wheel actions are ~15-20x SMALLER than P-ctrl wheel actions:
  - VLA |w| ≈ 0.035-0.047 (across ALL episodes, success AND failure)
  - P-ctrl |w| ≈ 0.5-0.87

This is NOT a per-quadrant issue. VLA outputs tiny wheel actions everywhere.
The 53% SR is likely due to:
  1. Arm movements coupling to base motion (passive locomotion)
  2. Accumulated small wheel corrections over many steps
  3. Some goals happen to be reachable with minimal wheel movement

This script:
1. Measures wheel action magnitudes across success/failure episodes
2. Compares VLA vs P-ctrl wheel magnitude distributions
3. Tests: what if we amplify VLA wheel actions by 10x?
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 176] Device: {DEVICE}")


class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.float32)
        pixel_values = pixel_values.to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state
        return hidden


class GoalConditionedPolicy(nn.Module):
    """SAME ARCHITECTURE as eval_phase174_wheel_fix.py."""
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.clip_encoder = CLIPSpatialEncoder(device)
        self.vision_proj = nn.Linear(768, hidden).to(device)
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 256), nn.ReLU(), nn.Linear(256, 128),
        ).to(device)
        self.goal_proj = nn.Linear(128, 256).to(device)
        self.q_proj = nn.Linear(256, hidden).to(device)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        ).to(device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.time_net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 256),
        ).to(device)
        self.action_head = nn.Sequential(
            nn.Linear(256 + 128 + hidden + 256, hidden),
            nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, action_dim),
        ).to(device)
        self.skip = nn.Linear(action_dim, action_dim, bias=False).to(device)

    def forward(self, image, state, noisy_action, t):
        clip_feat = self.clip_encoder(image)
        clip_proj = self.vision_proj(clip_feat)
        goal_emb = self.goal_mlp(state[:, 9:11])
        goal_q = self.goal_proj(goal_emb)
        state_feat = self.state_net(state)
        q = self.q_proj(state_feat + goal_q)
        q = q.unsqueeze(1)
        cross_out, _ = self.cross_attn(q, clip_proj, clip_proj)
        cross_out = self.cross_norm(cross_out + q)
        t_feat = self.time_net(t)
        combined = torch.cat([state_feat, goal_emb, cross_out.squeeze(1), t_feat], dim=-1)
        v_pred = self.action_head(combined)
        v_pred = v_pred + self.skip(noisy_action)
        return v_pred

    def infer(self, image, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * (i * dt)
            v = self.forward(image, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


def load_policy(ckpt_path, device=DEVICE):
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9,
                                   cross_heads=8, hidden=512, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('policy_state_dict', ckpt)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device).eval()
    policy.device = device
    print(f"[LOAD] epoch={ckpt.get('epoch','?')}, eval_sr={ckpt.get('eval_sr','?')}")
    return policy


def resize_for_clip(img):
    pil = Image.fromarray(img)
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_np = (np.array(pil_resized).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return img_np


def generate_restricted_goals(n_episodes, seed=42):
    """Same as eval_phase174_wheel_fix.py."""
    rng = np.random.default_rng(seed)
    goals = []
    for _ in range(n_episodes):
        gx = rng.uniform(-0.1, 0.5)
        gy = rng.uniform(-0.3, 0.3)
        goals.append([gx, gy])
    return np.array(goals, dtype=np.float32)


def run_episode_amplified(sim, goal, goal_norm, policy, amplifier=1.0, max_steps=200):
    """Run VLA with amplified wheel actions."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    base_id = sim.model.body('base').id
    threshold = 0.15

    arm_actions_list = []
    wheel_actions_list = []

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, arm_actions_list, wheel_actions_list

        img = sim.render()
        img_t = resize_for_clip(img)

        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
        wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
        state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

        img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(DEVICE)
        state_tensor = torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            raw_action = policy.infer(img_tensor, state_tensor, num_steps=4)
        raw_np = raw_action.squeeze(0).cpu().numpy()

        raw_wheel = raw_np[6:9]
        wheel_action = raw_wheel * 0.0834 * amplifier  # AMPLIFIED
        wheel_action = np.clip(wheel_action, -0.0417, 0.0417)

        arm_action = np.clip(raw_np[:6], -0.5, 0.5)

        arm_actions_list.append(arm_action.copy())
        wheel_actions_list.append(wheel_action.copy())

        action = np.concatenate([arm_action, wheel_action])
        sim.step(action)

    return False, max_steps, dist, arm_actions_list, wheel_actions_list


def run_pctrl_episode(sim, goal, goal_norm, max_steps=200):
    """P-controller baseline."""
    from sim_lekiwi_urdf import twist_to_contact_wheel_speeds
    base_id = sim.model.body('base').id
    threshold = 0.15
    kP = 0.1
    max_speed = 0.25

    wheel_actions_list = []

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, wheel_actions_list

        dx = goal[0] - base_pos[0]
        dy = goal[1] - base_pos[1]
        d = np.linalg.norm([dx, dy])
        if d > 1e-6:
            v_mag = min(kP * d, max_speed)
            vx, vy = (dx / d) * v_mag, (dy / d) * v_mag
        else:
            vx, vy = 0.0, 0.0

        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        wheel_action = wheel_speeds / 12.0
        wheel_actions_list.append(wheel_action.copy())

        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
        action = np.concatenate([arm_pos, wheel_action]).astype(np.float32)
        sim.step(action)

    return False, max_steps, dist, wheel_actions_list


def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF
    import time

    print("\n" + "=*(" * 30)
    print("Phase 176: VLA Wheel Magnitude Root Cause")
    print("=*(" * 30)

    # Load policy
    print("\n[INFO] Loading policy...")
    policy = load_policy("results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt", DEVICE)

    goals = generate_restricted_goals(30, seed=42)

    print("\n[TEST 1] Baseline VLA (no amplification)")
    print("-" * 60)
    baseline_results = []
    for i, goal in enumerate(goals):
        goal_norm = np.clip(goal / 1.0, -1.0, 1.0)
        sim = LeKiWiSimURDF()
        succ, steps, dist, arm_acts, wheel_acts = run_episode_amplified(
            sim, goal, goal_norm, policy, amplifier=1.0, max_steps=200
        )
        baseline_results.append({
            'episode': i, 'goal': goal.tolist(),
            'vla_success': succ, 'vla_steps': steps, 'vla_dist': dist,
            'arm_acts': arm_acts, 'wheel_acts': wheel_acts
        })
        if i % 10 == 0:
            print(f"  Episode {i}/30: {'SUCC' if succ else 'FAIL'} (dist={dist:.3f})")

    baseline_sr = sum(r['vla_success'] for r in baseline_results) / 30 * 100
    print(f"\nBaseline VLA SR: {baseline_sr:.1f}% ({sum(r['vla_success'] for r in baseline_results)}/30)")

    # Analyze wheel magnitude distribution
    print("\n[ANALYSIS] VLA Wheel Action Magnitudes")
    print("-" * 60)
    all_baseline_wheel_mags = []
    for r in baseline_results:
        if r['wheel_acts']:
            w = np.array(r['wheel_acts'])
            mag = np.linalg.norm(w[-1])  # last step magnitude
            all_baseline_wheel_mags.append(mag)

    print(f"Baseline VLA |wheel| range: [{min(all_baseline_wheel_mags):.4f}, {max(all_baseline_wheel_mags):.4f}]")
    print(f"Baseline VLA |wheel| mean: {np.mean(all_baseline_wheel_mags):.4f}")

    # Compare with P-ctrl
    print("\n[ANALYSIS] P-ctrl Wheel Action Magnitudes")
    print("-" * 60)
    all_pctrl_wheel_mags = []
    for i, goal in enumerate(goals[:10]):  # Just first 10 for comparison
        goal_norm = np.clip(goal / 1.0, -1.0, 1.0)
        sim = LeKiWiSimURDF()
        succ, steps, dist, pctrl_wheel_acts = run_pctrl_episode(
            sim, goal, goal_norm, max_steps=200
        )
        if pctrl_wheel_acts:
            w = np.array(pctrl_wheel_acts)
            mag = np.linalg.norm(w[-1])
            all_pctrl_wheel_mags.append(mag)
            if i < 5:
                print(f"  Ep {i}: P-ctrl |wheel|={mag:.4f}, VLA |wheel|={all_baseline_wheel_mags[i]:.4f}")

    print(f"\nP-ctrl |wheel| mean: {np.mean(all_pctrl_wheel_mags):.4f}")
    print(f"VLA |wheel| mean: {np.mean(all_baseline_wheel_mags):.4f}")
    ratio = np.mean(all_pctrl_wheel_mags) / (np.mean(all_baseline_wheel_mags) + 1e-8)
    print(f"Ratio: {ratio:.1f}x")

    # Test: amplifier sweep
    print("\n[TEST 2] Amplifier Sweep (5 goals)")
    print("-" * 60)
    test_goals = [goals[i] for i in [0, 1, 2, 7, 8]]  # Mix of fail/succ

    for amp in [2.0, 5.0, 10.0, 15.0, 20.0]:
        results_amp = []
        for i, goal in enumerate(test_goals):
            goal_norm = np.clip(goal / 1.0, -1.0, 1.0)
            sim = LeKiWiSimURDF()
            succ, steps, dist, _, _ = run_episode_amplified(
                sim, goal, goal_norm, policy, amplifier=amp, max_steps=200
            )
            results_amp.append({'goal': goal.tolist(), 'success': succ, 'steps': steps, 'dist': dist})

        n_succ = sum(r['success'] for r in results_amp)
        print(f"  amp={amp:5.1f}x: {n_succ}/5 successes")

    # Save results
    results_data = {
        'phase': 176,
        'baseline_sr': baseline_sr,
        'baseline_results': [
            {k: v for k, v in r.items() if k not in ['arm_acts', 'wheel_acts']}
            for r in baseline_results
        ],
        'pctrl_wheel_mags': all_pctrl_wheel_mags,
        'vla_wheel_mags': all_baseline_wheel_mags,
        'pctrl_vla_ratio': float(ratio),
    }

    ts = time.strftime('%Y%m%d_%H%M')
    with open(f'results/phase176_{ts}.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[INFO] Saved to results/phase176_{ts}.json")
    print("\nPhase 176 complete.")


if __name__ == "__main__":
    main()
