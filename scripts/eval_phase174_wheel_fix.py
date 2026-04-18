#!/usr/bin/env python3
"""
Phase 174: Full VLA Eval with CORRECTED Wheel Action Scale
==========================================================
Tests VLA on 30 restricted goals with the CORRECTED wheel action scale.

ROOT CAUSE (Phase 174):
- diagnose_vla_failures.py was using wheel_action = raw_wheel * 1.0
- But training data wheel_action = wheel_speeds/12.0 with max=0.5/12.0=0.0417
- VLA raw wheel is in [-0.5, 0.5] normalized space
- Scale factor: 0.0417/0.5 = 0.0834
- WRONG: 12x too large wheel actions!

CORRECT: wheel_action = raw_wheel * 0.0834

Quick test on 5 failed episodes: 4/5 SUCC (vs 0/5 with wrong scale)

Usage:
    python3 scripts/eval_phase174_wheel_fix.py
"""
import sys, os, json, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import mujoco

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 174] Device: {DEVICE}")


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
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    return img_np.transpose(2, 0, 1)


# FIXED: wheel action scale = 0.0834 (was 1.0 in diagnose_vla_failures.py)
WHEEL_SCALE = 0.0834  # 0.5/12.0 / 0.5 = 0.0417/0.5


def generate_restricted_goals(n_episodes, seed=42):
    """Restricted quadrant: x ∈ [-0.1, 0.5], y ∈ [-0.3, 0.3]"""
    rng = np.random.default_rng(seed)
    goals = []
    for _ in range(n_episodes):
        gx = rng.uniform(-0.1, 0.5)
        gy = rng.uniform(-0.3, 0.3)
        goals.append([gx, gy])
    return np.array(goals, dtype=np.float32)


def run_vla_episode_fixed_scale(sim, goal, goal_norm, policy, max_steps=200):
    """Run VLA with CORRECTED wheel action scale."""
    base_id = sim.model.body('base').id
    threshold = 0.15

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist

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

        # FIXED: correct wheel scale (was 1.0, should be 0.0834)
        raw_wheel = raw_np[6:9]
        wheel_action = raw_wheel * WHEEL_SCALE
        wheel_action = np.clip(wheel_action, -0.0417, 0.0417)  # cap at P-ctrl max

        # Arm: use VLA output as-is (clip to reasonable range)
        arm_action = np.clip(raw_np[:6], -0.5, 0.5)

        action = np.concatenate([arm_action, wheel_action])
        sim.step(action)

    return False, max_steps, dist


def run_pctrl_episode(sim, goal, goal_norm, max_steps=200):
    """P-controller baseline."""
    from sim_lekiwi_urdf import twist_to_contact_wheel_speeds

    base_id = sim.model.body('base').id
    threshold = 0.15
    kP = 0.1
    max_speed = 0.25

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist

        dx = goal[0] - base_pos[0]
        dy = goal[1] - base_pos[1]
        d = np.linalg.norm([dx, dy])
        if d < 0.05:
            wheel_speeds = np.zeros(3)
        else:
            v_mag = min(kP * d, max_speed)
            ws = twist_to_contact_wheel_speeds(v_mag * dx / d, v_mag * dy / d)
            wheel_speeds = np.clip(ws, -0.5, 0.5)

        action = np.zeros(9, dtype=np.float32)
        action[6:9] = wheel_speeds
        sim.step(action)

    return False, max_steps, dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ep", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str,
        default="results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt")
    args = parser.parse_args()

    from sim_lekiwi_urdf import LeKiWiSimURDF

    print(f"\n[Phase 174] VLA Eval with CORRECTED Wheel Scale")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  N episodes: {args.n_ep}, seed={args.seed}")
    print(f"  WHEEL_SCALE: {WHEEL_SCALE} (FIXED from 1.0)")
    print(f"  P-ctrl baseline: kP=0.1, max_speed=0.25")

    policy = load_policy(Path(args.ckpt), device=DEVICE)

    goals = generate_restricted_goals(args.n_ep, seed=args.seed)
    goal_norms = np.clip(goals / 1.0, -1.0, 1.0)

    vla_ok_count = 0
    pctrl_ok_count = 0
    results = []

    print(f"\n{'='*70}")
    for i, (goal, goal_norm) in enumerate(zip(goals, goal_norms)):
        # VLA episode
        sim_vla = LeKiWiSimURDF()
        sim_vla.reset()
        vla_ok, vla_steps, vla_dist = run_vla_episode_fixed_scale(
            sim_vla, goal, goal_norm, policy, args.max_steps
        )

        # P-ctrl episode
        sim_pctrl = LeKiWiSimURDF()
        sim_pctrl.reset()
        pctrl_ok, pctrl_steps, pctrl_dist = run_pctrl_episode(
            sim_pctrl, goal, goal_norm, args.max_steps
        )

        vla_ok_count += int(vla_ok)
        pctrl_ok_count += int(pctrl_ok)

        status = "VLA_WIN" if vla_ok and not pctrl_ok else ("TIE" if vla_ok == pctrl_ok else "PCTRL_WIN")
        vla_str = "SUCC" if vla_ok else "FAIL"
        pctrl_str = "SUCC" if pctrl_ok else "FAIL"
        print(f"  Ep {i:02d} ({goal[0]:+.3f},{goal[1]:+.3f}): VLA={vla_str}({vla_steps}st) P={pctrl_str}({pctrl_steps}st) [{status}]")

        results.append({
            "episode": i, "goal": goal.tolist(),
            "vla_success": vla_ok, "vla_steps": vla_steps, "vla_final_dist": float(vla_dist),
            "pctrl_success": pctrl_ok, "pctrl_steps": pctrl_steps, "pctrl_final_dist": float(pctrl_dist),
            "status": status,
        })

    vla_sr = vla_ok_count / args.n_ep * 100
    pctrl_sr = pctrl_ok_count / args.n_ep * 100

    print(f"\n{'='*70}")
    print(f"[SUMMARY] Phase 174 — CORRECTED Wheel Scale")
    print(f"  VLA SR:    {vla_ok_count}/{args.n_ep} = {vla_sr:.1f}%")
    print(f"  P-ctrl SR: {pctrl_ok_count}/{args.n_ep} = {pctrl_sr:.1f}%")
    print(f"  WHEEL_SCALE: {WHEEL_SCALE} (was 1.0 — 12x too large!)")
    print(f"  vs Phase 173: VLA 53.3% with WRONG scale")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M")
    output = {
        "summary": {"vla_sr": vla_sr, "pctrl_sr": pctrl_sr, "n_episodes": args.n_ep,
                    "seed": args.seed, "wheel_scale": WHEEL_SCALE},
        "results": results,
    }
    out_path = ROOT / "results" / f"phase174_eval_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
