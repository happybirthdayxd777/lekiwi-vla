#!/usr/bin/env python3
"""
Phase 173: VLA Failure Mode Diagnostic
=====================================
Run VLA on 30 restricted goals. For each episode record:
- Goal position
- VLA success/failure + steps
- P-ctrl success/failure + steps
- First/last VLA actions
- Trajectory (base positions)

This helps identify WHY VLA fails:
1. Does VLA fail on all goals, or specific regions?
2. Does VLA take systematically wrong actions (visual confusion)?
3. Does VLA have wheel saturation/oscillation issues?
4. Does VLA fail in specific goal quadrants?

Usage:
    python3 scripts/diagnose_vla_failures.py --n_ep 30
"""
import sys, os, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 173] Device: {DEVICE}")


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
            nn.Linear(goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(device)
        self.goal_proj = nn.Linear(128, 256).to(device)
        self.q_proj = nn.Linear(256, hidden).to(device)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.time_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        ).to(device)
        # Time embedding: 1 → 128 → 256
        self.time_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        ).to(device)

        # Output: state(256) + goal(128) + cross(512) + time(256) = 1152 → action
        self.action_head = nn.Sequential(
            nn.Linear(256 + 128 + hidden + 256, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        ).to(device)

        # Skip connection for action
        self.skip = nn.Linear(action_dim, action_dim, bias=False).to(device)

    def encode_goal(self, goal):
        return self.goal_mlp(goal)

    def forward(self, image, state, noisy_action, t):
        """Flow matching forward pass. Returns predicted velocity."""
        clip_feat = self.clip_encoder(image)  # [B, 50, 768]
        clip_proj = self.vision_proj(clip_feat)  # [B, 50, 512]
        goal_emb = self.goal_mlp(state[:, 9:11])  # [B, 128]
        goal_q = self.goal_proj(goal_emb)  # [B, 256]
        state_feat = self.state_net(state)  # [B, 256]

        q = self.q_proj(state_feat + goal_q)  # [B, 512]
        q = q.unsqueeze(1)  # [B, 1, 512]
        cross_out, _ = self.cross_attn(q, clip_proj, clip_proj)
        cross_out = self.cross_norm(cross_out + q)  # [B, 1, 512]
        t_feat = self.time_net(t)  # [B, 256]

        combined = torch.cat([
            state_feat,  # [B, 256]
            goal_emb,  # [B, 128]
            cross_out.squeeze(1),  # [B, 512]
            t_feat,  # [B, 256]
        ], dim=-1)  # [B, 1152]

        v_pred = self.action_head(combined)  # [B, 9]
        v_pred = v_pred + self.skip(noisy_action)  # residual
        return v_pred

    def infer(self, image, state, num_steps=4):
        """Denoise from pure noise to action in num_steps."""
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * (i * dt)
            v = self.forward(image, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)

    def inference(self, image, state, goal):
        """Legacy single-step inference (uses num_steps=4 denoising)."""
        return self.infer(image, state, num_steps=4)


def load_policy(ckpt_path, device=DEVICE):
    from transformers import CLIPModel
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


def generate_restricted_goals(n_episodes, seed=42):
    """Restricted quadrant: x ∈ [-0.1, 0.5], y ∈ [-0.3, 0.3]"""
    rng = np.random.default_rng(seed)
    goals = []
    for _ in range(n_episodes):
        gx = rng.uniform(-0.1, 0.5)
        gy = rng.uniform(-0.3, 0.3)
        goals.append([gx, gy])
    return np.array(goals, dtype=np.float32)


def run_vla_episode(sim, goal, goal_norm, policy, max_steps=200):
    """Run VLA policy. Returns (success, steps, final_dist, trajectory, actions)."""
    base_id = sim.model.body('base').id
    threshold = 0.15
    trajectory = []
    all_actions = []

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        trajectory.append(base_pos.tolist())

        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            # Success: use last action
            return True, step + 1, dist, trajectory, all_actions

        # Capture current image
        img = sim.render()
        img_t = resize_for_clip(img)

        # Build state: arm_pos(6) + wheel_vel(3) + goal(2) = 11D (MATCHES training)
        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
        wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
        state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

        # VLA inference
        img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(DEVICE)
        state_tensor = torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            raw_action = policy.infer(img_tensor, state_tensor, num_steps=4)
        raw_action_np = raw_action.squeeze(0).cpu().numpy()

        # De-normalize: wheel action [-0.5, 0.5] → servo units (MATCHES eval_matched_goals.py)
        wheel_denorm = raw_action_np[6:9] / 0.5 * 6.0
        wheel_denorm = np.clip(wheel_denorm, -6.0, 6.0)
        arm_action = raw_action_np[:6]
        wheel_action = wheel_denorm / 12.0

        full_action = np.concatenate([arm_action, wheel_action])
        all_actions.append(raw_action_np.tolist())
        sim.step(full_action)

    return False, max_steps, dist, trajectory, all_actions


def run_pctrl_episode(sim, goal, goal_norm, max_steps=200):
    """Run P-controller. Returns (success, steps, final_dist, trajectory)."""
    from sim_lekiwi_urdf import twist_to_contact_wheel_speeds

    base_id = sim.model.body('base').id
    threshold = 0.15
    trajectory = []
    kP = 0.1
    max_speed = 0.25

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        trajectory.append(base_pos.tolist())

        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, trajectory

        # P-controller action
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

    return False, max_steps, dist, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_ep", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str,
        default="results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt")
    args = parser.parse_args()

    import mujoco
    from sim_lekiwi_urdf import LeKiWiSimURDF

    ckpt_path = Path(args.ckpt)
    print(f"\n[Phase 173] VLA Failure Diagnostic")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  N episodes: {args.n_ep}, seed={args.seed}")

    policy = load_policy(ckpt_path, device=DEVICE)

    goals = generate_restricted_goals(args.n_ep, seed=args.seed)
    goal_norms = np.clip(goals / 1.0, -1.0, 1.0)

    results = []
    print(f"\n{'='*80}")
    for i, (goal, goal_norm) in enumerate(zip(goals, goal_norms)):
        print(f"\n[Episode {i:02d}] goal=({goal[0]:+.3f}, {goal[1]:+.3f})")

        # VLA episode
        sim_vla = LeKiWiSimURDF()
        sim_vla.reset()
        vla_ok, vla_steps, vla_dist, vla_traj, vla_acts = run_vla_episode(
            sim_vla, goal, goal_norm, policy, args.max_steps
        )

        # P-ctrl episode
        sim_pctrl = LeKiWiSimURDF()
        sim_pctrl.reset()
        pctrl_ok, pctrl_steps, pctrl_dist, pctrl_traj = run_pctrl_episode(
            sim_pctrl, goal, goal_norm, args.max_steps
        )

        # Categorize failure
        if vla_ok and pctrl_ok:
            category = "TIE_SUCCESS"
        elif not vla_ok and not pctrl_ok:
            category = "TIE_FAIL"
        elif vla_ok and not pctrl_ok:
            category = "VLA_WIN"
        else:
            category = "VLA_FAIL_PCTRL_OK"

        print(f"  VLA:   {'SUCC' if vla_ok else 'FAIL'} ({vla_steps}st, {vla_dist:.3f}m)")
        print(f"  P-ctrl: {'SUCC' if pctrl_ok else 'FAIL'} ({pctrl_steps}st, {pctrl_dist:.3f}m)")
        print(f"  Category: {category}")

        if not vla_ok and len(vla_acts) > 0:
            first_act = vla_acts[0]
            last_act = vla_acts[-1]
            print(f"  VLA first action: arm=[{first_act[0]:+.3f},{first_act[1]:+.3f},{first_act[2]:+.3f},{first_act[3]:+.3f},{first_act[4]:+.3f},{first_act[5]:+.3f}]")
            print(f"    wheel=[{first_act[6]:+.3f},{first_act[7]:+.3f},{first_act[8]:+.3f}]")
            print(f"  VLA last action:  arm=[{last_act[0]:+.3f},{last_act[1]:+.3f},{last_act[2]:+.3f},{last_act[3]:+.3f},{last_act[4]:+.3f},{last_act[5]:+.3f}]")
            print(f"    wheel=[{last_act[6]:+.3f},{last_act[7]:+.3f},{last_act[8]:+.3f}]")

            # Check for wheel saturation (oscillation indicator)
            wheel_acts = np.array([a[6:9] for a in vla_acts[-10:]])
            print(f"  VLA last10 wheel acts: mean=[{wheel_acts.mean(axis=0)[0]:+.3f},{wheel_acts.mean(axis=0)[1]:+.3f},{wheel_acts.mean(axis=0)[2]:+.3f}]")
            print(f"    std=[{wheel_acts.std(axis=0)[0]:.3f},{wheel_acts.std(axis=0)[1]:.3f},{wheel_acts.std(axis=0)[2]:.3f}]")

        if not pctrl_ok:
            # P-ctrl failure = physics limit reached (far goal)
            pctrl_traj_arr = np.array(pctrl_traj)
            final_pos = pctrl_traj_arr[-1]
            print(f"  P-ctrl final pos: ({final_pos[0]:+.3f}, {final_pos[1]:+.3f}), dist_to_goal={pctrl_dist:.3f}")

        ep_result = {
            "episode": i,
            "goal": goal.tolist(),
            "vla_success": vla_ok,
            "vla_steps": vla_steps,
            "vla_final_dist": float(vla_dist),
            "pctrl_success": pctrl_ok,
            "pctrl_steps": pctrl_steps,
            "pctrl_final_dist": float(pctrl_dist),
            "category": category,
            "vla_trajectory": vla_traj,
            "pctrl_trajectory": pctrl_traj,
        }
        if len(vla_acts) > 0:
            ep_result["vla_first_action"] = vla_acts[0]
            ep_result["vla_last_action"] = vla_acts[-1]
        results.append(ep_result)

    # Summary statistics
    vla_srs = sum(r["vla_success"] for r in results)
    pctrl_srs = sum(r["pctrl_success"] for r in results)
    categories = {}
    for r in results:
        c = r["category"]
        categories[c] = categories.get(c, 0) + 1

    print(f"\n{'='*80}")
    print(f"[SUMMARY] Phase 173 — VLA Failure Diagnostic")
    print(f"  VLA SR:   {vla_srs}/{args.n_ep} = {vla_srs/args.n_ep*100:.1f}%")
    print(f"  P-ctrl SR: {pctrl_srs}/{args.n_ep} = {pctrl_srs/args.n_ep*100:.1f}%")
    print(f"  Categories: {categories}")

    # Failure analysis by goal position
    fail_vla = [r for r in results if not r["vla_success"]]
    if fail_vla:
        fail_goals = np.array([r["goal"] for r in fail_vla])
        print(f"\n[VLA FAILURES] x∈[{fail_goals[:,0].min():.3f}, {fail_goals[:,0].max():.3f}], "
              f"y∈[{fail_goals[:,1].min():.3f}, {fail_goals[:,1].max():.3f}]")

        # Check wheel action patterns in failures
        wheel_acts_all = []
        for r in fail_vla:
            if "vla_last_action" in r:
                wheel_acts_all.append(r["vla_last_action"][6:9])
        if wheel_acts_all:
            wa = np.array(wheel_acts_all)
            print(f"  Failure last-wheel-actions: mean=[{wa.mean(axis=0)[0]:+.3f},"
                  f"{wa.mean(axis=0)[1]:+.3f},{wa.mean(axis=0)[2]:+.3f}]")
            print(f"    std=[{wa.std(axis=0)[0]:.3f},{wa.std(axis=0)[1]:.3f},{wa.std(axis=0)[2]:.3f}]")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_path = ROOT / "results" / f"phase173_diagnostic_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "vla_sr": vla_srs / args.n_ep,
                "pctrl_sr": pctrl_srs / args.n_ep,
                "categories": categories,
                "n_episodes": args.n_ep,
                "seed": args.seed,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
