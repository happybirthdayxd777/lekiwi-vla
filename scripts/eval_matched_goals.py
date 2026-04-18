#!/usr/bin/env python3
"""
Phase 156 — Matched-Goal VLA vs P-ctrl Evaluation
==================================================
Runs VLA and P-ctrl on IDENTICAL goals to fairly compare performance.
This directly addresses PRIORITY 1 from Phase 155c.

Key insight: Previous evals (phase155b) used independent random goals for
VLA vs P-ctrl — making comparison meaningless due to goal distribution variance.

Design:
  1. Pre-generate N goals with fixed seed (seed=42)
  2. Run VLA on all goals → record per-goal success/failure
  3. Run P-ctrl on SAME goals → record per-goal success/failure
  4. Compare: per-goal wins/losses/ties + aggregate SR

Also tests goal-restricted eval (PRIORITY 2):
  - Filter to reachable quadrant: x ∈ [-0.1, 0.5], y ∈ [-0.3, 0.3]
  - P-ctrl should be ~100% on reachable goals
  - Fair VLA vs P-ctrl comparison on this subset

Usage:
    python3 scripts/eval_matched_goals.py --n_ep 30
    python3 scripts/eval_matched_goals.py --n_ep 30 --restricted
    python3 scripts/eval_matched_goals.py --n_ep 100 --seed 999
"""

import sys, os, time, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ── Setup ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 156] Device: {DEVICE}")


# ── CLIP Spatial Encoder ──────────────────────────────────────────────────────
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
            hidden = outputs.last_hidden_state  # [B, 50, 768]
        return hidden


# ── GoalConditionedPolicy ──────────────────────────────────────────────────────
class GoalConditionedPolicy(nn.Module):
    """Phase 152: Same architecture as sweep_epochs_lr.py."""
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
        ).to(device)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        ).to(device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, batch_first=True
        ).to(device)
        self.cross_norm = nn.LayerNorm(hidden)
        fusion_dim = 770 + 256 + 128 + hidden + 256  # 1922
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        ).to(device)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        ).to(device)
        self.output_bounds = (0.0, 0.5)

    def forward(self, image, state, noisy_action, t):
        B = image.shape[0]
        device = self.device
        vision_tokens = self.clip_encoder(image)
        cls_token = vision_tokens[:, 0, :]
        state_feat = self.state_net(state)
        goal_xy = state[:, 9:11]
        goal_emb = self.goal_mlp(goal_xy)
        t_feat = self.time_mlp(t)
        vision_proj = self.vision_proj(vision_tokens)
        goal_emb_proj = self.goal_proj(goal_emb)
        q_features = state_feat + goal_emb_proj + t_feat
        q_proj = self.q_proj(q_features)
        cross_out, _ = self.cross_attn(q_proj.unsqueeze(1), vision_proj, vision_proj)
        cross_out = cross_out.squeeze(1)
        cls_with_goal = torch.cat([cls_token, goal_xy.to(device)], dim=1)
        fusion_in = torch.cat([cls_with_goal, state_feat, goal_emb, cross_out, t_feat], dim=1)
        fusion_feat = self.fusion_net(fusion_in)
        action_delta = self.action_head(fusion_feat)
        action = noisy_action.to(device) + action_delta
        arm_action = action[:, :6]
        wheel_action = torch.clamp(action[:, 6:9], self.output_bounds[0], self.output_bounds[1])
        action = torch.cat([arm_action, wheel_action], dim=1)
        return action

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        """4-step Euler ODE inference."""
        action = torch.randn(image.shape[0], 9, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vision_tokens = self.clip_encoder(image)
            cls_token = vision_tokens[:, 0, :]
            state_feat = self.state_net(state)
            goal_xy = state[:, 9:11]
            goal_emb = self.goal_mlp(goal_xy)
            t_feat = self.time_mlp(t)
            vision_proj = self.vision_proj(vision_tokens)
            goal_emb_proj = self.goal_proj(goal_emb)
            q_features = state_feat + goal_emb_proj + t_feat
            q_proj = self.q_proj(q_features)
            cross_out, _ = self.cross_attn(q_proj.unsqueeze(1), vision_proj, vision_proj)
            cross_out = cross_out.squeeze(1)
            cls_with_goal = torch.cat([cls_token, goal_xy.to(self.device)], dim=1)
            fusion_in = torch.cat([cls_with_goal, state_feat, goal_emb, cross_out, t_feat], dim=1)
            fusion_feat = self.fusion_net(fusion_in)
            action_delta = self.action_head(fusion_feat)
            action = action - dt * action_delta
            arm_action = action[:, :6]
            wheel_action = torch.clamp(action[:, 6:9], self.output_bounds[0], self.output_bounds[1])
            action = torch.cat([arm_action, wheel_action], dim=1)
        return action


# ── Omni-wheel IK (from eval_phase155b.py) ─────────────────────────────────────
def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    R = 0.05
    WHEEL_POSITIONS = np.array([
        [ 0.1732,  0.0],
        [-0.0866,  0.15],
        [-0.0866, -0.15],
    ], dtype=np.float64)
    _JOINT_AXES = np.array([
        [-0.866025,  0.0,  0.5],
        [ 0.866025,  0.0,  0.5],
        [ 0.0,       0.0, -1.0],
    ], dtype=np.float64)
    wheel_speeds = np.zeros(3, dtype=np.float64)
    for i in range(3):
        wheel_vel = np.array([vx - wz * WHEEL_POSITIONS[i, 1],
                              vy + wz * WHEEL_POSITIONS[i, 0], 0.0])
        angular_speed = np.dot(wheel_vel, _JOINT_AXES[i]) / R
        wheel_speeds[i] = angular_speed
    return wheel_speeds


# ── Load policy ────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, device=DEVICE):
    print(f"\n[LOAD] {ckpt_path}")
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9,
                                   device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("policy_state_dict", ckpt)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    policy.device = device
    print(f"[LOAD] epoch={ckpt.get('epoch','?')}, eval_sr={ckpt.get('eval_sr','?')}")
    return policy


# ── Resize helper ──────────────────────────────────────────────────────────────
def resize_for_clip(img):
    pil = Image.fromarray(img)
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    return img_np.transpose(2, 0, 1)


# ── Generate matched goals ─────────────────────────────────────────────────────
def generate_goals(n_episodes, seed=42, restricted=False):
    """
    Generate N random goals for evaluation.
    With restricted=True: filter to physically reachable quadrant.
    """
    rng = np.random.default_rng(seed)
    goals = []
    while len(goals) < n_episodes:
        if restricted:
            # Reachable quadrant (empirical — avoid -X far field)
            gx = rng.uniform(-0.1, 0.5)
            gy = rng.uniform(-0.3, 0.3)
        else:
            gx = rng.uniform(-0.5, 0.5)
            gy = rng.uniform(-0.5, 0.5)
        goals.append([gx, gy])
    return np.array(goals, dtype=np.float32)


# ── Run episode with policy ────────────────────────────────────────────────────
def run_episode(sim, goal, goal_norm, policy, max_steps=200, use_pctrl=False):
    """
    Run one episode. Returns (success, steps, final_dist).
    
    Parameters
    ----------
    sim : LeKiWiSimURDF
    goal : np.ndarray [2] — raw goal position
    goal_norm : np.ndarray [2] — normalized goal for policy
    policy : GoalConditionedPolicy or None (for P-ctrl)
    use_pctrl : bool — if True, use P-controller instead of policy
    """
    base_id = sim.model.body('base').id
    threshold = 0.15

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist

        if use_pctrl:
            # P-controller
            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            v_desired = np.array([dx, dy]) * 2.0
            vx, vy = np.clip(v_desired, -0.3, 0.3)
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            wheel_speeds = np.clip(wheel_speeds, -6.0, 6.0)
            arm_action = np.zeros(6)
            wheel_action = wheel_speeds / 12.0
            full_action = np.concatenate([arm_action, wheel_action])
        else:
            # VLA policy
            img = sim.render()
            img_chw = resize_for_clip(img)
            img_t = torch.from_numpy(img_chw[np.newaxis, ...]).to(DEVICE)

            arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
            wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
            state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
            state_t = torch.from_numpy(state_11d[np.newaxis, ...]).to(DEVICE)

            raw_action = policy.infer(img_t, state_t, num_steps=4)[0]
            raw_action_np = raw_action.cpu().numpy()

            wheel_denorm = raw_action_np[6:9] / 0.5 * 6.0
            wheel_denorm = np.clip(wheel_denorm, -6.0, 6.0)
            arm_action = raw_action_np[:6]
            wheel_action = wheel_denorm / 12.0
            full_action = np.concatenate([arm_action, wheel_action])

        obs, _, _, _ = sim.step(full_action)

    # Failed
    base_pos = sim.data.xpos[base_id, :2]
    dist = np.linalg.norm(base_pos - goal)
    return False, max_steps, dist


# ── Matched evaluation ────────────────────────────────────────────────────────
def matched_evaluation(policy, n_episodes=30, max_steps=200, seed=42,
                       restricted=False, verbose=True):
    """
    Run VLA and P-ctrl on IDENTICAL goals.
    Returns per-goal comparison results.
    """
    from sim_lekiwi_urdf import LeKiWiSimURDF

    goals = generate_goals(n_episodes, seed=seed, restricted=restricted)
    goal_norms = np.clip(goals / 1.0, -1.0, 1.0)

    if verbose:
        print(f"\n[MATCHED EVAL] {n_episodes} episodes, seed={seed}, restricted={restricted}")
        print(f"  Goals: x∈[{goals[:,0].min():.2f}, {goals[:,0].max():.2f}], "
              f"y∈[{goals[:,1].min():.2f}, {goals[:,1].max():.2f}]")

    results = {
        "goals": goals.tolist(),
        "vla": {"success": [], "steps": [], "dist": []},
        "pctrl": {"success": [], "steps": [], "dist": []},
    }

    vla_wins = 0
    pctrl_wins = 0
    ties = 0

    t0 = time.time()
    for i, (goal, goal_norm) in enumerate(zip(goals, goal_norms)):
        # VLA episode
        sim_vla = LeKiWiSimURDF()
        sim_vla.reset()
        vla_ok, vla_steps, vla_dist = run_episode(
            sim_vla, goal, goal_norm, policy, max_steps, use_pctrl=False
        )
        results["vla"]["success"].append(vla_ok)
        results["vla"]["steps"].append(vla_steps)
        results["vla"]["dist"].append(float(vla_dist))

        # P-ctrl episode (fresh sim)
        sim_pctrl = LeKiWiSimURDF()
        sim_pctrl.reset()
        pctrl_ok, pctrl_steps, pctrl_dist = run_episode(
            sim_pctrl, goal, goal_norm, None, max_steps, use_pctrl=True
        )
        results["pctrl"]["success"].append(pctrl_ok)
        results["pctrl"]["steps"].append(pctrl_steps)
        results["pctrl"]["dist"].append(float(pctrl_dist))

        # Per-goal comparison
        if vla_ok and not pctrl_ok:
            vla_wins += 1
            outcome = "VLA WIN"
        elif pctrl_ok and not vla_ok:
            pctrl_wins += 1
            outcome = "P-CTRL WIN"
        elif vla_ok and pctrl_ok:
            ties += 1
            outcome = f"TIE (VLA:{vla_steps}st, P:{pctrl_steps}st)"
        else:
            ties += 1
            outcome = f"TIE FAIL"

        if verbose and (i < 10 or not vla_ok or not pctrl_ok):
            print(f"  ep{i:02d}: {outcome} | VLA:{int(vla_ok)}({vla_steps}st,{vla_dist:.3f}m) "
                  f"P-ctrl:{int(pctrl_ok)}({pctrl_steps}st,{pctrl_dist:.3f}m) | "
                  f"goal=({goal[0]:+.2f},{goal[1]:+.2f})")

    elapsed = time.time() - t0

    # Aggregate stats
    vla_sr = sum(results["vla"]["success"]) / n_episodes
    pctrl_sr = sum(results["pctrl"]["success"]) / n_episodes

    print(f"\n{'='*60}")
    print(f"[MATCHED RESULT] seed={seed}, restricted={restricted}")
    print(f"  VLA   SR: {vla_sr*100:.1f}% ({sum(results['vla']['success'])}/{n_episodes})")
    print(f"  P-ctrl SR: {pctrl_sr*100:.1f}% ({sum(results['pctrl']['success'])}/{n_episodes})")
    print(f"  Gap: {(pctrl_sr - vla_sr)*100:+.1f}pp (positive = VLA worse)")
    print(f"  Per-goal: VLA wins={vla_wins}, P-ctrl wins={pctrl_wins}, ties={ties}")
    print(f"  VLA mean steps: {np.mean(results['vla']['steps']):.1f}")
    print(f"  P-ctrl mean steps: {np.mean(results['pctrl']['steps']):.1f}")
    print(f"  Eval time: {elapsed:.0f}s")
    print(f"{'='*60}")

    return results, {
        "vla_sr": vla_sr,
        "pctrl_sr": pctrl_sr,
        "gap_pp": (pctrl_sr - vla_sr) * 100,
        "vla_wins": vla_wins,
        "pctrl_wins": pctrl_wins,
        "ties": ties,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 156: Matched-goal VLA vs P-ctrl eval")
    parser.add_argument("--ckpt", type=str,
                        default="~/hermes_research/lekiwi_vla/results/phase154_sweep_lr2e-05_ep3_20260418_0754/best_policy.pt",
                        help="Path to best_policy.pt")
    parser.add_argument("--n_ep", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for goal generation")
    parser.add_argument("--restricted", action="store_true",
                        help="Use goal-restricted eval (reachable quadrant only)")
    parser.add_argument("--pctrl_only", action="store_true", help="Only run P-ctrl (no VLA)")
    args = parser.parse_args()

    import mujoco
    from sim_lekiwi_urdf import LeKiWiSimURDF

    if args.pctrl_only:
        print(f"\n[P-CTRL ONLY] seed={args.seed}, restricted={args.restricted}")
        goals = generate_goals(args.n_ep, seed=args.seed, restricted=args.restricted)
        goal_norms = np.clip(goals / 1.0, -1.0, 1.0)

        successes = 0
        steps_list = []
        dists = []
        t0 = time.time()
        for i, (goal, goal_norm) in enumerate(zip(goals, goal_norms)):
            sim = LeKiWiSimURDF()
            sim.reset()
            ok, steps, dist = run_episode(sim, goal, goal_norm, None, args.max_steps, use_pctrl=True)
            successes += int(ok)
            steps_list.append(steps)
            dists.append(dist)
            print(f"  ep{i:02d}: {'SUCC' if ok else 'FAIL'} ({steps}st, {dist:.3f}m) "
                  f"goal=({goal[0]:+.2f},{goal[1]:+.2f})")

        elapsed = time.time() - t0
        sr = successes / args.n_ep
        print(f"\n[P-CTRL RESULT] SR={sr*100:.1f}% ({successes}/{args.n_ep})")
        print(f"  Mean steps: {np.mean(steps_list):.1f}")
        print(f"  Mean dist: {np.mean(dists):.3f}m")
        print(f"  Time: {elapsed:.0f}s")
        return

    ckpt_path = Path(os.path.expanduser(args.ckpt))
    print(f"\n[Phase 156] Matched-goal VLA vs P-ctrl eval")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  N episodes: {args.n_ep}, seed={args.seed}, restricted={args.restricted}")

    policy = load_policy(ckpt_path, device=DEVICE)

    results, summary = matched_evaluation(
        policy, n_episodes=args.n_ep, max_steps=args.max_steps,
        seed=args.seed, restricted=args.restricted
    )

    # Save results
    eval_label = f"phase156_matched_seed{args.seed}"
    if args.restricted:
        eval_label += "_restricted"
    output_path = ROOT / "results" / f"{eval_label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "checkpoint": str(ckpt_path),
        "summary": summary,
        "goals": results["goals"],
        "vla": results["vla"],
        "pctrl": results["pctrl"],
        "config": {
            "n_episodes": args.n_ep,
            "seed": args.seed,
            "restricted": args.restricted,
            "max_steps": args.max_steps,
            "eval_time": time.strftime("%Y%m%d_%H%M"),
        }
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
