#!/usr/bin/env python3
"""
Phase 155b — Statistical Significance Eval for Best VLA Checkpoint
===================================================================
Evaluates the Phase 154 best checkpoint: lr=2e-5, ep=3 → 70% SR (10ep eval).
Best checkpoint: results/phase154_sweep_lr2e-05_ep3_20260418_0754/best_policy.pt

Uses GoalConditionedPolicy from sweep_epochs_lr.py with state_dim=11, goal_dim=2.

Statistical significance via 10 independent episodes (threshold=0.15m).
P-controller baseline: 100% SR (15ep eval)

Usage:
    python3 scripts/eval_phase155b.py
    python3 scripts/eval_phase155b.py --ckpt ~/hermes_research/lekiwi_vla/results/phase154_sweep_lr2e-05_ep3_20260418_0754/best_policy.pt --n_ep 10
"""

import sys, os, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import mujoco

# ── Setup ────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 155b] Device: {DEVICE}")

# ── CLIP Spatial Encoder (same as sweep_epochs_lr.py) ─────────────────────────
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
        """images: [B, 3, 224, 224] in [0,1]. Returns: [B, 50, 768] spatial tokens."""
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.float32)
        pixel_values = pixel_values.to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, 50, 768]
        return hidden


# ── GoalConditionedPolicy (same as sweep_epochs_lr.py) ────────────────────────
class GoalConditionedPolicy(nn.Module):
    """
    Phase 152: Strengthened goal MLP (2→256→128) + direct goal concat to CLIP [CLS].
    Same architecture as sweep_epochs_lr.py — used for Phase 154 eval.
    state_dim=11 (arm_pos6 + wheel_vel3 + goal_xy2), goal_dim=2, action_dim=9.
    """
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
            t = torch.full([image.shape[0], 1.0 - i * dt], device=self.device)
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


# ── Omni-wheel IK ──────────────────────────────────────────────────────────────
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


# ── Resize helper ─────────────────────────────────────────────────────────────
def resize_for_clip(img):
    pil = Image.fromarray(img)
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    return img_np.transpose(2, 0, 1)  # [3, 224, 224]


# ── Evaluate ────────────────────────────────────────────────────────────────────
def evaluate_policy(policy, n_episodes=10, threshold=0.15, max_steps=200, render=False):
    """Evaluate GoalConditionedPolicy on LeKiWiSimURDF with random goals."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    successes = 0
    steps_list = []
    distances = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        # Random goal in [-0.5, 0.5] × [-0.5, 0.5]
        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])
        goal_norm = np.clip(np.array([gx, gy]) / 1.0, -1.0, 1.0)  # normalize to [-1,1]

        if render:
            cam = mujoco.viewer.launch_passive(sim.model, sim.data)
        else:
            cam = None

        for step in range(max_steps):
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(base_pos - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step + 1)
                distances.append(dist)
                if cam:
                    cam.close()
                break

            # Capture observation
            img = sim.render()
            img_chw = resize_for_clip(img)
            img_t = torch.from_numpy(img_chw[np.newaxis, ...]).to(DEVICE)

            # Build 11D state: arm_pos(6) + wheel_vel(3) + goal_xy(2)
            arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
            wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
            state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
            state_t = torch.from_numpy(state_11d[np.newaxis, ...]).to(DEVICE)

            # Policy inference (4-step Euler)
            raw_action = policy.infer(img_t, state_t, num_steps=4)[0]  # (9,)
            raw_action_np = raw_action.cpu().numpy()

            # Denormalize: wheel portion is in [0, 0.5] → scale to physical units
            wheel_denorm = raw_action_np[6:9] / 0.5 * 6.0  # [-6, 6] rad/s
            wheel_denorm = np.clip(wheel_denorm, -6.0, 6.0)

            arm_action = raw_action_np[:6]
            wheel_action = wheel_denorm / 12.0  # normalize for sim step
            full_action = np.concatenate([arm_action, wheel_action])

            obs, _, _, _ = sim.step(full_action)

            if cam:
                cam.sync()
        else:
            # Failed to reach goal within max_steps
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(base_pos - goal)
            steps_list.append(max_steps)
            distances.append(dist)
            if cam:
                cam.close()

    sr = successes / n_episodes
    mean_steps = np.mean(steps_list) if steps_list else max_steps
    std_steps = np.std(steps_list) if len(steps_list) > 1 else 0.0
    mean_dist = np.mean(distances)

    # 95% CI for SR (Wilson score interval)
    z = 1.96
    p = sr
    n = n_episodes
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    ci_low = max(0.0, center - half)
    ci_high = min(1.0, center + half)

    return {
        "success_rate": sr,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_episodes": n_episodes,
        "successes": successes,
        "mean_steps": float(mean_steps),
        "std_steps": float(std_steps),
        "mean_final_dist": float(mean_dist),
        "steps_list": steps_list,
        "distances": [float(d) for d in distances],
    }


# ── P-controller baseline ────────────────────────────────────────────────────
def evaluate_pcontroller(n_episodes=15, threshold=0.15, max_steps=200):
    """P-controller baseline (always works = 100% SR when threshold=0.15)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    successes = 0
    steps_list = []
    distances = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])

        for step in range(max_steps):
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(base_pos - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step + 1)
                distances.append(dist)
                break

            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            v_desired = np.array([dx, dy]) * 2.0
            vx, vy = np.clip(v_desired, -0.3, 0.3)
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            wheel_speeds = np.clip(wheel_speeds, -6.0, 6.0)
            arm_action = np.zeros(6)
            wheel_action = wheel_speeds / 12.0
            full_action = np.concatenate([arm_action, wheel_action])
            obs, _, _, _ = sim.step(full_action)
        else:
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(base_pos - goal)
            steps_list.append(max_steps)
            distances.append(dist)

    sr = successes / n_episodes
    return {
        "success_rate": sr,
        "n_episodes": n_episodes,
        "successes": successes,
        "mean_steps": float(np.mean(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "mean_final_dist": float(np.mean(distances)),
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 155b: Statistical significance eval")
    parser.add_argument("--ckpt", type=str,
                        default="~/hermes_research/lekiwi_vla/results/phase154_sweep_lr2e-05_ep3_20260418_0754/best_policy.pt",
                        help="Path to best_policy.pt")
    parser.add_argument("--n_ep", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--threshold", type=float, default=0.15, help="Success threshold (m)")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable MuJoCo rendering")
    parser.add_argument("--pctrl_only", action="store_true", help="Only run P-controller baseline")
    args = parser.parse_args()

    import mujoco
    from sim_lekiwi_urdf import LeKiWiSimURDF

    if args.pctrl_only:
        print("\n[P-CTRL BASELINE]")
        result = evaluate_pcontroller(n_episodes=args.n_ep, threshold=args.threshold, max_steps=args.max_steps)
        print(f"  SR: {result['success_rate']*100:.0f}% ({result['successes']}/{result['n_episodes']})")
        print(f"  Mean steps: {result['mean_steps']:.1f} ± {result['std_steps']:.1f}")
        print(f"  Mean final dist: {result['mean_final_dist']:.3f}m")
        return

    ckpt_path = Path(os.path.expanduser(args.ckpt))
    print(f"\n[Phase 155b] Evaluating: {ckpt_path.name}")
    print(f"[Config] n_ep={args.n_ep}, threshold={args.threshold}m, max_steps={args.max_steps}")

    policy = load_policy(ckpt_path, device=DEVICE)

    print(f"\n[GOAL-CONDITIONED VLA] Running {args.n_ep} episodes...")
    t0 = time.time()
    result = evaluate_policy(policy, n_episodes=args.n_ep, threshold=args.threshold,
                           max_steps=args.max_steps, render=args.render)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"[VLA RESULT] {result['success_rate']*100:.0f}% SR ({result['successes']}/{result['n_episodes']} eps)")
    print(f"  95% CI: [{result['ci_95_low']*100:.1f}%, {result['ci_95_high']*100:.1f}%]")
    print(f"  Mean steps: {result['mean_steps']:.1f} ± {result['std_steps']:.1f}")
    print(f"  Mean final dist: {result['mean_final_dist']:.3f}m")
    print(f"  Eval time: {elapsed:.0f}s")
    print(f"{'='*50}")

    # P-controller baseline for comparison
    print(f"\n[P-CTRL BASELINE] Running {args.n_ep} episodes...")
    pctrl = evaluate_pcontroller(n_episodes=args.n_ep, threshold=args.threshold, max_steps=args.max_steps)
    print(f"  SR: {pctrl['success_rate']*100:.0f}% ({pctrl['successes']}/{pctrl['n_episodes']})")
    print(f"  Mean steps: {pctrl['mean_steps']:.1f} ± {pctrl['std_steps']:.1f}")

    # Gap
    gap = pctrl['success_rate'] - result['success_rate']
    print(f"\n[VLA vs P-ctrl gap]: {gap*100:.0f}pp")

    # Save results
    output_path = ROOT / "results" / "phase155b_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "checkpoint": str(ckpt_path),
        "vla": result,
        "pctrl": pctrl,
        "gap_pp": float(gap * 100),
        "config": {
            "n_episodes": args.n_ep,
            "threshold": args.threshold,
            "max_steps": args.max_steps,
            "eval_time": time.strftime("%Y%m%d_%H%M"),
        }
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    main()
