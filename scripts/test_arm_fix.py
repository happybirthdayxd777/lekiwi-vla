#!/usr/bin/env python3
"""
Phase 174: Arm Saturation Fix — Test Post-Processing on Failed Episodes
=======================================================================
Replays the 14 VLA failures from Phase 173 with arm action post-processing.

Arm saturation root cause (Phase 173):
- VLA outputs arm j4=±0.5, j5=±0.5 (clip limit) in ALL 14 failures
- This physically obstructs the camera → wrong visual → wrong correction
- The wheel actions are actually correct (biased toward +X from step 0)

Fix strategy: post-process VLA actions to clip or zero arm actions.
Test multiple variants to find the best fix.

Usage:
    python3 scripts/test_arm_fix.py --variant clip_03
    python3 scripts/test_arm_fix.py --variant zero_arm
    python3 scripts/test_arm_fix.py --all_variants
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


def postprocess_arm(raw_action_np, variant="clip_03"):
    """Post-process VLA arm actions to fix saturation."""
    arm_action = raw_action_np[:6].copy()
    wheel_action = raw_action_np[6:9].copy()

    if variant == "raw":
        pass  # No change
    elif variant == "zero_arm":
        arm_action = np.zeros(6)
    elif variant == "clip_03":
        arm_action = np.clip(arm_action, -0.3, 0.3)
    elif variant == "clip_02":
        arm_action = np.clip(arm_action, -0.2, 0.2)
    elif variant == "clip_01":
        arm_action = np.clip(arm_action, -0.1, 0.1)
    elif variant == "decay_j4j5":
        # Zero out j4 and j5 (indices 3,4) which are the saturating joints
        arm_action[3] = 0.0
        arm_action[4] = 0.0
    elif variant == "neutral_pose":
        # Set arm to a neutral "home" pose instead of zero
        arm_action = np.array([0.0, -0.2, 0.1, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return np.concatenate([arm_action, wheel_action])


def run_vla_episode_with_fix(sim, goal, goal_norm, policy, variant, max_steps=200):
    """Run VLA policy with arm post-processing. Returns (success, steps, final_dist, traj)."""
    base_id = sim.model.body('base').id
    threshold = 0.15
    trajectory = []

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        trajectory.append(base_pos.tolist())

        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, trajectory

        img = sim.render()
        img_t = resize_for_clip(img)

        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
        wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
        state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

        img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(DEVICE)
        state_tensor = torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            raw_action = policy.infer(img_tensor, state_tensor, num_steps=4)
        raw_action_np = raw_action.squeeze(0).cpu().numpy()

        # Post-process arm actions
        full_action = postprocess_arm(raw_action_np, variant)

        # De-normalize wheel action for simulation (MATCHES diagnose_vla_failures.py)
        wheel_denorm = full_action[6:9] / 0.5 * 6.0
        wheel_denorm = np.clip(wheel_denorm, -6.0, 6.0)
        arm_action_sim = full_action[:6]
        wheel_action_sim = wheel_denorm / 12.0

        sim_action = np.concatenate([arm_action_sim, wheel_action_sim])
        sim.step(sim_action)

    return False, max_steps, dist, trajectory


def load_phase173_results():
    """Load the Phase 173 diagnostic results to extract failed episodes."""
    json_path = ROOT / "results" / "phase173_diagnostic_20260419_0442.json"
    with open(json_path) as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="clip_03",
        choices=["raw", "zero_arm", "clip_03", "clip_02", "clip_01", "decay_j4j5", "neutral_pose"])
    parser.add_argument("--ckpt", type=str,
        default="results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt")
    parser.add_argument("--all_variants", action="store_true")
    args = parser.parse_args()

    import mujoco
    from sim_lekiwi_urdf import LeKiWiSimURDF

    # Load Phase 173 data
    phase173 = load_phase173_results()
    failed_episodes = [r for r in phase173["results"] if r["category"] == "VLA_FAIL_PCTRL_OK"]
    print(f"\n[Phase 174] Testing arm saturation fix on {len(failed_episodes)} failed episodes from Phase 173")
    print(f"  Checkpoint: {args.ckpt}")

    policy = load_policy(Path(args.ckpt), device=DEVICE)

    variants_to_test = [args.variant]
    if args.all_variants:
        variants_to_test = ["raw", "zero_arm", "clip_03", "clip_02", "clip_01", "decay_j4j5", "neutral_pose"]

    all_results = {}
    for variant in variants_to_test:
        print(f"\n{'='*60}")
        print(f"[Variant: {variant}]")
        print(f"{'='*60}")

        success_count = 0
        variant_results = []

        for ep_data in failed_episodes:
            ep_idx = ep_data["episode"]
            goal = np.array(ep_data["goal"], dtype=np.float32)
            goal_norm = np.clip(goal / 1.0, -1.0, 1.0)

            # Run episode with this variant
            sim = LeKiWiSimURDF()
            sim.reset()
            ok, steps, dist, traj = run_vla_episode_with_fix(
                sim, goal, goal_norm, policy, variant, max_steps=200
            )

            success_count += int(ok)
            variant_results.append({
                "episode": ep_idx,
                "goal": goal.tolist(),
                "original_vla_failed": True,
                "fixed_success": ok,
                "steps": steps,
                "final_dist": float(dist),
                "original_dist": ep_data["vla_final_dist"],
                "improvement": float(ep_data["vla_final_dist"] - dist),
            })

            status = "SUCC" if ok else "FAIL"
            print(f"  Ep {ep_idx:02d} ({goal[0]:+.3f},{goal[1]:+.3f}): {status} "
                  f"(steps={steps:3d}, dist={dist:.3f}m, "
                  f"orig_dist={ep_data['vla_final_dist']:.3f}m)")

        sr = success_count / len(failed_episodes) * 100
        print(f"\n  >>> {variant}: {success_count}/{len(failed_episodes)} = {sr:.1f}% SR on previously FAILED episodes")
        all_results[variant] = {
            "sr": sr,
            "successes": success_count,
            "total": len(failed_episodes),
            "episodes": variant_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print("[SUMMARY] Arm Saturation Fix Results")
    print(f"{'='*60}")
    print(f"Baseline (Phase 173): VLA 0/{len(failed_episodes)} = 0.0% SR on failed episodes")
    print(f"P-ctrl baseline:       {len(failed_episodes)}/{len(failed_episodes)} = 100.0% SR")
    print()
    for variant, res in sorted(all_results.items(), key=lambda x: -x[1]["sr"]):
        print(f"  {variant:20s}: {res['successes']:2d}/{res['total']} = {res['sr']:5.1f}% SR")

    # Save
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_path = ROOT / "results" / f"phase174_arm_fix_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({
            "all_results": all_results,
            "n_failed_from_phase173": len(failed_episodes),
            "pctrl_baseline": 100.0,
        }, f, indent=2)
    print(f"\n[Saved] {output_path}")


if __name__ == "__main__":
    import time as time_module
    main()
