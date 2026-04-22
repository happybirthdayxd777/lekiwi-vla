#!/usr/bin/env python3
"""
Phase 266: Evaluate Stage 3 Curriculum Checkpoint (s3_epoch9.pt)
================================================================
Same evaluation methodology as Phase 265 (s3_epoch6.pt) for fair comparison.
Uses wheel-only action application (matching training: arm stays at current pos).
"""

import os, sys, time, json
import numpy as np
import torch
from pathlib import Path

WORKDIR = Path(__file__).parent.parent.resolve()
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(WORKDIR / "scripts"))

DEVICE = "cpu"
print(f"[Phase 266] Device: {DEVICE}")


# ─── Policy Architecture (matching train_curriculum_stage3.py + eval s3_epoch6) ─

class CLIPVisionEncoder(torch.nn.Module):
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
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            return outputs.last_hidden_state  # [B, 50, 768]


class GoalConditionedPolicy(torch.nn.Module):
    """Same architecture as eval_stage3_s3epoch6.py — must match exactly."""
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.encoder = CLIPVisionEncoder(device)
        self.clip = self.encoder.clip

        self.goal_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )
        self.goal_q_proj = torch.nn.Linear(128, 768)

        self.state_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )

        self.cross_attn = torch.nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = torch.nn.LayerNorm(768)

        self.flow_head = torch.nn.Sequential(
            torch.nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, action_dim)
        )

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 256), torch.nn.SiLU()
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] GoalConditionedPolicy: {n_params:,} params")

    def forward(self, images, state, noisy_action, timestep):
        B = images.shape[0]
        clip_tokens = self.encoder(images)  # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]
        cls_token = clip_tokens[:, 0:1, :]  # [B, 1, 768]
        state_feat = self.state_net(state)  # [B, 128]
        t_feat = self.time_mlp(timestep)  # [B, 256]
        combined = torch.cat([
            cls_token,
            state_feat.unsqueeze(1),
            cross_out,
            t_feat.unsqueeze(1),
            noisy_action.unsqueeze(1)
        ], dim=-1).squeeze(1)  # [B, 768+128+768+256+9 = 1929]
        v_pred = self.flow_head(combined)
        return v_pred  # no skip for stage3

    def infer(self, images, state, num_steps=4):
        """Same as eval_stage3_s3epoch6.py: Euler integration with fixed t=0.5."""
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(state.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(state.device) * 0.5
            v = self.forward(images, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


# ─── Wheel Normalization (matching eval_stage3_s3epoch6.py) ────────────────────

def normalize_action(raw_action):
    """Policy (unbounded) → LeKiWi native units (wheel range [-0.5, 0.5])."""
    raw = np.asarray(raw_action, dtype=np.float32)
    raw_clipped = np.clip(raw, -1.0, 1.0)
    wheel_limits = np.array([[-0.5, 0.5]] * 3)
    denormed = (raw_clipped + 1.0) / 2.0 * (wheel_limits[:, 1] - wheel_limits[:, 0]) + wheel_limits[:, 0]
    return denormed


# ─── P-controller baseline ─────────────────────────────────────────────────────

def evaluate_pcontroller_baseline(n_episodes=20, max_steps=200, threshold=0.15):
    """P-controller baseline (this should be ~85% with correct physics)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

    successes = 0
    fail_dists = []
    steps_list = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])
        sim.goal_xy = goal

        ep_success = False
        for step in range(max_steps):
            pos = sim.data.xpos[base_id, :2]
            rel = goal - pos
            dist = np.linalg.norm(rel)
            if dist < threshold:
                successes += 1
                steps_list.append(step)
                ep_success = True
                break

            des_vx, des_vy = 2.0 * rel
            wheel_speeds = twist_to_contact_wheel_speeds(des_vx, des_vy)
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)

        if not ep_success:
            fail_dists.append(np.linalg.norm(sim.data.xpos[base_id, :2] - goal))

    sr = successes / n_episodes
    mean_fail_dist = np.mean(fail_dists) if fail_dists else 0.0
    mean_steps = np.mean(steps_list) if steps_list else 0.0
    print(f"\n  P-controller: {successes}/{n_episodes} = {100*sr:.0f}% SR | mean_fail_dist={mean_fail_dist:.3f} | mean_steps={mean_steps:.0f}")
    return sr, mean_fail_dist, mean_steps


# ─── VLA Evaluation ──────────────────────────────────────────────────────────

def evaluate_vla_policy(policy, n_episodes=20, max_steps=200, threshold=0.15):
    """Evaluate Stage 3 VLA on LeKiWiSimURDF."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    successes = 0
    fail_dists = []
    steps_list = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])
        sim.goal_xy = goal

        ep_success = False
        for step in range(max_steps):
            img_np_full = sim.render()
            from PIL import Image
            img_pil = Image.fromarray(img_np_full)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[6:9]
            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()

            # Apply only wheel actions (matching eval_stage3_s3epoch6.py)
            wheel_speeds = normalize_action(raw_action[6:9])
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)

            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step)
                print(f"  Ep {ep}: SUCCESS step={step} dist={dist:.3f}")
                ep_success = True
                break

        if not ep_success:
            fail_dists.append(dist)
            print(f"  Ep {ep}: FAIL final_dist={dist:.3f}")

    sr = successes / n_episodes
    mean_fail_dist = np.mean(fail_dists) if fail_dists else 0.0
    mean_steps = np.mean(steps_list) if steps_list else 0.0
    print(f"\n  Stage3 VLA (s3_epoch9): {successes}/{n_episodes} = {100*sr:.0f}% SR | mean_fail_dist={mean_fail_dist:.3f} | mean_steps={mean_steps:.0f}")
    return sr, mean_fail_dist, mean_steps


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ckpt_path = WORKDIR / "results/phase264_curriculum_train/s3_epoch9.pt"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        for p in (WORKDIR / "results/phase264_curriculum_train").glob("*.pt"):
            print(f"  {p.name}")
        return

    print(f"[Phase 266] Loading s3_epoch9.pt from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")

    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, device=DEVICE)
    policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    policy.to(DEVICE)
    policy.eval()
    print("[OK] Policy loaded")

    print("\n[1/2] Evaluating P-controller baseline (20 episodes)...")
    p_sr, p_mfd, p_ms = evaluate_pcontroller_baseline(n_episodes=20)

    print("\n[2/2] Evaluating VLA (s3_epoch9)...")
    vla_sr, vla_mfd, vla_ms = evaluate_vla_policy(policy, n_episodes=20)

    print(f"\n{'='*60}")
    print(f"Phase 266 Summary: s3_epoch9.pt vs s3_epoch6.pt (Phase 265)")
    print(f"{'='*60}")
    print(f"  P-controller:       {p_sr*100:.0f}% SR | mfd={p_mfd:.3f} | ms={p_ms:.0f}")
    print(f"  VLA (s3_epoch9):    {vla_sr*100:.0f}% SR | mfd={vla_mfd:.3f} | ms={vla_ms:.0f}")
    print(f"  Phase 265 (s3_epoch6): VLA=15% SR")
    print(f"  Improvement (epoch6→9): {(vla_sr - 0.15)*100:+.0f}-points")

    # Save results
    out = {
        "phase": 266,
        "checkpoint": str(ckpt_path.name),
        "vla_sr": vla_sr * 100,
        "p_sr": p_sr * 100,
        "vla_mfd": vla_mfd,
        "p_mfd": p_mfd,
        "vla_ms": vla_ms,
        "p_ms": p_ms,
        "phase265_s3epoch6_vla_sr": 15,
        "phase265_s3epoch6_p_sr": 85,
        "improvement_vs_epoch6": (vla_sr - 0.15) * 100,
    }
    out_path = WORKDIR / "results/phase266_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Results saved: {out_path}")


if __name__ == "__main__":
    main()
