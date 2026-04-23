#!/usr/bin/env python3
"""
Phase 284: Stage3 s3_epoch6 — Full 50-Goal Evaluation
======================================================
Compare s3_epoch6 vs s3_epoch9 on 50 goals (same eval protocol as phase282).
s3_epoch6 had 15% SR in 20-goal eval (phase265), better than s3_epoch9's 2%.
This 50-goal eval determines if it's actually better.

Checkpoints:
  - s3_epoch6: results/phase264_curriculum_train/s3_epoch6.pt  (loss=0.2558)
  - s3_epoch9: results/phase264_curriculum_train/s3_epoch9.pt  (loss=0.2324, overfitted)
"""
import os, sys, time, json
import numpy as np
import torch
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(WORKDIR / "scripts"))

DEVICE = "cpu"
print(f"[Phase 284] Device: {DEVICE}")


# ─── Policy Architecture (matching train_curriculum_stage3.py) ──────────────────

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

        self.action_dim = action_dim

    def forward(self, images, state, x, t):
        """Matching train_curriculum_stage3.py forward()."""
        B = images.shape[0]
        img_feat = self.encoder(images)  # [B, 50, 768]
        goal_q = self.goal_q_proj(self.goal_mlp(state[:, :2]))  # [B, 768]
        goal_q = goal_q.unsqueeze(1)                             # [B, 1, 768]

        attn_out, _ = self.cross_attn(goal_q, img_feat, img_feat)
        attn_out = self.cross_norm(attn_out.squeeze(1))          # [B, 768]

        goal_q = goal_q.squeeze(1)                               # [B, 768]
        state_feat = self.state_net(state[:, :11])              # [B, 128]
        time_emb = self.time_mlp(t)                             # [B, 256]

        combined = torch.cat([attn_out, goal_q, state_feat, time_emb, x], dim=-1)
        return self.flow_head(combined)

    def infer(self, images, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(state.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(state.device) * 0.5
            pred = self.forward(images, state, x, t)
            x = x + pred * dt
        return x


def normalize_action(raw_action):
    """Map raw VLA output [-1, 1] → wheel native [-0.5, 0.5] rad/s."""
    return np.clip(raw_action, -1.0, 1.0) * 0.5


def evaluate_pcontroller_baseline(n_episodes=50, max_steps=300, threshold=0.10):
    """P-controller baseline (same as phase282)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

    np.random.seed(42)
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


def evaluate_vla_policy(policy, n_episodes=50, max_steps=300, threshold=0.10, label="VLA"):
    """Evaluate Stage 3 VLA on LeKiWiSimURDF (same as phase282)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    np.random.seed(42)
    successes = 0
    fail_dists = []
    steps_list = []
    wheel_action_mags = []

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
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]

            # Wheel normalization: raw [-1,1] → native [-0.5, 0.5] rad/s
            wheel_action = normalize_action(action_np[6:9])
            if ep == 0 and step < 5:
                print(f"    Step {step}: raw_wheel={action_np[6:9]} → norm={wheel_action}")

            full_action = np.zeros(9)
            full_action[6:9] = wheel_action
            sim.step(full_action)

            if step == 0:
                wheel_action_mags.append(np.linalg.norm(wheel_action))

            pos = sim.data.xpos[base_id, :2]
            if np.linalg.norm(pos - goal) < threshold:
                successes += 1
                steps_list.append(step)
                ep_success = True
                break

        if not ep_success:
            fail_dists.append(np.linalg.norm(sim.data.xpos[base_id, :2] - goal))

        if (ep + 1) % 10 == 0:
            print(f"  {label}: {successes}/{ep+1} = {100*successes/(ep+1):.0f}% SR (through episode {ep+1})")

    sr = successes / n_episodes
    mean_fail_dist = np.mean(fail_dists) if fail_dists else 0.0
    mean_steps = np.mean(steps_list) if steps_list else 0.0
    mean_wheel_mag = np.mean(wheel_action_mags) if wheel_action_mags else 0.0
    print(f"\n  {label}: {successes}/{n_episodes} = {100*sr:.0f}% SR | mean_fail_dist={mean_fail_dist:.3f} | mean_steps={mean_steps:.0f} | mean_wheel_mag={mean_wheel_mag:.4f}")
    return sr, mean_fail_dist, mean_steps, mean_wheel_mag


def main():
    print("=" * 60)
    print("Phase 284: s3_epoch6 vs s3_epoch9 — 50-Goal Evaluation")
    print("=" * 60)

    # Load s3_epoch6
    ckpt6_path = WORKDIR / "results/phase264_curriculum_train/s3_epoch6.pt"
    print(f"\n[INFO] Loading s3_epoch6: {ckpt6_path}")
    policy6 = GoalConditionedPolicy(device=DEVICE)
    state_dict6 = torch.load(ckpt6_path, map_location=DEVICE, weights_only=False)
    policy6.load_state_dict(state_dict6, strict=False)
    policy6.to(DEVICE)
    policy6.eval()
    print("[INFO] s3_epoch6 loaded OK")

    # Load s3_epoch9
    ckpt9_path = WORKDIR / "results/phase264_curriculum_train/s3_epoch9.pt"
    print(f"\n[INFO] Loading s3_epoch9: {ckpt9_path}")
    policy9 = GoalConditionedPolicy(device=DEVICE)
    state_dict9 = torch.load(ckpt9_path, map_location=DEVICE, weights_only=False)
    policy9.load_state_dict(state_dict9, strict=False)
    policy9.to(DEVICE)
    policy9.eval()
    print("[INFO] s3_epoch9 loaded OK")

    # P-controller baseline
    t0 = time.time()
    p_sr, p_mfd, p_ms = evaluate_pcontroller_baseline(n_episodes=50, threshold=0.10)
    p_time = time.time() - t0
    print(f"[P-controller] {100*p_sr:.0f}% SR | {p_time:.0f}s")

    # s3_epoch6
    t1 = time.time()
    s6_sr, s6_mfd, s6_ms, s6_wm = evaluate_vla_policy(policy6, n_episodes=50, threshold=0.10, label="s3_epoch6")
    s6_time = time.time() - t1
    print(f"[s3_epoch6] {100*s6_sr:.0f}% SR | {s6_time:.0f}s")

    # s3_epoch9 (same eval for fair comparison)
    t2 = time.time()
    s9_sr, s9_mfd, s9_ms, s9_wm = evaluate_vla_policy(policy9, n_episodes=50, threshold=0.10, label="s3_epoch9")
    s9_time = time.time() - t2
    print(f"[s3_epoch9] {100*s9_sr:.0f}% SR | {s9_time:.0f}s")

    result = {
        "phase": 284,
        "n_goals": 50,
        "success_radius": 0.10,
        "seed": 42,
        "p_controller": {
            "successes": int(p_sr * 50),
            "success_rate": round(100 * p_sr, 1),
            "mean_fail_dist": round(p_mfd, 4),
            "mean_steps": round(p_ms, 1)
        },
        "s3_epoch6": {
            "checkpoint": "phase264_curriculum_train/s3_epoch6.pt",
            "loss": 0.2558,
            "successes": int(s6_sr * 50),
            "success_rate": round(100 * s6_sr, 1),
            "mean_fail_dist": round(s6_mfd, 4),
            "mean_steps": round(s6_ms, 1),
            "mean_wheel_action_mag": round(s6_wm, 4)
        },
        "s3_epoch9": {
            "checkpoint": "phase264_curriculum_train/s3_epoch9.pt",
            "loss": 0.2324,
            "successes": int(s9_sr * 50),
            "success_rate": round(100 * s9_sr, 1),
            "mean_fail_dist": round(s9_mfd, 4),
            "mean_steps": round(s9_ms, 1),
            "mean_wheel_action_mag": round(s9_wm, 4)
        }
    }

    out_path = WORKDIR / "results/phase284_s3epoch6_vs_s3epoch9.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[RESULT] {out_path}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
