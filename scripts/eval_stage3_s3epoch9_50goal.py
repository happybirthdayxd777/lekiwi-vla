#!/usr/bin/env python3
"""
Phase 282: Stage3 s3_epoch9 — 50-Goal Evaluation (sr=0.10m, seed=42)
====================================================================
"""
import os, sys, time, json
import numpy as np
import torch
from pathlib import Path

WORKDIR = Path(__file__).parent if '__file__' in dir() else Path("/Users/i_am_ai/hermes_research/lekiwi_vla")
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(WORKDIR / "scripts"))

DEVICE = "cpu"
print(f"[Phase 282] Device: {DEVICE}")

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
        vision_out = self.encoder(images)  # [B, 50, 768]
        goal_feat = self.goal_mlp(state[:, :2])  # [B, 2] → [B, 128]
        goal_q = self.goal_q_proj(goal_feat)     # [B, 768]
        goal_q = goal_q.unsqueeze(1)              # [B, 1, 768]

        state_feat = self.state_net(state[:, :11])  # [B, 11] → [B, 128]
        state_q = state_feat.unsqueeze(1)            # [B, 1, 128]

        # Cross-attention: vision as K/V, goal+state as Q
        vision_q = vision_out
        attended, _ = self.cross_attn(goal_q, vision_q, vision_q)
        attended = self.cross_norm(attended.squeeze(1))  # [B, 768]

        t_feat = self.time_mlp(t)  # [B, 1] → [B, 256]

        combined = torch.cat([attended, goal_feat, state_feat, t_feat, x], dim=-1)
        v_pred = self.flow_head(combined)
        return v_pred

    def infer(self, images, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(state.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(state.device) * 0.5
            v = self.forward(images, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


def normalize_action(raw_action):
    """Policy (unbounded) → LeKiWi native units (wheel range [-0.5, 0.5])."""
    raw = np.asarray(raw_action, dtype=np.float32)
    raw_clipped = np.clip(raw, -1.0, 1.0)
    wheel_limits = np.array([[-0.5, 0.5]] * 3)
    denormed = (raw_clipped + 1.0) / 2.0 * (wheel_limits[:, 1] - wheel_limits[:, 0]) + wheel_limits[:, 0]
    return denormed


def evaluate_pcontroller_baseline(n_episodes=50, max_steps=300, threshold=0.10):
    """P-controller baseline."""
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


def evaluate_vla_policy(policy, n_episodes=50, max_steps=300, threshold=0.10):
    """Evaluate Stage 3 VLA on LeKiWiSimURDF."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

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
            img_np_full = sim.render()
            from PIL import Image
            img_pil = Image.fromarray(img_np_full)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

            arm_pos = sim.data.qpos[7:13]
            wheel_pos = sim.data.qpos[6:9]
            wheel_vel = sim.data.qvel[6:9]
            base_xy = sim.data.xpos[base_id, :2]

            state = np.concatenate([arm_pos, wheel_pos, wheel_vel, base_xy, goal])
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]

            # Wheel normalization: raw → native [-0.5, 0.5]
            wheel_action = normalize_action(action_np[6:9])
            full_action = np.zeros(9)
            full_action[6:9] = wheel_action
            sim.step(full_action)

            pos = sim.data.xpos[base_id, :2]
            if np.linalg.norm(pos - goal) < threshold:
                successes += 1
                steps_list.append(step)
                ep_success = True
                break

        if not ep_success:
            fail_dists.append(np.linalg.norm(sim.data.xpos[base_id, :2] - goal))

        if (ep + 1) % 10 == 0:
            print(f"  VLA: {successes}/{ep+1} = {100*successes/(ep+1):.0f}% SR (through episode {ep+1})")

    sr = successes / n_episodes
    mean_fail_dist = np.mean(fail_dists) if fail_dists else 0.0
    mean_steps = np.mean(steps_list) if steps_list else 0.0
    print(f"\n  Stage3 VLA (s3_epoch9): {successes}/{n_episodes} = {100*sr:.0f}% SR | mean_fail_dist={mean_fail_dist:.3f} | mean_steps={mean_steps:.0f}")
    return sr, mean_fail_dist, mean_steps


def main():
    ckpt_path = WORKDIR / "results/phase264_curriculum_train/s3_epoch9.pt"
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    policy = GoalConditionedPolicy(device=DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(DEVICE)
    policy.eval()
    print("[INFO] Policy loaded OK")

    t0 = time.time()
    p_sr, p_mfd, p_ms = evaluate_pcontroller_baseline(n_episodes=50, threshold=0.10)
    print(f"[P-controller] {50*p_sr:.0f}% SR | {time.time()-t0:.0f}s")

    t1 = time.time()
    vla_sr, vla_mfd, vla_ms = evaluate_vla_policy(policy, n_episodes=50, threshold=0.10)
    print(f"[Stage3 VLA] {50*vla_sr:.0f}% SR | {time.time()-t1:.0f}s")

    result = {
        "phase": 282,
        "checkpoint": "s3_epoch9.pt",
        "n_goals": 50,
        "success_radius": 0.10,
        "seed": 42,
        "p_controller": {
            "successes": int(p_sr * 50),
            "success_rate": round(100 * p_sr, 1)
        },
        "stage3_vla": {
            "successes": int(vla_sr * 50),
            "success_rate": round(100 * vla_sr, 1)
        }
    }

    out_path = WORKDIR / "results/phase282_s3epoch9_50goal_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[RESULT] {out_path}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
