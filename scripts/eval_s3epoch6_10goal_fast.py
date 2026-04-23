#!/usr/bin/env python3
"""
Phase 284: Stage3 s3_epoch6 — Quick 10-Goal Diagnostic (50 steps each)
======================================================================
Fast eval to check wheel action magnitude before running full 50-goal.
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
            return outputs.last_hidden_state

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
            torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Dropout(0.1), torch.nn.Linear(hidden, action_dim)
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 256), torch.nn.SiLU()
        )
        self.action_dim = action_dim

    def forward(self, images, state, x, t):
        B = images.shape[0]
        img_feat = self.encoder(images)
        goal_q = self.goal_q_proj(self.goal_mlp(state[:, :2])).unsqueeze(1)
        attn_out, _ = self.cross_attn(goal_q, img_feat, img_feat)
        attn_out = self.cross_norm(attn_out.squeeze(1))
        goal_q = goal_q.squeeze(1)
        state_feat = self.state_net(state[:, :11])
        time_emb = self.time_mlp(t)
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
    return np.clip(raw_action, -1.0, 1.0) * 0.5


def quick_eval(policy, label, n_eps=10, max_steps=50, threshold=0.10):
    """Quick 10-goal eval, 50 steps max — just to get wheel action magnitude."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    np.random.seed(42)
    successes = 0
    raw_wheel_mags = []
    norm_wheel_mags = []

    for ep in range(n_eps):
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

            raw_wheel = action_np[6:9]
            norm_wheel = normalize_action(raw_wheel)

            if ep == 0 and step < 3:
                print(f"    {label} ep0 step{step}: raw={raw_wheel} norm={norm_wheel}")

            raw_wheel_mags.append(np.linalg.norm(raw_wheel))
            norm_wheel_mags.append(np.linalg.norm(norm_wheel))

            full_action = np.zeros(9)
            full_action[6:9] = norm_wheel
            sim.step(full_action)

            pos = sim.data.xpos[base_id, :2]
            if np.linalg.norm(pos - goal) < threshold:
                successes += 1
                ep_success = True
                break

        if (ep + 1) % 5 == 0:
            print(f"  {label}: {successes}/{ep+1} SR through ep {ep+1}")

    sr = successes / n_eps
    mean_raw_mag = np.mean(raw_wheel_mags)
    mean_norm_mag = np.mean(norm_wheel_mags)
    print(f"\n  {label}: {successes}/{n_eps}={100*sr:.0f}% SR | mean_raw_mag={mean_raw_mag:.4f} | mean_norm_mag={mean_norm_mag:.4f}")
    return sr, mean_raw_mag, mean_norm_mag


def main():
    print("=" * 60)
    print("Phase 284: s3_epoch6 vs s3_epoch9 — Quick 10-Goal Diagnostic")
    print("=" * 60)

    # Load s3_epoch6
    ckpt6 = WORKDIR / "results/phase264_curriculum_train/s3_epoch6.pt"
    print(f"\n[INFO] Loading s3_epoch6...")
    policy6 = GoalConditionedPolicy(device=DEVICE)
    state_dict6 = torch.load(ckpt6, map_location=DEVICE, weights_only=False)
    policy6.load_state_dict(state_dict6, strict=False)
    policy6.to(DEVICE).eval()
    print("[INFO] s3_epoch6 loaded OK")

    # Load s3_epoch9
    ckpt9 = WORKDIR / "results/phase264_curriculum_train/s3_epoch9.pt"
    print(f"\n[INFO] Loading s3_epoch9...")
    policy9 = GoalConditionedPolicy(device=DEVICE)
    state_dict9 = torch.load(ckpt9, map_location=DEVICE, weights_only=False)
    policy9.load_state_dict(state_dict9, strict=False)
    policy9.to(DEVICE).eval()
    print("[INFO] s3_epoch9 loaded OK")

    t1 = time.time()
    s6_sr, s6_raw, s6_norm = quick_eval(policy6, "s3_epoch6", n_eps=10, max_steps=50)
    print(f"[s3_epoch6] done in {time.time()-t1:.0f}s")

    t2 = time.time()
    s9_sr, s9_raw, s9_norm = quick_eval(policy9, "s3_epoch9", n_eps=10, max_steps=50)
    print(f"[s3_epoch9] done in {time.time()-t2:.0f}s")

    result = {
        "phase": 284,
        "mode": "quick_10goal_50step",
        "s3_epoch6": {"sr": round(100*s6_sr, 1), "mean_raw_wheel_mag": round(s6_raw, 4), "mean_norm_wheel_mag": round(s6_norm, 4)},
        "s3_epoch9": {"sr": round(100*s9_sr, 1), "mean_raw_wheel_mag": round(s9_raw, 4), "mean_norm_wheel_mag": round(s9_norm, 4)},
    }
    out = WORKDIR / "results/phase284_quick10_diag.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[RESULT] {out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
