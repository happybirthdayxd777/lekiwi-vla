#!/usr/bin/env python3
"""
Phase 151: Test the qvel[0:3] → qvel[6:9] FIX
=============================================
Tests whether fixing the wheel velocity index in eval improves VLA SR.

BEFORE (buggy): wheel_vel = sim.data.qvel[0:3]  # BASE velocity (wrong!)
AFTER  (fixed): wheel_vel = sim.data.qvel[6:9]  # WHEEL angular velocity (correct!)

The policy was trained on phase63 data where actions[6:9] = wheel velocities.
The eval was using BASE velocity (qvel[0:3]) instead of WHEEL velocity (qvel[6:9]).
This is a fundamental state representation mismatch.
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/hermes_research/lekiwi_vla'))
sys.path.insert(0, os.path.expanduser('~/hermes_research/lekiwi_vla/src'))
os.chdir(os.path.expanduser('~/hermes_research/lekiwi_vla'))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

DEVICE = "cpu"
print(f"[Phase 151] Device: {DEVICE}")

# ── Policy class (inline to avoid exec issues) ─────────────────────────────────

class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state
        return hidden


class CrossAttentionPolicy(nn.Module):
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.clip_encoder = CLIPSpatialEncoder(device=device)
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 128), nn.SiLU(), nn.Linear(128, 64)
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(768)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )
        total_dim = 768 + 256 + 768 + 256  # 2048
        self.action_head = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)
        self.to(device)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] CrossAttentionPolicy: {n_params:,} params")

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.clip_encoder(images)
        goal_emb = self.goal_mlp(state[:, -2:])
        state_feat = self.state_net(state)
        goal_q = nn.Linear(64, 768, device=self.device)(goal_emb.unsqueeze(1))
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)
        cls_token = clip_tokens[:, 0:1, :]
        combined = torch.cat([cls_token, state_feat.unsqueeze(1), cross_out], dim=-1)
        t_feat = self.time_mlp(timestep)
        combined = torch.cat([combined, t_feat.unsqueeze(1)], dim=-1)
        v_pred = self.action_head(combined).squeeze(1)
        return v_pred + self.skip(noisy_action)

    def infer(self, images, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * 0.5
            v = self.forward(images, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


def normalize_action(raw_action):
    """Policy output (bounded [-0.5, 0.5]) → sim native units."""
    raw = np.asarray(raw_action, dtype=np.float32)
    raw_clipped = np.clip(raw, -0.5, 0.5)
    return raw_clipped


# ── Test: Compare OLD (qvel[0:3]) vs NEW (qvel[6:9]) ─────────────────────────

def run_eval(n_episodes=3, use_fixed_qvel=True, threshold=0.15):
    """Run eval with either buggy (qvel[0:3]) or fixed (qvel[6:9]) wheel velocity."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    print(f"\n[EVAL] {'FIXED' if use_fixed_qvel else 'BUGGY'} qvel — use qvel[{'6:9' if use_fixed_qvel else '0:3'}] for wheel velocity")

    # Load policy
    policy_path = "results/phase145_jacobian_train/final_policy.pt"
    ckpt = torch.load(policy_path, map_location=DEVICE, weights_only=False)
    if 'policy_state_dict' in ckpt:
        ckpt = ckpt['policy_state_dict']

    policy = CrossAttentionPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
    policy.load_state_dict(ckpt, strict=False)
    policy.to(DEVICE)
    policy.eval()

    successes = 0
    steps_list = []
    dists = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])

        for step in range(200):
            img_np = sim.render()
            img_pil = Image.fromarray(img_np)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            arm_pos = sim.data.qpos[7:13]

            # THE FIX: use qvel[6:9] for wheel velocities
            if use_fixed_qvel:
                wheel_vel = sim.data.qvel[6:9]  # CORRECT: wheel angular velocities
            else:
                wheel_vel = sim.data.qvel[0:3]  # BUG: base velocity (wrong!)

            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()

            wheel_speeds = normalize_action(raw_action[6:9])
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)

            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step + 1)
                print(f"  ep{ep}: SUCCESS step={step}")
                break
        else:
            dists.append(dist)
            print(f"  ep{ep}: FAIL final_dist={dist:.3f}")

    sr = successes / n_episodes
    mean_steps = np.mean(steps_list) if steps_list else 200
    mean_dist = np.mean(dists) if dists else 0.0
    print(f"  {'FIXED' if use_fixed_qvel else 'BUGGY'}: {successes}/{n_episodes} = {100*sr:.0f}% SR, mean_steps={mean_steps:.0f}, mean_dist={mean_dist:.3f}")
    return sr, mean_steps, mean_dist


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 151: qvel[0:3] → qvel[6:9] FIX Test")
    print("=" * 60)

    # First test with buggy qvel[0:3] (OLD)
    sr_buggy, steps_buggy, dist_buggy = run_eval(n_episodes=3, use_fixed_qvel=False)

    # Then test with fixed qvel[6:9] (NEW)
    sr_fixed, steps_fixed, dist_fixed = run_eval(n_episodes=3, use_fixed_qvel=True)

    print("\n" + "=" * 60)
    print("COMPARISON:")
    print(f"  BUGGY (qvel[0:3]): {100*sr_buggy:.0f}% SR, dist={dist_buggy:.3f}")
    print(f"  FIXED (qvel[6:9]): {100*sr_fixed:.0f}% SR, dist={dist_fixed:.3f}")
    print("=" * 60)
