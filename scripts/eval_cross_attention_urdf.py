#!/usr/bin/env python3
"""
Phase 134: Cross-Attention VLA on URDF Sim (Correct Physics)
============================================================
Phase 132 discovered that LeKiWiSim (primitive) has broken locomotion
while LeKiWiSimURDF (mesh) has correct physics and P-controller = 100% SR.

This script evaluates the Phase 131 Cross-Attention VLA on URDF sim
to determine its TRUE success rate with correct physics.

Architecture (from train_cross_attention_vla.py):
  - CLIP spatial tokens [B, 50, 768] via clip_encoder
  - Goal MLP: 2 → 128 → 64
  - State net: 11 → 256 (with LayerNorm + SiLU)
  - Cross-attention: goal (Q) attends to CLIP tokens (K,V)
  - Concat[cls(768) + state_feat(256) + cross_out(768) + time_feat(256)] = 2048
  - Action head: 2048 → 512 → 512 → 9 + skip connection
  - Flow matching: 4-step Euler denoising
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 134] Device: {DEVICE}")


# ─── CLIP Spatial Encoder (matching train script) ────────────────────────────

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
        """images: [B, 3, 224, 224] in [0,1]. Returns: [B, 50, 768] spatial tokens."""
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, 50, 768]
        return hidden


# ─── Cross-Attention Policy (matching train script exactly) ─────────────────

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
        clip_tokens = self.clip_encoder(images)  # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 64]
        state_feat = self.state_net(state)       # [B, 256]
        goal_q = nn.Linear(64, 768, device=self.device)(goal_emb.unsqueeze(1))  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]
        cls_token = clip_tokens[:, 0:1, :]         # [B, 1, 768]
        combined = torch.cat([
            cls_token,
            state_feat.unsqueeze(1),
            cross_out,
        ], dim=-1)  # [B, 1, 1792]
        t_feat = self.time_mlp(timestep)
        combined = torch.cat([combined, t_feat.unsqueeze(1)], dim=-1)  # [B, 1, 2048]
        v_pred = self.action_head(combined).squeeze(1)  # [B, 9]
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


# ─── Normalize Action (matching eval_p126_policy.py Phase 129 fix) ─────────

def normalize_action(raw_action):
    """Policy (unbounded) → LeKiWi native units.
    CLIP-FM outputs are unbounded. Clip to [-1,1], then denorm to wheel range.
    """
    raw = np.asarray(raw_action, dtype=np.float32)
    raw_clipped = np.clip(raw, -1.0, 1.0)
    wheel_limits = np.array([[-0.5, 0.5]] * 3)
    denormed = (raw_clipped + 1.0) / 2.0 * (wheel_limits[:, 1] - wheel_limits[:, 0]) + wheel_limits[:, 0]
    return denormed


# ─── Evaluation on URDF Sim ──────────────────────────────────────────────────

def evaluate_cross_attn_vla_on_urdf(policy, n_episodes=10, max_steps=200, threshold=0.15):
    """Evaluate CrossAttention VLA on LeKiWiSimURDF (correct physics)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

    successes = 0
    dists = []
    policy.eval()

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        # Random goal in reachable range
        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])

        for step in range(max_steps):
            # Render image for policy (sim.render() returns numpy array directly)
            img_np_full = sim.render()  # [H, W, 3] uint8
            # Resize to 224x224 using PIL
            from PIL import Image
            img_pil = Image.fromarray(img_np_full)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            # Get state: arm_pos(6) + wheel_vel(3) + goal_xy(2) = 11D
            # Goals in training data are ABSOLUTE world positions
            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[0:3]
            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)

            # Policy inference
            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()

            # Apply only wheel actions (actions 6:9)
            wheel_speeds = normalize_action(raw_action[6:9])
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)

            # Check success
            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < threshold:
                successes += 1
                print(f"  Ep {ep}: SUCCESS step={step} dist={dist:.3f}")
                break
        else:
            print(f"  Ep {ep}: FAIL final_dist={dist:.3f}")
            dists.append(dist)

    sr = successes / n_episodes
    mean_dist = np.mean(dists) if dists else 0.0
    print(f"\n  CrossAttn VLA (URDF): {successes}/{n_episodes} = {100*sr:.0f}% SR, mean_dist={mean_dist:.3f}")
    return sr, mean_dist


def evaluate_pcontroller_on_urdf(n_episodes=10, max_steps=200, threshold=0.15):
    """P-controller baseline on URDF sim (this should be ~100% SR)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

    successes = 0
    steps_list = []

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
                print(f"  Ep {ep}: P-CTRL SUCCESS step={step}")
                break

            # P-controller toward goal
            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            d = np.linalg.norm([dx, dy])
            if d > 0.01:
                v_mag = min(1.5 * d, 0.3)
                vx, vy = v_mag * dx / d, v_mag * dy / d
            else:
                vx, vy = 0.0, 0.0
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            action = np.zeros(9)
            action[6:9] = np.clip(wheel_speeds, -0.5, 0.5)
            sim.step(action)
        else:
            steps_list.append(max_steps)
            print(f"  Ep {ep}: P-CTRL FAIL final_dist={dist:.3f}")

    sr = successes / n_episodes
    mean_steps = np.mean(steps_list)
    print(f"\n  P-controller (URDF): {successes}/{n_episodes} = {100*sr:.0f}% SR, mean_steps={mean_steps:.0f}")
    return sr, mean_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results/phase131/final_policy.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 134 — Cross-Attention VLA on URDF Sim (Correct Physics)")
    print("=" * 60)

    # Load policy
    print(f"\n[1] Loading policy from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if 'policy_state_dict' in ckpt:
        ckpt = ckpt['policy_state_dict']

    policy = CrossAttentionPolicy(state_dim=11, goal_dim=2, action_dim=9, device=args.device)
    # Load only matching keys
    model_state = policy.state_dict()
    loaded_keys = {k: v for k, v in ckpt.items() if k in model_state}
    missing = set(model_state.keys()) - set(loaded_keys.keys())
    unexpected = set(ckpt.keys()) - set(model_state.keys())
    if missing:
        print(f"  WARNING: {len(missing)} keys missing from checkpoint")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (architecture mismatch)")
    policy.load_state_dict(loaded_keys, strict=False)
    policy.to(args.device)
    policy.eval()
    print("[2] Policy loaded OK")

    # Baseline P-controller
    print(f"\n[3] P-controller baseline on URDF...")
    p_sr, p_steps = evaluate_pcontroller_on_urdf(n_episodes=args.episodes, max_steps=args.max_steps, threshold=args.threshold)

    # CrossAttn VLA
    print(f"\n[4] CrossAttn VLA on URDF...")
    vla_sr, vla_dist = evaluate_cross_attn_vla_on_urdf(policy, n_episodes=args.episodes, max_steps=args.max_steps, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("PHASE 134 RESULTS")
    print("=" * 60)
    print(f"  P-controller (URDF):    {100*p_sr:.0f}% SR")
    print(f"  CrossAttn VLA (URDF):  {100*vla_sr:.0f}% SR")
    print(f"  VLA vs P-gap:          {100*(p_sr - vla_sr):.0f}%-points below baseline")
    print("=" * 60)

    # Save results
    results = {
        "phase": 134,
        "architecture": "cross_attention_on_urdf",
        "pcontroller_sr": float(p_sr),
        "pcontroller_mean_steps": float(p_steps),
        "vla_sr": float(vla_sr),
        "vla_mean_dist": float(vla_dist),
        "note": "Phase 131 CrossAttn VLA evaluated on URDF sim (correct physics)",
    }
    out_path = Path("results/phase134_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
