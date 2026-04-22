#!/usr/bin/env python3
"""
Phase 266 Quick Eval: s3_epoch9.pt (10 goals, 100 steps each)
Fair comparison with s3_epoch6 (20 goals, 200 steps) at same methodology.
"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path

WORKDIR = Path(__file__).parent.parent.resolve()
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(WORKDIR / "scripts"))
DEVICE = "cpu"

class CLIPVisionEncoder(torch.nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32).to(device)
        for p in self.clip.parameters(): p.requires_grad = False
    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            return self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True).last_hidden_state

class GoalConditionedPolicy(torch.nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.encoder = CLIPVisionEncoder(device)
        self.clip = self.encoder.clip
        self.goal_mlp = torch.nn.Sequential(torch.nn.Linear(2, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
                                           torch.nn.Linear(256, 128), torch.nn.SiLU())
        self.goal_q_proj = torch.nn.Linear(128, 768)
        self.state_net = torch.nn.Sequential(torch.nn.Linear(state_dim, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
                                             torch.nn.Linear(256, 128), torch.nn.SiLU())
        self.cross_attn = torch.nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = torch.nn.LayerNorm(768)
        self.flow_head = torch.nn.Sequential(
            torch.nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, action_dim))
        self.time_mlp = torch.nn.Sequential(torch.nn.Linear(1, 128), torch.nn.SiLU(),
                                            torch.nn.Linear(128, 256), torch.nn.SiLU())
    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)
        goal_emb = self.goal_mlp(state[:, -2:])
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)
        cls_token = clip_tokens[:, 0:1, :]
        state_feat = self.state_net(state)
        t_feat = self.time_mlp(timestep)
        combined = torch.cat([cls_token, state_feat.unsqueeze(1), cross_out, t_feat.unsqueeze(1), noisy_action.unsqueeze(1)], dim=-1).squeeze(1)
        return self.flow_head(combined)
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
    raw = np.asarray(raw_action, dtype=np.float32)
    raw_clipped = np.clip(raw, -1.0, 1.0)
    wheel_limits = np.array([[-0.5, 0.5]] * 3)
    return (raw_clipped + 1.0) / 2.0 * (wheel_limits[:, 1] - wheel_limits[:, 0]) + wheel_limits[:, 0]

def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    from PIL import Image

    ckpt = WORKDIR / "results/phase264_curriculum_train/s3_epoch9.pt"
    print(f"[Phase 266 Quick] Loading {ckpt.name}")
    ckpt_data = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    print(f"  Epoch: {ckpt_data.get('epoch')}, Loss: {ckpt_data.get('loss'):.4f}")

    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, device=DEVICE)
    policy.load_state_dict(ckpt_data["policy_state_dict"], strict=False)
    policy.to(DEVICE).eval()
    print("[OK] Policy loaded")

    # P-controller baseline
    print("\n[P-controller] 10 goals, 100 steps each...")
    p_successes, p_fail_dists, p_steps = 0, [], []
    rng = np.random.default_rng(42)
    for ep in range(10):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id
        goal = rng.uniform([-0.5, -0.3], [0.5, 0.3])
        sim.goal_xy = goal
        for step in range(100):
            pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(pos - goal)
            if dist < 0.15:
                p_successes += 1; p_steps.append(step)
                print(f"  P-ctrl Ep{ep}: SUCCESS step={step}")
                break
            rel = goal - pos
            ws = twist_to_contact_wheel_speeds(2.0*rel[0], 2.0*rel[1])
            a = np.zeros(9); a[6:9] = ws; sim.step(a)
        else:
            p_fail_dists.append(np.linalg.norm(sim.data.xpos[base_id, :2] - goal))
            print(f"  P-ctrl Ep{ep}: FAIL dist={p_fail_dists[-1]:.3f}")
    p_sr = p_successes / 10 * 100
    print(f"\n  P-ctrl: {p_successes}/10 = {p_sr:.0f}% SR")

    # VLA eval
    print("\n[VLA s3_epoch9] 10 goals, 100 steps each...")
    v_successes, v_fail_dists, v_steps = 0, [], []
    rng = np.random.default_rng(42)
    for ep in range(10):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id
        goal = rng.uniform([-0.5, -0.3], [0.5, 0.3])
        sim.goal_xy = goal
        for step in range(100):
            img = sim.render()
            img_s = np.array(Image.fromarray(img).resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_s.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[6:9]
            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                raw = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()
            ws = normalize_action(raw[6:9])
            a = np.zeros(9); a[6:9] = ws; sim.step(a)
            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < 0.15:
                v_successes += 1; v_steps.append(step)
                print(f"  VLA Ep{ep}: SUCCESS step={step}")
                break
        else:
            v_fail_dists.append(dist)
            print(f"  VLA Ep{ep}: FAIL dist={v_fail_dists[-1]:.3f}")
    v_sr = v_successes / 10 * 100
    print(f"\n  VLA (s3_epoch9): {v_successes}/10 = {v_sr:.0f}% SR")

    print(f"\n{'='*50}")
    print(f"Phase 266 Quick Results (10-goal, 100-step)")
    print(f"  P-ctrl: {p_sr:.0f}% SR")
    print(f"  VLA s3_epoch9: {v_sr:.0f}% SR")
    print(f"  Phase265 (s3_epoch6): VLA=15%")
    print(f"  Improvement: {v_sr-15:+.0f}-points")

    out = {"phase": 266, "checkpoint": ckpt.name, "p_sr": p_sr, "vla_sr": v_sr,
           "p_successes": p_successes, "vla_successes": v_successes,
           "improvement_vs_s3epoch6": v_sr - 15}
    with open(WORKDIR / "results/phase266_quick_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Saved: results/phase266_quick_eval.json")

if __name__ == "__main__":
    main()
