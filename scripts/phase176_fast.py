#!/usr/bin/env python3
"""
Phase 176: VLA Wheel Magnitude Root Cause (FAST VERSION)
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 176] Device: {DEVICE}")


class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.float32)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state


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
        q = self.q_proj(state_feat + goal_q).unsqueeze(1)
        cross_out, _ = self.cross_attn(q, clip_proj, clip_proj)
        cross_out = self.cross_norm(cross_out + q)
        t_feat = self.time_net(t)
        combined = torch.cat([state_feat, goal_emb, cross_out.squeeze(1), t_feat], dim=-1)
        return self.action_head(combined) + self.skip(noisy_action)

    def infer(self, image, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        for i in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * (i / num_steps)
            v = self.forward(image, state, x, t)
            x = (x + v / num_steps).clamp(-0.5, 0.5)
        return x


def load_policy(path, device=DEVICE):
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt.get('policy_state_dict', ckpt), strict=False)
    policy.to(device).eval()
    print(f"[LOAD] epoch={ckpt.get('epoch','?')}, eval_sr={ckpt.get('eval_sr','?')}")
    return policy


def resize_for_clip(img):
    pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
    return (np.array(pil).astype(np.float32) / 255.0).transpose(2, 0, 1)


def make_state(sim, goal_norm):
    arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
    wheel_vel = np.array([sim.data.qvel[sim._jvel_idx[n]] for n in ["w1","w2","w3"]], dtype=np.float32)
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)


def run_episode(sim, goal, goal_norm, policy, amplifier=1.0, max_steps=200):
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    base_id = sim.model.body('base').id
    threshold = 0.15
    last_w = None
    
    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, last_w

        img_t = resize_for_clip(sim.render())
        state_11d = make_state(sim, goal_norm)
        
        with torch.no_grad():
            raw = policy.infer(
                torch.from_numpy(img_t).unsqueeze(0).to(DEVICE),
                torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE),
            ).squeeze(0).cpu().numpy()
        
        raw_wheel = raw[6:9]
        wheel_action = np.clip(raw_wheel * 0.0834 * amplifier, -0.0417, 0.0417)
        last_w = wheel_action.copy()
        arm_action = np.clip(raw[:6], -0.5, 0.5)
        sim.step(np.concatenate([arm_action, wheel_action]))

    return False, max_steps, dist, last_w


def run_pctrl(sim, goal, goal_norm, max_steps=200):
    from sim_lekiwi_urdf import twist_to_contact_wheel_speeds
    base_id = sim.model.body('base').id
    kP, max_speed = 0.1, 0.25
    threshold = 0.15
    last_w = None
    
    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist, last_w

        dx, dy = goal - base_pos
        d = np.linalg.norm([dx, dy])
        if d > 1e-6:
            v_mag = min(kP * d, max_speed)
            vx, vy = dx/d * v_mag, dy/d * v_mag
        else:
            vx, vy = 0.0, 0.0
        
        wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
        wheel_action = wheel_speeds / 12.0
        last_w = wheel_action.copy()
        arm_pos = np.array([sim.data.qpos[sim._jpos_idx[n]] for n in ["j0","j1","j2","j3","j4","j5"]], dtype=np.float32)
        sim.step(np.concatenate([arm_pos, wheel_action]).astype(np.float32))

    return False, max_steps, dist, last_w


def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF
    print("\n" + "=*(" * 15 + " Phase 176 " + "=*(" * 15)
    
    policy = load_policy("results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt", DEVICE)

    # Test goals: 5 fails, 5 successes from phase174
    test_episodes = [
        (0, [0.364, -0.037], 'FAIL'),   # ep00 fail
        (1, [0.415, 0.118], 'SUCC'),    # ep01 succ
        (7, [0.166, -0.164], 'FAIL'),   # ep07 fail
        (8, [0.233, -0.262], 'FAIL'),   # ep08 fail
        (10, [0.355, -0.087], 'FAIL'),  # ep10 fail
    ]

    print("\n[TEST] Amplifier Sweep")
    print("-" * 70)
    print(f"{'Amp':>6} | {'ep00':>6} {'ep01':>6} {'ep07':>6} {'ep08':>6} {'ep10':>6} | {'N/5':>4}")
    
    results_by_amp = {}
    for amp in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        row = []
        n_succ = 0
        for ep_id, goal, baseline in test_episodes:
            goal_norm = np.clip(np.array(goal) / 1.0, -1.0, 1.0)
            sim = LeKiWiSimURDF()
            succ, steps, dist, _ = run_episode(sim, goal, goal_norm, policy, amplifier=amp)
            row.append('SUCC' if succ else 'FAIL')
            n_succ += int(succ)
        results_by_amp[amp] = n_succ
        print(f"{amp:>6.1f} | {'  '.join(row)} | {n_succ}/5")

    # P-ctrl baseline
    print("\n[P-CTRL] Baseline for comparison")
    print("-" * 70)
    for ep_id, goal, baseline in test_episodes:
        goal_norm = np.clip(np.array(goal) / 1.0, -1.0, 1.0)
        sim = LeKiWiSimURDF()
        succ, steps, dist, last_w = run_pctrl(sim, goal, goal_norm)
        vla_w = None
        for a2 in [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
            if results_by_amp.get(a2, 0) >= sum(1 for _, _, _ in test_episodes[:1]):
                pass
        print(f"  Ep {ep_id:02d} ({baseline}): P-ctrl={'SUCC' if succ else 'FAIL'} (dist={dist:.3f})")

    # Save
    ts = time.strftime('%Y%m%d_%H%M')
    with open(f'results/phase176_{ts}.json', 'w') as f:
        json.dump({'phase': 176, 'amplifier_results': results_by_amp, 'test_episodes': [(e, g, b) for e, g, b in test_episodes]}, f)
    print(f"\n[INFO] Saved results/phase176_{ts}.json")


if __name__ == "__main__":
    main()
