#!/usr/bin/env python3
"""
Phase 176: TEST w1 SIGN FLIP = ROOT CAUSE

Hypothesis: VLA has w1 sign inverted. Flipping w1 should fix failures.
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


def run_episode_flip(sim, goal, goal_norm, policy, flip_w1=False, max_steps=200):
    from sim_lekiwi_urdf import LeKiWiSimURDF
    base_id = sim.model.body('base').id
    threshold = 0.15
    
    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            return True, step + 1, dist

        img_t = resize_for_clip(sim.render())
        state_11d = make_state(sim, goal_norm)
        
        with torch.no_grad():
            raw = policy.infer(
                torch.from_numpy(img_t).unsqueeze(0).to(DEVICE),
                torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE),
            ).squeeze(0).cpu().numpy()
        
        raw_wheel = raw[6:9].copy()
        if flip_w1:
            raw_wheel[0] = -raw_wheel[0]  # FLIP w1!
        
        wheel_action = np.clip(raw_wheel * 0.0834, -0.0417, 0.0417)
        arm_action = np.clip(raw[:6], -0.5, 0.5)
        sim.step(np.concatenate([arm_action, wheel_action]))

    return False, max_steps, dist


def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    print("\n" + "=" * 50)
    print("Phase 176: w1 SIGN FLIP TEST")
    print("=" * 50)
    
    policy = load_policy("results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt", DEVICE)

    # Test on 10 episodes from phase174 eval
    test_episodes = [
        (0, [0.364, -0.037]),   # ep00 FAIL baseline
        (1, [0.415, 0.118]),    # ep01 SUCC baseline
        (7, [0.166, -0.164]),   # ep07 FAIL baseline
        (8, [0.233, -0.262]),   # ep08 FAIL baseline
        (10, [0.355, -0.087]),  # ep10 FAIL baseline
        (12, [0.367, -0.183]),  # ep12 FAIL baseline
        (17, [0.182, -0.186]),  # ep17 FAIL baseline
        (21, [0.320, -0.113]),  # ep21 FAIL baseline
        (24, [0.309, -0.216]),  # ep24 FAIL baseline
        (25, [0.020, -0.296]),  # ep25 FAIL baseline
    ]

    print("\n[TEST] 10 episodes: baseline vs w1-flip")
    print("-" * 55)
    print(f"{'Ep':>4} | {'Baseline':>10} {'w1-Flip':>10} | {'Change':>10}")
    print("-" * 55)
    
    n_baseline_succ = 0
    n_flip_succ = 0
    
    for ep_id, goal in test_episodes:
        goal_norm = np.clip(np.array(goal) / 1.0, -1.0, 1.0)
        
        # Baseline
        sim = LeKiWiSimURDF()
        baseline_succ, _, _ = run_episode_flip(sim, goal, goal_norm, policy, flip_w1=False)
        
        # w1 flip
        sim = LeKiWiSimURDF()
        flip_succ, _, _ = run_episode_flip(sim, goal, goal_norm, policy, flip_w1=True)
        
        n_baseline_succ += int(baseline_succ)
        n_flip_succ += int(flip_succ)
        
        b_str = 'SUCC' if baseline_succ else 'FAIL'
        f_str = 'SUCC' if flip_succ else 'FAIL'
        chg = 'FIXED!' if (not baseline_succ and flip_succ) else ('BROKEN!' if (baseline_succ and not flip_succ) else 'same')
        print(f"{ep_id:>4} | {b_str:>10} {f_str:>10} | {chg:>10}")

    print("-" * 55)
    print(f"Baseline: {n_baseline_succ}/10  |  w1-flip: {n_flip_succ}/10")
    
    if n_flip_succ > n_baseline_succ:
        print("\n✓ w1 FLIP FIXES FAILURES! This confirms w1 sign inversion as root cause.")
    elif n_flip_succ == n_baseline_succ:
        print("\n? w1 flip has no effect — root cause may be elsewhere.")
    else:
        print("\n✗ w1 flip makes things WORSE — not the root cause.")

    # Save
    ts = time.strftime('%Y%m%d_%H%M')
    with open(f'results/phase176_w1flip_{ts}.json', 'w') as f:
        json.dump({
            'phase': 176,
            'baseline_sr': n_baseline_succ / 10 * 100,
            'flip_sr': n_flip_succ / 10 * 100,
            'test_episodes': test_episodes,
        }, f, indent=2)
    print(f"\n[INFO] Results saved to results/phase176_w1flip_{ts}.json")


if __name__ == "__main__":
    main()
