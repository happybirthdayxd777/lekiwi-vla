#!/usr/bin/env python3
"""
Phase 191: Evaluate Phase 190 VLA policy (GoalConditionedPolicy, 11D state).
Best checkpoint: results/phase190_vision_train/epoch_14.pt

Key: phase189 data was CORRUPTED by *200 scaling → wheel speeds saturate to ±0.5
This eval tests the trained policy on the SAME corrupted data to establish baseline.

Architecture (from train_phase190.py):
  - CLIP ViT-B/32 spatial tokens [B, 50, 768]
  - Goal MLP: 2 → 256 → 128
  - State net: 11D → 256 → 128
  - Cross-attention: goal(Q) attends to CLIP(K,V) → [B, 1, 768]
  - 4-step Euler flow matching

State: 11D = arm_pos(6) + wheel_vel(3) + goal_norm(2)
Action: 9D = arm_torque(6) + wheel_speed(3)

Compare: VLA policy vs P-controller baseline
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import time

sys.path.insert(0, '/Users/i_am_ai/hermes_research/lekiwi_vla')
os.chdir('/Users/i_am_ai/hermes_research/lekiwi_vla')

from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds

DEVICE = 'cpu'  # Phase 190 trained on CPU
CKPT_PATH = 'results/phase190_vision_train/epoch_14.pt'
MAX_STEPS = 200
GOAL_THRESHOLD = 0.1
N_EPISODES = 30
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── CLIP Vision Encoder (same as train_phase190.py) ─────────────────────────

class CLIPVisionEncoder(torch.nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[CLIP] Loading CLIP ViT-B/32...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        """images: [B, 3, 224, 224] in normalized form. Returns: [B, 50, 768]"""
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            return outputs.last_hidden_state  # [B, 50, 768]


# ── GoalConditioned Policy (same as train_phase190.py) ───────────────────────

class GoalConditionedPolicy(torch.nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.encoder = CLIPVisionEncoder(device=device)

        self.goal_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )

        self.state_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128), torch.nn.SiLU()
        )

        self.goal_q_proj = torch.nn.Linear(128, 768)
        self.cross_attn = torch.nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = torch.nn.LayerNorm(768)

        self.flow_head = torch.nn.Sequential(
            torch.nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 256), torch.nn.SiLU()
        )

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)
        goal_emb = self.goal_mlp(state[:, -2:])

        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)

        state_feat = self.state_net(state)
        t_emb = self.time_mlp(timestep)

        cls_token = clip_tokens[:, 0:1, :]
        x = torch.cat([
            cls_token,
            cross_out,
            state_feat.unsqueeze(1),
            t_emb.unsqueeze(1),
            noisy_action.unsqueeze(1),
        ], dim=-1)

        x = x.squeeze(1)
        return self.flow_head(x)

    def infer(self, images, state, num_steps=4):
        self.eval()
        with torch.no_grad():
            x = torch.zeros_like(state[:, :self.action_dim])
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((images.shape[0], 1), i * dt, device=state.device)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ── Image preprocessing (same as train_phase190.py) ─────────────────────────

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img: np.ndarray) -> torch.Tensor:
    """raw_img: (H, W, 3) uint8 → (3, 224, 224) normalized float32"""
    img = Image.fromarray(raw_img)
    img = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


# ── P-controller (baseline) ──────────────────────────────────────────────────

def p_controller_action(sim, goal_xy, kP=0.5):
    base_xy = sim.data.qpos[:2].copy()
    dx, dy = goal_xy[0] - base_xy[0], goal_xy[1] - base_xy[1]
    dist = np.linalg.norm([dx, dy])
    if dist < 0.005:
        return [0.0, 0.0, 0.0]
    vx, vy = kP * dx, kP * dy
    return twist_to_contact_wheel_speeds(vx, vy, 0.0)


def run_episode_pctrl(sim, goal, max_steps=200):
    sim.reset()
    for _ in range(15):
        sim.step([0]*9)
    for step in range(max_steps):
        ctrl = p_controller_action(sim, goal)
        action_np = np.array(list([0]*6) + list(ctrl))
        sim.step(action_np)
        dist = np.linalg.norm(sim.data.qpos[:2] - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist
    return False, max_steps, dist


# ── VLA inference ─────────────────────────────────────────────────────────────

def load_vla():
    print(f"[VLA] Loading {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    policy = GoalConditionedPolicy(
        state_dim=config.get('state_dim', 11),
        action_dim=config.get('action_dim', 9),
        hidden=config.get('hidden', 512),
        device=DEVICE
    )
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()
    print(f"  Loaded epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}")
    return policy


def run_episode_vla(sim, policy, goal, max_steps=200):
    goal_arr = np.array(goal, dtype=np.float32)
    goal_norm = np.clip(goal_arr / 0.5, -1, 1).astype(np.float32)
    sim.reset()
    for _ in range(15):
        sim.step([0]*9)

    for step in range(max_steps):
        # Render image
        img = sim.render()
        img_t = preprocess_image(np.array(img)).unsqueeze(0).to(DEVICE)

        obs = sim._obs()
        arm = obs['arm_positions']
        wheel_v = obs['wheel_velocities']

        # 11D state: arm_pos(6) + wheel_vel(3) + goal_norm(2)
        state11 = np.concatenate([
            np.clip(arm / 2.0, -1, 1),      # arm_pos → normalized
            np.clip(wheel_v / 0.5, -1, 1),  # wheel_vel → normalized
            goal_norm                           # goal_norm (already [-1,1])
        ]).astype(np.float32)
        state_t = torch.from_numpy(state11).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)

        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, -1, 1).astype(np.float32)

        # action[0:6] = arm_torque, action[6:9] = wheel_speed (normalized [-1,1])
        # Scale wheel to actual rad/s: * 0.5
        wheel_ctrl = action_np[6:9] * 0.5
        wheel_ctrl = np.clip(wheel_ctrl, -0.5, 0.5)
        sim.step(action_np)

        dist = np.linalg.norm(sim.data.qpos[:2] - goal)
        if dist < GOAL_THRESHOLD:
            return True, step+1, dist

    return False, max_steps, dist


# ── Test goals ────────────────────────────────────────────────────────────────

def get_test_goals(n=30, seed=42):
    rng = np.random.RandomState(seed)
    goals = []
    for _ in range(n):
        angle = rng.uniform(0, 2*np.pi)
        dist = rng.uniform(0.2, 0.45)
        goals.append((dist * np.cos(angle), dist * np.sin(angle)))
    return goals


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 191: Evaluate Phase 190 VLA vs P-controller Baseline")
    print("=" * 60)

    sim = LeKiWiSimURDF()
    goals = get_test_goals(N_EPISODES, SEED)

    # Load VLA policy
    policy = load_vla()

    results = {
        'vla': {'success': [], 'steps': [], 'final_dist': []},
        'pctrl': {'success': [], 'steps': [], 'final_dist': []},
    }

    t_start = time.time()

    for i, goal in enumerate(goals):
        print(f"\n[Goal {i+1}/{N_EPISODES}] {goal[0]:.3f}, {goal[1]:.3f}")

        # VLA
        ok, steps, dist = run_episode_vla(sim, policy, goal, MAX_STEPS)
        results['vla']['success'].append(ok)
        results['vla']['steps'].append(steps)
        results['vla']['final_dist'].append(float(dist))
        print(f"  VLA:    {'✓' if ok else '✗'} steps={steps}, dist={dist:.4f}")

        # P-ctrl
        ok, steps, dist = run_episode_pctrl(sim, goal, MAX_STEPS)
        results['pctrl']['success'].append(ok)
        results['pctrl']['steps'].append(steps)
        results['pctrl']['final_dist'].append(float(dist))
        print(f"  P-ctrl: {'✓' if ok else '✗'} steps={steps}, dist={dist:.4f}")

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for label, data in results.items():
        sr = np.mean(data['success']) * 100
        avg_steps = np.mean([s for s, ok in zip(data['steps'], data['success']) if ok])
        avg_dist = np.mean(data['final_dist'])
        print(f"\n{label.upper()}")
        print(f"  Success Rate: {sr:.1f}% ({sum(data['success'])}/{N_EPISODES})")
        print(f"  Avg Steps (success): {avg_steps:.1f}")
        print(f"  Avg Final Dist: {avg_dist:.4f}")

    vla_sr = np.mean(results['vla']['success']) * 100
    p_sr   = np.mean(results['pctrl']['success']) * 100

    print(f"\n[Phase 191] VLA SR={vla_sr:.1f}% vs P-ctrl SR={p_sr:.1f}%")
    print(f"[Phase 191] elapsed={elapsed:.0f}s")

    # Save results
    output_path = 'results/phase191_vla_eval.json'
    with open(output_path, 'w') as f:
        json.dump({
            'vla_success_rate': vla_sr,
            'pctrl_success_rate': p_sr,
            'vla_results': results['vla'],
            'pctrl_results': results['pctrl'],
            'goals': goals,
            'elapsed_seconds': elapsed,
            'checkpoint': CKPT_PATH,
        }, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()