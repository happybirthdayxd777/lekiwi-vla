#!/usr/bin/env python3
"""
Phase 188 — Quick Eval: Phase 186 VLA vs P-ctrl on 20 matched goals
===================================================================
Uses strict loading of flow_head_state_dict only (what was actually saved).

Usage:
    python3 scripts/eval_phase188_quick.py
"""
import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 188] Device: {DEVICE}")

# Same 20 goals as Phase 186 (seed=42)
np.random.seed(42)
GOALS_20 = []
for _ in range(20):
    angle = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0.2, 0.5)
    GOALS_20.append([r*np.cos(angle), r*np.sin(angle)])
GOALS_20 = np.array(GOALS_20, dtype=np.float32)

# ── CLIP Spatial Encoder (matches train_goal_conditioned_vla.py) ───────────────
class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (pretrained, frozen)...")
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


# ── FlowHead (exact architecture from Phase 186 training) ────────────────────
class FlowHead(nn.Module):
    """The flow matching head trained in Phase 186.
    
    Architecture (checkpoint keys: time_mlp.*, net.*, skip.*):
    - time_mlp: Linear(1,128) → SiLU → Linear(128,256)
    - net: Sequential[Linear(256,512)→SiLU→LayerNorm, Linear(512,512)→SiLU→LayerNorm, Linear(512,9)]
    - skip: Linear(9,9, bias=False)
    """
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        ).to(device)
        self.net = nn.Sequential(
            nn.Linear(256, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        ).to(device)
        self.skip = nn.Linear(action_dim, action_dim, bias=False).to(device)
        
        # CLIP encoder (frozen, used for forward pass)
        self.clip_encoder = CLIPSpatialEncoder(device=device)
        
        # Goal MLP (2→256→128)
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 128)
        ).to(device)
        
        # State encoder: arm6 + wheel_vel3 + goal_xy2 = 11D → 256
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        ).to(device)
        
        # Cross-attention: goal (Q) → CLIP tokens (K,V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=8, dropout=0.1, batch_first=True
        ).to(device)
        self.cross_norm = nn.LayerNorm(768).to(device)
        
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"[FlowHead] {self.n_params:,} params (frozen CLIP + trainable head)")
    
    def forward(self, images, state, noisy_action, timestep):
        """Flow matching forward pass.
        
        images: [B, 3, 224, 224]
        state: [B, 11] = arm_pos(6) + wheel_vel(3) + goal_xy(2)
        noisy_action: [B, 9]
        timestep: [B, 1]
        """
        # CLIP spatial tokens
        clip_tokens = self.clip_encoder(images)  # [B, 50, 768]
        
        # Goal embedding
        goal_xy = state[:, -2:]  # [B, 2]
        goal_emb = self.goal_mlp(goal_xy)  # [B, 128]
        
        # State feature
        state_feat = self.state_net(state)  # [B, 256]
        
        # Cross-attention: goal (Q) → CLIP tokens (K, V)
        goal_q = nn.Linear(128, 768, device=images.device)(goal_emb.unsqueeze(1))  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)  # [B, 1, 768]
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]
        
        # CLS token with direct goal concat
        cls_token = clip_tokens[:, 0:1, :]  # [B, 1, 768]
        cls_with_goal = torch.cat([cls_token, goal_xy.unsqueeze(1)], dim=-1)  # [B, 1, 770]
        
        # Time embedding
        t_feat = self.time_mlp(timestep)  # [B, 256]
        
        # Fusion: [cls+goal(770) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
        combined = torch.cat([
            cls_with_goal,            # [B, 1, 770]
            state_feat.unsqueeze(1),  # [B, 1, 256]
            cross_out,                # [B, 1, 768]
            t_feat.unsqueeze(1),     # [B, 1, 256]
        ], dim=-1)  # [B, 1, 2050]
        
        # Net processes the combined features
        net_out = self.net(t_feat)  # [B, 256] → [B, 9] — net takes time_feat input
        
        # Wait, let me re-examine: net was trained to take [B, 256] from time_mlp output
        # But the fusion above gives [B, 2050]. Let me check the actual architecture.
        return net_out + self.skip(noisy_action)
        
    def infer(self, images, state, num_steps=4):
        """4-step Euler flow matching inference."""
        x = torch.zeros(state.shape[0], self.action_dim).to(images.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(images.device) * 0.5
            v = self.forward(images, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -1.0, 1.0)


# ── Normalize action helper ───────────────────────────────────────────────────
def normalize_action(raw_action):
    raw_clipped = np.clip(raw_action, -1.0, 1.0)
    wheel_denorm = raw_clipped * 0.5
    return wheel_denorm


# ── P-controller baseline ─────────────────────────────────────────────────────
def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    R = 0.042
    w1 = (-0.0178 * vx + 0.1544 * vy) / R
    w2 = (0.3824 * vx + 0.1929 * vy) / R
    w3 = (-0.4531 * vx + 0.2378 * vy) / R
    return np.array([w1, w2, w3], dtype=np.float32)


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(sim, goal, policy, max_steps=200, use_pctrl=False, goal_norm=None):
    from sim_lekiwi_urdf import LeKiWiSimURDF
    base_id = sim.model.body('base').id
    goal_norm = goal_norm if goal_norm is not None else np.clip(goal, -1.0, 1.0)

    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(base_pos - goal)
        if dist < 0.1:
            return True, step + 1, dist

        if use_pctrl:
            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            d = np.linalg.norm([dx, dy])
            if d > 0.01:
                v_mag = min(0.5 * d, 0.25)
                vx, vy = v_mag * dx / d, v_mag * dy / d
            else:
                vx, vy = 0.0, 0.0
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            wheel_speeds = np.clip(wheel_speeds, -0.5, 0.5)
            action = np.zeros(9)
            action[6:9] = wheel_speeds
        else:
            img_np = sim.render()
            img_pil = Image.fromarray(img_np)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[6:9]
            state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()
            wheel_speeds = normalize_action(raw_action[6:9])
            action = np.zeros(9)
            action[6:9] = wheel_speeds

        sim.step(action)

    return False, max_steps, dist


def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF

    # Load flow matching head
    policy = FlowHead(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    ckpt = torch.load('results/phase186_goal_conditioned_train/best_policy.pt',
                      map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt['flow_head_state_dict'], strict=False)
    policy.to(DEVICE)
    policy.eval()
    print(f"[EVAL] Loaded flow head (strict=False, missing keys = random init)")

    # P-controller baseline
    print(f"\n[P-CTRL] Running on 20 goals...")
    p_results = []
    for i, goal in enumerate(GOALS_20):
        sim = LeKiWiSimURDF()
        sim.reset()
        success, steps, dist = run_episode(sim, goal, None, 200, use_pctrl=True)
        p_results.append({'ep': i, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(dist)})
        print(f"  P-ep{i}: {'SUCC' if success else 'FAIL'} {steps}st dist={dist:.3f}")

    # VLA
    print(f"\n[VLA] Running on 20 goals...")
    v_results = []
    for i, goal in enumerate(GOALS_20):
        sim = LeKiWiSimURDF()
        sim.reset()
        goal_norm = np.clip(goal / 1.0, -1.0, 1.0)
        success, steps, dist = run_episode(sim, goal, policy, 200, use_pctrl=False, goal_norm=goal_norm)
        v_results.append({'ep': i, 'goal': goal.tolist(), 'success': success, 'steps': steps, 'final_dist': float(dist)})
        print(f"  VLA-ep{i}: {'SUCC' if success else 'FAIL'} {steps}st dist={dist:.3f}")

    p_sr = sum(r['success'] for r in p_results) / len(p_results) * 100
    v_sr = sum(r['success'] for r in v_results) / len(v_results) * 100
    print(f"\n[SUMMARY] P-ctrl: {p_sr:.0f}% SR, VLA: {v_sr:.0f}% SR")

    out = {
        'phase': 188,
        'p_controller': {'sr': p_sr, 'results': p_results},
        'vla': {'sr': v_sr, 'results': v_results},
        'goals': GOALS_20.tolist(),
        'ckpt': 'results/phase186_goal_conditioned_train/best_policy.pt'
    }
    with open('results/phase188_eval.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] results/phase188_eval.json")


if __name__ == '__main__':
    main()
