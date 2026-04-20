#!/usr/bin/env python3
"""
Phase 240: Definitive VLA Evaluation — sr=0.10 vs sr=0.15
========================================================
Uses the CORRECT architecture matching Phase196/Phase227 training.

Architecture (verified from checkpoint weights):
  - CLIP ViT-B/32 (openai/clip-vit-base-patch32): [B, 50, 768] tokens
  - Goal MLP: 2 → 256 → 128
  - State net: 11 → 256 → 128
  - Cross-attention: goal(Q, 768) attends to CLIP(K/V, 768) → [B, 1, 768]
  - Flow head: [1929 → 512 → 512 → 9], where 1929 = 768(cls)+768(cross)+128(state)+256(time)+9(action)

Data (Phase196): states=(5562,11) arm_pos(6)+wheel_vel(3)+goal_norm(2), actions=(5562,9)
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

import numpy as np
import mujoco
import torch
import torch.nn as nn
from transformers import CLIPModel
from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV
from PIL import Image

DEVICE = 'cpu'
np.random.seed(42)

# ── VLA Policy (matching Phase196 architecture exactly) ───────────────────────
class CLIPVisionEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        from transformers import AutoModel
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32
        )
        self.clip.to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
    
    def forward(self, images):
        outputs = self.clip.vision_model(images)
        return outputs.last_hidden_state  # [B, 50, 768]

class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device='cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        
        self.encoder = CLIPVisionEncoder(device=device)
        
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )
        self.goal_q_proj = nn.Linear(128, 768)
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(768)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU()
        )
        
        self.flow_head = nn.Sequential(
            nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim)
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)
    
    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)  # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)  # [B, 1, 768]
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
        ], dim=-1).squeeze(1)
        return self.flow_head(x) + self.skip(noisy_action)
    
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

def load_policy(pt_path):
    """Load VLA policy with correct architecture."""
    ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get('policy_config', {'state_dim': 11, 'action_dim': 9, 'hidden': 512})
    policy = GoalConditionedPolicy(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden=config['hidden'],
        device=DEVICE
    )
    sd = ckpt.get('policy_state_dict', ckpt)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:3]}...")
    policy.to(DEVICE).eval()
    print(f"  Loaded {pt_path} (epoch {ckpt.get('epoch','?')}, loss={ckpt.get('loss',0):.3f})")
    return policy

# ── Observation ────────────────────────────────────────────────────────────────
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)

def make_obs(sim, goal_xy):
    arm_pos = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
    wheel_vel = sim.data.qvel[9:12].copy()
    state_vec = np.concatenate([arm_pos, wheel_vel, np.array(goal_xy)/0.525]).astype(np.float32)
    
    img = sim.render()
    pil_img = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2, 0, 1)
    img_t = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)
    return img_t, state_t

def p_controller_obs(goal_xy, sim, kP=2.0):
    """Contact-Jacobian P-controller (oracle baseline)."""
    base_xy = sim.data.xpos[sim.model.body('base').id, :2]
    err = np.array(goal_xy) - base_xy
    dist = np.linalg.norm(err)
    if dist < 0.05:
        return np.zeros(9)
    v_desired = kP * err
    wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
    arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
    return np.concatenate([arm, wheel_speeds])

def run_episode(sim, policy_fn, goal_xy, use_vla=False, max_steps=200, sr=0.10):
    base_body_id = sim.model.body('base').id
    for step in range(max_steps):
        base_xy = sim.data.xpos[base_body_id, :2]
        dist = np.linalg.norm(np.array(goal_xy) - base_xy)
        if dist < sr:
            return True, step+1, dist
        if use_vla:
            img_t, state_t = make_obs(sim, goal_xy)
            with torch.no_grad():
                action = policy_fn.infer(img_t, state_t).squeeze(0).cpu().numpy()
        else:
            action = policy_fn(goal_xy, sim)
        sim.step(action)
    base_xy = sim.data.xpos[base_body_id, :2]
    return False, max_steps, np.linalg.norm(np.array(goal_xy) - base_xy)

# ── Goals ──────────────────────────────────────────────────────────────────────
np.random.seed(42)
goals_20 = [(np.random.uniform(-0.3, 0.4), np.random.uniform(-0.25, 0.25)) for _ in range(20)]

def eval_policy(name, policy_fn, use_vla=False, sr=0.10):
    successes = 0
    results = []
    for i, g in enumerate(goals_20):
        sim = LeKiWiSimURDF()
        sim.reset()
        s, steps, d = run_episode(sim, policy_fn, np.array(g), use_vla=use_vla, sr=sr)
        successes += int(s)
        results.append({"goal": g, "success": s, "dist": d, "steps": steps})
        mark = '✓' if s else '✗'
        print(f"  [{i+1:2d}/20] g=({g[0]:+.2f},{g[1]:+.2f}) d={d:.3f}m {mark}")
    print(f"  -> {name}: {successes}/20 = {successes*5.0:.0f}% SR (sr={sr}m)")
    return successes * 5.0, successes, results

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Phase 240: Definitive VLA Cross-Radius Evaluation (20 goals)")
print("=" * 70)

results = {}

# P-controller (oracle baseline)
for sr in [0.10, 0.15]:
    label = f"P-ctrl sr={sr}m"
    print(f"\n[{label}]")
    sr_pct, n, _ = eval_policy("P-controller CJ", lambda g, s: p_controller_obs(g, s), use_vla=False, sr=sr)
    results[("P-controller", sr)] = (sr_pct, n)

# Load and eval VLA policies
policies = [
    ("results/phase196_contact_jacobian_train/epoch_14.pt", "VLA Phase196"),
    ("results/phase227_contact_jacobian_train/epoch_30.pt", "VLA Phase227"),
]

for pt_path, label in policies:
    if not os.path.exists(pt_path):
        print(f"  SKIP: {pt_path} not found")
        continue
    try:
        policy = load_policy(pt_path)
    except Exception as e:
        print(f"  Load error {pt_path}: {e}")
        continue
    
    for sr in [0.10, 0.15]:
        print(f"\n[{label} sr={sr}m]")
        sr_pct, n, _ = eval_policy(label, policy, use_vla=True, sr=sr)
        results[(label, sr)] = (sr_pct, n)

# Summary
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Policy':<25} {'sr=0.10m':>10} {'sr=0.15m':>10}")
print(f"{'-'*45}")
for label, _ in [("P-controller CJ", None), ("VLA Phase196", None), ("VLA Phase227", None)]:
    r10 = results.get((label, 0.10), (0, 0))[0]
    r15 = results.get((label, 0.15), (0, 0))[0]
    r10_str = f"{r10:.0f}%" if r10 else "N/A"
    r15_str = f"{r15:.0f}%" if r15 else "N/A"
    print(f"{label:<25} {r10_str:>10} {r15_str:>10}")
print("=" * 70)
