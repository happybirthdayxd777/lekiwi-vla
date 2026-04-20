#!/usr/bin/env python3
"""Phase 240 Quick Check: 10-goal eval to verify the script works."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

import numpy as np
import mujoco
import torch
from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV
from PIL import Image

np.random.seed(42)
DEVICE = 'cpu'

def make_vla_policy(pt_path):
    class VLA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.img_enc = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 8, stride=4), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 4, stride=2), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, stride=2), torch.nn.ReLU(),
                torch.nn.Conv2d(128, 256, 3, stride=2), torch.nn.ReLU(),
                torch.nn.Flatten(),
            )
            self.state_enc = torch.nn.Sequential(
                torch.nn.Linear(11, 256), torch.nn.SiLU(), torch.nn.LayerNorm(256),
            )
            self.cross_attn = torch.nn.MultiheadAttention(256, 4, batch_first=True)
            self.fuse = torch.nn.Sequential(
                torch.nn.Linear(768 + 768 + 128 + 256 + 9, 512), torch.nn.SiLU(),
            )
            self.head = torch.nn.Linear(512, 9)
            self.skip = torch.nn.Linear(9, 9, bias=False)
        
        def forward(self, img, state, noisy_action, timestep):
            img_feat = self.img_enc(img).flatten(1)
            state_feat = self.state_enc(state)
            img_seq = img_feat.unsqueeze(1)
            attn_out, _ = self.cross_attn(img_seq, img_seq, img_seq)
            fused = torch.cat([attn_out.squeeze(1), state_feat, noisy_action], dim=1)
            out = self.fuse(fused)
            return self.head(out) + self.skip(noisy_action)
        
        def infer(self, img, state, num_steps=4):
            action = torch.zeros(9)
            for _ in range(num_steps):
                with torch.no_grad():
                    action = self.forward(img, state, action, torch.tensor([0.0]))
            return action
    
    policy = VLA()
    try:
        ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)
        if 'policy_state_dict' in ckpt:
            policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
        else:
            policy.load_state_dict(ckpt, strict=False)
        print(f"  Loaded: {pt_path}")
    except Exception as e:
        print(f"  Load error: {e}")
        return None
    policy.to(DEVICE).eval()
    return policy

def make_obs(sim, goal_norm):
    arm_pos = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
    wheel_vel = sim.data.qvel[9:12].copy()
    state_vec = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
    img = sim.render()
    pil_img = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - img_mean) / img_std
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE), torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)

print("=" * 60)
print("Phase 240: Quick 10-goal Check (sr=0.10 and sr=0.15)")
print("=" * 60)

np.random.seed(99)  # Different seed for quick check
goals_10 = [(np.random.uniform(-0.3, 0.4), np.random.uniform(-0.25, 0.25)) for _ in range(10)]

def run_ep(sim, policy_fn, goal_xy, use_vla=False, max_steps=200, sr=0.10):
    base_body_id = sim.model.body('base').id
    goal_norm = np.array(goal_xy) / 0.525
    arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
    for step in range(max_steps):
        base_xy = sim.data.xpos[base_body_id, :2]
        dist = np.linalg.norm(goal_xy - base_xy)
        if dist < sr:
            return True, step+1, dist
        if use_vla:
            img_t, state_t = make_obs(sim, goal_norm)
            with torch.no_grad():
                action = policy_fn(img_t, state_t).squeeze(0).cpu().numpy()
        else:
            err = goal_xy - base_xy
            v_desired = 2.0 * err
            wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
            action = np.concatenate([arm, wheel_speeds])
        sim.step(action)
    base_xy = sim.data.xpos[base_body_id, :2]
    return False, max_steps, np.linalg.norm(goal_xy - base_xy)

def eval_quick(name, policy_fn, use_vla=False, sr=0.10):
    successes = 0
    for i, g in enumerate(goals_10):
        g = np.array(g)
        sim = LeKiWiSimURDF()
        sim.reset()
        s, steps, d = run_ep(sim, policy_fn, g, use_vla=use_vla, sr=sr)
        successes += int(s)
        mark = '✓' if s else '✗'
        print(f"  [{i+1}/10] g=({g[0]:+.2f},{g[1]:+.2f}) d={d:.3f}m {mark}")
    print(f"  → {name}: {successes}/10 = {successes*10}% SR (sr={sr}m)")
    return successes * 10

# P-ctrl
for sr in [0.10, 0.15]:
    print(f"\n[P-ctrl sr={sr}m]")
    eval_quick("P-ctrl", None, use_vla=False, sr=sr)

# VLA Phase196
p196 = make_vla_policy("results/phase196_contact_jacobian_train/epoch_14.pt")
if p196:
    for sr in [0.10, 0.15]:
        print(f"\n[VLA Phase196 sr={sr}m]")
        eval_quick("VLA p196", lambda img, st: p196.infer(img, st), use_vla=True, sr=sr)

# VLA Phase227
p227 = make_vla_policy("results/phase227_contact_jacobian_train/epoch_30.pt")
if p227:
    for sr in [0.10, 0.15]:
        print(f"\n[VLA Phase227 sr={sr}m]")
        eval_quick("VLA p227", lambda img, st: p227.infer(img, st), use_vla=True, sr=sr)

print("\n" + "=" * 60)
