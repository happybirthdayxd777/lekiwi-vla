#!/usr/bin/env python3
"""Phase 216: Quick eval Phase 196 VLA (epoch 14) vs Contact-Jacobian P-ctrl"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV, ARM_JOINTS, WHEEL_JOINTS
from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from PIL import Image

ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_14.pt', map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
policy.to(DEVICE).eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(raw_img):
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.from_numpy(arr.transpose(2,0,1))

arm_default = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

print("=== Phase 216: Phase 196 VLA Eval ===")
np.random.seed(42)

results = {}
for mode, label in [('pctrl', 'P-ctrl (CJ kP=2.0)'), ('vla', 'VLA-e14')]:
    successes = 0
    for g_i in range(10):
        goal = np.array([np.random.uniform(-0.3, 0.4), np.random.uniform(-0.25, 0.25)])
        sim = LeKiWiSimURDF(); sim.reset()
        base_id = sim.model.body('base').id
        for step in range(150):
            base_xy = sim.data.xpos[base_id, :2]
            if np.linalg.norm(goal - base_xy) < 0.10:
                successes += 1; break
            if mode == 'pctrl':
                v = 2.0 * (goal - base_xy)
                ws = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v, -0.5, 0.5)
                action = np.concatenate([arm_default, ws])
            else:
                # FIXED Phase 222: wheel vels are qvel[6:9], NOT qvel[9:12]=ARM_vel
                wv = sim.data.qvel[6:9].copy()
                gn = np.clip(goal / 0.525, -1, 1)
                sv = np.concatenate([arm_default, wv, gn]).astype(np.float32)
                img = preprocess(sim.render().astype(np.uint8))
                with torch.no_grad():
                    a = policy.infer(img.unsqueeze(0).to(DEVICE),
                        torch.from_numpy(sv).unsqueeze(0).to(DEVICE), num_steps=4)
                    a = a.squeeze(0).cpu().numpy()
                a[3:] = np.clip(a[3:], -0.5, 0.5); action = a
            sim.step(action)
    sr = 100 * successes / 10
    results[mode] = sr
    print(f"  {label}: {successes}/10 = {sr:.0f}% SR")

print(f"\nResult: VLA SR={results['vla']:.0f}% vs P-ctrl={results['pctrl']:.0f}%")
if results['vla'] >= results['pctrl']:
    print("VLA matches/beats P-ctrl baseline!")
elif results['vla'] >= 20:
    print("VLA is functional (>=20% SR)")
else:
    print("VLA underperforms — architecture may need redesign")