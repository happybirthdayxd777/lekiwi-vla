#!/usr/bin/env python3
"""Phase 18 — Goal Distributional Gap Validation (FAST version).
Tests goal_aware policy on ID vs OOD goals with fewer steps.
"""
import sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '.')
os.chdir('.')

from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

print("=" * 60)
print("Phase 18 — Goal Distributional Gap Validation (FAST)")
print("=" * 60)

# Load 11D goal_aware policy
ckpt_path = Path('results/goal_aware_50ep/final_policy.pt')
policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy.load_state_dict(ckpt.get('policy_state_dict', ckpt), strict=False)
policy.to('cpu').eval()
policy.device = 'cpu'
print(f"Loaded: {ckpt_path.name} (state_dim=11)")

sim = LeKiWiSimURDF()

# 5 ID goals, 5 OOD goals
id_goals = [(0.2, 0.0), (-0.2, 0.0), (0.3, 0.3), (-0.3, -0.3), (0.4, 0.0)]
ood_goals = [(0.5, 0.0), (0.6, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.5, 0.5)]

MAX_STEPS = 100  # fast eval

def resize_for_clip(img):
    pil = Image.fromarray(img)
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    img_chw = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_chw).unsqueeze(0).cpu()

def eval_goal(goal_x, goal_y):
    sim.reset()
    sim.set_target(np.array([goal_x, goal_y, 0.0]))
    img = sim.render()
    img_t = resize_for_clip(img)

    for step in range(MAX_STEPS):
        result = sim.step(np.zeros(9))
        obs = result[0] if isinstance(result, tuple) else result
        arm_pos = obs['arm_positions']
        wheel_vel = obs['wheel_velocities']
        goal_norm = np.array([goal_x / 1.0, goal_y / 1.0])
        state = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
        state_t = torch.from_numpy(state).unsqueeze(0).cpu()

        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()

        result2 = sim.step(action)
        obs2 = result2[0] if isinstance(result2, tuple) else result2

        robot_pos = sim.data.qpos[:2]
        dist = np.linalg.norm(robot_pos - np.array([goal_x, goal_y]))
        if dist < 0.15:
            return True, dist, step + 1

        img = sim.render()
        img_t = resize_for_clip(img)

    return False, dist, MAX_STEPS

print("\n--- IN-DISTRIBUTION (ID) ---")
id_results = []
for gx, gy in id_goals:
    ok, d, steps = eval_goal(gx, gy)
    id_results.append((gx, gy, ok, d, steps))
    print(f"  ({gx:+.1f},{gy:+.1f}): success={ok}, dist={d:.3f}m, steps={steps}")

id_sr = sum(r[2] for r in id_results) / len(id_results)
id_md = np.mean([r[3] for r in id_results])
print(f"  ID: SR={id_sr*100:.0f}%, mean_dist={id_md:.3f}m")

print("\n--- OUT-OF-DISTRIBUTION (OOD) ---")
ood_results = []
for gx, gy in ood_goals:
    ok, d, steps = eval_goal(gx, gy)
    ood_results.append((gx, gy, ok, d, steps))
    print(f"  ({gx:+.1f},{gy:+.1f}): success={ok}, dist={d:.3f}m, steps={steps}")

ood_sr = sum(r[2] for r in ood_results) / len(ood_results)
ood_md = np.mean([r[3] for r in ood_results])
print(f"  OOD: SR={ood_sr*100:.0f}%, mean_dist={ood_md:.3f}m")

print("\n=== CONCLUSION ===")
print(f"  ID  SR={id_sr*100:.0f}%, mean_dist={id_md:.3f}m")
print(f"  OOD SR={ood_sr*100:.0f}%, mean_dist={ood_md:.3f}m")
gap = id_md - ood_md
print(f"  ID-OOD gap: {gap:.3f}m (negative = OOD better than ID)")
if id_sr > ood_sr:
    print("  → OOD worse than ID: distributional gap confirmed")
elif id_sr == ood_sr == 0:
    print("  → Both 0%: need more training data or architecture fix")
else:
    print("  → No clear distributional gap")

import json
result = {
    "phase": 18,
    "policy": str(ckpt_path),
    "id_results": [{"goal": (r[0],r[1]), "success": r[2], "dist": float(r[3]), "steps": r[4]} for r in id_results],
    "ood_results": [{"goal": (r[0],r[1]), "success": r[2], "dist": float(r[3]), "steps": r[4]} for r in ood_results],
    "id_sr": float(id_sr), "id_mean_dist": float(id_md),
    "ood_sr": float(ood_sr), "ood_mean_dist": float(ood_md),
}
with open("data/goal_gap_eval.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved: data/goal_gap_eval.json")
