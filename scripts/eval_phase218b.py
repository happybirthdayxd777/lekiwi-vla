#!/usr/bin/env python3
"""
Phase 218b: Compare phase190 (epoch_27, 30ep full training) vs phase196 (epoch_14, 14ep interrupted).
Both architectures are identical (11D state, 9D action, 512 hidden).
Goal: determine if more training epochs give better generalization.
"""
import sys, os, numpy as np, torch, time
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV, ARM_JOINTS, WHEEL_JOINTS
from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    'phase190_e27': 'results/phase190_vision_train/best_policy.pt',  # epoch 27/30, full train
    'phase196_e14': 'results/phase196_contact_jacobian_train/epoch_14.pt',  # epoch 14/30, interrupted
}
N_GOALS = 10
SEED = 99  # different from phase218's seed=42 for diversity
SUCCESS_R = 0.10
MAX_STEPS = 200

# ── Preprocessing ──────────────────────────────────────────────────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(raw_img):
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))

ARM_DEFAULT = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

# ── Goals ─────────────────────────────────────────────────────────────────────
np.random.seed(SEED)
GOALS = [
    np.array([np.random.uniform(-0.35, 0.40), np.random.uniform(-0.25, 0.25)])
    for _ in range(N_GOALS)
]

# ── Load policies ─────────────────────────────────────────────────────────────
policies = {}
for name, path in CHECKPOINTS.items():
    print(f"Loading {name} from {path}...")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
    policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
    policy.to('cpu').eval()
    policies[name] = policy
    print(f"  epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss', ckpt.get('losses', ['?'])[-1] if isinstance(ckpt.get('losses'), list) else '?')}")

def run_policy(sim, goal, policy):
    base_id = sim.model.body('base').id
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(goal - base_xy)
        if dist < SUCCESS_R:
            return {'success': True, 'steps': step, 'final_dist': dist}
        # FIXED Phase 222: wheel vels are qvel[6:9], NOT qvel[9:12]=ARM_vel
        wv = sim.data.qvel[6:9].copy()
        gn = np.clip(goal / 0.525, -1, 1)
        sv = np.concatenate([ARM_DEFAULT, wv, gn]).astype(np.float32)
        img = preprocess(sim.render().astype(np.uint8))
        with torch.no_grad():
            a = policy.infer(
                img.unsqueeze(0).to('cpu'),
                torch.from_numpy(sv).unsqueeze(0).to('cpu'),
                num_steps=4
            )
            a = a.squeeze(0).cpu().numpy()
        a[3:] = np.clip(a[3:], -0.5, 0.5)
        sim.step(a)
    return {'success': False, 'steps': MAX_STEPS, 'final_dist': dist}

# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\n=== Phase 218b: Phase190(30ep) vs Phase196(14ep) — {N_GOALS} goals, seed={SEED} ===\n")

results = {name: [] for name in policies}
t_start = time.time()

for g_i, goal in enumerate(GOALS):
    print(f"Goal {g_i+1}/{N_GOALS}: {goal.round(3)}")
    for name, policy in policies.items():
        sim = LeKiWiSimURDF(); sim.reset()
        r = run_policy(sim, goal, policy)
        results[name].append(r)
        flag = "✓" if r['success'] else f"✗ dist={r['final_dist']:.3f}"
        print(f"  {name}: {flag} (steps={r['steps']})")

elapsed = time.time() - t_start

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for name, res in results.items():
    n_ok = sum(r['success'] for r in res)
    sr = 100 * n_ok / N_GOALS
    avg_steps = np.mean([r['steps'] for r in res])
    avg_dist = np.mean([r['final_dist'] for r in res])
    print(f"\n{name}:")
    print(f"  SR: {n_ok}/{N_GOALS} = {sr:.0f}%")
    print(f"  Avg steps: {avg_steps:.1f}, Avg final dist: {avg_dist:.4f}m")

print(f"\nTime: {elapsed:.1f}s")

# Save
import json
out = {
    'phase': '218b',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seed': SEED,
    'n_goals': N_GOALS,
    'checkpoints': {k: v for k, v in CHECKPOINTS.items()},
    'results': {name: [{'success': r['success'], 'steps': r['steps'],
                       'final_dist': float(r['final_dist']),
                       'goal': [float(x) for x in GOALS[i]]}
                      for i, r in enumerate(res)]
                for name, res in results.items()},
}
with open('results/phase218b_eval.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved: results/phase218b_eval.json")