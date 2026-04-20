#!/usr/bin/env python3
"""
Phase 218: Comprehensive Phase 196 VLA evaluation — 10 goals, detailed metrics

Goal: Get a proper 10-goal success rate comparison between:
  1. VLA-e14 (epoch_14.pt) — vision-based policy
  2. Contact-Jacobian P-controller — oracle baseline

Each goal is tested with both controllers for fair comparison.
Also tracks: time-to-success, final distance, trajectory quality.
"""
import sys, os, numpy as np, torch, time
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV, ARM_JOINTS, WHEEL_JOINTS
from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = 'results/phase196_contact_jacobian_train/epoch_14.pt'
DATA_H5    = 'data/phase196_clean_50ep.h5'
N_GOALS    = 10
SEED       = 42
SUCCESS_R  = 0.10   # meters
MAX_STEPS  = 200
DEVICE_CPU = 'cpu'

# ── Preprocessing (same as training) ─────────────────────────────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(raw_img: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))

# ── Load policy ────────────────────────────────────────────────────────────────
print("Loading VLA policy...")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE_CPU, weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE_CPU)
policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
policy.to(DEVICE_CPU).eval()
print(f"  Policy epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')}")

# ── Default arm pose ──────────────────────────────────────────────────────────
ARM_DEFAULT = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

# ── Goal pool (fixed for reproducibility) ──────────────────────────────────────
np.random.seed(SEED)
GOALS = [
    np.array([np.random.uniform(-0.35, 0.40), np.random.uniform(-0.25, 0.25)])
    for _ in range(N_GOALS)
]

# ── Controllers ───────────────────────────────────────────────────────────────
def run_pcontroller(sim: LeKiWiSimURDF, goal: np.ndarray) -> dict:
    """Contact-Jacobian P-controller oracle"""
    base_id = sim.model.body('base').id
    history = []
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(goal - base_xy)
        history.append({'step': step, 'dist': dist, 'base_xy': base_xy.copy()})
        if dist < SUCCESS_R:
            return {'success': True, 'steps': step, 'final_dist': dist, 'history': history}
        v = 2.0 * (goal - base_xy)
        ws = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v, -0.5, 0.5)
        action = np.concatenate([ARM_DEFAULT, ws])
        sim.step(action)
    return {'success': False, 'steps': MAX_STEPS, 'final_dist': dist, 'history': history}


def run_vla(sim: LeKiWiSimURDF, goal: np.ndarray, policy) -> dict:
    """VLA-e14 policy with 4-step flow matching inference"""
    base_id = sim.model.body('base').id
    history = []
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(goal - base_xy)
        history.append({'step': step, 'dist': dist, 'base_xy': base_xy.copy()})
        if dist < SUCCESS_R:
            return {'success': True, 'steps': step, 'final_dist': dist, 'history': history}

        # Build state vector
        # FIXED Phase 222: wheel vels are qvel[6:9], NOT qvel[9:12]=ARM_vel
        wv = sim.data.qvel[6:9].copy()
        gn = np.clip(goal / 0.525, -1, 1)
        sv = np.concatenate([ARM_DEFAULT, wv, gn]).astype(np.float32)

        # Render image
        img = preprocess(sim.render().astype(np.uint8))

        # Inference
        with torch.no_grad():
            a = policy.infer(
                img.unsqueeze(0).to(DEVICE_CPU),
                torch.from_numpy(sv).unsqueeze(0).to(DEVICE_CPU),
                num_steps=4
            )
            a = a.squeeze(0).cpu().numpy()

        a[3:] = np.clip(a[3:], -0.5, 0.5)
        sim.step(a)
    return {'success': False, 'steps': MAX_STEPS, 'final_dist': dist, 'history': history}


# ── Run evaluation ─────────────────────────────────────────────────────────────
print(f"\n=== Phase 218: Phase 196 VLA vs P-ctrl ({N_GOALS} goals, seed={SEED}) ===\n")

results = {'pctrl': [], 'vla': []}
t_start = time.time()

for g_i, goal in enumerate(GOALS):
    print(f"Goal {g_i+1}/{N_GOALS}: goal_xy={goal.round(3)}")

    # P-controller
    sim_p = LeKiWiSimURDF(); sim_p.reset()
    r_p = run_pcontroller(sim_p, goal)
    results['pctrl'].append(r_p)
    p_str = "✓" if r_p['success'] else f"✗ dist={r_p['final_dist']:.3f}"

    # VLA
    sim_v = LeKiWiSimURDF(); sim_v.reset()
    r_v = run_vla(sim_v, goal, policy)
    results['vla'].append(r_v)
    v_str = "✓" if r_v['success'] else f"✗ dist={r_v['final_dist']:.3f}"

    print(f"  P-ctrl: {p_str} (steps={r_p['steps']}) | VLA: {v_str} (steps={r_v['steps']})")

elapsed = time.time() - t_start

# ── Summary ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for mode, label in [('pctrl', 'P-ctrl (CJ kP=2.0)'), ('vla', 'VLA-e14')]:
    res = results[mode]
    n_success = sum(r['success'] for r in res)
    sr = 100 * n_success / N_GOALS
    avg_steps = np.mean([r['steps'] for r in res])
    avg_final_dist = np.mean([r['final_dist'] for r in res])
    print(f"\n{label}:")
    print(f"  Success Rate: {n_success}/{N_GOALS} = {sr:.0f}%")
    print(f"  Avg steps (success): {avg_steps:.1f}")
    print(f"  Avg final dist: {avg_final_dist:.4f}m")
    for i, r in enumerate(res):
        flag = "✓" if r['success'] else f"✗ {r['final_dist']:.3f}m"
        print(f"    Goal {i+1}: {flag} (steps={r['steps']})")

print(f"\nTotal time: {elapsed:.1f}s")
print("\n" + "="*60)

# ── Verdict ─────────────────────────────────────────────────────────────────────
p_sr = 100 * sum(r['success'] for r in results['pctrl']) / N_GOALS
v_sr = 100 * sum(r['success'] for r in results['vla']) / N_GOALS

if v_sr >= p_sr:
    print("✅ VLA matches or beats P-controller oracle!")
elif v_sr >= 20:
    print("⚠️  VLA is functional but underperforms oracle by {:.0f}%".format(p_sr - v_sr))
else:
    print("❌ VLA significantly underperforms — architecture issue")

# Save results
import json
out = {
    'phase': 218,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seed': SEED,
    'n_goals': N_GOALS,
    'success_radius': SUCCESS_R,
    'max_steps': MAX_STEPS,
    'checkpoint': CHECKPOINT,
    'policy_epoch': int(ckpt.get('epoch', -1)),
    'policy_loss': float(ckpt.get('loss', -1)),
    'results': {
        'pctrl': [{'success': r['success'], 'steps': r['steps'], 'final_dist': float(r['final_dist']),
                   'goal': [float(x) for x in GOALS[i]]}
                  for i, r in enumerate(results['pctrl'])],
        'vla': [{'success': r['success'], 'steps': r['steps'], 'final_dist': float(r['final_dist']),
                 'goal': [float(x) for x in GOALS[i]]}
                for i, r in enumerate(results['vla'])],
    },
    'summary': {
        'pctrl_sr': float(p_sr),
        'vla_sr': float(v_sr),
        'elapsed_s': float(elapsed),
    }
}

with open('results/phase218_eval.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to results/phase218_eval.json")