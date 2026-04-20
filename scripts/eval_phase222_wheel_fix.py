#!/usr/bin/env python3
"""
Phase 222: 50-goal statistical evaluation of Phase196 VLA (epoch_14)
FIXED wheel velocity bug — was reading qvel[9:12]=ARM velocities as "wheel_vel"

Bug history:
- collect_phase196_clean.py line 111: wheel_vel = sim.data.qvel[9:12]  ← ARM velocities j0,j1,j2
- eval_phase216.py, eval_phase218.py, eval_phase218b.py: same bug
- CORRECT: sim.data.qvel[sim._jvel_idx["w1"]] etc. or qvel[6:9]

This eval uses CORRECT wheel velocity indexing.
"""
import sys, os, numpy as np, torch, time, json
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV, ARM_JOINTS, WHEEL_JOINTS
from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = 'results/phase196_contact_jacobian_train/epoch_14.pt'
N_GOALS = 50
SEED = 42
SUCCESS_R = 0.10   # meters — goal reached if base within this distance
MAX_STEPS = 200

# ── Preprocessing ──────────────────────────────────────────────────────────────
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(raw_img):
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))

# Fixed arm default (same as training/collect)
ARM_DEFAULT = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

# ── Load Policy ────────────────────────────────────────────────────────────────
print(f"Loading policy from {CHECKPOINT}...")
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
policy.to('cpu').eval()
print(f"  epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?')}")

# ── Goals ─────────────────────────────────────────────────────────────────────
np.random.seed(SEED)
GOALS = [
    np.array([np.random.uniform(-0.35, 0.40), np.random.uniform(-0.25, 0.25)])
    for _ in range(N_GOALS)
]

# ── Run Policy ────────────────────────────────────────────────────────────────
def run_vla_policy(sim, goal, policy):
    """Run VLA policy with CORRECT wheel velocity indexing."""
    base_id = sim.model.body('base').id
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(goal - base_xy)
        if dist < SUCCESS_R:
            return {'success': True, 'steps': step, 'final_dist': dist}

        # CORRECT wheel velocities: qvel[6:9] = w1, w2, w3
        # (NOT qvel[9:12] which are ARM velocities j0, j1, j2)
        wheel_vel = sim.data.qvel[6:9].copy()

        # Build state: arm_pos(6) + wheel_vel(3) + goal_norm(2) = 11D
        goal_norm = np.clip(goal / 0.525, -1, 1)
        state_vec = np.concatenate([ARM_DEFAULT, wheel_vel, goal_norm]).astype(np.float32)

        # Render and infer
        img = preprocess(sim.render().astype(np.uint8))
        with torch.no_grad():
            action = policy.infer(
                img.unsqueeze(0).to('cpu'),
                torch.from_numpy(state_vec).unsqueeze(0).to('cpu'),
                num_steps=4
            ).squeeze(0).cpu().numpy()

        # Clip wheel actions to [-0.5, 0.5] (MuJoCo stability)
        action[3:] = np.clip(action[3:], -0.5, 0.5)
        sim.step(action)

    return {'success': False, 'steps': MAX_STEPS, 'final_dist': dist}


def run_pctrl(sim, goal):
    """Contact-Jacobian P-controller baseline."""
    base_id = sim.model.body('base').id
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        dist = np.linalg.norm(goal - base_xy)
        if dist < SUCCESS_R:
            return {'success': True, 'steps': step, 'final_dist': dist}

        # CORRECT wheel velocities for state (used for logging)
        wheel_vel = sim.data.qvel[6:9].copy()

        # P-controller
        v = 2.0 * (goal - base_xy)
        wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v, -0.5, 0.5)
        action = np.concatenate([ARM_DEFAULT, wheel_speeds])
        sim.step(action)

    return {'success': False, 'steps': MAX_STEPS, 'final_dist': dist}


# ── Execute ────────────────────────────────────────────────────────────────────
print(f"\n=== Phase 222: Phase196 VLA 50-goal Eval (FIXED wheel vel) ===")
print(f"Success radius: {SUCCESS_R}m, Max steps: {MAX_STEPS}, Seed: {SEED}")

vla_results = []
pctrl_results = []
t_start = time.time()

for g_i, goal in enumerate(GOALS):
    # VLA
    sim_vla = LeKiWiSimURDF()
    sim_vla.reset()
    r_vla = run_vla_policy(sim_vla, goal, policy)
    vla_results.append({**r_vla, 'goal': goal.tolist()})

    # P-ctrl (oracle baseline)
    sim_pc = LeKiWiSimURDF()
    sim_pc.reset()
    r_pc = run_pctrl(sim_pc, goal)
    pctrl_results.append({**r_pc, 'goal': goal.tolist()})

    flag_vla = "✓" if r_vla['success'] else f"✗ {r_vla['final_dist']:.3f}m"
    flag_pc  = "✓" if r_pc['success']  else f"✗ {r_pc['final_dist']:.3f}m"
    print(f"  Goal {g_i+1:2d}/{N_GOALS}: VLA {flag_vla} ({r_vla['steps']}st) | "
          f"P-ctrl {flag_pc} ({r_pc['steps']}st) | goal={goal.round(2)}")

elapsed = time.time() - t_start

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)

for label, results in [('VLA (phase196_e14)', vla_results), ('P-ctrl (CJ kP=2.0)', pctrl_results)]:
    n_ok = sum(r['success'] for r in results)
    sr = 100 * n_ok / N_GOALS
    avg_steps = np.mean([r['steps'] for r in results])
    avg_dist = np.mean([r['final_dist'] for r in results])
    print(f"\n{label}:")
    print(f"  SR: {n_ok}/{N_GOALS} = {sr:.0f}%")
    print(f"  Avg steps: {avg_steps:.1f}, Avg final dist: {avg_dist:.4f}m")

print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f}min)")

# ── Save Results ──────────────────────────────────────────────────────────────
output = {
    'phase': '222',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'seed': SEED,
    'n_goals': N_GOALS,
    'success_radius': SUCCESS_R,
    'max_steps': MAX_STEPS,
    'checkpoint': CHECKPOINT,
    'bug_fixed': 'wheel_vel was qvel[9:12]=ARM_vel, corrected to qvel[6:9]=wheel_vel',
    'vla_results': vla_results,
    'pctrl_results': pctrl_results,
    'elapsed_seconds': elapsed,
}
with open('results/phase222_eval.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: results/phase222_eval.json")
