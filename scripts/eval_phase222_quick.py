#!/usr/bin/env python3
"""
Phase 222: Quick 20-goal eval of Phase196 VLA (FIXED wheel velocities)
This gives preliminary statistical results while the 50-goal version runs.
"""
import sys, os, numpy as np, torch, time, json
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV, ARM_JOINTS
from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT = 'results/phase196_contact_jacobian_train/epoch_14.pt'
N_GOALS = 20
SEED = 42
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

# ── Run ────────────────────────────────────────────────────────────────────────
def run_vla(sim, goal):
    base_id = sim.model.body('base').id
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        if np.linalg.norm(goal - base_xy) < SUCCESS_R:
            return True, step, np.linalg.norm(goal - base_xy)
        # CORRECT: wheel_vel = qvel[6:9], NOT qvel[9:12]
        wheel_vel = sim.data.qvel[6:9].copy()
        goal_norm = np.clip(goal / 0.525, -1, 1)
        state_vec = np.concatenate([ARM_DEFAULT, wheel_vel, goal_norm]).astype(np.float32)
        img = preprocess(sim.render().astype(np.uint8))
        with torch.no_grad():
            action = policy.infer(img.unsqueeze(0).to('cpu'),
                torch.from_numpy(state_vec).unsqueeze(0).to('cpu'), num_steps=4
            ).squeeze(0).cpu().numpy()
        action[3:] = np.clip(action[3:], -0.5, 0.5)
        sim.step(action)
    return False, MAX_STEPS, np.linalg.norm(sim.data.xpos[base_id, :2] - goal)

def run_pctrl(sim, goal):
    base_id = sim.model.body('base').id
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_id, :2]
        if np.linalg.norm(goal - base_xy) < SUCCESS_R:
            return True, step, np.linalg.norm(goal - base_xy)
        v = 2.0 * (goal - base_xy)
        ws = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v, -0.5, 0.5)
        sim.step(np.concatenate([ARM_DEFAULT, ws]))
    return False, MAX_STEPS, np.linalg.norm(sim.data.xpos[base_id, :2] - goal)

print(f"\n=== Phase 222: Phase196 VLA 20-goal Eval (FIXED wheel vel) ===")
vla_ok, pc_ok = 0, 0
t_start = time.time()
for g_i, goal in enumerate(GOALS):
    sv = LeKiWiSimURDF(); sv.reset()
    ok, st, dist = run_vla(sv, goal)
    vla_ok += ok
    sp = LeKiWiSimURDF(); sp.reset()
    ok2, st2, dist2 = run_pctrl(sp, goal)
    pc_ok += ok2
    print(f"  Goal {g_i+1:2d}: VLA {'✓'+str(st) if ok else '✗'+str(dist)[:6]} | P-ctrl {'✓'+str(st2) if ok2 else '✗'+str(dist2)[:6]} | {goal.round(2)}")
elapsed = time.time() - t_start
print(f"\n{'='*60}\nVLA SR: {vla_ok}/{N_GOALS} = {100*vla_ok/N_GOALS:.0f}%\nP-ctrl SR: {pc_ok}/{N_GOALS} = {100*pc_ok/N_GOALS:.0f}%\nTime: {elapsed:.1f}s ({elapsed/60:.1f}min)\n{'='*60}")

out = {'phase': '222_quick', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
       'n_goals': N_GOALS, 'seed': SEED, 'success_radius': SUCCESS_R,
       'vla_sr': vla_ok/N_GOALS, 'pctrl_sr': pc_ok/N_GOALS,
       'elapsed': elapsed, 'goals': [g.tolist() for g in GOALS]}
with open('results/phase222_quick.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved: results/phase222_quick.json")
