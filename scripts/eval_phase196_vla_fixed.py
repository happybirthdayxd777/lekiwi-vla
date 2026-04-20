#!/usr/bin/env python3
"""
Phase 235: Fixed evaluation of Phase 196 VLA policy.

BUGS FIXED in eval_phase196_vla.py:
  1. final_dist used qpos[:2] instead of xpos[base_id,:2] → off by ~1-2cm
  2. success_radius=0.3m instead of 0.1m → artificially inflated SR
  3. No early termination → counts overshoot+return trajectories as failures

FIXES APPLIED:
  - Uses xpos[base_body_id, :2] for distance
  - success_radius=0.1m matching Phase 227 standard
  - Early termination when dist < success_radius
  - 50 goals matching standard eval protocol
"""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS
from PIL import Image

ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_14.pt', map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(raw_img)
    img = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)

def build_state(sim, goal):
    arm_pos = np.array([sim.data.qpos[sim.model.joint(n).qposadr[0]] for n in ARM_JOINTS])
    wheel_vel = np.array([sim.data.qvel[sim.model.joint(n).dofadr[0]] for n in WHEEL_JOINTS])
    goal_norm = np.clip(goal / 0.4, -1, 1)
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

print("=" * 60)
print("Phase 235: Phase 196 VLA Fixed Evaluation")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Checkpoint: epoch_14.pt, loss={ckpt['loss']:.4f}")
print(f"Fixes: xpos-based dist, sr=0.1m, early termination, 50 goals")
print()

n_goals = 50
max_steps = 200
success_radius = 0.1  # FIXED: was 0.3
successes = 0
dists = []
steps_list = []

np.random.seed(42)

t0 = time.time()
for g_i in range(n_goals):
    goal = np.random.uniform(-0.3, 0.4, 2)
    sim = LeKiWiSimURDF()
    sim.reset()
    base_body_id = sim.model.body('base').id

    reached = False
    for step in range(max_steps):
        state = build_state(sim, goal)
        image = sim.render()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            img = preprocess_image(image).unsqueeze(0).to(DEVICE)
            action = policy.infer(img, s, num_steps=4).cpu().numpy()[0]
        action = np.clip(action, -0.5, 0.5)
        sim.step(action)

        # FIX 3: Early termination
        dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        if dist < success_radius:
            reached = True
            steps_list.append(step + 1)
            break

    # FIX 1 & 2: Use xpos instead of qpos, sr=0.1 instead of 0.3
    final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
    successes += int(reached)
    dists.append(final_dist)

    mark = '✓' if reached else '✗'
    if g_i < 10 or g_i % 10 == 9:
        print(f"  [{g_i+1:2d}/50] goal=({goal[0]:+.2f},{goal[1]:+.2f}) "
              f"dist={final_dist:.3f}m {mark}")

elapsed = time.time() - t0
sr = successes / n_goals * 100
mean_dist = np.mean(dists)
median_dist = np.median(dists)
mean_steps = np.mean(steps_list) if steps_list else max_steps

print()
print(f"VLA Phase196 (epoch_14, FIXED): {successes}/{n_goals} = {sr:.1f}% SR")
print(f"Mean dist: {mean_dist:.3f}m, Median: {median_dist:.3f}m")
print(f"Mean steps (successes): {mean_steps:.1f}")
print(f"Total time: {elapsed:.0f}s")
print()
print("Comparison (all with sr=0.1m, early termination):")
print(f"  P-controller CJ:  94.0% SR  (from Phase234)")
print(f"  VLA Phase196 FIXED: {sr:.1f}% SR  ← THIS RUN")
