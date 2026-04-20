#!/usr/bin/env python3
"""Phase 226: Re-evaluate Phase196 VLA at success_r=0.15m — fair comparison.

Previous evals used success_r=0.10m but the VLA was trained with success_r=0.15m.
This gives a more fair comparison.
"""
import sys, os, torch, numpy as np, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS
from PIL import Image

SUCCESS_RADIUS = 0.15  # fair comparison: same as training
N_EPISODES = 50
MAX_STEPS = 200
SEED = 42

ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_14.pt', map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img):
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

print("=" * 65)
print(f"Phase 226: VLA eval at success_r={SUCCESS_RADIUS}m (fair comparison)")
print(f"Checkpoint: phase196_contact_jacobian_train/epoch_14.pt")
print(f"Episodes: {N_EPISODES}, max_steps: {MAX_STEPS}, seed: {SEED}")
print("=" * 65)

np.random.seed(SEED)
goals = [np.random.uniform(-0.35, 0.40, 2) for _ in range(N_EPISODES)]

successes = 0
dists = []
steps_list = []
episodes_data = []
t0 = time.time()

for i, goal in enumerate(goals):
    sim = LeKiWiSimURDF()
    sim.reset()
    success = False
    ep_steps = 0
    for step in range(MAX_STEPS):
        state = build_state(sim, goal)
        image = sim.render()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            img = preprocess_image(image).unsqueeze(0).to(DEVICE)
            action = policy.infer(img, s, num_steps=4).cpu().numpy()[0]
        action = np.clip(action, -0.5, 0.5)
        sim.step(action)
        ep_steps = step + 1
        dist_now = np.linalg.norm(sim.data.qpos[:2] - goal)
        if dist_now < SUCCESS_RADIUS:
            success = True
            break
    
    final_dist = np.linalg.norm(sim.data.qpos[:2] - goal)
    successes += int(success)
    dists.append(final_dist)
    steps_list.append(ep_steps)
    episodes_data.append({
        "goal": goal.tolist(),
        "final_dist": float(final_dist),
        "steps": ep_steps,
        "success": success
    })
    
    elapsed = time.time() - t0
    eta = elapsed / (i+1) * (N_EPISODES - i - 1)
    print(f"  [{i+1:2d}/{N_EPISODES}] goal=({goal[0]:.2f},{goal[1]:.2f}) "
          f"dist={final_dist:.3f}m steps={ep_steps} "
          f"{'✓ SUCCESS' if success else '✗ FAIL'} | ETA {eta:.0f}s")

total_elapsed = time.time() - t0
print()
print("=" * 65)
print(f"PHASE 226 RESULTS (success_r={SUCCESS_RADIUS}m, {N_EPISODES} goals)")
print(f"VLA phase196_e14:  {successes}/{N_EPISODES} = {successes/N_EPISODES*100:.1f}% SR")
print(f"Mean final dist:   {np.mean(dists):.3f}m")
print(f"Median final dist: {np.median(dists):.3f}m")
print(f"Mean steps:        {np.mean(steps_list):.1f}")
print(f"Elapsed:           {total_elapsed/60:.1f} min")
print("=" * 65)

results = {
    "phase": 226,
    "checkpoint": "phase196_contact_jacobian_train/epoch_14.pt",
    "success_radius": SUCCESS_RADIUS,
    "n_episodes": N_EPISODES,
    "max_steps": MAX_STEPS,
    "seed": SEED,
    "success_rate": successes / N_EPISODES,
    "successes": successes,
    "mean_final_dist": float(np.mean(dists)),
    "median_final_dist": float(np.median(dists)),
    "mean_steps": float(np.mean(steps_list)),
    "elapsed_sec": total_elapsed,
    "episodes": episodes_data
}

out_path = f"results/phase226_eval_sr015.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {out_path}")
