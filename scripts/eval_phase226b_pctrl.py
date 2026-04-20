#!/usr/bin/env python3
"""Phase 226b: P-controller eval at success_r=0.15m for fair comparison."""
import sys, os, numpy as np, json, time
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
os.chdir(os.path.expanduser("~/hermes_research/lekiwi_vla"))

from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV

SUCCESS_RADIUS = 0.15
N_EPISODES = 50
MAX_STEPS = 200
SEED = 42
KP = 2.0
ARM_NEUTRAL = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

print("=" * 65)
print(f"Phase 226b: P-controller eval at success_r={SUCCESS_RADIUS}m")
print(f"kP={KP}, Episodes: {N_EPISODES}, seed: {SEED}")
print("=" * 65)

np.random.seed(SEED)
goals = [np.random.uniform([-0.35, -0.30], [0.40, 0.30]) for _ in range(N_EPISODES)]

successes = 0
dists = []
steps_list = []
t0 = time.time()

for i, goal in enumerate(goals):
    sim = LeKiWiSimURDF()
    sim.reset()
    success = False
    ep_steps = MAX_STEPS
    
    base_body_id = sim.model.body("base").id
    
    for step in range(MAX_STEPS):
        base_xy = sim.data.xpos[base_body_id, :2]
        err = goal - base_xy
        dist = np.linalg.norm(err)
        if dist < SUCCESS_RADIUS:
            success = True
            ep_steps = step + 1
            break
        v_desired = KP * err
        wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
        action = np.concatenate([ARM_NEUTRAL, wheel_speeds])
        sim.step(action)
    
    final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
    successes += int(success)
    dists.append(final_dist)
    steps_list.append(ep_steps)
    elapsed = time.time() - t0
    eta = elapsed / (i+1) * (N_EPISODES - i - 1)
    print(f"  [{i+1:2d}/{N_EPISODES}] goal=({goal[0]:.2f},{goal[1]:.2f}) "
          f"dist={final_dist:.3f}m steps={ep_steps} "
          f"{'✓ SUCCESS' if success else '✗ FAIL'} | ETA {eta:.0f}s")

total_elapsed = time.time() - t0
print()
print("=" * 65)
print(f"P-ctrl CJ kP={KP}: {successes}/{N_EPISODES} = {successes/N_EPISODES*100:.1f}% SR (sr={SUCCESS_RADIUS}m)")
print(f"Mean final dist: {np.mean(dists):.3f}m, mean steps: {np.mean(steps_list):.1f}")
print(f"Elapsed: {total_elapsed:.1f}s")
print("=" * 65)

out = {
    "phase": "226b", "policy": f"pctrl_cj_kP{KP}",
    "success_radius": SUCCESS_RADIUS,
    "n_episodes": N_EPISODES, "seed": SEED,
    "success_rate": successes/N_EPISODES,
    "successes": successes,
    "mean_final_dist": float(np.mean(dists)),
    "mean_steps": float(np.mean(steps_list)),
    "elapsed_sec": total_elapsed
}
with open("results/phase226b_pctrl_sr015.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/phase226b_pctrl_sr015.json")
