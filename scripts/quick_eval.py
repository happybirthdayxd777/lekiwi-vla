#!/usr/bin/env python3
"""Quick eval: Run the FIXED reach_target eval on trained policy."""
import sys, os, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
os.chdir('.')

from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.improve_reward import TaskEvaluator
from scripts.train_task_oriented import CLIPFlowMatchingPolicy

print("Loading policy...")
ckpt_path = Path('results/task_oriented_goaldirected/checkpoint_epoch_50.pt')
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
policy.load_state_dict(ckpt.get('policy_state_dict', ckpt), strict=False)
policy.to('cpu').eval()
print("Policy ready")

sim = LeKiWiSimURDF()
evaluator = TaskEvaluator(sim, policy=policy, device='cpu')

results = []
for ep in range(5):
    s, st, d = evaluator.reach_target(
        target=(0.5, 0.0), start=(0.0, 0.0), threshold=0.1, max_steps=300
    )
    results.append({'success': s, 'steps': st, 'dist': d})
    print(f"  Episode {ep+1}: success={s}, dist={d:.3f}m, steps={st}")

sr = sum(r['success'] for r in results) / len(results)
md = np.mean([r['dist'] for r in results])
print(f"\n=== CLIP-FM FIXED EVAL ===")
print(f"Success rate: {sr*100:.0f}%")
print(f"Mean final distance: {md:.3f}m (threshold=0.1m)")
print(f"(Before fix: 0% success, 0.877m mean_dist from broken station-keeping eval)")

import json
Path("data/fixed_eval_results.json").parent.mkdir(parents=True, exist_ok=True)
with open("data/fixed_eval_results.json", "w") as f:
    json.dump({"policy": "clip_fm", "checkpoint": str(ckpt_path), "success_rate": float(sr),
              "mean_dist": float(md), "results": results}, f, indent=2)
print("Saved to data/fixed_eval_results.json")
