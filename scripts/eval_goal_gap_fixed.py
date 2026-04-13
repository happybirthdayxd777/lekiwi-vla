#!/usr/bin/env python3
"""
Phase 27 — eval_goal_gap_fixed.py
=================================
Fixed evaluation script that uses CORRECT state indexing.
Tests goal_aware policy on LeKiWiSim (primitive) backend.

Key fixes:
1. State construction: use obs['arm_positions'] + obs['wheel_velocities']
   (NOT qpos[0:6] + qvel[0:3] which gives base pose/vel, not arm/wheel)
2. Use LeKiWiSim (primitive) backend for stable physics
3. Use correct goal-aware state (11D = arm_pos 6 + wheel_vel 3 + goal_xy 2)

Usage:
  python3 scripts/eval_goal_gap_fixed.py --episodes 5 --max-steps 200
"""
import sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_lekiwi import LeKiwiSim
from scripts.train_task_oriented import CLIPFlowMatchingPolicy


def resize_for_clip(img_pil):
    """Resize PIL Image to 224x224 for CLIP ViT-B/32."""
    pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    img_chw = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_chw).unsqueeze(0).cpu()


def make_state(obs, goal_x, goal_y):
    """Build 11D goal-aware state from obs dict."""
    arm_pos = obs['arm_positions']      # 6D
    wheel_vel = obs['wheel_velocities']  # 3D
    goal_norm = np.array([goal_x / 1.0, goal_y / 1.0])  # 2D
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)


def evaluate_goal(sim, policy, goal_x, goal_y, max_steps=200, threshold=0.15):
    """Evaluate one goal. Returns (success, final_dist, steps)."""
    sim_local = LeKiwiSim()
    sim_local.set_target(np.array([goal_x, goal_y, 0.02]))
    
    # Warmup steps for renderer
    for _ in range(50):
        sim_local.step(np.zeros(9))
    
    img_pil = sim_local.render()
    img_t = resize_for_clip(img_pil)
    
    for step in range(max_steps):
        obs = sim_local._obs()
        state = make_state(obs, goal_x, goal_y)
        state_t = torch.from_numpy(state).unsqueeze(0).cpu()
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()
        
        sim_local.step(action)
        
        robot_pos = sim_local.data.qpos[:2]
        dist = np.linalg.norm(robot_pos - np.array([goal_x, goal_y]))
        
        if dist < threshold:
            return True, float(dist), step + 1
        
        if step < max_steps - 1:
            img_pil = sim_local.render()
            img_t = resize_for_clip(img_pil)
    
    return False, float(dist), max_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default='results/goal_aware_50ep/final_policy.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=0.15)
    args = parser.parse_args()

    # Load policy
    policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
    ckpt = torch.load(args.policy, map_location='cpu', weights_only=False)
    policy.load_state_dict(ckpt.get('policy_state_dict', ckpt), strict=False)
    policy.to('cpu').eval()
    print(f"Loaded: {Path(args.policy).name} (state_dim=11, goal-aware)")
    print(f"Backend: LeKiWiSim (primitive)")
    
    # Goals: 3 ID + 2 OOD
    id_goals = [(0.2, 0.0), (-0.2, 0.0), (0.3, 0.3)]
    ood_goals = [(0.5, 0.0), (0.5, 0.5)]
    
    print(f"\n=== IN-DISTRIBUTION (ID) ===")
    id_results = []
    for gx, gy in id_goals:
        ok, d, steps = evaluate_goal(None, policy, gx, gy, args.max_steps, args.threshold)
        id_results.append((gx, gy, ok, d, steps))
        print(f"  ({gx:+.1f},{gy:+.1f}): success={ok}, dist={d:.3f}m, steps={steps}")
    
    id_sr = sum(r[2] for r in id_results) / len(id_results)
    id_md = np.mean([r[3] for r in id_results])
    print(f"  ID: SR={id_sr*100:.0f}%, mean_dist={id_md:.3f}m")
    
    print(f"\n=== OUT-OF-DISTRIBUTION (OOD) ===")
    ood_results = []
    for gx, gy in ood_goals:
        ok, d, steps = evaluate_goal(None, policy, gx, gy, args.max_steps, args.threshold)
        ood_results.append((gx, gy, ok, d, steps))
        print(f"  ({gx:+.1f},{gy:+.1f}): success={ok}, dist={d:.3f}m, steps={steps}")
    
    ood_sr = sum(r[2] for r in ood_results) / len(ood_results)
    ood_md = np.mean([r[3] for r in ood_results])
    print(f"  OOD: SR={ood_sr*100:.0f}%, mean_dist={ood_md:.3f}m")
    
    print(f"\n=== SUMMARY ===")
    print(f"  ID  SR={id_sr*100:.0f}%, mean_dist={id_md:.3f}m")
    print(f"  OOD SR={ood_sr*100:.0f}%, mean_dist={ood_md:.3f}m")
    
    gap = id_md - ood_md
    print(f"  ID-OOD gap: {gap:.3f}m")
    
    if id_sr > ood_sr:
        print("  → OOD worse than ID: distributional gap confirmed")
    elif id_sr == ood_sr == 0:
        print("  → Both 0%: need more training data or architecture fix")
    else:
        print("  → No clear distributional gap")
    
    # Save results
    import json
    result = {
        "phase": 27,
        "policy": str(args.policy),
        "backend": "LeKiWiSim (primitive)",
        "state_fix": "CORRECT: obs['arm_positions'] + obs['wheel_velocities'] + goal_xy",
        "id_results": [{"goal": (r[0],r[1]), "success": r[2], "dist": float(r[3]), "steps": r[4]} for r in id_results],
        "ood_results": [{"goal": (r[0],r[1]), "success": r[2], "dist": float(r[3]), "steps": r[4]} for r in ood_results],
        "id_sr": float(id_sr), "id_mean_dist": float(id_md),
        "ood_sr": float(ood_sr), "ood_mean_dist": float(ood_md),
    }
    out_path = Path("data/goal_gap_eval_fixed_phase27.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
