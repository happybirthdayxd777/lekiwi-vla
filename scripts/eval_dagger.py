#!/usr/bin/env python3
"""
Phase 252: Quick eval of DAgger policy vs Phase227 VLA vs P-controller
"""
import sys, os, json, time
from pathlib import Path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS, _CONTACT_JACOBIAN_PSEUDO_INV
from scripts.train_phase227 import GoalConditionedPolicy, DEVICE

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img):
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return arr.transpose(2, 0, 1)

def build_state(sim, goal):
    arm_pos = np.array([sim.data.qpos[sim.model.joint(n).qposadr[0]] for n in ARM_JOINTS])
    wheel_vel = np.array([sim.data.qvel[sim.model.joint(n).dofadr[0]] for n in WHEEL_JOINTS])
    goal_norm = np.clip(goal / 0.4, -1, 1)
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

def eval_policy(policy, n_goals, max_steps, success_radius, seed, policy_name, verbose=True):
    np.random.seed(seed)
    successes = 0
    episodes = []

    for ep_i in range(n_goals):
        goal = np.array([
            np.random.uniform(-0.40, 0.40),
            np.random.uniform(-0.34, 0.34)
        ])
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id

        arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])

        # Warmup for first episode
        if ep_i == 0:
            _ = sim.render()
        sim.step(np.zeros(9))

        steps = 0
        for step in range(max_steps):
            base_xy = sim.data.xpos[base_body_id, :2]
            dist = np.linalg.norm(goal - base_xy)

            if dist < success_radius:
                break

            img = sim.render().astype(np.uint8)
            state = build_state(sim, goal)
            img_t = torch.from_numpy(preprocess_image(img)).unsqueeze(0).float().to(DEVICE)
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)

            if policy is None:
                # P-controller
                err = goal - base_xy
                wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ (2.0 * err), -0.5, 0.5)
                action = np.concatenate([arm, wheel_speeds]).astype(np.float32)
            else:
                with torch.no_grad():
                    action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy()[0]
                action = np.clip(action, -0.5, 0.5)

            sim.step(action)
            steps += 1

        final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        success = final_dist < success_radius
        successes += int(success)
        episodes.append({'goal': goal.tolist(), 'final_dist': float(final_dist),
                         'steps': steps, 'success': success})

        if verbose:
            status = '✅' if success else '❌'
            print(f"  {policy_name} ep{ep_i+1}: goal={goal.round(2)}, dist={final_dist:.3f}m, steps={steps} {status}")

    sr = successes / n_goals
    mean_steps = np.mean([e['steps'] for e in episodes])
    mean_dist = np.mean([e['final_dist'] for e in episodes])
    return {'policy': policy_name, 'success_rate': sr, 'successes': successes,
            'n_episodes': n_goals, 'success_radius': success_radius,
            'mean_final_dist': float(mean_dist), 'mean_steps': float(mean_steps),
            'episodes': episodes}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_goals', type=int, default=15)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--success_radius', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"Phase 246: DAgger Policy Evaluation")
    print(f"=" * 60)
    print(f"Config: {args.n_goals} goals, {args.max_steps} steps, sr={args.success_radius}m, seed={args.seed}")
    print()

    results = {}
    total_start = time.time()

    # P-controller baseline
    print("[P-controller baseline]")
    r = eval_policy(None, args.n_goals, args.max_steps, args.success_radius, args.seed, "P-ctrl CJ kP=2.0")
    results['p_controller'] = r
    print(f"  → {r['successes']}/{args.n_goals} = {r['success_rate']*100:.0f}% SR, mean_dist={r['mean_final_dist']:.3f}m")
    print()

    # Phase227 VLA (best checkpoint)
    print("[Phase227 VLA (best_policy.pt)]")
    p227 = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE).to(DEVICE)
    ckpt = torch.load('results/phase227_contact_jacobian_train/best_policy.pt', map_location=DEVICE, weights_only=False)
    p227.load_state_dict(ckpt['policy_state_dict'], strict=False)
    p227.eval()
    r = eval_policy(p227, args.n_goals, args.max_steps, args.success_radius, args.seed, "VLA Phase227")
    results['vla_phase227'] = r
    print(f"  → {r['successes']}/{args.n_goals} = {r['success_rate']*100:.0f}% SR, mean_dist={r['mean_final_dist']:.3f}m")
    print()

    # DAgger-246 policy (for comparison)
    if os.path.exists('results/dagger_phase246_train/best_policy.pt'):
        print("[DAgger-246 policy (best_policy.pt)]")
        p_dagger246 = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE).to(DEVICE)
        ckpt = torch.load('results/dagger_phase246_train/best_policy.pt', map_location=DEVICE, weights_only=False)
        p_dagger246.load_state_dict(ckpt['policy_state_dict'], strict=False)
        p_dagger246.eval()
        r = eval_policy(p_dagger246, args.n_goals, args.max_steps, args.success_radius, args.seed, "VLA DAgger-246")
        results['vla_dagger246'] = r
        print(f"  → {r['successes']}/{args.n_goals} = {r['success_rate']*100:.0f}% SR, mean_dist={r['mean_final_dist']:.3f}m")
        print()

    # DAgger-252 policy (new run)
    if os.path.exists('results/dagger_phase252_train/best_policy.pt'):
        print("[DAgger-252 policy (best_policy.pt)]")
        p_dagger252 = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE).to(DEVICE)
        ckpt = torch.load('results/dagger_phase252_train/best_policy.pt', map_location=DEVICE, weights_only=False)
        p_dagger252.load_state_dict(ckpt['policy_state_dict'], strict=False)
        p_dagger252.eval()
        r = eval_policy(p_dagger252, args.n_goals, args.max_steps, args.success_radius, args.seed, "VLA DAgger-252")
        results['vla_dagger252'] = r
        print(f"  → {r['successes']}/{args.n_goals} = {r['success_rate']*100:.0f}% SR, mean_dist={r['mean_final_dist']:.3f}m")
        print()

    total_time = time.time() - total_start

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {r['policy']}: {r['successes']}/{args.n_goals} = {r['success_rate']*100:.0f}% SR")
    print(f"\nTotal time: {total_time/60:.1f} min")

    # Save results
    out_dir = Path('results/dagger_phase252_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'eval_results.json'

    import json as json_module
    from pathlib import Path
    def make_json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(x) for x in obj]
        return obj

    with open(out_file, 'w') as f:
        json_module.dump(make_json_safe(results), f, indent=2)
    print(f"\n✅ Saved: {out_file}")

if __name__ == "__main__":
    main()