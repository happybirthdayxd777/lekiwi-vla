#!/usr/bin/env python3
"""
Phase 227: Evaluate Phase 227 VLA on 50 random goals
====================================================

Compares:
  - Phase 227 VLA (Q2-extended, epoch_XX.pt)
  - Phase 196 VLA (original, epoch_14.pt)
  - P-controller CJ kP=2.0 (oracle baseline)

Focus: Q2 goals with large gy > 0.24m (the previously OOD region)

Usage:
  python3 scripts/eval_phase227.py --vla results/phase227_contact_jacobian_train/epoch_30.pt --n_episodes 50
"""
import sys, os, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS, _CONTACT_JACOBIAN_PSEUDO_INV
from scripts.train_phase227 import GoalConditionedPolicy, DEVICE
from PIL import Image

np.random.seed(None)

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

def run_evaluation(policy, n_episodes, max_steps, success_radius, seed, policy_name, verbose=True):
    """Run VLA policy evaluation."""
    np.random.seed(seed)
    successes = 0
    episodes = []

    for ep_i in range(n_episodes):
        goal = np.array([
            np.random.uniform(-0.35, 0.40),
            np.random.uniform(-0.28, 0.28)
        ])
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id

        # Warmup step to fix render-black bug (LeKiWiSimURDF returns black at t=0.002s)
        if ep_i == 0:
            _ = sim.render()  # First render is black, warmup
        sim.step(np.zeros(9))  # Physics warmup step

        for step in range(max_steps):
            state = build_state(sim, goal)
            image = sim.render()
            img_tensor = torch.from_numpy(preprocess_image(image)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = policy.infer(img_tensor,
                                     torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                                     num_steps=4).cpu().numpy()[0]
            sim.step(np.clip(action, -0.5, 0.5))

            # Early termination when goal reached
            dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
            if dist < success_radius:
                break

        final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        success = bool(final_dist < success_radius)
        successes += int(success)
        episodes.append({
            'goal': list(goal),
            'final_dist': float(final_dist),
            'steps': step + 1,
            'success': success,
        })

        if verbose and (ep_i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (ep_i + 1) * (n_episodes - ep_i - 1)
            print(f"  [{ep_i+1}/{n_episodes}] goal=({goal[0]:.2f},{goal[1]:.2f}) dist={final_dist:.3f}m {'✓' if success else '✗'} | ETA {eta:.0f}s")

    sr = successes / n_episodes
    mean_dist = np.mean([e['final_dist'] for e in episodes])
    median_dist = np.median([e['final_dist'] for e in episodes])
    mean_steps = np.mean([e['steps'] for e in episodes])

    return {
        'policy': policy_name,
        'success_rate': sr,
        'successes': successes,
        'n_episodes': n_episodes,
        'success_radius': success_radius,
        'mean_final_dist': float(mean_dist),
        'median_final_dist': float(median_dist),
        'mean_steps': float(mean_steps),
        'episodes': episodes,
    }

def run_pcontroller(n_episodes, max_steps, success_radius, seed, verbose=True):
    """
    Run P-controller baseline.

    Phase 234 FIX: Two bugs in eval_phase227.py caused P-controller to show 8% SR
    instead of the true ~94% SR:

    BUG 1 (CRITICAL): No early termination.
    - eval_phase227.py ran all 200 steps THEN checked final distance.
    - The controller OVERSHOOTS the goal (inertia) then oscillates.
    - By step 200, it's far from the goal even though it passed through it at ~step 100.
    - FIX: Early termination when dist < success_radius.

    BUG 2: Uses qpos[:2] instead of xpos[base_body_id, :2].
    - Minor (~1cm) but xpos is canonical world position (Phase 195 style).
    """
    np.random.seed(seed)
    successes = 0
    episodes = []

    for ep_i in range(n_episodes):
        goal = np.array([
            np.random.uniform(-0.35, 0.40),
            np.random.uniform(-0.28, 0.28)
        ])
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id
        arm = np.array([0., -0.5, 1., 0.5, 0., 0.])
        actual_steps = max_steps

        for step in range(max_steps):
            # Phase 195 style: use xpos (world position) NOT qpos
            base_xy = sim.data.xpos[base_body_id, :2].copy()
            err = goal - base_xy
            v_desired = 2.0 * err
            ws = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
            sim.step(np.concatenate([arm, ws]))

            # Phase 234 FIX BUG 1: Early termination — the P-controller DOES reach
            # the goal (typically by step 60-130), but overshoots and oscillates.
            # Without early exit, final_dist after 200 steps is ~0.5m even for
            # successful trajectories that passed through the goal at step ~100.
            if np.linalg.norm(err) < success_radius:
                actual_steps = step + 1
                break

        final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        success = bool(final_dist < success_radius)
        successes += int(success)
        episodes.append({
            'goal': list(goal),
            'final_dist': float(final_dist),
            'steps': actual_steps,
            'success': success,
        })

        if verbose and (ep_i + 1) % 10 == 0:
            print(f"  [{ep_i+1}/{n_episodes}] P-ctrl dist={final_dist:.3f}m {'✓' if success else '✗'}")

    sr = successes / n_episodes
    mean_dist = np.mean([e['final_dist'] for e in episodes])
    median_dist = np.median([e['final_dist'] for e in episodes])
    mean_steps = np.mean([e['steps'] for e in episodes])

    return {
        'policy': 'P-controller CJ kP=2.0',
        'success_rate': sr,
        'successes': successes,
        'n_episodes': n_episodes,
        'success_radius': success_radius,
        'mean_final_dist': float(mean_dist),
        'median_final_dist': float(median_dist),
        'mean_steps': float(mean_steps),
        'episodes': episodes,
    }

def analyze_q2_performance(results, name):
    """Analyze Q2 (gx<0, gy>0) performance specifically."""
    episodes = results['episodes']
    q2_eps = [e for e in episodes if e['goal'][0] < 0 and e['goal'][1] > 0]
    q2_success = sum(1 for e in q2_eps if e['success'])
    q2_total = len(q2_eps)

    q2_gy = [e['goal'][1] for e in q2_eps]
    q2_gy_above_024 = sum(1 for e in q2_eps if e['goal'][1] > 0.24)

    print(f"\n  [{name}] Q2 (gx<0, gy>0) Performance:")
    print(f"    Q2 SR: {q2_success}/{q2_total} = {q2_success/q2_total*100:.1f}%")
    print(f"    Q2 gy range: [{min(q2_gy):.3f}, {max(q2_gy):.3f}]")
    print(f"    Q2 with gy > 0.24m (OLD OOD region): {q2_gy_above_024}")

    q2_fails = [e for e in q2_eps if not e['success']]
    if q2_fails:
        print(f"    Q2 FAILURES ({len(q2_fails)}):")
        for e in q2_fails[:5]:
            gx, gy = e['goal']
            print(f"      goal=({gx:+.2f},{gy:+.2f}) gy={gy:.3f} final_dist={e['final_dist']:.3f}m")
        if len(q2_fails) > 5:
            print(f"      ... and {len(q2_fails)-5} more")

    return q2_success, q2_total, q2_gy_above_024


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vla', type=str, required=True, help='Path to Phase 227 VLA checkpoint')
    parser.add_argument('--phase196', type=str, default='results/phase196_contact_jacobian_train/epoch_14.pt',
                       help='Path to Phase 196 baseline')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--success_radius', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 227: VLA Evaluation (50-goal, Q2-Focused)")
    print("=" * 70)

    global_start = time.time()

    # Phase 227 VLA
    print(f"\n[Loading Phase 227 VLA: {args.vla}]")
    vla227 = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
    vla227_ckpt = torch.load(args.vla, map_location='cpu', weights_only=False)
    vla227.load_state_dict(vla227_ckpt['policy_state_dict'])
    vla227.eval()
    print(f"  Loaded epoch {vla227_ckpt.get('epoch', '?')}, loss={vla227_ckpt.get('loss', 0):.4f}")

    # Phase 196 VLA baseline
    print(f"\n[Loading Phase 196 VLA: {args.phase196}]")
    vla196 = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
    vla196_ckpt = torch.load(args.phase196, map_location='cpu', weights_only=False)
    vla196.load_state_dict(vla196_ckpt['policy_state_dict'])
    vla196.eval()
    print(f"  Loaded epoch {vla196_ckpt.get('epoch', '?')}, loss={vla196_ckpt.get('loss', 0):.4f}")

    n_episodes = args.n_episodes
    max_steps = args.max_steps
    success_radius = args.success_radius
    seed = args.seed

    print(f"\nConfig: {n_episodes} goals, {max_steps} steps, sr={success_radius}m, seed={seed}")
    print()

    # Generate shared goals for fair comparison
    np.random.seed(seed)
    shared_goals = [
        np.array([np.random.uniform(-0.35, 0.40), np.random.uniform(-0.28, 0.28)])
        for _ in range(n_episodes)
    ]

    # P-controller (always fresh sim per goal)
    print(f"\n[Evaluating P-controller CJ kP=2.0]")
    start_time = time.time()
    pctrl_results = run_pcontroller(n_episodes, max_steps, success_radius, seed, verbose=True)
    print(f"  P-controller: {pctrl_results['successes']}/{n_episodes} = {pctrl_results['success_rate']*100:.1f}% SR")
    print(f"  Mean final dist: {pctrl_results['mean_final_dist']:.3f}m")
    print(f"  Mean steps: {pctrl_results['mean_steps']:.1f}")

    # Phase 196 VLA
    print(f"\n[Evaluating Phase 196 VLA (baseline)]")
    start_time = time.time()
    vla196_results = run_evaluation(vla196, n_episodes, max_steps, success_radius, seed, "VLA Phase196", verbose=True)
    print(f"  Phase196 VLA: {vla196_results['successes']}/{n_episodes} = {vla196_results['success_rate']*100:.1f}% SR")
    analyze_q2_performance(vla196_results, "Phase196 VLA")

    # Phase 227 VLA
    print(f"\n[Evaluating Phase 227 VLA (Q2-extended)]")
    start_time = time.time()
    vla227_results = run_evaluation(vla227, n_episodes, max_steps, success_radius, seed, "VLA Phase227", verbose=True)
    print(f"  Phase227 VLA: {vla227_results['successes']}/{n_episodes} = {vla227_results['success_rate']*100:.1f}% SR")
    analyze_q2_performance(vla227_results, "Phase227 VLA")

    total_time = time.time() - global_start
    print(f"\nTotal evaluation time: {total_time/60:.1f} min")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Policy':<30} {'SR':>8} {'MeanDist':>10} {'MedianDist':>11} {'MeanSteps':>10}")
    print("-" * 70)
    print(f"{'P-controller CJ kP=2.0':<30} {pctrl_results['success_rate']*100:>7.1f}% {pctrl_results['mean_final_dist']:>9.3f}m {pctrl_results['median_final_dist']:>10.3f}m {pctrl_results['mean_steps']:>9.1f}")
    print(f"{'VLA Phase196 (original)':<30} {vla196_results['success_rate']*100:>7.1f}% {vla196_results['mean_final_dist']:>9.3f}m {vla196_results['median_final_dist']:>10.3f}m {vla196_results['mean_steps']:>9.1f}")
    print(f"{'VLA Phase227 (Q2-extended)':<30} {vla227_results['success_rate']*100:>7.1f}% {vla227_results['mean_final_dist']:>9.3f}m {vla227_results['median_final_dist']:>10.3f}m {vla227_results['mean_steps']:>9.1f}")

    # Save results
    output_dir = args.output or f"results/phase227_eval"
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'phase': 227,
        'config': {
            'n_episodes': n_episodes,
            'max_steps': max_steps,
            'success_radius': success_radius,
            'seed': seed,
        },
        'p_controller': pctrl_results,
        'vla_phase196': vla196_results,
        'vla_phase227': vla227_results,
        'total_time_sec': total_time,
    }

    with open(f"{output_dir}/phase227_results.json", 'w') as f:
        # Phase 243 FIX: Ensure all numpy types are converted to Python native types for JSON
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        json.dump(make_json_safe(results), f, indent=2)

    with open(f"{output_dir}/phase227_eval_log.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Phase 227: VLA Evaluation (50-goal, Q2-Focused)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Config: {n_episodes} goals, {max_steps} steps, sr={success_radius}m, seed={seed}\n")
        f.write(f"\nSUMMARY:\n")
        f.write(f"  P-controller: {pctrl_results['successes']}/{n_episodes} = {pctrl_results['success_rate']*100:.1f}%\n")
        f.write(f"  Phase196 VLA: {vla196_results['successes']}/{n_episodes} = {vla196_results['success_rate']*100:.1f}%\n")
        f.write(f"  Phase227 VLA: {vla227_results['successes']}/{n_episodes} = {vla227_results['success_rate']*100:.1f}%\n")
        f.write(f"\nTotal time: {total_time/60:.1f} min\n")

    print(f"\nResults saved: {output_dir}/phase227_results.json")
