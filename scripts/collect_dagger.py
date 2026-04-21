#!/usr/bin/env python3
"""
Phase 252: DAgger Data Collection for LeKiWi VLA
=================================================

Implements DAgger (Ross & Bagnell 2013): collect VLA failures, then use P-controller
as expert to provide corrective actions for those failure states.

Strategy:
1. Run VLA policy in simulation until step 30
2. If dist > threshold (VLA stuck), switch to P-controller expert
3. Record (image, state, expert_action) for ALL steps
4. After episode: save trajectories with "vla" and "expert" action labels

Key insight from Phase 243 analysis: VLA fails when |goal| > ~0.3m
→ Target large-displacement goals to collect failurecorrection pairs

Usage:
  python3 scripts/collect_dagger.py --n_episodes 30 --output data/dagger_phase246_30ep.h5

Expected outcome: DAgger data + Phase196/227 data → retrain → improved VLA on large |g|
"""
import os, sys, argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
import torch
from PIL import Image
from pathlib import Path

from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS, _CONTACT_JACOBIAN_PSEUDO_INV

# Import VLA policy from Phase 227 training
from scripts.train_phase227 import GoalConditionedPolicy, DEVICE

# ImageNet normalization (inlined from eval_phase227.py)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Config ──────────────────────────────────────────────────────────────────

class PController:
    """Contact-Jacobian P-controller — the expert demonstrator."""
    def __init__(self, kP=2.0, wheel_clip=0.5):
        self.kP = kP
        self.wheel_clip = wheel_clip

    def compute(self, goal_xy, base_xy):
        err = goal_xy - base_xy
        v_desired = self.kP * err
        wheel_speeds = _CONTACT_JACOBIAN_PSEUDO_INV @ v_desired
        wheel_speeds = np.clip(wheel_speeds, -self.wheel_clip, self.wheel_clip)
        return np.array(wheel_speeds, dtype=np.float32)


def build_state(sim, goal):
    """Build 11D state: arm_pos(6) + wheel_vel(3) + goal_norm(2)."""
    arm_pos = np.array([
        sim.data.qpos[sim.model.joint(n).qposadr[0]]
        for n in ARM_JOINTS
    ])
    wheel_vel = np.array([
        sim.data.qvel[sim.model.joint(n).dofadr[0]]
        for n in WHEEL_JOINTS
    ])
    goal_norm = np.clip(goal / 0.4, -1, 1)
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)


def preprocess_image(raw_img):
    img = Image.fromarray(raw_img).resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return arr.transpose(2, 0, 1)


# ── DAgger Episode Collection ────────────────────────────────────────────────

def collect_dagger_episode(vla_policy, p_controller, sim,
                            goal_range=0.40,  # Target LARGE goals (|g| > 0.3m)
                            max_steps=250,
                            dagger_threshold_step=30,
                            dagger_stuck_dist=0.25,
                            seed=None):
    """
    Collect one DAgger episode.

    At each step:
      - If step < dagger_threshold_step: use VLA action
      - If VLA is "stuck" (dist > threshold after step 30): use P-controller expert
      - Record BOTH vla_action and expert_action for DAgger training

    Returns dict with trajectories for both VLA and expert.
    """
    if seed is not None:
        np.random.seed(seed)

    # Large goal (target the failure mode: |g| > 0.3m)
    goal = np.array([
        np.random.uniform(-goal_range, goal_range),
        np.random.uniform(-goal_range * 0.85, goal_range * 0.85)
    ], dtype=np.float32)

    sim.reset()
    base_body_id = sim.model.body('base').id
    arm_pos = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0], dtype=np.float32)

    # Trajectory buffers
    obs_tensors = []    # Preprocessed image tensors (for VLA)
    raw_images = []     # Original images (for reference)
    states = []         # 11D states
    vla_actions = []    # Actions chosen by VLA
    expert_actions = [] # Actions from P-controller (the expert)
    labels = []         # 0=VLA expert, 1=expert (for DAgger loss weighting)
    goals = []          # Goal norm vectors
    rewards = []

    use_expert = False  # Once stuck, stay in expert mode for entire episode
    expert_mode_reason = ""

    for step in range(max_steps):
        base_xy = sim.data.xpos[base_body_id, :2].copy()
        dist = np.linalg.norm(goal - base_xy)
        state = build_state(sim, goal)

        # Render BEFORE action
        raw_img = sim.render().astype(np.uint8)
        img_tensor = torch.from_numpy(preprocess_image(raw_img)).unsqueeze(0).to(DEVICE)

        # ── Decide action ───────────────────────────────────────────────────
        if step < dagger_threshold_step:
            # VLA phase
            with torch.no_grad():
                vla_action = vla_policy.infer(img_tensor,
                    torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                    num_steps=4).cpu().numpy()[0]
            expert_action = vla_action.copy()  # No correction needed yet
            action_source = "vla"
            label = 0  # VLA action — DAgger will train to match expert eventually
            use_expert = False

        else:
            # Check if VLA is stuck
            if not use_expert and dist > dagger_stuck_dist:
                use_expert = True
                expert_mode_reason = f"step={step}, dist={dist:.3f}m > {dagger_stuck_dist}m"

            if use_expert:
                # Expert correction — P-controller
                p_wheel_speeds = p_controller.compute(goal, base_xy)
                expert_action = np.concatenate([arm_pos, p_wheel_speeds]).astype(np.float32)

                # VLA action (what policy originally would have done)
                with torch.no_grad():
                    vla_action = vla_policy.infer(img_tensor,
                        torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                        num_steps=4).cpu().numpy()[0]
                action_source = "expert"
                label = 1  # Expert action — DAgger loss prioritizes these
            else:
                # Still VLA but past threshold step — keep running VLA
                with torch.no_grad():
                    vla_action = vla_policy.infer(img_tensor,
                        torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                        num_steps=4).cpu().numpy()[0]
                expert_action = vla_action.copy()
                action_source = "vla"
                label = 0
                use_expert = False

        # Full action (arm + wheels)
        action = np.clip(vla_action, -0.5, 0.5)

        # Record
        obs_tensors.append(img_tensor.squeeze(0).cpu().numpy())
        raw_images.append(raw_img)  # Store RAW image for training (not preprocessed!)
        states.append(state)
        vla_actions.append(vla_action)
        expert_actions.append(expert_action)
        labels.append(label)
        goals.append(state[-2:])  # goal_norm from state

        # Reward
        reward = 1.0 if dist < 0.10 else 0.0
        rewards.append(reward)

        # Step simulation
        sim.step(action)

        if reward > 0.5:
            break

    final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2].copy() - goal)

    return {
        'obs': np.array(obs_tensors, dtype=np.float32),      # (N, 3, 224, 224)
        'raw_images': np.array(raw_images, dtype=np.uint8),  # (N, H, W, 3)
        'states': np.array(states, dtype=np.float32),        # (N, 11)
        'vla_actions': np.array(vla_actions, dtype=np.float32),      # (N, 9)
        'expert_actions': np.array(expert_actions, dtype=np.float32), # (N, 9)
        'labels': np.array(labels, dtype=np.int32),          # (N,) 0=VLA, 1=expert
        'goals': np.array(goals, dtype=np.float32),          # (N, 2)
        'rewards': np.array(rewards, dtype=np.float32),      # (N,)
        'goal_raw': goal,
        'final_dist': final_dist,
        'expert_mode_reason': expert_mode_reason,
        'n_expert_steps': int(sum(labels)),  # steps where expert was used
        'success': final_dist < 0.10,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DAgger data collection for LeKiWi VLA')
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=250)
    parser.add_argument('--goal_range', type=float, default=0.40,
                        help='Goal range in meters (use >0.35 to target failure modes)')
    parser.add_argument('--output', type=str, default='data/dagger_phase246_30ep.h5')
    parser.add_argument('--vla_checkpoint', type=str,
                        default='results/phase227_contact_jacobian_train/best_policy.pt')
    parser.add_argument('--dagger_threshold_step', type=int, default=30,
                        help='Switch to expert after this many steps')
    parser.add_argument('--dagger_stuck_dist', type=float, default=0.25,
                        help='Switch to expert when dist > this (meters)')
    parser.add_argument('--seed', type=int, default=246)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAgger Data Collection — Phase 246")
    print("=" * 60)
    print(f"  VLA checkpoint: {args.vla_checkpoint}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Goal range: ±{args.goal_range}m (targeting large |g| failure mode)")
    print(f"  DAgger threshold: step > {args.dagger_threshold_step} AND dist > {args.dagger_stuck_dist}m")
    print(f"  Output: {args.output}")
    print()

    # ── Load VLA policy ──────────────────────────────────────────────────────
    print("[Loading VLA policy]")
    policy = GoalConditionedPolicy().to(DEVICE)
    ckpt = torch.load(args.vla_checkpoint, map_location=DEVICE, weights_only=True)
    # Checkpoint format: {'epoch': N, 'loss': X, 'policy_state_dict': {...}}
    if 'policy_state_dict' in ckpt:
        policy.load_state_dict(ckpt['policy_state_dict'])
    else:
        policy.load_state_dict(ckpt)
    policy.eval()
    print(f"  Loaded: {args.vla_checkpoint}")
    print(f"  Device: {DEVICE}")

    # Expert controller
    p_controller = PController(kP=2.0, wheel_clip=0.5)
    print(f"  Expert: P-controller CJ kP=2.0")
    print()

    # ── Collect episodes ──────────────────────────────────────────────────────
    all_obs = []
    all_states = []
    all_vla_actions = []
    all_expert_actions = []
    all_labels = []
    all_goals = []
    all_rewards = []
    episode_starts = [0]
    expert_step_count = 0

    success_count = 0

    for ep in range(args.n_episodes):
        print(f"  Episode {ep+1}/{args.n_episodes}...", end=" ", flush=True)

        sim = LeKiWiSimURDF()
        ep_data = collect_dagger_episode(
            policy, p_controller, sim,
            goal_range=args.goal_range,
            max_steps=args.max_steps,
            dagger_threshold_step=args.dagger_threshold_step,
            dagger_stuck_dist=args.dagger_stuck_dist,
            seed=args.seed + ep if args.seed else None,
        )

        n_steps = len(ep_data['states'])
        all_obs.append(ep_data['obs'])
        all_states.append(ep_data['states'])
        all_vla_actions.append(ep_data['vla_actions'])
        all_expert_actions.append(ep_data['expert_actions'])
        all_labels.append(ep_data['labels'])
        all_goals.append(ep_data['goals'])
        all_rewards.append(ep_data['rewards'])
        episode_starts.append(episode_starts[-1] + n_steps)

        expert_step_count += ep_data['n_expert_steps']
        ep_success = ep_data['success']
        success_count += int(ep_success)
        sr = success_count / (ep + 1) * 100

        if ep_data['expert_mode_reason']:
            reason = f" [{ep_data['expert_mode_reason']}]"
        else:
            reason = ""
        print(f"{n_steps} steps, expert={ep_data['n_expert_steps']}, "
              f"success={ep_success}, SR={sr:.0f}%{reason}")

    # ── Concatenate ──────────────────────────────────────────────────────────
    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    vla_actions = np.concatenate(all_vla_actions, axis=0).astype(np.float32)
    expert_actions = np.concatenate(all_expert_actions, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    goals = np.concatenate(all_goals, axis=0).astype(np.float32)
    rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    episode_starts = np.array(episode_starts, dtype=np.int64)

    n_total = len(obs)
    n_expert = int(labels.sum())
    print(f"\n[Data Summary]")
    print(f"  Total frames: {n_total}")
    print(f"  Expert frames (DAgger corrections): {n_expert} ({n_expert/n_total*100:.1f}%)")
    print(f"  VLA frames: {n_total - n_expert} ({(n_total-n_expert)/n_total*100:.1f}%)")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Success rate: {success_count}/{args.n_episodes} = {success_count/args.n_episodes*100:.0f}%")
    print(f"  Obs shape: {obs.shape}")
    print(f"  States shape: {states.shape}")
    print(f"  VLA actions shape: {vla_actions.shape}")
    print(f"  Expert actions shape: {expert_actions.shape}")
    print(f"  Labels shape: {labels.shape}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n[Saving to {args.output}]")
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('obs', data=obs, compression='gzip')           # (N, 3, 224, 224)
        f.create_dataset('states', data=states, compression='gzip')       # (N, 11)
        f.create_dataset('vla_actions', data=vla_actions, compression='gzip')   # (N, 9)
        f.create_dataset('expert_actions', data=expert_actions, compression='gzip') # (N, 9)
        f.create_dataset('labels', data=labels)                            # (N,) 0=VLA, 1=expert
        f.create_dataset('goals', data=goals, compression='gzip')         # (N, 2)
        f.create_dataset('rewards', data=rewards, compression='gzip')       # (N,)
        f.create_dataset('episode_starts', data=episode_starts)            # (N_ep+1,)
        f.attrs['n_episodes'] = args.n_episodes
        f.attrs['n_expert_steps'] = n_expert
        f.attrs['dagger_threshold_step'] = args.dagger_threshold_step
        f.attrs['dagger_stuck_dist'] = args.dagger_stuck_dist

    print(f"  ✅ Saved")

    # ── Quick correlation check ──────────────────────────────────────────────
    print(f"\n[Expert Action Correlations]")
    wheel_expert = expert_actions[:, 6:9]  # (N, 3)
    gx, gy = goals[:, 0], goals[:, 1]
    for i, name in enumerate(['w1', 'w2', 'w3']):
        cx = np.corrcoef(wheel_expert[:, i], gx)[0, 1]
        cy = np.corrcoef(wheel_expert[:, i], gy)[0, 1]
        print(f"  Corr({name}, gx) = {cx:+.3f}, Corr({name}, gy) = {cy:+.3f}")

    print(f"\n[DAgger Data Collection Complete]")


if __name__ == "__main__":
    main()