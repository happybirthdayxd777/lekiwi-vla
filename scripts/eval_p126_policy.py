#!/usr/bin/env python3
"""
Phase 127 — VLA Policy Evaluation on Goal-Directed Task
======================================================
Tests CLIP-FM policy (task_oriented_50ep) on goal reaching with P126 dataset goals.

Key findings from Phase 126:
- PController achieves 65% SR in data collection (vs 0% GridSearchController)
- k_omni=15.0 overlay active in sim_lekiwi_urdf.py
- P126 dataset: 4000 frames, 42.4% positive, goal-conditioned (goal_positions stored)

This script tests:
1. Does the task_oriented_50ep policy work on goal-reaching?
2. What's the success rate vs P-controller baseline?
3. Does goal conditioning help?

Usage:
  python3 scripts/eval_p126_policy.py --episodes 5 --max-steps 200
  python3 scripts/eval_p126_policy.py --episodes 10 --policy fresh_train_5k
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

from sim_lekiwi import LeKiwiSim

# ── Parse train_clip_fm.py to get policy architecture ────────────────────────

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "train_clip_fm.py"
with open(SCRIPT_PATH) as f:
    src = f.read()

# ─── CLIPVisionEncoder ──────────────────────────────────────────────────────

class CLIPVisionEncoder(torch.nn.Module):
    """CLIP ViT-B/32 frozen encoder → 512-dim pooled visual features."""
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        for p in self.clip.parameters():
            p.requires_grad = False
        self.proj = torch.nn.Linear(768, 512).to(device)

    def forward(self, images):
        """images: [B, 3, 224, 224] in [0, 1]"""
        with torch.no_grad():
            img_tensor = images.to(self.device)
            outputs = self.clip.get_image_features(pixel_values=img_tensor)
            vis = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0].mean(dim=1)
        return self.proj(vis)


# ─── FlowMatchingHead ──────────────────────────────────────────────────────

class FlowMatchingHead(torch.nn.Module):
    """Flow Matching action head: predicts velocity for action denoising."""
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim
        total_dim = vision_dim + state_dim + action_dim + 256  # 786
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 256),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(total_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, action_dim),
        )
        self.skip = torch.nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)  # [B, 256]
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


# ─── CLIPFlowMatchingPolicy ─────────────────────────────────────────────────

class CLIPFlowMatchingPolicy(torch.nn.Module):
    """CLIP ViT-B/32 frozen + Flow Matching head."""
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead(vision_dim=hidden, state_dim=state_dim,
                                          action_dim=action_dim, hidden=hidden)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        """4-step Euler ODE inference."""
        action = torch.randn(image.shape[0], self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── Helpers ──────────────────────────────────────────────────────────────

def resize_for_clip(img):
    """Convert img (numpy array [H,W,3]) to CLIP-ready tensor [1,3,224,224]."""
    from PIL import Image
    if isinstance(img, np.ndarray):
        img_pil = Image.fromarray(img.astype(np.uint8), 'RGB')
    else:
        img_pil = img
    pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    img_chw = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_chw).unsqueeze(0).cpu()


def make_state_9d(obs):
    """Build 9D state: arm_pos(6) + wheel_vel(3)."""
    arm_pos = obs['arm_positions']       # 6D
    wheel_vel = obs['wheel_velocities']   # 3D
    return np.concatenate([arm_pos, wheel_vel]).astype(np.float32)


def make_state_11d(obs, goal_x, goal_y):
    """Build 11D goal-aware state."""
    base = make_state_9d(obs)
    goal_norm = np.array([goal_x / 1.0, goal_y / 1.0])
    return np.concatenate([base, goal_norm]).astype(np.float32)


def normalize_action(raw_action, device="cpu"):
    """Policy (unbounded) → LeKiWi native units.
    
    Phase 128 FIX: TWO issues found:
    1. CLIP-FM flow matching produces UNBOUNDED outputs (wheel raw: [-4.7, 1.1])
       → clip to [-1, 1] BEFORE denormalization
    2. LEKIWI_WHEEL_LIMITS was [-5, 5] but P-controller clips at [-0.5, 0.5]
       Training data uses wheel_speeds clipped to [-0.5, 0.5]
       → Fix limits to match actual actuator range
    """
    LEKIWI_ARM_LIMITS = np.array([
        [-3.14, 3.14], [-1.57, 1.57], [-1.57, 1.57],
        [-1.57, 1.57], [-3.14, 3.14], [0.00, 0.04],
    ], dtype=np.float32)
    # Phase 128 FIX: P-controller clips at [-0.5, 0.5], matching real wheel actuators
    LEKIWI_WHEEL_LIMITS = np.array([[-0.5, 0.5]] * 3, dtype=np.float32)
    # Phase 128: clip to [-1, 1] to bound flow matching output
    raw_clipped = np.clip(raw_action, -1.0, 1.0)
    arm   = raw_clipped[:6]
    wheel = raw_clipped[6:9]
    arm_n   = LEKIWI_ARM_LIMITS[:, 0]   + (arm   + 1) / 2 * (LEKIWI_ARM_LIMITS[:, 1]   - LEKIWI_ARM_LIMITS[:, 0])
    wheel_n = LEKIWI_WHEEL_LIMITS[:, 0] + (wheel + 1) / 2 * (LEKIWI_WHEEL_LIMITS[:, 1] - LEKIWI_WHEEL_LIMITS[:, 0])
    return np.concatenate([arm_n, wheel_n]).astype(np.float32)


def pcontroller_base_action(obs, goal_x, goal_y, kP=1.5, max_speed=0.3):
    """P-controller using contact-Jacobian J_c^+ IK."""
    J_c = np.array([[-0.0467, 1.3399, -2.5397], [1.3865, 0.5885, 1.2485]])
    Jc_pinv = np.linalg.pinv(J_c)
    base_pos = obs['base_position'][:2]
    err = np.array([goal_x, goal_y]) - base_pos
    vx = np.clip(err[0] * kP, -max_speed, max_speed)
    vy = np.clip(err[1] * kP, -max_speed, max_speed)
    v_world = np.array([vx, vy])
    wheel_speeds = Jc_pinv @ v_world
    return np.concatenate([np.zeros(6), wheel_speeds])


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate_policy_single(sim, policy, goal_x, goal_y, max_steps=200, threshold=0.15, use_goal_state=False):
    """Evaluate one goal. Returns (success, final_dist, steps, reward_sum)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    sim_local = LeKiWiSimURDF()
    sim_local.reset()
    base_id = sim_local.model.body('base').id

    # Warmup
    for _ in range(30):
        sim_local.step(np.zeros(9))

    img_pil = sim_local.render()
    img_t = resize_for_clip(img_pil)
    obs = sim_local._obs()

    reward_sum = 0.0
    for step in range(max_steps):
        state_9d = make_state_9d(obs)

        if use_goal_state and hasattr(policy, 'state_dim') and policy.state_dim == 11:
            state_t = torch.from_numpy(make_state_11d(obs, goal_x, goal_y)).unsqueeze(0).cpu()
        else:
            state_t = torch.from_numpy(state_9d).unsqueeze(0).cpu()

        # If policy has 11d but we're using 9d policy, just use 9d
        try:
            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()
            action = normalize_action(raw_action)
        except Exception as e:
            # Fallback to zero action
            action = np.zeros(9)

        result = sim_local.step(action)
        reward_sum += 0.0

        robot_pos = sim_local.data.xpos[base_id, :2]
        dist = np.linalg.norm(robot_pos - np.array([goal_x, goal_y]))

        if dist < threshold:
            return True, float(dist), step + 1, reward_sum

        if step < max_steps - 1:
            img_pil = sim_local.render()
            img_t = resize_for_clip(img_pil)
            obs = sim_local._obs()

    return False, float(dist), max_steps, reward_sum


def evaluate_pcontroller_baseline(goals, max_steps=200, threshold=0.15):
    """P-controller baseline (contact-Jacobian IK) — the data collection controller."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    successes = 0
    total_dist = 0.0
    steps_list = []

    for i, (gx, gy) in enumerate(goals):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        for step in range(max_steps):
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(np.array([gx, gy]) - base_pos)
            if dist < threshold:
                successes += 1
                steps_list.append(step + 1)
                break
            # P-controller: compute vx, vy toward goal
            dx, dy = gx - base_pos[0], gy - base_pos[1]
            d = np.linalg.norm([dx, dy])
            if d > 0.01:
                v_mag = min(1.5 * d, 0.3)
                vx, vy = v_mag * dx / d, v_mag * dy / d
            else:
                vx, vy = 0.0, 0.0
            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            action = np.zeros(9)
            action[6:9] = np.clip(wheel_speeds, -0.5, 0.5)
            sim.step(action)
        else:
            steps_list.append(max_steps)
            total_dist += np.linalg.norm(sim.data.xpos[base_id, :2] - np.array([gx, gy]))

    return successes, np.mean(steps_list), total_dist / len(goals)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA policy on goal-reaching")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--policy", type=str, default="task_oriented_50ep",
                        choices=["task_oriented_50ep", "goal_aware_50ep", "fresh_train_5k", "none"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load goals from P126 dataset
    data_path = Path(__file__).parent.parent / "data" / "lekiwi_goal_p126_20ep.h5"
    f = h5py.File(data_path, 'r')
    all_goals = f['goal_positions'][:]
    f.close()

    # Sample random goals
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_goals))[:args.episodes]
    goals = all_goals[indices]

    print(f"=" * 60)
    print(f"Phase 127 — VLA Policy Evaluation")
    print(f"=" * 60)
    print(f"Episodes: {args.episodes}, Max steps: {args.max_steps}, Threshold: {args.threshold}m")
    print(f"Policy: {args.policy}")
    print(f"Goals: {goals[:3]}...")
    print()

    # P-controller baseline (data collection controller)
    print("[1] P-controller baseline (contact-Jacobian IK)...")
    pcontroller_successes, pcontroller_mean_steps, pcontroller_mean_dist = \
        evaluate_pcontroller_baseline(goals, args.max_steps, args.threshold)
    print(f"  P-controller: {pcontroller_successes}/{args.episodes} ({pcontroller_successes/args.episodes*100:.0f}%) SR, "
          f"mean steps={pcontroller_mean_steps:.0f}, mean dist={pcontroller_mean_dist:.4f}")
    print()

    # VLA policy evaluation
    if args.policy != "none":
        print(f"[2] Loading policy: {args.policy}...")
        policy_path = Path(__file__).parent.parent / "results" / args.policy

        # Try final_policy.pt first, then checkpoint
        final_path = policy_path / "final_policy.pt"
        ckpt_path = None
        if args.policy == "fresh_train_5k":
            ckpt_path = policy_path / "final_clean.pt"
        else:
            for ep in [30, 20, 10]:
                p = policy_path / f"checkpoint_epoch_{ep}.pt"
                if p.exists():
                    ckpt_path = p
                    break

        load_path = final_path if final_path.exists() else (ckpt_path if ckpt_path else None)
        if load_path is None:
            print(f"  ERROR: No checkpoint found in {policy_path}")
            print(f"  Available files: {list(policy_path.iterdir())}")
            return

        print(f"  Loading: {load_path.name}")
        ckpt = torch.load(load_path, map_location='cpu', weights_only=False)

        if 'policy_state_dict' in ckpt:
            state_dict = ckpt['policy_state_dict']
            epoch = ckpt.get('epoch', '?')
        else:
            state_dict = ckpt
            epoch = '?'

        print(f"  State dict keys: {len(state_dict)}, Epoch: {epoch}")

        # Create policy and load weights
        policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device=args.device)
        policy.load_state_dict(state_dict, strict=False)
        policy.eval()
        print(f"  Policy created and weights loaded.")

        print(f"\n[3] Running VLA policy evaluation...")
        vla_successes = 0
        vla_steps_list = []
        vla_dists = []

        for i, (gx, gy) in enumerate(goals):
            success, dist, steps, _ = evaluate_policy_single(
                None, policy, gx, gy, args.max_steps, args.threshold
            )
            if success:
                vla_successes += 1
                vla_steps_list.append(steps)
            else:
                vla_steps_list.append(args.max_steps)
            vla_dists.append(dist)
            print(f"  Episode {i+1}/{args.episodes}: {'✓' if success else '✗'} dist={dist:.3f} steps={steps}")

        vla_sr = vla_successes / args.episodes
        print(f"\n  VLA policy ({args.policy}): {vla_successes}/{args.episodes} ({vla_sr*100:.0f}%) SR, "
              f"mean steps={np.mean(vla_steps_list):.0f}, mean dist={np.mean(vla_dists):.4f}")
    else:
        vla_sr = 0.0

    print()
    print(f"=" * 60)
    print(f"RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"  P-controller baseline: {pcontroller_successes}/{args.episodes} ({pcontroller_successes/args.episodes*100:.0f}%)")
    if args.policy != "none":
        print(f"  VLA policy ({args.policy}): {vla_successes}/{args.episodes} ({vla_sr*100:.0f}%)")
        print(f"  Delta: {(vla_sr - pcontroller_successes/args.episodes)*100:+.0f}pp")
    print()
    print(f"CONCLUSION:")
    if vla_sr >= pcontroller_successes / args.episodes:
        print(f"  VLA policy MEETS OR BEATS P-controller baseline")
    else:
        print(f"  VLA policy UNDERPERFORMS P-controller baseline")
        print(f"  Possible causes:")
        print(f"    - k_omni=15.0 overlay contaminating training data")
        print(f"    - State dim mismatch (9D policy vs 11D state needed)")
        print(f"    - Policy not trained long enough (only 30 epochs)")
        print(f"    - Need goal-conditioned state in policy")


if __name__ == "__main__":
    main()