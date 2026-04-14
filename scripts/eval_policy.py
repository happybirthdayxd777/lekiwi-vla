#!/usr/bin/env python3
"""
Evaluate trained policies on LeKiWi simulation.
Supports both SimpleCNN-Flow Matching and CLIP-Flow Matching architectures.

Usage:
  # SimpleCNN-Flow Matching
  python3 scripts/eval_policy.py --arch simple_cnn_fm --checkpoint /tmp/fm_real/final_policy.pt --episodes 10

  # CLIP-Flow Matching
  python3 scripts/eval_policy.py --arch clip_fm --checkpoint /tmp/clip_fm_test/final_policy.pt --episodes 10

  # Random baseline
  python3 scripts/eval_policy.py --policy random --episodes 10

  # Custom steps + JSON output
  python3 scripts/eval_policy.py --arch clip_fm --checkpoint results/fresh_train_5k/checkpoint_epoch_10.pt --episodes 10 --max-steps 100 --eval-output data/clip_fm_eval.json
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from sim_lekiwi_urdf import LeKiWiSimURDF


# ─── SimpleCNN Vision Encoder (from train_flow_matching_real.py) ──────────────

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, embed_dim),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class FlowMatchingMLP(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 128))
        total_dim = vision_dim + state_dim + action_dim + 128  # 658
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class SimpleCNNFlowMatchingPolicy(nn.Module):
    """SimpleCNN + Flow Matching (8M params)."""
    def __init__(self, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=hidden)
        self.flow_head = FlowMatchingMLP(hidden, state_dim, action_dim, hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=next(self.parameters()).device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=image.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── CLIP Vision Encoder ─────────────────────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """CLIP ViT-B/32 frozen encoder → 768 → 512."""
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
        self.proj = nn.Linear(768, 512).to(device)

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            out = self.clip.vision_model(pixel_values=pixel_values)
            pooled = out.pooler_output
        return self.proj(pooled)


class CLIPFlowMatchingHead(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        total_dim = vision_dim + state_dim + action_dim + 256  # 786
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class CLIPFlowMatchingPolicy(nn.Module):
    """CLIP ViT-B/32 + Flow Matching (151M frozen + 970K trainable)."""
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = CLIPFlowMatchingHead(vision_dim=hidden, state_dim=state_dim,
                                              action_dim=action_dim, hidden=hidden)
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device = device

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# ─── Random Baseline ─────────────────────────────────────────────────────────

class RandomPolicy:
    def __init__(self, action_dim=9):
        self.action_dim = action_dim
    def infer(self, image, state, num_steps=4):
        return torch.rand(1, self.action_dim) * 2 - 1


# ─── Evaluation ──────────────────────────────────────────────────────────────

def make_policy(arch, checkpoint, device, infer_state_dim=None):
    """Load a policy from checkpoint based on architecture.

    Args:
        arch: 'simple_cnn_fm' or 'clip_fm'
        checkpoint: path to .pt file
        device: compute device
        infer_state_dim: if None, auto-detect from checkpoint weight shape.
                         For clip_fm goal-aware: 11. For standard: 9.
    """
    if arch == "simple_cnn_fm":
        policy = SimpleCNNFlowMatchingPolicy(state_dim=9, action_dim=9)
    elif arch == "clip_fm":
        # Auto-detect state_dim from checkpoint to support both goal-aware (11D)
        # and standard (9D) policies trained with train_task_oriented.py.
        if infer_state_dim is None:
            ckpt_sd = torch.load(checkpoint, map_location=device, weights_only=False)
            raw_sd = ckpt_sd.get("policy_state_dict", ckpt_sd)
            # flow_head.net.0.weight shape: [hidden, vision+state+action+time_feat]
            # CLIP-FM: total_dim = 512 + state_dim + 9 + 256 = 777 + state_dim
            #   9D  → 786, 11D → 788
            w0_shape = raw_sd.get("flow_head.net.0.weight", None)
            if w0_shape is not None:
                total_dim = w0_shape.shape[1]
                infer_state_dim = total_dim - 512 - 9 - 256  # 9 or 11
                print(f"  [INFO] Auto-detected clip_fm state_dim={infer_state_dim} from checkpoint")
            else:
                infer_state_dim = 9  # fallback
        policy = CLIPFlowMatchingPolicy(state_dim=infer_state_dim, action_dim=9, device=device)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("policy_state_dict", ckpt)   # handle both raw sd and wrapped ckpt
    policy.load_state_dict(sd)
    policy.to(device)
    policy.eval()
    return policy


def evaluate(policy, device, episodes=10, max_steps=200, verbose=True,
             goal_pos=None, use_goal_aware=False):
    """Run episodes and collect metrics.

    Args:
        policy: loaded policy (9D or 11D state_dim based on training)
        device: compute device
        episodes: number of evaluation episodes
        max_steps: max steps per episode
        goal_pos: tuple (x, y) — fixed goal for 9D eval; None for random goals per episode
        use_goal_aware: if True, policy expects 11D state [arm_pos(6)+wheel_vel(3)+goal_xy(2)]
                        if False, policy expects 9D state [arm_pos(6)+wheel_vel(3)]
                        If goal_pos is not None and policy is 11D, goal_pos is used directly.
                        If goal_pos is None and policy is 11D, a random goal is sampled per episode.
    """
    from sim_lekiwi_urdf import LeKiWiSimURDF
    sim = LeKiWiSimURDF()
    all_rewards = []
    all_distances = []
    all_success = []
    goal_threshold = 0.15  # meters

    for ep in range(episodes):
        sim.reset()
        # Sample goal for this episode
        if goal_pos is not None:
            gx, gy = goal_pos
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.3, 0.7)
            gx, gy = radius * np.cos(angle), radius * np.sin(angle)
        if hasattr(sim, 'set_target'):
            sim.set_target(np.array([gx, gy]))

        # Warmup step: URDF sim render is black at frame-0; one step seeds the renderer
        sim.step(np.zeros(9, dtype=np.float32))
        total_reward = 0.0
        start_pos = sim.data.qpos[:2].copy()
        reached_goal = False

        for step in range(max_steps):
            img_raw = sim.render()
            # Handle both PIL Image (primitive sim) and np.ndarray (URDF sim)
            # NOTE: numpy arrays also have .resize() method with DIFFERENT signature,
            # so isinstance check is required, not hasattr
            from PIL import Image as PILImage
            if isinstance(img_raw, PILImage.Image):
                # PIL Image path (LeKiwiSim)
                img_pil = img_raw.resize((224, 224))
                img_np  = np.array(img_pil, dtype=np.float32) / 255.0
            else:
                # numpy array path (LeKiWiSimURDF)
                img_np  = np.array(img_raw, dtype=np.float32)
                if img_np.ndim == 2:
                    img_np = np.stack([img_np]*3, axis=-1)
                elif img_np.ndim == 0:
                    # Empty/corrupt frame — use black placeholder
                    img_np = np.zeros((224, 224, 3), dtype=np.float32)
                else:
                    # URDF sim can return all-black frames at step 0; use cv2 or scipy
                    # for fast reliable resize, avoiding PIL Image conversion quirks
                    try:
                        import cv2
                        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LINEAR)
                    except ImportError:
                        import scipy.ndimage as ndimage
                        img_np = ndimage.zoom(img_np, (224/img_np.shape[0], 224/img_np.shape[1], 1), order=1)
                img_np = img_np.astype(np.float32) / 255.0
            img_t   = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # Use correct joint indices via _jpos_idx / _jvel_idx lookup
            # qpos layout: [freejoint_base(7), wheel_w1/w2/w3(3), arm_j0..j5(6)]
            # arm j0..j5 → qpos[10:16], wheel w1..w3 → qvel[6:9]
            # CRITICAL FIX (Phase 33): was using qpos[7:13] + qvel[9:12] which gave
            # wheel positions for arm + wrong qvel indices. Matches _obs() output now.
            arm_pos = sim.data.qpos[10:16]
            wheel_v = sim.data.qvel[6:9]

            # Build state: 9D or 11D depending on policy
            state_9d = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
            policy_state_dim = getattr(policy, 'state_dim', 9)
            if use_goal_aware or policy_state_dim > 9:
                # 11D: append goal position (normalized to [-1,1] range for [-0.8,0.8] goals)
                # NOTE: For 11D goal-aware policies, we always embed the goal regardless of
                # whether goal is fixed (--goal_x/--goal_y) or random. The policy was trained
                # with goal conditioning and needs goal input at every step.
                goal_norm = np.array([gx / 0.8, gy / 0.8], dtype=np.float32)
                state_9d = np.concatenate([state_9d, goal_norm])

            state_t = torch.from_numpy(state_9d).float().unsqueeze(0).to(device)

            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = np.clip(action.cpu().numpy()[0], -1, 1)

            sim.step(action_np)
            reward = sim.get_reward()
            total_reward += reward

            # Check goal reached
            cur_pos = sim.data.qpos[:2]
            if np.linalg.norm(cur_pos - np.array([gx, gy])) < goal_threshold:
                reached_goal = True

        end_pos = sim.data.qpos[:2]
        dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
        all_rewards.append(total_reward)
        all_distances.append(dist)
        all_success.append(1.0 if reached_goal else 0.0)
        if verbose:
            status = "✓ GOAL" if reached_goal else f"✗ dist={dist:.3f}m"
            print(f"  Episode {ep+1:2d}: reward={total_reward:+.3f} {status}")

    return {
        "mean_reward":    np.mean(all_rewards),
        "std_reward":     np.std(all_rewards),
        "mean_distance":  np.mean(all_distances),
        "std_distance":   np.std(all_distances),
        "success_rate":   np.mean(all_success) * 100,
        "all_rewards":    all_rewards,
        "all_success":    all_success,
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",     type=str,   default=None, help="'random' for baseline")
    parser.add_argument("--arch",       type=str,   default="simple_cnn_fm",
                        choices=["simple_cnn_fm", "clip_fm"])
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--episodes",   type=int,   default=10)
    parser.add_argument("--max-steps",  type=int,   default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--device",     type=str,   default="mps")
    parser.add_argument("--eval-output", type=str,  default=None,
                        help="Path to save evaluation metrics as JSON (e.g., data/eval_results.json)")
    parser.add_argument("--goal_x",     type=float, default=None,
                        help="Fixed goal X position (for goal-aware policy eval)")
    parser.add_argument("--goal_y",     type=float, default=None,
                        help="Fixed goal Y position (for goal-aware policy eval)")
    parser.add_argument("--goal_aware", action="store_true",
                        help="Force 11D goal-aware state even without --goal_x/--goal_y")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    if args.policy == "random":
        print(f"  Policy: RANDOM BASELINE")
        policy = RandomPolicy(action_dim=9)
        device = "cpu"
        use_goal_aware = False
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for trained policies (or use --policy random)")
        arch_name = args.arch
        print(f"  Architecture: {arch_name}")
        print(f"  Checkpoint:  {args.checkpoint}")
        policy = make_policy(args.arch, args.checkpoint, args.device)
        device = args.device
        # Use goal-aware if explicitly flagged or if goal_x/y provided
        use_goal_aware = args.goal_aware or (args.goal_x is not None)
        if use_goal_aware:
            print(f"  Mode: GOAL-AWARE (state_dim={policy.state_dim})")
        else:
            print(f"  Mode: STANDARD (state_dim={policy.state_dim})")

    # Fixed goal or random goals
    goal_pos = (args.goal_x, args.goal_y) if args.goal_x is not None else None
    if goal_pos:
        print(f"  Goal: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    else:
        print(f"  Goal: random (0.3–0.7m radius)")

    print(f"  Episodes: {args.episodes} | Steps/ep: {args.max_steps} | Device: {device}")
    print(f"{'='*60}\n")

    print(f"[Running {args.episodes} episodes...]")
    metrics = evaluate(policy, device, episodes=args.episodes, max_steps=args.max_steps,
                      goal_pos=goal_pos, use_goal_aware=use_goal_aware)

    print(f"\n{'='*40}")
    print(f"  Mean reward:   {metrics['mean_reward']:+.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Mean distance: {metrics['mean_distance']:.3f} ± {metrics['std_distance']:.3f}m")
    print(f"  Success rate:  {metrics['success_rate']:.1f}%")
    print(f"  Best reward:   {max(metrics['all_rewards']):+.3f}")
    print(f"  Worst reward:  {min(metrics['all_rewards']):+.3f}")
    print(f"{'='*40}")
    print("✓ Evaluation complete")

    # Save metrics to JSON
    if args.eval_output:
        out_path = Path(args.eval_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "policy":    args.policy if args.policy == "random" else args.arch,
            "checkpoint": args.checkpoint or "n/a",
            "device":    device,
            "episodes":  args.episodes,
            "max_steps": args.max_steps,
            "mean_reward":   float(metrics["mean_reward"]),
            "std_reward":    float(metrics["std_reward"]),
            "mean_distance": float(metrics["mean_distance"]),
            "std_distance":  float(metrics["std_distance"]),
            "success_rate":  float(metrics["success_rate"]),
            "all_rewards":   [float(r) for r in metrics["all_rewards"]],
            "all_success":   [float(s) for s in metrics["all_success"]],
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {out_path}")

if __name__ == "__main__":
    main()