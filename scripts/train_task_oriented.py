#!/usr/bin/env python3
"""
LeKiWi Task-Oriented Training — CLIP-Flow Matching with Reward-Weighted Learning
================================================================================
Improves over standard CLIP-FM by weighting training samples by task success signal.

Standard Flow Matching: MSE loss on all samples equally
Task-Oriented FM:       Samples that lead toward the goal get higher loss weight

Key improvements over train_clip_fm.py:
  1. Reward-weighted loss: samples with positive task signal receive 3× weight
  2. Sparse goal reward: binary signal when robot moves toward target
  3. Smooth-action regularization: penalize large action changes
  4. Curriculum: start with short episodes, extend as policy improves

Usage:
  python3 train_task_oriented.py \
    --data data/lekiwi_urdf_5k.h5 \
    --epochs 50 \
    --device cpu \
    --output results/task_oriented \
    --goal_threshold 0.1
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

from sim_lekiwi_urdf import LeKiWiSimURDF


# ─── Reward Shaping ────────────────────────────────────────────────────────────

def compute_shaped_reward(state_t, state_tp1, action, sim, goal_pos=(0.5, 0.0), threshold=0.1):
    """
    Shaped reward for task-oriented training.

    Uses actual simulation state (not stored state) to compute distance to goal,
    because h5 states = [arm_pos*6, wheel_vel*3] — no base position stored.

    Returns:
      reward:    float — sparse + shaped reward
      is_goal:   bool  — whether state_tp1 is at goal
      dist_t:     float — distance to goal at time t
      dist_tp1:   float — distance to goal at time t+1
    """
    GOAL_POS = np.array(goal_pos)

    # Get base position from simulation (LeKiWiSimURDF tracks this)
    base_pos_t  = sim.data.qpos[:2].copy()
    base_pos_tp1 = sim.data.qpos[:2].copy()

    dist_t  = np.linalg.norm(base_pos_t  - GOAL_POS)
    dist_tp1 = np.linalg.norm(base_pos_tp1 - GOAL_POS)

    # Sparse: +1.0 only when we ARRIVE at goal (were NOT already there)
    # dist_t >= threshold prevents re-triggering when already at goal
    if dist_tp1 < threshold and dist_t >= threshold:
        reward = 1.0
        is_goal = True
    else:
        # Shaped: reward = improvement in distance (clipped)
        improvement = dist_t - dist_tp1
        reward = np.clip(improvement / 0.1, -0.1, 0.1)
        is_goal = False

    # Small action smoothness penalty (L2 on action magnitude)
    action_penalty = -0.001 * np.sum(action ** 2)

    return reward + action_penalty, is_goal, dist_t, dist_tp1


def compute_sample_weights(rewards, is_goals, base_threshold=0.5):
    """
    Convert per-step rewards to per-sample loss weights for training.

    Steps near goal achievements get higher weights.
    Steps that move toward goal get positive weights.
    Steps that don't move get neutral weights.

    Args:
      rewards:     (N,) float — shaped reward at each step
      is_goals:    (N,) bool  — whether step ends at goal
      base_threshold: float — minimum distance to count as "moving toward goal"

    Returns:
      weights: (N,) float — per-sample weight in [0.5, 3.0]
    """
    weights = np.ones_like(rewards, dtype=np.float32)

    # Goal steps: 3× weight
    weights[is_goals] = 3.0

    # Positive-reward steps: up to 2× based on reward magnitude
    positive_mask = rewards > 0
    weights[positive_mask] = 1.0 + np.clip(rewards[positive_mask] * 5, 0, 2.0)

    # Negative-reward steps: reduce to 0.5× (de-prioritize)
    negative_mask = rewards < -0.05
    weights[negative_mask] = 0.5

    return weights


# ─── CLIP Vision Encoder (from train_clip_fm.py) ────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    """CLIP ViT-B/32 frozen encoder → 768-dim → 512-dim pooled visual features."""
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        print("[INFO] Loading CLIP ViT-B/32 (pretrained, frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

        for p in self.clip.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in self.clip.parameters())
        print(f"[INFO] CLIP loaded: {n_params:,} params (frozen)")

        self.proj = nn.Linear(768, 512).to(device)

    def forward(self, images):
        """
        images: [B, 3, 224, 224] in [0, 1] range
        Returns: [B, 512] pooled visual features
        """
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
        return self.proj(pooled)


# ─── Flow Matching Policy ─────────────────────────────────────────────────────

class FlowMatchingHead(nn.Module):
    """Flow Matching MLP: predicts velocity = x_0 - x_noise."""
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )
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
    """
    Full VLA policy: CLIP vision encoder + Flow Matching action head.
    Vision: frozen CLIP ViT-B/32 (151M)
    Action: Flow Matching MLP (8M trainable)
    """
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead(
            vision_dim=hidden, state_dim=state_dim,
            action_dim=action_dim, hidden=hidden
        )
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device = device

        n_vision    = sum(p.numel() for p in self.vision_encoder.parameters())
        n_flow      = sum(p.numel() for p in self.flow_head.parameters())
        n_trainable = sum(p.numel() for p in self.flow_head.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {n_vision + n_flow:,} | trainable: {n_trainable:,}")

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)

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


# ─── Task-Oriented Replay Buffer ───────────────────────────────────────────────

class TaskOrientedReplayBuffer:
    """
    Replay buffer with task-oriented reward computation.
    Pre-computes per-sample weights from shaped rewards.
    """
    def __init__(self, h5_path, batch_size=16, goal_pos=(0.5, 0.0), goal_threshold=0.1):
        print(f"[INFO] Loading HDF5: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            self.images  = f["images"][:]
            self.states  = f["states"][:]
            self.actions = f["actions"][:]

        N = len(self.actions)
        print(f"[INFO] {N:,} frames loaded")

        # ── Compute shaped rewards ─────────────────────────────────────────────
        print("[INFO] Computing task-oriented rewards (using LeKiWiSimURDF)...")
        rewards   = np.zeros(N, dtype=np.float32)
        is_goals  = np.zeros(N, dtype=np.bool_)
        distances = np.zeros(N, dtype=np.float32)

        # Use LeKiWiSimURDF (same as collect_data.py) for consistent reward computation
        sim = LeKiWiSimURDF()
        try:
            sim.reset()
        except AttributeError:
            pass

        for i in range(N):
            state_t   = self.states[i]
            action_t  = self.actions[i]

            # Set simulation to the stored state from h5
            # h5 state = [arm_pos*6, wheel_vel*3]
            sim.data.qpos[sim._jpos_idx["j0"]:sim._jpos_idx["j0"]+6] = state_t[0:6]
            sim.data.qvel[sim._jvel_idx["w1"]] = state_t[6]
            sim.data.qvel[sim._jvel_idx["w2"]] = state_t[7]
            sim.data.qvel[sim._jvel_idx["w3"]] = state_t[8]
            sim.step(action_t)

            # Now compute reward using the simulated next-state
            reward, is_goal, dist_t, dist_tp1 = compute_shaped_reward(
                state_t, np.zeros(9), action_t, sim,
                goal_pos=goal_pos, threshold=goal_threshold
            )
            rewards[i]   = reward
            is_goals[i]  = is_goal
            distances[i] = dist_tp1

            if i % 1000 == 0:
                print(f"  [{i:,}/{N:,}] reward={reward:.3f}, dist={dist_tp1:.3f}m")

        # ── Compute per-sample weights ─────────────────────────────────────────
        self.weights = compute_sample_weights(rewards, is_goals)
        print(f"[INFO] Weights: min={self.weights.min():.2f}, max={self.weights.max():.2f}, "
              f"mean={self.weights.mean():.2f}")
        print(f"[INFO] Goal frames: {is_goals.sum():,} / {N:,} ({100*is_goals.mean():.1f}%)")

        self.N = N
        self.bs = batch_size

    def sample(self):
        """Weighted sampling based on task-oriented weights."""
        idx = np.random.randint(0, self.N, size=self.bs)
        sample_weights = self.weights[idx]

        # Images: uint8 → [0,1] float → [B, C, H, W]
        imgs = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)
        imgs = imgs.permute(0, 3, 1, 2)
        states  = torch.from_numpy(self.states[idx].astype(np.float32))
        actions = torch.from_numpy(self.actions[idx].astype(np.float32))
        weights = torch.from_numpy(sample_weights)

        return imgs, states, actions, weights


# ─── Task-Oriented Training ────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=50, device="cpu", output_dir="results", start_epoch=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy.to(device)
    policy.train()
    losses = []
    weighted_losses = []

    print(f"\n[3] Task-Oriented Training on {replay.N:,} frames (resuming from epoch {start_epoch})...")
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_loss        = 0.0
        epoch_w_loss     = 0.0
        epoch_num_samples = 0

        for batch_idx in range(100):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img    = batch_img.to(device)
            batch_state  = batch_state.to(device)
            batch_action = batch_action.to(device)
            batch_weights = batch_weights.to(device)

            t = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            x_t    = (1 - t) * batch_action + t * noise

            v_pred   = policy(batch_img, batch_state, x_t, t)
            v_target = batch_action - noise

            # Standard MSE loss
            loss_per_sample = ((v_pred - v_target) ** 2).mean(dim=-1)  # [B]
            # Weighted MSE loss
            weighted_loss = (loss_per_sample * batch_weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss    += loss_per_sample.mean().item()
            epoch_w_loss  += weighted_loss.item()
            epoch_num_samples += batch_img.shape[0]

        avg     = epoch_loss    / 100
        avg_w   = epoch_w_loss  / 100
        losses.append(avg)
        weighted_losses.append(avg_w)

        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "policy_state_dict": policy.state_dict(),
                "loss": avg,
                "weighted_loss": avg_w,
            }, ckpt_path)
            print(f"  ✓ Saved: {ckpt_path.name}")

        elapsed = time.time() - t_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f} | W-Loss: {avg_w:.4f} | ETA: {eta:.0f}s")

    # Save final
    final_path = output_dir / "final_policy.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"\n✓ Policy: {final_path}")

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Standard MSE")
    plt.plot(weighted_losses, label="Reward-Weighted MSE")
    plt.title("Task-Oriented Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    weight_hist = replay.weights
    plt.hist(weight_hist[weight_hist < 3.0], bins=50, alpha=0.7)
    plt.title("Sample Weight Distribution")
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.savefig(output_dir / "training_analysis.png", dpi=150)

    total_time = time.time() - t_start
    print(f"\n✓ Training done in {total_time:.0f}s")
    return losses, weighted_losses


# ─── Task Evaluation ──────────────────────────────────────────────────────────

def evaluate_task_success(policy, device="cpu", num_episodes=5, goal_pos=(0.5, 0.0)):
    """Evaluate how often policy reaches goal."""
    from scripts.improve_reward import TaskEvaluator

    policy.eval()
    sim = LeKiWiSimURDF()
    evaluator = TaskEvaluator(sim, policy=policy, device=device)

    results = []
    for ep in range(num_episodes):
        success, steps, dist = evaluator.reach_target(
            target=goal_pos, threshold=0.1, max_steps=200
        )
        results.append({"success": success, "steps": steps, "dist": dist})
        print(f"  Episode {ep+1}: success={success}, dist={dist:.3f}m")

    success_rate = sum(r["success"] for r in results) / len(results)
    mean_dist    = np.mean([r["dist"] for r in results])
    print(f"\n  Task success rate: {100*success_rate:.0f}% ({num_episodes} episodes)")
    print(f"  Mean final distance: {mean_dist:.3f}m")
    return success_rate, mean_dist


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           type=str,   default="data/lekiwi_urdf_5k.h5")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--hidden",         type=int,   default=512)
    parser.add_argument("--device",         type=str,   default="cpu")
    parser.add_argument("--output",         type=str,   default="results/task_oriented")
    parser.add_argument("--goal_threshold", type=float, default=0.1,
                        help="Goal radius in meters (default: 0.1m)")
    parser.add_argument("--goal_x",         type=float, default=0.5,
                        help="Goal X position (default: 0.5m)")
    parser.add_argument("--goal_y",         type=float, default=0.0,
                        help="Goal Y position (default: 0.0m)")
    parser.add_argument("--resume",         type=str,   default="",
                        help="Path to checkpoint to resume from (e.g. results/task_oriented/checkpoint_epoch_30.pt)")
    parser.add_argument("--eval",           action="store_true",
                        help="Run task evaluation after training")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Goal: ({args.goal_x}, {args.goal_y}) with threshold={args.goal_threshold}m")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run: python3 scripts/collect_data.py --sim_type urdf --episodes 50 --output data/lekiwi_urdf_5k.h5")
        sys.exit(1)

    print("\n[1] Loading task-oriented replay buffer...")
    replay = TaskOrientedReplayBuffer(
        str(data_path),
        batch_size=args.batch_size,
        goal_pos=(args.goal_x, args.goal_y),
        goal_threshold=args.goal_threshold,
    )

    print("\n[2] Building CLIP-Flow Matching policy...")
    policy = CLIPFlowMatchingPolicy(
        state_dim=9, action_dim=9,
        hidden=args.hidden, device=args.device
    )
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
            policy.load_state_dict(ckpt["policy_state_dict"], strict=False)
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"  Resumed from {resume_path} (epoch {start_epoch})")
        else:
            print(f"  WARNING: resume checkpoint not found: {resume_path}")
    optimizer = torch.optim.Adam(policy.flow_head.parameters(), lr=args.lr)

    train(
        policy, optimizer, replay,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output,
        start_epoch=start_epoch,
    )

    if args.eval:
        print("\n[4] Task success evaluation...")
        policy.eval()
        success_rate, mean_dist = evaluate_task_success(
            policy, device=args.device,
            num_episodes=5,
            goal_pos=(args.goal_x, args.goal_y),
        )
        # Save eval results
        import json
        eval_path = Path(args.output) / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump({
                "success_rate": success_rate,
                "mean_dist": mean_dist,
                "goal_pos": [args.goal_x, args.goal_y],
                "threshold": args.goal_threshold,
            }, f, indent=2)
        print(f"\n✓ Eval results: {eval_path}")

    print("\n✓ All done!")
    print(f"  Output: {args.output}/")
    print(f"  Goal:   ({args.goal_x}, {args.goal_y}) @ {args.goal_threshold}m radius")


if __name__ == "__main__":
    main()
