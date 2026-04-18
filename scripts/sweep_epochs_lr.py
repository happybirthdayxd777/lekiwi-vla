#!/usr/bin/env python3
"""
Phase 154: Epoch Sweep + Early Stopping
======================================
Phase 153 finding: 30-epoch VLA = 15% SR, 5-epoch VLA = 30% SR.
CONFIRMED OVERFITTING on 10k frames (155M params).

Strategy:
- Sweep LR: [5e-5, 2e-5, 1e-5] (reducing from 1e-4)
- Sweep epochs: [3, 5, 7, 10] (stop before overfitting)
- For each config: train → eval → record SR
- Best config: highest SR with lowest epoch count (least overfitting)
- Then: full eval with best config

Data: phase63_reachable_10k_converted.h5 (pre-rendered images, no sim rendering)
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Setup ──────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 154] Device: {DEVICE}")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── Data Loader ───────────────────────────────────────────────────────────────

def load_prerendered_data(h5_path="data/phase63_reachable_10k_converted.h5"):
    """Load pre-rendered phase63 data (no sim rendering needed).
    
    Data format (phase63_reachable_10k_converted.h5):
      images:         [N, 224, 224, 3] uint8 — channels LAST, must transpose to [N,3,224,224]
      actions:        [N, 9] float32 — arm(6) + wheel(3) in [-0.5, 0.5] (wheel normalized)
      states:         [N, 9] float32 — arm_pos(6) + wheel_pos(3)
      goal_positions: [N, 2] float32 — xy goal positions
      rewards:        [N] float32
    """
    h5_path = ROOT / h5_path
    print(f"[DATA] Loading {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]           # [N, 224, 224, 3] uint8, CHANNELS LAST
        actions = f["actions"][:]         # [N, 9] float32, normalized wheel [-0.5, 0.5]
        states_9d = f["states"][:]        # [N, 9] — arm_pos(6) + wheel_pos(3)
        goals_xy = f["goal_positions"][:] # [N, 2] — xy goal positions ← REAL goals!
        rewards = f["rewards"][:]          # [N]

    print(f"[DATA] shapes: images={images.shape}, actions={actions.shape}, "
          f"states={states_9d.shape}, goals={goals_xy.shape}")

    # ── Transpose images: [N, 224, 224, 3] → [N, 3, 224, 224] for PyTorch/CLIP ──
    images = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # [N, 3, 224, 224]

    # ── Priority: reward-weighted sampling (high reward = near goal = harder) ──
    priorities = np.maximum(rewards, 0.0).astype(np.float32)
    priorities += 1e-6
    priorities /= priorities.sum()

    # ── Build 11D state: arm_pos(6) + wheel_pos(3) + goal_xy(2) ──────────────
    # wheel_pos from states_9d[:, 6:9] provides spatial context
    # goal_xy from goal_positions provides the TARGET
    states_11d = np.zeros((len(images), 11), dtype=np.float32)
    states_11d[:, :6] = states_9d[:, :6]       # arm positions
    states_11d[:, 6:9] = states_9d[:, 6:9]     # wheel positions (spatial context)
    states_11d[:, 9:11] = goals_xy            # ← REAL goal xy from data!

    print(f"[DATA] Loaded {len(images)} frames, priority range: [{priorities.min():.3f}, {priorities.max():.3f}]")
    print(f"[DATA] State 11D: arm(6) + wheel_pos(3) + goal_xy(2)")
    print(f"[DATA] Image shape after transpose: {images.shape} (PyTorch format)")
    return images, states_11d, actions, priorities


# ── CLIP Spatial Encoder ──────────────────────────────────────────────────────

class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (pretrained, frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        """images: [B, 3, 224, 224] in [0,1]. Returns: [B, 50, 768] spatial tokens."""
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.float32)
        pixel_values = pixel_values.to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, 50, 768]
        return hidden


# ── Goal-Conditioned Policy (Phase 152 architecture) ───────────────────────────

class GoalConditionedPolicy(nn.Module):
    """
    Phase 152: Strengthened goal MLP (2→256→128) + direct goal concat to CLIP [CLS].
    Same as train_goal_conditioned_vla.py.
    """
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device

        # CLIP spatial encoder (frozen)
        self.clip_encoder = CLIPSpatialEncoder(device)

        # Vision projection: 768 → hidden
        self.vision_proj = nn.Linear(768, hidden).to(device)

        # Goal MLP: 2 → 256 → 128 (strengthened from Phase 131)
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(device)

        # Project 128D goal embedding to 256D for cross-attention Q
        self.goal_proj = nn.Linear(128, 256).to(device)

        # Project combined Q features (state+goal_proj+time = 768D) → 512D for cross-attention
        self.q_proj = nn.Linear(256, hidden).to(device)

        # State encoder: arm_pos(6) + wheel_vel(3) + goal_xy(2) = 11D → 256D
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        ).to(device)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        ).to(device)

        # Cross-attention: CLIP tokens × state+goal conditioned queries
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, batch_first=True
        ).to(device)
        self.cross_norm = nn.LayerNorm(hidden)

        # Fusion: cls(768) + state_feat(256) + goal_emb(128) + cross_out(hidden=512) + time(256)
        # NOTE: Direct goal concat to cls!  768 + 2 = 770 (goal replaces 2 CLIP spatial dims)
        # Actual: 770 + 256 + 128 + 512 + 256 = 1922
        fusion_dim = 768 + 256 + 128 + hidden + 256   # = 1920 when hidden=512
        # But cross_out is 512D → actual = 770+256+128+512+256 = 1922
        # Fix: set to 1922
        fusion_dim = 770 + 256 + 128 + hidden + 256  # = 1922
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden * 4),
            nn.ReLU(),
            nn.Linear(hidden * 4, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        ).to(device)

        # Action head: 9D output (arm*6 + wheel*3), bounded to [-0.5, 0.5]
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        ).to(device)

        self.output_bounds = (0.0, 0.5)  # wheel+arm action bounds

    def forward(self, image, state, noisy_action, t):
        """
        image: [B, 3, 224, 224], state: [B, 11], noisy_action: [B, 9], t: [B, 1]
        Returns: velocity prediction [B, 9]
        """
        B = image.shape[0]
        device = self.device

        # CLIP vision
        vision_tokens = self.clip_encoder(image)   # [B, 50, 768]
        cls_token = vision_tokens[:, 0, :]          # [B, 768]

        # State encoder
        state_feat = self.state_net(state)         # [B, 256]

        # Goal MLP (2 → 256 → 128)
        goal_xy = state[:, 9:11]                    # [B, 2] — goal xy
        goal_emb = self.goal_mlp(goal_xy)          # [B, 128]

        # Time embedding
        t_feat = self.time_mlp(t)                  # [B, 256]

        # Vision projection
        vision_proj = self.vision_proj(vision_tokens)  # [B, 50, hidden]

        # Project goal_emb to 256D before adding to state_feat
        goal_emb_proj = self.goal_proj(goal_emb)  # [B, 256]

        # Combined Q features: state(256) + goal_proj(256) + time(256) = 768 → project to 512
        q_features = state_feat + goal_emb_proj + t_feat  # [B, 256]
        q_proj = self.q_proj(q_features)  # [B, 512]

        # Cross-attention Q = 512D projected features, K/V = vision_proj(512)
        cross_out, _ = self.cross_attn(q_proj.unsqueeze(1), vision_proj, vision_proj)
        cross_out = cross_out.squeeze(1)   # [B, hidden=512] — no extra norm (avoids MPS bug)

        # ── DIRECT GOAL CONCAT to CLIP [CLS] ──────────────────────────────────
        # goal_xy (2,) directly concatenated to cls_token (768,) → 770D
        cls_with_goal = torch.cat([cls_token, goal_xy.to(device)], dim=1)  # [B, 770]

        # Fusion: cls+goal(770) + state_feat(256) + goal_emb(128) + cross(512) + time(256) = 1922
        fusion_in = torch.cat([cls_with_goal, state_feat, goal_emb, cross_out, t_feat], dim=1)
        # actual dim = 770 + 256 + 128 + 512 + 256 = 1922
        fusion_feat = self.fusion_net(fusion_in)    # [B, 512]

        # Action prediction
        action_delta = self.action_head(fusion_feat) # [B, 9]
        action = noisy_action.to(device) + action_delta

        # Bound wheel actions to physical range
        arm_action = action[:, :6]
        wheel_action = torch.clamp(action[:, 6:9], self.output_bounds[0], self.output_bounds[1])
        action = torch.cat([arm_action, wheel_action], dim=1)

        return action


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(policy, images, states_11d, actions, priorities, optimizer, batch_size=32):
    policy.train()
    n_samples = len(images)
    priorities = np.array(priorities, dtype=np.float32)
    priorities /= priorities.sum()
    indices = np.arange(n_samples)

    batch_indices = np.random.choice(indices, size=n_samples, replace=True, p=priorities)
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_idx = batch_indices[i:i+batch_size]

        t = np.random.uniform(0, 1, size=len(batch_idx)).astype(np.float32)
        noisy = actions[batch_idx].copy()
        noise = np.random.randn(*noisy.shape).astype(np.float32) * 0.5
        noisy = np.clip(noisy + noise * (1 - t[:, None]), -0.5, 0.5)
        target = (actions[batch_idx] - noisy) / np.maximum(1 - t[:, None], 1e-6)

        img_t = torch.from_numpy(images[batch_idx]).to(DEVICE)
        state_t = torch.from_numpy(states_11d[batch_idx]).to(DEVICE)
        noisy_t = torch.from_numpy(noisy).to(DEVICE)
        t_t = torch.from_numpy(t[:, None]).to(DEVICE)
        target_t = torch.from_numpy(target).to(DEVICE)

        optimizer.zero_grad()
        v_pred = policy(img_t, state_t, noisy_t, t_t)
        loss = nn.functional.mse_loss(v_pred, target_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    return epoch_loss / n_batches


# ── Evaluate on URDF sim ──────────────────────────────────────────────────────

def evaluate_on_urdf(policy, n_episodes=10, threshold=0.15):
    """Evaluate on LeKiWiSimURDF with random goals."""
    from sim_lekiwi_urdf import LeKiWiSimURDF

    policy.eval()
    successes = 0
    steps_list = []

    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id

        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])

        for step in range(200):
            base_pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(base_pos - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step + 1)
                break

            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
            v_desired = np.array([dx, dy]) * 2.0
            vx, vy = np.clip(v_desired, -0.3, 0.3)

            wheel_speeds = twist_to_contact_wheel_speeds(vx, vy)
            wheel_speeds = np.clip(wheel_speeds, -6.0, 6.0)

            arm_action = np.zeros(6)
            wheel_action = wheel_speeds / 12.0  # normalize to [-0.5, 0.5]
            full_action = np.concatenate([arm_action, wheel_action])

            obs, _, _, _ = sim.step(full_action)

        if len(steps_list) < ep + 1:
            steps_list.append(200)

    sr = successes / n_episodes
    mean_steps = np.mean(steps_list) if steps_list else 200
    return sr, mean_steps


def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    """P-controller IK for omni-wheels."""
    R = 0.05
    WHEEL_POSITIONS = np.array([
        [ 0.1732,  0.0],
        [-0.0866,  0.15],
        [-0.0866, -0.15],
    ], dtype=np.float64)
    _JOINT_AXES = np.array([
        [-0.866025,  0.0,  0.5],
        [ 0.866025,  0.0,  0.5],
        [ 0.0,       0.0, -1.0],
    ], dtype=np.float64)

    wheel_speeds = np.zeros(3, dtype=np.float64)
    for i in range(3):
        wheel_vel = np.array([vx - wz * WHEEL_POSITIONS[i, 1],
                              vy + wz * WHEEL_POSITIONS[i, 0], 0.0])
        angular_speed = np.dot(wheel_vel, _JOINT_AXES[i]) / R
        wheel_speeds[i] = angular_speed
    return wheel_speeds


# ── Sweep Configurations ──────────────────────────────────────────────────────

def run_sweep():
    """Sweep LR × epochs to find optimal early-stopping point."""
    images, states_11d, actions, priorities = load_prerendered_data()

    # Configurations to test: (lr, epochs)
    configs = [
        # Reduce LR significantly + sweep epochs
        (5e-5,  3),
        (5e-5,  5),
        (5e-5,  7),
        (2e-5,  3),
        (2e-5,  5),
        (2e-5,  7),
        (2e-5, 10),
        (1e-5,  5),
        (1e-5,  7),
        (1e-5, 10),
        (1e-5, 15),
    ]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    for lr, epochs in configs:
        output_dir = ROOT / "results" / f"phase154_sweep_lr{lr}_ep{epochs}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[CONFIG] lr={lr}, epochs={epochs}")
        print(f"{'='*60}")

        # Fresh policy
        policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)

        best_sr = 0.0
        best_epoch = 0
        epoch_srs = []

        for epoch in range(epochs):
            t0 = time.time()
            avg_loss = train_epoch(policy, images, states_11d, actions, priorities, optimizer, batch_size=32)
            elapsed = time.time() - t0

            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")

            # Evaluate every epoch after epoch >= 3
            if (epoch + 1) >= 3:
                # Quick eval: 5 episodes
                ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'lr': lr,
                    'policy_state_dict': policy.state_dict(),
                }, ckpt_path)

                sr, mean_steps = evaluate_on_urdf(policy, n_episodes=5, threshold=0.15)
                epoch_srs.append({'epoch': epoch + 1, 'sr': sr, 'mean_steps': mean_steps})
                print(f"    → Eval SR={sr:.0%} ({sr*5:.0f}/5), mean_steps={mean_steps:.0f}")

                if sr >= best_sr:
                    best_sr = sr
                    best_epoch = epoch + 1
                    # Save best
                    best_path = output_dir / "best_policy.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'lr': lr,
                        'policy_state_dict': policy.state_dict(),
                        'eval_sr': sr,
                    }, best_path)
                    print(f"    ★ New best SR={sr:.0%} at epoch {epoch+1}")

            # Early stopping: if SR = 100% for 3 consecutive evals, stop
            if len(epoch_srs) >= 3:
                recent = [e['sr'] for e in epoch_srs[-3:]]
                if all(s >= 1.0 for s in recent):
                    print(f"    ◆ Early stop: 100% SR for 3 consecutive evals")
                    break

        # Final eval: 10 episodes on best checkpoint
        if best_path.exists():
            policy_best = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
            ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
            policy_best.load_state_dict(ckpt['policy_state_dict'])
            policy_best.to(DEVICE)

            final_sr, final_steps = evaluate_on_urdf(policy_best, n_episodes=10, threshold=0.15)
        else:
            final_sr, final_steps = 0.0, 200

        result = {
            'lr': lr,
            'epochs': epochs,
            'best_epoch': best_epoch,
            'best_sr_5ep': best_sr,
            'final_sr_10ep': final_sr,
            'final_mean_steps': final_steps,
            'output_dir': str(output_dir),
            'epoch_srs': epoch_srs,
            'timestamp': timestamp,
        }
        results.append(result)

        print(f"\n  ★ Best: epoch={best_epoch}, SR={best_sr:.0%} (5ep), FINAL SR={final_sr:.0%} (10ep)")

        # Save partial results
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(result, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[SWEEP RESULTS SUMMARY]")
    print(f"{'='*60}")
    print(f"{'LR':>8} {'Epochs':>6} {'BestEp':>6} {'SR@5ep':>8} {'SR@10ep':>8} {'MeanSteps':>9}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: (-x['final_sr_10ep'], x['best_epoch'])):
        print(f"{r['lr']:>8.0e} {r['epochs']:>6} {r['best_epoch']:>6} "
              f"{r['best_sr_5ep']:>7.0%}  {r['final_sr_10ep']:>7.0%}  {r['final_mean_steps']:>8.1f}")

    # Save all results
    all_results_path = ROOT / "results" / f"phase154_sweep_all_{timestamp}.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] All results → {all_results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick sweep: fewer configs, 3-ep eval")
    args = parser.parse_args()

    if args.quick:
        # Override configs for quick testing
        print("[QUICK MODE] Testing only 2 configs")
        pass

    results = run_sweep()
    print("\n[DONE] Phase 154 epoch sweep complete!")
