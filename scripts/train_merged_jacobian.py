#!/usr/bin/env python3
"""
Phase 170: Train on MERGED Data — phase63 images + CORRECTED jacobian kP=0.1 actions
===================================================================================

KEY INSIGHT from Phase 167/169:
- Previous jacobian_pctrl_50ep_p143.h5 used kP=1.5 (WRONG - causes IK saturation)
- jacobian_pctrl_50ep_kP01.h5 uses CORRECT kP=0.1 (matches eval P-controller)
- jacobian_kP01 collected with: kP=0.1, max_speed=0.25, no wheel clip
- Result: 30% SR (URDF physics limit), but CORRECT action directions

Previous Phase 158 used kP=1.5 data → VLA 10-30% SR (learned wrong actions)
This Phase 170 uses kP=0.1 data → VLA should approach P-ctrl baseline (40% SR)

Data alignment strategy:
- phase63: has images [N=10000, 224×224×3]
- jacobian_kP01: has CORRECT actions [N=10000, 9]
- Align by finding matching goal positions
- Result: phase63 images + jacobian_kP01 actions (CORRECT directions)

Architecture: GoalConditionedPolicy (same as Phase 154)
Loss: Flow matching MSE
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 158] Device: {DEVICE}")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_merged_data():
    """Load phase63 images + jacobian actions (aligned by episode).

    Returns:
        images:    [N, 3, 224, 224] float32 — from phase63
        states_11d: [N, 11] float32 — arm6 + wheel3 + goal2
        actions:  [N, 9] float32 — BEST actions (jacobian where matched)
        priorities: [N] float32 — reward-based sampling weights
        match_mask: [N] bool — True where jacobian action was used

    Phase 171: Uses jacobian_pctrl_100ep_kP01.h5 (100 episodes, v1+v2 combined).
    This gives 2x more correct kP=0.1 P-controller actions for alignment.
    """
    # Phase 171: Use combined 100-episode dataset (v1+v2) for 2x more correct kP=0.1 data
    f63 = h5py.File(ROOT / "data/phase63_reachable_10k_converted.h5", "r")
    fj  = h5py.File(ROOT / "data/jacobian_pctrl_100ep_kP01.h5", "r")
    
    # Load phase63
    images_raw = f63["images"][:]          # [N, 224, 224, 3] uint8
    states_9d  = f63["states"][:]          # [N, 9] arm6 + wheel_pos3
    actions_63 = f63["actions"][:]         # [N, 9] — GridSearch actions (lower quality)
    goals_63   = f63["goal_positions"][:]  # [N, 2]
    rewards_63 = f63["rewards"][:]         # [N]
    f63.close()
    
    # Load jacobian
    actions_j  = fj["actions"][:]          # [N_jac, 9] — Jacobian P-ctrl actions (BETTER)
    rewards_j  = fj["rewards"][:]           # [N_jac]
    goals_j    = fj["goal_positions"][:]   # [N_jac, 2]
    ep_starts_j_raw = list(fj["episode_starts"][:])  # last element may be end marker
    N_jac = len(actions_j)
    fj.close()
    
    # Remove trailing end-marker from jacobian episode starts (if last == N_jac)
    if ep_starts_j_raw[-1] == N_jac:
        ep_starts_j = ep_starts_j_raw[:-1]
    else:
        ep_starts_j = ep_starts_j_raw
    
    # ── Find matching episodes between datasets ──────────────────────────────
    
    N = len(images_raw)  # 10000
    
    # Build merged actions: start with phase63, override with jacobian where confident
    merged_actions = actions_63.copy()
    match_mask = np.zeros(N, dtype=bool)
    
    # Find episode boundaries in phase63
    # Episode start: arm_pos[0] ≈ 0 (robot at rest)
    ep_starts_63 = [0]
    for i in range(1, N):
        if np.abs(states_9d[i, 0]) < 0.01 and np.abs(states_9d[i-1, 0]) > 0.05:
            ep_starts_63.append(i)
    ep_starts_63 = np.array(ep_starts_63)
    
    print(f"[DATA] phase63 episodes: {len(ep_starts_63)}, jacobian episodes: {len(ep_starts_j)}")
    
    # Match episodes by goal position
    n_matched = 0
    # rewards_j_aligned: rewards aligned to phase63 index space (for priority merge)
    rewards_j_aligned = rewards_63.copy()  # default to phase63 rewards
    
    for ji, j_start in enumerate(ep_starts_j):
        j_end = ep_starts_j[ji+1] if ji+1 < len(ep_starts_j) else N_jac
        j_goal = goals_j[j_start]  # goal for this episode
        
        # Find phase63 episode with same goal
        best_match = -1
        best_diff = float('inf')
        
        for pi, p_start in enumerate(ep_starts_63):
            p_end = ep_starts_63[pi+1] if pi+1 < len(ep_starts_63) else N
            if p_end - p_start < 50:  # skip tiny episodes
                continue
            p_goal = goals_63[p_start]
            diff = np.linalg.norm(j_goal - p_goal)
            if diff < best_diff:
                best_diff = diff
                best_match = pi
        
        if best_match >= 0 and best_diff < 0.15:  # goal match threshold
            p_start = ep_starts_63[best_match]
            p_end = ep_starts_63[best_match+1] if best_match+1 < len(ep_starts_63) else N
            n_frames = min(j_end - j_start, p_end - p_start)
            
            if n_frames > 50:  # only merge substantial episodes
                # Use jacobian actions for this episode
                # Index by p_start (phase63 index) since merged_actions is 10000-sized (phase63)
                merged_actions[p_start:p_start+n_frames] = actions_j[j_start:j_start+n_frames]
                match_mask[p_start:p_start+n_frames] = True
                # Align jacobian rewards to phase63 index space
                rewards_j_aligned[p_start:p_start+n_frames] = rewards_j[j_start:j_start+n_frames]
                n_matched += 1
    
    print(f"[DATA] Matched {n_matched} episodes ({match_mask.sum()} frames)")
    print(f"[DATA] Using jacobian actions: {match_mask.sum()} frames")
    print(f"[DATA] Keeping phase63 actions: {(~match_mask).sum()} frames")
    
    # ── Transpose images: [N, 224, 224, 3] → [N, 3, 224, 224] ───────────────
    images = images_raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    
    # ── Build 11D state: arm_pos(6) + wheel_pos(3) + goal_xy(2) ────────────
    states_11d = np.zeros((N, 11), dtype=np.float32)
    states_11d[:, :6]  = states_9d[:, :6]    # arm positions
    states_11d[:, 6:9]  = states_9d[:, 6:9]   # wheel positions
    states_11d[:, 9:11] = goals_63            # goal xy
    
    # ── Priority: reward-weighted sampling (prefer high-reward frames) ────────
    # Use jacobian rewards where available (aligned to phase63 index space)
    rewards_for_priority = rewards_j_aligned  # already aligned to phase63 index space
    priorities = np.maximum(rewards_for_priority, 0.0).astype(np.float32)
    priorities += 1e-6
    priorities /= priorities.sum()
    
    # ── Normalize actions to [-0.5, 0.5] (matching inference range) ──────────────
    # phase63 wheel actions: [-1, 1], jacobian wheel actions: [-0.5, 0.5]
    # For training consistency: scale ALL actions to [-0.5, 0.5]
    # Use arm actions from data as-is (they're already in reasonable range)
    # Scale wheel actions: wheel part is [:3] and [3:6] for arm, [6:9] for wheels
    
    arm_actions = merged_actions[:, :6]  # arm actions (already bounded)
    wheel_actions = merged_actions[:, 6:9]  # wheel actions (need scaling)
    
    # Scale wheel actions from [-1, 1] → [-0.5, 0.5]
    # If already in [-0.5, 0.5], just clip
    wheel_scaled = np.clip(wheel_actions, -0.5, 0.5)
    
    merged_actions_norm = np.concatenate([arm_actions, wheel_scaled], axis=1)
    
    print(f"[DATA] Loaded {N} frames")
    print(f"[DATA] Reward stats: mean={rewards_for_priority.mean():.3f}, max={rewards_for_priority.max():.3f}")
    print(f"[DATA] Image shape: {images.shape}")
    print(f"[DATA] Action range (normalized): [{merged_actions_norm.min():.3f}, {merged_actions_norm.max():.3f}]")
    
    return images, states_11d, merged_actions_norm, priorities, match_mask


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


# ── GoalConditionedPolicy (Phase 152/154 architecture) ─────────────────────────

class GoalConditionedPolicy(nn.Module):
    """Phase 152: Same architecture as sweep_epochs_lr.py / eval_matched_goals.py."""
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
            nn.ReLU(),
        ).to(device)

        # Cross-attention: query from state+goal, key/value from vision
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)

        # Time embedding
        self.time_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        ).to(device)

        # Output: state(256) + goal_proj(128) + cross(512) + time(256) = 1152 → action
        self.action_head = nn.Sequential(
            nn.Linear(256 + 128 + hidden + 256, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        ).to(device)

        # Skip connection for action
        self.skip = nn.Linear(action_dim, action_dim, bias=False).to(device)

        self.to(device)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] GoalConditionedPolicy: {n_params:,} params")

    def forward(self, image, state, noisy_action, t):
        """Flow matching forward pass.
        
        Args:
            image:        [B, 3, 224, 224] — RGB image
            state:        [B, 11] — arm_pos(6) + wheel_pos(3) + goal(2)
            noisy_action: [B, 9] — noisy action at timestep t
            t:            [B, 1] — timestep in [0, 1]
        Returns:
            v_pred: [B, 9] — predicted velocity to denoise noisy_action
        """
        # Encode image: [B, 50, 768]
        clip_feat = self.clip_encoder(image)
        
        # Project to hidden dim: [B, 50, 512]
        clip_proj = self.vision_proj(clip_feat)
        
        # Goal embedding: [B, 128]
        goal_emb = self.goal_mlp(state[:, 9:11])  # goal xy
        goal_q = self.goal_proj(goal_emb)           # [B, 256]
        
        # State encoding: [B, 256]
        state_feat = self.state_net(state)          # [B, 256]
        
        # Query = state + goal projection: [B, 256]
        q = self.q_proj(state_feat + goal_q)       # [B, 512]
        q = q.unsqueeze(1)                          # [B, 1, 512]
        
        # Cross-attention with CLIP visual features
        cross_out, _ = self.cross_attn(q, clip_proj, clip_proj)
        cross_out = self.cross_norm(cross_out + q)   # [B, 1, 512]
        
        # Time embedding: [B, 256]
        t_feat = self.time_net(t)                   # [B, 256]
        
        # Concatenate: state(256) + goal(128) + cross(512) + time(256) = 1152
        combined = torch.cat([
            state_feat,          # [B, 256]
            goal_emb,            # [B, 128]
            cross_out.squeeze(1), # [B, 512]
            t_feat,              # [B, 256]
        ], dim=-1)                                   # [B, 1152]
        
        # Predict velocity
        v_pred = self.action_head(combined)         # [B, 9]
        v_pred = v_pred + self.skip(noisy_action)  # residual connection
        
        return v_pred

    def infer(self, image, state, num_steps=4):
        """Denoise from pure noise to action in num_steps."""
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)  # start from zero noise
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * (i * dt)
            v = self.forward(image, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(policy, images, states_11d, actions, priorities, optimizer, batch_size=32):
    policy.train()
    n_samples = len(images)
    
    # Priority-weighted sampling with replacement
    idx = np.random.choice(n_samples, size=n_samples, replace=True, p=priorities)
    
    epoch_loss = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        batch_idx = idx[i:i+batch_size]
        B = len(batch_idx)
        
        # Sample timestep t ~ Uniform[0, 1]
        t = np.random.uniform(0, 1, size=B).astype(np.float32)
        
        # Create noisy action: action_t = (1 - t) * action_0 + t * noise
        # Here we use flow matching: noisy = action + sigma * noise, target = -noise/sigma
        noise = np.random.randn(B, 9).astype(np.float32) * 0.5
        noisy = np.clip(actions[batch_idx] + noise * (1 - t[:, None]), -0.5, 0.5)
        
        # Target: velocity to correct noise (flow matching target)
        target = (actions[batch_idx] - noisy) / np.maximum(1 - t[:, None], 1e-6)
        
        # To tensor
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


# ── Quick Eval ─────────────────────────────────────────────────────────────────

def evaluate_on_urdf(policy, n_episodes=10, threshold=0.15):
    """Evaluate policy on URDF sim (same as eval_matched_goals.py)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    successes = 0
    steps_list = []
    
    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id
        
        # Random goal
        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])
        
        for step in range(200):
            # Render
            img_np = sim.render()
            img_pil = __import__('PIL').Image.fromarray(img_np)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # State: arm_pos(6) + wheel_vel(3) + goal_xy(2)
            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[6:9]
            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)
            
            # Policy inference
            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()
            
            # Apply action (wheel only for now)
            wheel_speeds = np.clip(raw_action[6:9], -0.5, 0.5)
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)
            
            # Check success
            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < threshold:
                successes += 1
                steps_list.append(step)
                break
        else:
            steps_list.append(200)
    
    sr = successes / n_episodes
    mean_steps = np.mean(steps_list)
    return sr, mean_steps


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)  # Known best from Phase 154
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=3)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output) if args.output else \
        ROOT / "results" / f"phase158_merged_jacobian_lr{args.lr}_ep{args.epochs}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[Phase 158] Merged Jacobian Training")
    print(f"  epochs={args.epochs}, lr={args.lr}, batch={args.batch}")
    print(f"  output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load merged data
    images, states_11d, actions, priorities, match_mask = load_merged_data()
    
    # Policy
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    
    n_samples = len(images)
    losses = []
    best_sr = 0.0
    best_epoch = 0
    
    print(f"\n[TRAIN] Starting {args.epochs} epochs on {n_samples} frames...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = train_epoch(policy, images, states_11d, actions, priorities, optimizer, 
                               batch_size=args.batch)
        elapsed = time.time() - t0
        losses.append(avg_loss)
        
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")
        
        # Eval every `eval_every` epochs (starting from epoch 3)
        if (epoch + 1) >= 3 and (epoch + 1) % args.eval_every == 0:
            ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'lr': args.lr,
                'policy_state_dict': policy.state_dict(),
            }, ckpt_path)
            
            sr, mean_steps = evaluate_on_urdf(policy, n_episodes=args.n_eval, threshold=0.15)
            print(f"    → Eval SR={sr:.0%} ({sr*args.n_eval:.0f}/{args.n_eval}), mean_steps={mean_steps:.0f}")
            
            if sr >= best_sr:
                best_sr = sr
                best_epoch = epoch + 1
                best_path = output_dir / "best_policy.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'lr': args.lr,
                    'policy_state_dict': policy.state_dict(),
                    'eval_sr': sr,
                }, best_path)
                print(f"    ★ New best SR={sr:.0%} at epoch {epoch+1}")
    
    total_time = time.time() - start_time
    print(f"\n[DONE] Training complete in {total_time:.0f}s")
    print(f"  Best SR: {best_sr:.0%} at epoch {best_epoch}")
    
    # Save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title(f"Phase 158 Training Loss (merged jacobian data)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(output_dir / "loss_curve.png", dpi=100)
    plt.close()
    
    # Final eval: 30 episodes
    if best_epoch > 0:
        print(f"\n[FINAL EVAL] 30 episodes on best policy...")
        policy_best = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
        ckpt = torch.load(output_dir / "best_policy.pt", map_location=DEVICE, weights_only=False)
        policy_best.load_state_dict(ckpt['policy_state_dict'])
        policy_best.to(DEVICE)
        
        final_sr, final_steps = evaluate_on_urdf(policy_best, n_episodes=30, threshold=0.15)
        print(f"  Final SR: {final_sr:.0%} (30ep), mean_steps={final_steps:.0f}")
        
        # Save results
        results = {
            "phase": 158,
            "merged_data": True,
            "jacobian_frames": int(match_mask.sum()),
            "phase63_frames": int((~match_mask).sum()),
            "lr": args.lr,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_sr_10ep": best_sr,
            "final_sr_30ep": final_sr,
            "final_mean_steps": float(final_steps),
            "total_time_s": total_time,
            "output_dir": str(output_dir),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    print(f"\n[DONE] Results saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
