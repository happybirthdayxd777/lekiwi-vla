#!/usr/bin/env python3
"""
Phase 152: Goal-Conditioned VLA with STRENGTHENED Goal MLP + Direct Goal Concatenation
=====================================================================================
Previous VLAs failed because:
1. Goal MLP (2→128→64) was too weak — 64-dim embedding can't represent spatial position
2. Goal was only available through cross-attention, not to the state MLP

Architecture improvements:
1. Goal MLP: 2 → 256 → 128 (4x larger than Phase 131's 2→128→64)
2. Goal_xy concatenated DIRECTLY to CLIP [CLS] token (not just through cross-attention)
3. state_net sees: arm_pos(6) + wheel_vel(3) + goal_xy(2) = 11D (unchanged)
4. CLIP fusion: Concat[cls(768) + state_feat(256) + goal_emb(128) + cross_out(768) + time_feat(256)] = 2176
5. Skip connection: action + skip(wheel_vel)

Training data:
- phase63_reachable_10k_converted.h5: 10k frames with pre-rendered images
- jacobian_pctrl_50ep_p143.h5: priority sampling weights (reward-weighted)

Physics: k_omni=15.0 (active locomotion mechanism)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 152] Device: {DEVICE}")


# ─── CLIP Spatial Encoder ──────────────────────────────────────────────────────

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
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # [B, 50, 768]
        return hidden


# ─── Strengthened Goal-Conditioned Policy ─────────────────────────────────────

class GoalConditionedPolicy(nn.Module):
    """
    Improvements over Phase 131 CrossAttentionPolicy:
    1. STRONGER goal MLP: 2→256→128 (4x Phase 131's 2→128→64)
    2. DIRECT goal concat: goal_xy prepended to cls token
    3. goal_emb also fed to cross-attention Q
    4. State net unchanged (11D → 256)
    5. Fusion: [cls+goal_direct(768+2=770) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
    """
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        
        # CLIP spatial encoder (frozen)
        self.clip_encoder = CLIPSpatialEncoder(device=device)
        
        # STRONGER goal MLP: 2→256→128 (was 2→128→64)
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 256), nn.SiLU(), nn.Linear(256, 128)
        )
        
        # State encoder: arm6 + wheel_vel3 + goal_xy2 = 11D → 256
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        
        # Cross-attention: goal embedding (Q) attends to CLIP tokens (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(768)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )
        
        # Fusion: [cls(768) + goal_direct(2) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
        # Note: goal_direct (2) is concatenated to cls token
        total_dim = 768 + 2 + 256 + 768 + 256  # 2050
        self.action_head = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        
        # Skip connection: wheel velocity (more stable for locomotion)
        self.skip = nn.Linear(action_dim, action_dim, bias=False)
        
        self.to(device)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] GoalConditionedPolicy: {n_params:,} params")

    def forward(self, images, state, noisy_action, timestep):
        """
        images: [B, 3, 224, 224]
        state: [B, 11] = arm_pos(6) + wheel_vel(3) + goal_xy(2)
        noisy_action: [B, 9]
        timestep: [B, 1]
        """
        # CLIP spatial tokens
        clip_tokens = self.clip_encoder(images)  # [B, 50, 768]
        
        # Goal embedding (stronger MLP)
        goal_xy = state[:, -2:]  # [B, 2]
        goal_emb = self.goal_mlp(goal_xy)  # [B, 128]
        
        # State feature
        state_feat = self.state_net(state)  # [B, 256]
        
        # Cross-attention: goal (Q) → CLIP tokens (K, V)
        goal_q = nn.Linear(128, 768, device=self.device)(goal_emb.unsqueeze(1))  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)  # [B, 1, 768]
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]
        
        # CLS token with DIRECT goal concatenation (key improvement)
        cls_token = clip_tokens[:, 0:1, :]  # [B, 1, 768]
        cls_with_goal = torch.cat([cls_token, goal_xy.unsqueeze(1)], dim=-1)  # [B, 1, 770]
        
        # Fusion: [cls+goal(770) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
        t_feat = self.time_mlp(timestep)  # [B, 256]
        combined = torch.cat([
            cls_with_goal,           # [B, 1, 770]
            state_feat.unsqueeze(1), # [B, 1, 256]
            cross_out,               # [B, 1, 768]
            t_feat.unsqueeze(1),     # [B, 1, 256]
        ], dim=-1)  # [B, 1, 2050]
        
        v_pred = self.action_head(combined).squeeze(1)  # [B, 9]
        
        # Skip connection: action + skip(wheel_vel)
        # The skip carries the current wheel velocity, reducing learning burden
        return v_pred + self.skip(noisy_action)

    def infer(self, images, state, num_steps=4):
        """4-step Euler flow matching inference."""
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * 0.5
            v = self.forward(images, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_prerendered_data(h5_path="data/phase63_reachable_10k_converted.h5",
                           jacobian_h5="data/jacobian_pctrl_50ep_p143.h5"):
    """Load pre-rendered data for fast training.
    
    Primary source: phase63_reachable_10k_converted.h5
      - Images: [N, 224, 224, 3] uint8 (pre-rendered)
      - States: [N, 9] (arm6 + wheel3)
      - Actions: [N, 9] (arm6 + wheel3) — wheels clipped to [-0.5, 0.5]
      - Goals: [N, 2]
    
    Returns: images (N,3,224,224), states_11d (N,11), actions (N,9), priorities (N,)
    """
    print(f"[DATA] Loading pre-rendered data from {h5_path}...")
    f = h5py.File(h5_path, 'r')
    actions = f['actions'][:].astype(np.float32)      # (10000, 9)
    states_9d = f['states'][:].astype(np.float32)    # (10000, 9)
    goals = f['goal_positions'][:].astype(np.float32) # (10000, 2)
    images_raw = f['images'][:]                        # (10000, 224, 224, 3) uint8
    f.close()
    
    # Load rewards for priority sampling
    jac_f = h5py.File(jacobian_h5, 'r')
    jac_rewards = jac_f['rewards'][:]
    jac_f.close()
    
    # Normalize images: uint8 [0,255] → float32 [0,1] CHW
    images = images_raw.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    
    # Build 11D state: 9D (arm6 + wheel3) + 2D goal
    states_11d = np.concatenate([states_9d, goals], axis=1)
    
    # Priority sampling: upweight high-reward frames
    jac_min, jac_max = jac_rewards.min(), jac_rewards.max()
    if jac_max > jac_min:
        priorities = (jac_rewards - jac_min) / (jac_max - jac_min)
    else:
        priorities = np.ones_like(jac_rewards)
    
    print(f"  Actions: {actions.shape}, range=[{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  States: {states_11d.shape}, range=[{states_11d.min():.3f}, {states_11d.max():.3f}]")
    print(f"  Images: {images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
    print(f"  Goals: {goals.shape}, range=[{goals.min():.3f}, {goals.max():.3f}]")
    print(f"  Priority weights: mean={priorities.mean():.3f}, max={priorities.max():.3f}")
    
    return images, states_11d, actions, priorities


# ─── Normalize Action ──────────────────────────────────────────────────────────

def normalize_action(raw_action):
    """Policy output (bounded [-0.5, 0.5]) → sim native units.
    
    The policy infers velocity increments. After denoising, we get
    action values in [-0.5, 0.5]. These are ALREADY in native units
    (wheel speeds clipped to [-0.5, 0.5] during data collection).
    """
    raw = np.asarray(raw_action, dtype=np.float32)
    return np.clip(raw, -0.5, 0.5)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pre-rendered data (images loaded from disk, no sim rendering needed!)
    images, states_11d, actions, priorities = load_prerendered_data(
        h5_path="data/phase63_reachable_10k_converted.h5",
        jacobian_h5="data/jacobian_pctrl_50ep_p143.h5"
    )
    
    # Policy with STRENGTHENED goal conditioning
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    
    n_samples = len(images)
    # Priority-weighted sampling indices
    priorities = np.array(priorities, dtype=np.float32)
    priorities /= priorities.sum()
    indices = np.arange(n_samples)
    
    losses = []
    print(f"\n[TRAIN] Starting {args.epochs} epochs on {n_samples} frames (priority-weighted sampling)...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Priority-weighted shuffle
        batch_indices = np.random.choice(indices, size=n_samples, replace=True, p=priorities)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, args.batch):
            batch_idx = batch_indices[i:i+args.batch]
            
            # Sample timestep t ~ Uniform[0,1]
            t = np.random.uniform(0, 1, size=len(batch_idx)).astype(np.float32)
            
            # Sample random actions as noisy actions (Euler flow matching)
            noisy = actions[batch_idx].copy()
            noise = np.random.randn(*noisy.shape).astype(np.float32) * 0.5
            noisy = np.clip(noisy + noise * (1 - t[:, None]), -0.5, 0.5)
            
            # Target velocity
            target = (actions[batch_idx] - noisy) / np.maximum(1 - t[:, None], 1e-6)
            
            # To tensor
            img_t = torch.from_numpy(images[batch_idx]).to(DEVICE)
            state_t = torch.from_numpy(states_11d[batch_idx]).to(DEVICE)
            noisy_t = torch.from_numpy(noisy).to(DEVICE)
            t_t = torch.from_numpy(t[:, None]).to(DEVICE)
            target_t = torch.from_numpy(target).to(DEVICE)
            
            # Forward
            optimizer.zero_grad()
            v_pred = policy(img_t, state_t, noisy_t, t_t)
            loss = nn.functional.mse_loss(v_pred, target_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")
        
        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Saved {ckpt_path}")
    
    # Final save
    final_path = output_dir / "final_policy.pt"
    torch.save({
        'epoch': args.epochs,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"\n[DONE] Final policy saved to {final_path}")
    
    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Phase 152: GoalConditioned VLA — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()
    
    return policy, losses


# ─── Evaluation on URDF ───────────────────────────────────────────────────────

def twist_to_contact_wheel_speeds(vx, vy, wz=0.0):
    """P-controller: desired velocity → wheel angular velocities.
    
    From omni_controller.py kinematics.
    """
    R = 0.05  # wheel radius
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


def evaluate(policy_path, n_episodes=10, threshold=0.15):
    """Evaluate on URDF sim with CORRECTED qvel[6:9] (Phase 151 fix)."""
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    # Load policy
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=DEVICE)
    ckpt = torch.load(policy_path, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt['flow_head_state_dict'], strict=False)
    policy.to(DEVICE)
    policy.eval()
    print(f"[EVAL] Loaded policy from {policy_path}")
    
    # P-controller baseline
    print(f"\n[EVAL] P-controller baseline...")
    p_successes = 0
    p_steps_list = []
    
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
                p_successes += 1
                p_steps_list.append(step + 1)
                print(f"  P-ep{ep}: SUCCESS step={step}")
                break
            
            dx, dy = goal[0] - base_pos[0], goal[1] - base_pos[1]
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
            p_steps_list.append(200)
            print(f"  P-ep{ep}: FAIL final_dist={dist:.3f}")
    
    p_sr = p_successes / n_episodes
    p_mean_steps = np.mean(p_steps_list)
    print(f"\n  P-controller: {p_successes}/{n_episodes} = {100*p_sr:.0f}% SR, mean_steps={p_mean_steps:.0f}")
    
    # VLA policy
    print(f"\n[EVAL] VLA policy...")
    v_successes = 0
    v_dists = []
    v_steps_list = []
    
    for ep in range(n_episodes):
        sim = LeKiWiSimURDF()
        sim.reset()
        base_id = sim.model.body('base').id
        
        gx, gy = np.random.uniform(-0.5, 0.5, 2)
        goal = np.array([gx, gy])
        
        for step in range(200):
            # Render
            img_np = sim.render()
            from PIL import Image
            img_pil = Image.fromarray(img_np)
            img_small = np.array(img_pil.resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # State: arm_pos(6) + wheel_vel(3) + goal_xy(2) = 11D
            # FIX (Phase 151): qvel[6:9] = WHEEL angular velocities (not base velocity)
            arm_pos = sim.data.qpos[7:13]
            wheel_vel = sim.data.qvel[6:9]  # qvel[6:9] = w1, w2, w3
            state_11d = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_11d).float().unsqueeze(0).to(DEVICE)
            
            # Policy inference
            with torch.no_grad():
                raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()
            
            wheel_speeds = normalize_action(raw_action[6:9])
            action = np.zeros(9)
            action[6:9] = wheel_speeds
            sim.step(action)
            
            # Check
            dist = np.linalg.norm(sim.data.xpos[base_id, :2] - goal)
            if dist < threshold:
                v_successes += 1
                v_steps_list.append(step + 1)
                print(f"  VLA-ep{ep}: SUCCESS step={step}")
                break
        else:
            v_dists.append(dist)
            v_steps_list.append(200)
            print(f"  VLA-ep{ep}: FAIL final_dist={dist:.3f}")
    
    v_sr = v_successes / n_episodes
    v_mean_dist = np.mean(v_dists) if v_dists else 0.0
    v_mean_steps = np.mean(v_steps_list)
    print(f"\n  VLA policy: {v_successes}/{n_episodes} = {100*v_sr:.0f}% SR, mean_dist={v_mean_dist:.3f}, mean_steps={v_mean_steps:.0f}")
    
    # Save results
    results = {
        "phase": 152,
        "architecture": "goal_conditioned_vla_strengthened",
        "pcontroller_sr": float(p_sr),
        "pcontroller_mean_steps": float(p_mean_steps),
        "vla_sr": float(v_sr),
        "vla_mean_dist": float(v_mean_dist),
        "vla_mean_steps": float(v_mean_steps),
        "n_episodes": n_episodes,
        "threshold": threshold,
    }
    
    out_path = Path(args.output) / "eval_results.json"
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EVAL] Results saved to {out_path}")
    
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="results/phase152_goal_conditioned")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="Path to policy.pt to eval only (skip training)")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate(args.eval_only, n_episodes=args.n_episodes, threshold=args.threshold)
    else:
        policy, losses = train(args)
        print(f"\n[DONE] Training complete. Evaluating final policy...")
        evaluate(Path(args.output) / "final_policy.pt",
                 n_episodes=args.n_episodes, threshold=args.threshold)