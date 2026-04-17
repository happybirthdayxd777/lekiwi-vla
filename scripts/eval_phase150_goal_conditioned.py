#!/usr/bin/env python3
"""
Phase 153: Evaluate Phase 150 GoalConditionedPolicy (30 epochs) on URDF sim
==========================================================================
Phase 152 only trained 5 epochs → VLA=20% vs P-ctrl=30% (unlucky goal sampling)
Phase 150 has 30-epoch checkpoint → need to evaluate if extended training helps.

Architecture (GoalConditionedPolicy, same as Phase 152):
  - CLIP spatial tokens [B, 50, 768]
  - Goal MLP: 2→256→128 (strengthened)
  - State net: 11→256
  - Cross-attention: goal (Q) → CLIP tokens
  - Fusion: [cls+goal_direct(770) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
  - Action head: 2050→512→512→9
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 153] Device: {DEVICE}")


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


# ─── GoalConditionedPolicy (matching train_goal_conditioned_vla.py) ─────────

class GoalConditionedPolicy(nn.Module):
    """
    Phase 152: Strengthened Goal MLP + Direct Goal Concatenation.
    Architecture:
      - CLIP spatial: [B, 50, 768]
      - Goal MLP: 2→256→128
      - State net: 11→256
      - Cross-attention: goal (Q) → CLIP tokens (K,V)
      - cls+goal_direct(770) + state_feat(256) + cross_out(768) + time_feat(256) = 2050
      - Action head: 2050→512→512→9
    """
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        
        self.clip_encoder = CLIPSpatialEncoder(device=device)
        
        # STRONGER goal MLP: 2→256→128
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 256), nn.SiLU(), nn.Linear(256, 128)
        )
        
        # State encoder: arm6 + wheel_vel3 + goal_xy2 = 11D → 256
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(768)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256)
        )
        
        # Fusion: cls+goal_direct(770) + state_feat(256) + cross_out(768) + time_feat(256) = 2050
        total_dim = 768 + 2 + 256 + 768 + 256  # 2050
        self.action_head = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        
        # Skip connection: wheel velocity
        self.skip = nn.Linear(action_dim, action_dim, bias=False)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Policy] GoalConditionedPolicy: {n_params:,} params")
        self.to(device)

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.clip_encoder(images)  # [B, 50, 768]
        
        goal_xy = state[:, -2:]  # [B, 2]
        goal_emb = self.goal_mlp(goal_xy)        # [B, 128]
        
        state_feat = self.state_net(state)         # [B, 256]
        
        # Cross-attention: goal (Q) attends to CLIP tokens (K,V)
        goal_q = nn.Linear(128, 768, device=self.device)(goal_emb.unsqueeze(1))  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)  # [B, 1, 768]
        
        # CLS token + goal_direct (concatenated, not summed)
        cls_token = clip_tokens[:, 0:1, :]  # [B, 1, 768]
        cls_with_goal = torch.cat([cls_token, goal_xy.unsqueeze(1)], dim=-1)  # [B, 1, 770]
        
        # Time embedding
        t = timestep.float().unsqueeze(1) / 4.0  # normalized [0,1]
        time_feat = self.time_mlp(t)             # [B, 256]
        
        # Fusion: [cls+goal(770) + state_feat(256) + cross_out(768) + time_feat(256)] = 2050
        combined = torch.cat([
            cls_with_goal,          # [B, 1, 770]
            state_feat.unsqueeze(1),  # [B, 1, 256]
            cross_out,              # [B, 1, 768]
            time_feat.unsqueeze(1),   # [B, 1, 256]
        ], dim=-1).squeeze(1)  # [B, 2050]
        
        action = self.action_head(combined)
        # Skip connection: wheel velocity as bias
        wheel_vel = state[:, 6:9]  # [B, 3]
        wheel_skip = self.skip(wheel_vel)
        action[:, :3] = action[:, :3] + wheel_skip[:, :3]
        
        return action

    @torch.no_grad()
    def infer(self, images, state, num_steps=4):
        """Flow matching inference with 4-step Euler denoising."""
        noisy_action = torch.randn(images.size(0), 9, device=self.device)
        for step in range(num_steps):
            t = torch.full((images.size(0), 1), step / num_steps, device=self.device)
            pred = self.forward(images, state, noisy_action, t)
            if step < num_steps - 1:
                noisy_action = noisy_action - (1.0 / num_steps) * pred
            else:
                noisy_action = pred
        return noisy_action


# ─── Normalize Action (from Phase 128 fix) ───────────────────────────────────

LEKIWI_WHEEL_LIMITS = [[-0.5, 0.5]] * 3
LEKIWI_ARM_LIMITS = [[-1.57, 1.57]] * 6

def normalize_action(action):
    """Clip + normalize action to [-1, 1] for flow matching."""
    wheel_action = action[:, :3]
    arm_action = action[:, 3:]
    wheel_clipped = np.clip(wheel_action, -0.5, 0.5)
    arm_clipped = np.clip(arm_action, -1.57, 1.57)
    wheel_norm = (wheel_clipped - (-0.5)) / (0.5 - (-0.5)) * 2 - 1
    arm_norm = (arm_clipped - (-1.57)) / (1.57 - (-1.57)) * 2 - 1
    return np.concatenate([wheel_norm, arm_norm], axis=1)

def denormalize_action(norm_action):
    """Denormalize from [-1,1] to actual actuator values."""
    wheel_norm = norm_action[:, :3]
    arm_norm = norm_action[:, 3:]
    wheel = (wheel_norm + 1) / 2 * (0.5 - (-0.5)) + (-0.5)
    arm = (arm_norm + 1) / 2 * (1.57 - (-1.57)) + (-1.57)
    return np.concatenate([wheel, arm], axis=1)


# ─── URDF Sim + P-controller baseline ─────────────────────────────────────────

def make_urdf_sim():
    """Create LeKiWiSimURDF with k_omni=15.0 (same as training)."""
    import sim_lekiwi_urdf
    sim = sim_lekiwi_urdf.LeKiWiSimURDF()
    sim._k_omni = 15.0  # Match training physics
    return sim


def p_controller_goalDirected(state, goal, kp=3.0):
    """Jacobian P-controller (oracle, same as training data collector)."""
    vx, vy = goal[0] - state[0], goal[1] - state[1]
    dist = np.sqrt(vx**2 + vy**2)
    if dist < 0.15:
        return np.zeros(9)
    vx_clipped = np.clip(vx, -0.3, 0.3)
    vy_clipped = np.clip(vy, -0.3, 0.3)
    vz_angular = 0.0
    wheel_speeds = sim_lekiwi_urdf.twist_to_contact_wheel_speeds(vx_clipped, vy_clipped, vz_angular)
    wheel_speeds = np.clip(wheel_speeds, -0.5, 0.5)
    return np.concatenate([np.zeros(6), wheel_speeds])


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_policy(policy, sim, n_episodes=20, max_steps=200, threshold=0.15, seed=42):
    """Evaluate VLA policy vs P-controller baseline on URDF sim."""
    np.random.seed(seed)
    policy.eval()
    
    vla_successes = 0
    vla_dists = []
    vla_steps_list = []
    
    pctrl_successes = 0
    pctrl_dists = []
    pctrl_steps_list = []
    
    for ep in range(n_episodes):
        # Random goal in reachable area
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.2, 0.8)
        goal = np.array([np.cos(angle) * radius, np.sin(angle) * radius])
        
        # Evaluate VLA
        sim.reset()
        obs = sim.observe()
        success = False
        for step in range(max_steps):
            # Image
            img = sim.render(height=224, width=224)
            if img is None or img.size == 0:
                img = np.zeros((224, 224, 3), dtype=np.float32)
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
            
            # State: arm_pos(6) + wheel_vel(3) + goal_xy(2) = 11D
            arm_pos = obs['arm_qpos'][:6]
            wheel_vel = obs['qvel'][6:9]
            state_vec = np.concatenate([arm_pos, wheel_vel, goal])
            state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(DEVICE)
            
            # Policy inference
            raw_action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy().squeeze()
            norm_action = normalize_action(raw_action.reshape(1, 9))
            action = denormalize_action(norm_action).squeeze()
            
            # Step
            obs = sim.step(action)
            
            # Check success
            base_pos = obs['base_position'][:2]
            dist = np.linalg.norm(base_pos - goal)
            if dist < threshold:
                success = True
                break
        
        vla_successes += int(success)
        vla_dists.append(dist if success else np.linalg.norm(obs['base_position'][:2] - goal))
        vla_steps_list.append(step if success else max_steps)
        
        # Evaluate P-controller (reset sim)
        sim.reset()
        obs = sim.observe()
        success = False
        for step in range(max_steps):
            arm_pos = obs['arm_qpos'][:6]
            wheel_vel = obs['qvel'][6:9]
            state_vec = np.concatenate([arm_pos, wheel_vel, goal])
            action = p_controller_goalDirected(state_vec, goal)
            obs = sim.step(action)
            
            base_pos = obs['base_position'][:2]
            dist = np.linalg.norm(base_pos - goal)
            if dist < threshold:
                success = True
                break
        
        pctrl_successes += int(success)
        pctrl_dists.append(dist if success else np.linalg.norm(obs['base_position'][:2] - goal))
        pctrl_steps_list.append(step if success else max_steps)
    
    vla_sr = vla_successes / n_episodes
    pctrl_sr = pctrl_successes / n_episodes
    
    print(f"\n{'='*60}")
    print(f"Phase 153 EVALUATION — GoalConditionedPolicy (30 epochs)")
    print(f"{'='*60}")
    print(f"VLA policy:       {vla_successes:2d}/{n_episodes} = {vla_sr*100:.0f}% SR, mean_dist={np.mean(vla_dists):.3f}m, mean_steps={np.mean(vla_steps_list):.1f}")
    print(f"P-controller:     {pctrl_successes:2d}/{n_episodes} = {pctrl_sr*100:.0f}% SR, mean_dist={np.mean(pctrl_dists):.3f}m, mean_steps={np.mean(pctrl_steps_list):.1f}")
    print(f"VLA gap:          {(pctrl_sr - vla_sr)*100:.0f}%-points below baseline")
    print(f"{'='*60}")
    
    return {
        'vla_sr': vla_sr, 'vla_mean_dist': float(np.mean(vla_dists)),
        'vla_mean_steps': float(np.mean(vla_steps_list)),
        'pctrl_sr': pctrl_sr, 'pctrl_mean_dist': float(np.mean(pctrl_dists)),
        'pctrl_mean_steps': float(np.mean(pctrl_steps_list)),
        'n_episodes': n_episodes, 'threshold': threshold,
        'phase': 153, 'architecture': 'goal_conditioned_30epoch',
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='results/phase150_train/checkpoint_epoch_30.pt')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()
    
    import sim_lekiwi_urdf
    
    print(f"\n[1] Loading GoalConditionedPolicy from {args.ckpt}...")
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9, device=args.device)
    
    ckpt = torch.load(args.ckpt, map_location=args.device)
    loaded_state = ckpt.get('policy_state_dict', ckpt)
    missing, unexpected = policy.load_state_dict(loaded_state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} keys missing from checkpoint")
    policy.to(args.device)
    policy.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', 'unknown')} checkpoint")
    
    print(f"\n[2] Creating URDF sim (k_omni=15.0 to match training)...")
    sim = make_urdf_sim()
    
    print(f"\n[3] Evaluating VLA vs P-controller ({args.episodes} episodes)...")
    results = evaluate_policy(policy, sim, n_episodes=args.episodes, 
                              max_steps=args.max_steps, threshold=args.threshold, seed=args.seed)
    
    # Save results
    import json
    results_path = Path('results/phase153_goal_conditioned_30epoch')
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[4] Results saved to {results_path / 'eval_results.json'}")
