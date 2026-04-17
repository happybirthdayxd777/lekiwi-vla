#!/usr/bin/env python3
"""
Phase 130 — Evaluate Goal-Conditioned Policy Checkpoint
=======================================================
Quick eval of the 20-epoch checkpoint from test_goal_conditioning.py

The training ran to epoch 20 before timeout. Now we evaluate:
- Goal-conditioned VLA (state_dim=11, epoch 20): success rate
- P-controller baseline: success rate
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

from sim_lekiwi import LeKiwiSim

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 130 Eval] Device: {DEVICE}")

# ─── CLIP Vision Encoder (must match training) ─────────────────────────────

class CLIPVisionEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        for p in self.clip.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(768, 512).to(device)
    
    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
        return self.proj(pooled)


class FlowMatchingHead(nn.Module):
    def __init__(self, vision_dim=512, state_dim=11, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        total_dim = vision_dim + state_dim + action_dim + 256  # 512+11+9+256=788
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


class GoalConditionedPolicy(nn.Module):
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead(vision_dim=hidden, state_dim=state_dim, action_dim=action_dim, hidden=hidden)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    def forward(self, image, state, noisy_action, timestep):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, noisy_action, timestep)
    
    def infer(self, image, state, num_steps=4):
        self.eval()
        with torch.no_grad():
            action = torch.randn(image.shape[0], self.action_dim, device=self.device)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((image.shape[0], 1), step * dt, device=self.device)
                v = self.forward(image, state, action, t)
                action = action - dt * v
            return action


# ─── P-Controller (oracle baseline) ─────────────────────────────────────────

class PController:
    def __init__(self, kP=1.5, max_speed=0.05):
        self.kP = kP
        self.max_speed = max_speed
    
    def act(self, base_xy, goal):
        err = goal - base_xy
        vx = np.clip(self.kP * err[0], -self.max_speed, self.max_speed)
        vy = np.clip(self.kP * err[1], -self.max_speed, self.max_speed)
        # Omni-wheel IK (Phase 122 calibrated)
        w1 = 0.3824*vx + 0.1929*vy
        w2 = -0.4531*vx + 0.2378*vy
        w3 = 0.0178*vx + 0.1544*vy
        wheel = np.clip([w1, w2, w3], -0.5, 0.5)
        return np.concatenate([np.zeros(6), wheel])


# ─── Evaluation Functions ────────────────────────────────────────────────────

def evaluate_goal_conditioned_vla(policy, n_episodes=5, max_steps=200):
    """Evaluate goal-conditioned VLA with 11D state."""
    sim = LeKiwiSim()
    successes = []
    
    for ep in range(n_episodes):
        sim.reset()
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 0.6)
        goal = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        success = False
        for step in range(max_steps):
            img = np.array(sim.render().resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            arm = sim.data.qpos[0:6]
            whl = sim.data.qvel[0:3]
            base_xy = sim.data.qpos[7:9]
            # 11D state: arm(6) + wheel_vel(3) + goal_xy(2)
            state_11d = np.concatenate([arm, whl, goal])
            state_t = torch.from_numpy(state_11d.astype(np.float32)).unsqueeze(0).to(DEVICE)
            
            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]
            
            # Denormalize: tanh bounded scaling
            action_np[6:9] = 0.5 * np.tanh(action_np[6:9] / 0.5)
            sim.step(action_np)
            
            dist = np.linalg.norm(sim.data.qpos[7:9] - goal)
            if dist < 0.1:
                success = True
                break
        
        successes.append(success)
        dist_final = np.linalg.norm(sim.data.qpos[7:9] - goal)
        print(f"  VLA Episode {ep+1}: {'SUCCESS' if success else 'FAIL'} (final dist={dist_final:.3f})")
    
    sr = np.mean(successes)
    return sr, successes


def evaluate_pcontroller(n_episodes=5, max_steps=200):
    """P-controller baseline (oracle, knows goal position)."""
    sim = LeKiwiSim()
    ctrl = PController(kP=1.5, max_speed=0.05)
    successes = []
    
    for ep in range(n_episodes):
        sim.reset()
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 0.6)
        goal = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        success = False
        for step in range(max_steps):
            base_xy = sim.data.qpos[7:9]
            action_np = ctrl.act(base_xy, goal)
            action_np[6:9] = np.clip(action_np[6:9], -0.5, 0.5)
            sim.step(action_np)
            
            dist = np.linalg.norm(base_xy - goal)
            if dist < 0.1:
                success = True
                break
        
        successes.append(success)
        print(f"  P-ctrl Episode {ep+1}: {'SUCCESS' if success else 'FAIL'}")
    
    sr = np.mean(successes)
    return sr, successes


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ckpt_path = Path("results/phase130/checkpoint_epoch_20.pt")
    
    print("=" * 60)
    print("Phase 130 — Goal-Conditioned VLA Evaluation")
    print("=" * 60)
    
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found — run test_goal_conditioning.py first")
        return
    
    # Load checkpoint
    print(f"\n[1] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  Epoch: {ckpt['epoch']}, Loss range: [{min(ckpt['losses']):.4f}, {max(ckpt['losses']):.4f}]")
    
    # Build policy
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(DEVICE)
    policy.eval()
    print("  Policy loaded OK")
    
    # Evaluate VLA
    print("\n[2] Evaluating Goal-Conditioned VLA (5 episodes)...")
    sr_vla, _ = evaluate_goal_conditioned_vla(policy, n_episodes=5, max_steps=200)
    
    # Evaluate P-controller
    print("\n[3] P-controller baseline (5 episodes)...")
    sr_pctrl, _ = evaluate_pcontroller(n_episodes=5, max_steps=200)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — Phase 130")
    print("=" * 60)
    print(f"  VLA (11D state, goal-conditioned, epoch 20): {100*sr_vla:.0f}% SR")
    print(f"  P-controller (oracle):                       {100*sr_pctrl:.0f}% SR")
    print(f"\n  Comparison to Phase 127:")
    print(f"    OLD VLA (9D, no goal): 0% SR")
    print(f"    NEW VLA (11D, goal):   {100*sr_vla:.0f}% SR")
    print(f"\n  Conclusion:")
    if sr_vla > 0.6:
        print(f"  ✓ GOAL-CONDITIONING WORKS — VLA improved from 0% → {100*sr_vla:.0f}% SR")
    elif sr_vla > 0:
        print(f"  ~ Partial improvement — VLA improved from 0% → {100*sr_vla:.0f}% SR")
    else:
        print(f"  ✗ Goal-conditioning NOT sufficient — VLA still {100*sr_vla:.0f}% SR")
    
    # Save results
    results = {
        "phase": 130,
        "policy": "goal_conditioned_11d",
        "checkpoint": str(ckpt_path),
        "epoch": ckpt['epoch'],
        "vla_sr": float(sr_vla),
        "pcontroller_sr": float(sr_pctrl),
        "state_dim": 11,
        "improvement_vs_phase127": f"{100*sr_vla:.0f}% vs 0%",
    }
    with open("results/phase130_goal_conditioned_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/phase130_goal_conditioned_eval.json")

if __name__ == "__main__":
    main()