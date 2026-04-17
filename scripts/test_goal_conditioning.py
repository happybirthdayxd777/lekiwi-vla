#!/usr/bin/env python3
"""
Phase 130 — Goal-Conditioned VLA: State Extension Test
======================================================
Tests whether adding goal_xy (2D) to state_dim makes VLA goal-reaching.

Hypothesis: 9D policy (arm_pos + wheel_vel) fails goal-reaching because
it has no position information. Adding goal_xy (2D) → 11D state should
enable the policy to learn goal-directed behavior.

Test:
1. Create 11D state (9D + goal_xy) dataset from lekiwi_goal_5k.h5
2. Retrain policy with state_dim=11 for 100 epochs
3. Quick eval: 5 episodes, compare to P-controller baseline

If SR improves from 0%: goal-conditioning is the fix.
If SR still ~0%: need more architectural changes (attention, memory, etc.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
from PIL import Image
import time

from sim_lekiwi import LeKiwiSim

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 130] Device: {DEVICE}")

# ─── Extended ReplayBuffer with goal in state ──────────────────────────────

class GoalConditionedReplay:
    """Extends state from 9D to 11D by concatenating goal_xy."""
    def __init__(self, h5_path, batch_size=16, state_dim=11):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.state_dim = state_dim  # 9 + 2 = 11
        
        with h5py.File(h5_path, 'r') as h:
            self.n_frames = len(h['images'])
            self.states = h['states'][:].astype(np.float32)      # [N, 9]
            self.actions = h['actions'][:].astype(np.float32)     # [N, 9]
            self.goals = h['goal_positions'][:].astype(np.float32)  # [N, 2]
            self.images = h['images'][:]
            
        print(f"[Replay] {self.n_frames} frames, state_dim={state_dim} (9+2)")
        
        # Build extended states: [arm_pos(6) + wheel_vel(3) + goal_x(1) + goal_y(1)]
        self.extended_states = np.concatenate([self.states, self.goals], axis=1)
        print(f"[Replay] Extended state range: [{self.extended_states.min():.3f}, {self.extended_states.max():.3f}]")
    
    def sample(self):
        idx = np.random.randint(0, self.n_frames, self.batch_size)
        imgs = np.stack([self.images[i] for i in idx])  # [B, H, W, C]
        imgs = imgs.transpose(0, 3, 1, 2)  # [B, C, H, W] = [B, 3, 224, 224]
        # Return extended 11D state + 9D action
        return (torch.from_numpy(imgs.astype(np.float32) / 255.0),
                torch.from_numpy(self.extended_states[idx]),
                torch.from_numpy(self.actions[idx]))
    
    def __len__(self):
        return self.n_frames // self.batch_size


# ─── CLIP Vision Encoder (same as train_clip_fm.py) ─────────────────────────

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


# ─── Flow Matching Head (state_dim=11 now) ─────────────────────────────────

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


# ─── Goal-Conditioned Policy ────────────────────────────────────────────────

class GoalConditionedPolicy(nn.Module):
    """VLA with 11D state (9D + goal_xy)."""
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
        """4-step Euler ODE inference with goal-conditioned state."""
        self.eval()
        with torch.no_grad():
            action = torch.randn(image.shape[0], self.action_dim, device=self.device)
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((image.shape[0], 1), step * dt, device=self.device)
                v = self.forward(image, state, action, t)
                action = action - dt * v  # Euler integration
            return action


# ─── P-Controller Baseline ──────────────────────────────────────────────────

class PController:
    """Oracle P-controller for comparison. Uses goal info to navigate."""
    def __init__(self, kP=1.5, max_speed=0.05):
        self.kP = kP
        self.max_speed = max_speed
    
    def act(self, state, goal):
        """
        state: [9D] arm_pos(6) + wheel_vel(3) — NOTE: no base position in state!
        goal: [2D] goal_x, goal_y
        
        Since state has no base position, we need to get it from the sim.
        For eval, we'll pass base_pos from sim.
        """
        base_pos = state[-2:] if len(state) >= 11 else state[6:8]  # [x, y] from sim
        err = goal - base_pos
        vx = np.clip(self.kP * err[0], -self.max_speed, self.max_speed)
        vy = np.clip(self.kP * err[1], -self.max_speed, self.max_speed)
        # Omni-wheel inverse kinematics (from Phase 122)
        w1 = 0.3824*vx + 0.1929*vy
        w2 = -0.4531*vx + 0.2378*vy
        w3 = 0.0178*vx + 0.1544*vy
        wheel = np.clip([w1, w2, w3], -0.5, 0.5)
        return np.concatenate([np.zeros(6), wheel])


# ─── Evaluation ────────────────────────────────────────────────────────────

def evaluate_policy(policy, replay, n_episodes=5, max_steps=200):
    """Evaluate goal-reaching success rate."""
    sim = LeKiwiSim()
    successes = []
    
    for ep in range(n_episodes):
        sim.reset()
        # Random goal from reachable area
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 0.6)
        goal = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        success = False
        for step in range(max_steps):
            img = np.array(sim.render().resize((224, 224)), dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            # 11D state: arm(6) + wheel_vel(3) + goal_xy(2)
            arm = sim.data.qpos[0:6]
            whl = sim.data.qvel[0:3]
            base_xy = sim.data.qpos[7:9]
            state_11d = np.concatenate([arm, whl, goal, base_xy])
            state_t = torch.from_numpy(state_11d.astype(np.float32)).unsqueeze(0).to(DEVICE)
            
            action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]
            
            # Denormalize: wheels to [-0.5, 0.5]
            action_np[6:9] = 0.5 * np.tanh(action_np[6:9] / 0.5)
            sim.step(action_np)
            
            dist = np.linalg.norm(sim.data.qpos[7:9] - goal)
            if dist < 0.1:
                success = True
                break
        
        successes.append(success)
        print(f"  Episode {ep+1}: {'SUCCESS' if success else 'FAIL'} (dist={dist:.3f})")
    
    sr = np.mean(successes)
    print(f"\nGoal-Conditioned VLA SR: {100*sr:.0f}% ({sum(successes)}/{n_episodes})")
    return sr


def evaluate_pcontroller(replay, n_episodes=5, max_steps=200):
    """P-controller baseline with full state access (base_xy from sim)."""
    sim = LeKiWiSim()
    ctrl = PController(kP=1.5, max_speed=0.05)
    successes = []
    
    for ep in range(n_episodes):
        sim.reset()
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.3, 0.6)
        goal = np.array([r * np.cos(angle), r * np.sin(angle)])
        
        success = False
        for step in range(max_steps):
            arm = sim.data.qpos[0:6]
            whl = sim.data.qvel[0:3]
            base_xy = sim.data.qpos[7:9]
            state_9d = np.concatenate([arm, whl])  # 9D
            
            # Extended state for P-controller
            state_11d = np.concatenate([state_9d, goal, base_xy])
            action_np = ctrl.act(state_11d, goal)
            
            action_np[6:9] = np.clip(action_np[6:9], -0.5, 0.5)
            sim.step(action_np)
            
            dist = np.linalg.norm(base_xy - goal)
            if dist < 0.1:
                success = True
                break
        
        successes.append(success)
        print(f"  P-ctrl Episode {ep+1}: {'SUCCESS' if success else 'FAIL'}")
    
    sr = np.mean(successes)
    print(f"\nP-Controller SR: {100*sr:.0f}% ({sum(successes)}/{n_episodes})")
    return sr


# ─── Training ───────────────────────────────────────────────────────────────

def train_goal_conditioned(policy, replay, epochs=100, output_dir="results/phase130"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    policy.to(DEVICE)
    policy.train()
    optimizer = torch.optim.Adam(policy.flow_head.parameters(), lr=1e-4)
    
    print(f"\n[Training] {epochs} epochs on {len(replay)} batches...")
    t_start = time.time()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = min(50, len(replay))  # Reduced for faster iteration
        
        for batch_idx in range(n_batches):
            batch_img, batch_state, batch_action = replay.sample()
            batch_img = batch_img.to(DEVICE)
            batch_state = batch_state.to(DEVICE)
            batch_action = batch_action.to(DEVICE)
            
            t = (torch.rand(batch_img.shape[0], 1, device=DEVICE) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            x_t = (1 - t) * batch_action + t * noise
            
            v_pred = policy(batch_img, batch_state, x_t, t)
            v_target = batch_action - noise
            
            loss = ((v_pred - v_target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg = epoch_loss / n_batches
        losses.append(avg)
        
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            torch.save({
                "epoch": epoch,
                "policy_state_dict": policy.state_dict(),
                "losses": losses
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pt")
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg:.4f} | ETA: {(elapsed/(epoch+1))*(epochs-epoch-1):.0f}s")
    
    torch.save(policy.state_dict(), output_dir / "final_policy.pt")
    print(f"\n✓ Training done in {time.time()-t_start:.0f}s")
    return losses


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    data_path = "data/lekiwi_goal_5k.h5"
    
    print("=" * 60)
    print("Phase 130 — Goal-Conditioned VLA Test")
    print("=" * 60)
    
    # Step 1: Load extended replay
    print("\n[1] Loading goal-conditioned replay buffer...")
    replay = GoalConditionedReplay(data_path, state_dim=11)
    
    # Step 2: Train 30 epochs (quick test — if promising, train 100 later)
    print("\n[2] Training goal-conditioned policy (state_dim=11) for 30 epochs...")
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    losses = train_goal_conditioned(policy, replay, epochs=30)
    
    # Step 3: Evaluate
    print("\n[3] Evaluating goal-conditioned VLA...")
    sr_vla = evaluate_policy(policy, replay, n_episodes=5, max_steps=200)
    
    print("\n[4] P-controller baseline...")
    sr_pctrl = evaluate_pcontroller(replay, n_episodes=5, max_steps=200)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  VLA (11D state, goal-conditioned):  {100*sr_vla:.0f}% SR")
    print(f"  P-controller (oracle):              {100*sr_pctrl:.0f}% SR")
    print(f"\n  Conclusion:")
    if sr_vla > 0.5:
        print(f"  ✓ Goal-conditioning WORKS — VLA achieves {100*sr_vla:.0f}% SR")
    elif sr_vla > 0:
        print(f"  ~ Partial improvement — VLA achieves {100*sr_vla:.0f}% SR (vs 0% before)")
    else:
        print(f"  ✗ Goal-conditioning NOT sufficient — VLA still {100*sr_vla:.0f}% SR")
        print(f"     Need: attention, memory, or longer training")
    
    # Save results
    import json
    results = {
        "phase": 130,
        "vla_sr": float(sr_vla),
        "pcontroller_sr": float(sr_pctrl),
        "state_dim": 11,
        "epochs": 100,
        "conclusion": "goal_conditioning_works" if sr_vla > 0.5 else "needs_more"
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/phase130_goal_conditioned_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results/phase130_goal_conditioned_eval.json")

if __name__ == "__main__":
    main()