#!/usr/bin/env python3
"""
Phase 196: Train GoalConditioned VLA on Phase 196 CLEAN Contact-Jacobian data
==============================================================================
Train on CORRECT Contact-Jacobian P-controller data (kP=2.0, 100% SR baseline).

KEY FIX vs Phase 190/187:
  - Phase 187/189: Data collected with WRONG kinematic IK P-controller
  - Phase 190: Trained on Phase 189 data (CORRUPTED)
  - Phase 196: Data collected with CORRECT Contact-Jacobian P-controller
    (_CONTACT_JACOBIAN_PSEUDO_INV) which achieves 100% SR in eval

Controller used in Phase 196 data collection:
  - v_desired = kP * err  (kP=2.0)
  - wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
  - Same controller that achieves 100% SR in eval_jacobian_pcontroller.py

Architecture (unchanged from Phase 190):
  - CLIP ViT-B/32 spatial tokens [B, 50, 768]
  - Goal MLP: 2 → 256 → 128
  - State net: 11D → 256 → 128
  - Cross-attention: goal(Q) attends to CLIP(K,V) → [B, 1, 768]
  - Fusion: [cls, state_feat, cross_out] → [B, 2048] → [B, 9]
  - 4-step Euler flow matching

Data (phase196_clean_50ep.h5):
  - states: (5562, 11) — arm_pos(6) + wheel_vel(3) + goal_norm(2)
  - actions: (5562, 9) — arm_torque(6) + wheel_speed(3)
  - images: (5562, 640, 480, 3) — one per step
  - goals: (5562, 2) — normalized goal coordinates
  - rewards: (5562,) — binary (1.0 at goal, 0.0 elsewhere)
  - episode_starts: (51,)
  - 90% episode success rate (45/50 episodes)

NOTE: Phase 196 data has ~0.8% positive rewards (mostly near-goal frames),
unlike Phase 189 which had ~82.8% positive rewards (continuous reward).
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

DEVICE = "cpu"
print(f"[Phase 196] Device: {DEVICE}")


# ─── CLIP Vision Encoder ───────────────────────────────────────────────────────

class CLIPVisionEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        """
        images: [B, 3, 224, 224] in [0,1].
        Returns: [B, 50, 768] spatial tokens (CLS + 49 patches).
        """
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(
                pixel_values=pixel_values, output_hidden_states=True
            )
            return outputs.last_hidden_state  # [B, 50, 768]


# ─── GoalConditioned Policy ─────────────────────────────────────────────────────

class GoalConditionedPolicy(nn.Module):
    """
    CLIP-FM with goal-conditioned state (11D) + vision (per-step images).
    Same architecture as Phase 190.

    Architecture:
      - CLIP spatial tokens [B, 50, 768] — real per-step images from Phase 196
      - Goal MLP: 2 → 256 → 128
      - State net: 11D → 256 → 128
      - Cross-attention: goal(Q) attends to CLIP(K,V) → [B, 1, 768]
      - Fusion: [cls, state_feat, cross_out] → [B, 2048] → [B, 9]
      - 4-step Euler flow matching
    """
    def __init__(self, state_dim=11, action_dim=9, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.encoder = CLIPVisionEncoder(device=device)

        self.goal_mlp = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.SiLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.SiLU()
        )

        self.goal_q_proj = nn.Linear(128, 768)
        self.cross_attn = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(768)

        self.flow_head = nn.Sequential(
            nn.Linear(768 + 768 + 128 + 256 + action_dim, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim)
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU()
        )

    def forward(self, images, state, noisy_action, timestep):
        clip_tokens = self.encoder(images)  # [B, 50, 768]
        goal_emb = self.goal_mlp(state[:, -2:])  # [B, 128]
        goal_q = self.goal_q_proj(goal_emb).unsqueeze(1)  # [B, 1, 768]
        cross_out, _ = self.cross_attn(goal_q, clip_tokens, clip_tokens)
        cross_out = self.cross_norm(cross_out + goal_q)
        state_feat = self.state_net(state)
        t_emb = self.time_mlp(timestep)

        cls_token = clip_tokens[:, 0:1, :]
        x = torch.cat([
            cls_token,
            cross_out,
            state_feat.unsqueeze(1),
            t_emb.unsqueeze(1),
            noisy_action.unsqueeze(1),
        ], dim=-1)
        x = x.squeeze(1)
        return self.flow_head(x)  # [B, 9]

    def infer(self, images, state, num_steps=4):
        """4-step Euler flow matching inference."""
        self.eval()
        with torch.no_grad():
            x = torch.zeros_like(state[:, :self.action_dim])
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((images.shape[0], 1), i * dt, device=state.device)
                v = self.forward(images, state, x, t)
                x = x + v * dt
            return x


# ─── Replay Buffer ──────────────────────────────────────────────────────────────

class Phase196Replay:
    """
    Replay for Phase 196 data: 11D state + per-step images (5562 frames).

    Data keys (Phase 196 uses 'goals', NOT 'goal_positions'):
      - states: (N, 11) — arm_pos(6) + wheel_vel(3) + goal_norm(2)
      - actions: (N, 9) — arm_torque(6) + wheel_speed(3)
      - images: (N, 640, 480, 3) — one per step
      - goals: (N, 2) — normalized goal coordinates
      - rewards: (N,) — binary (1.0 at goal, 0.0 elsewhere)

    NOTE: Phase 196 has BINARY rewards (~0.8% positive), unlike Phase 189's
    continuous reward (~82.8% positive). We weight goal-near frames higher.
    """
    def __init__(self, h5_path, batch_size=32):
        self.batch_size = batch_size
        with h5py.File(h5_path, 'r') as f:
            self.states  = f['states'][:]            # (N, 11)
            self.actions = f['actions'][:]           # (N, 9)
            self.rewards = f['rewards'][:]           # (N,)
            self.goals   = f['goals'][:]             # (N, 2) — NOTE: 'goals' not 'goal_positions'
            self.images_raw = f['images'][:]         # (N, 640, 480, 3)

        N = len(self.actions)
        self.n = N

        # Binary reward: weight goal-near frames higher
        # Phase 196: ~0.8% positive rewards (mostly near-goal frames)
        is_goal_near = (self.rewards >= 0.5)
        self.weights = np.ones(N, dtype=np.float32)
        self.weights[is_goal_near] = 5.0  # Higher weight for goal-near frames
        print(f"[Phase196Replay] {N} frames, state=11D, images={self.images_raw.shape}")
        n_goal = is_goal_near.sum()
        print(f"[Phase196Replay] goal_near_frames(reward>=0.5)={n_goal}, weight={self.weights[is_goal_near][0] if n_goal > 0 else 'N/A'}x")

        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_image(self, raw_img: np.ndarray) -> torch.Tensor:
        from PIL import Image
        img = Image.fromarray(raw_img)
        img = img.resize((224, 224), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.img_mean) / self.img_std
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def sample(self):
        probs = self.weights.astype(np.float64) / float(self.weights.sum())
        idx = np.random.choice(len(self.actions), self.batch_size, p=probs)
        states  = torch.from_numpy(self.states[idx].astype(np.float32))
        actions = torch.from_numpy(self.actions[idx].astype(np.float32))
        weights = torch.from_numpy(self.weights[idx].astype(np.float32))
        imgs = torch.stack([self._preprocess_image(self.images_raw[i])
                           for i in idx], dim=0)
        return imgs, states, actions, weights


# ─── Training ───────────────────────────────────────────────────────────────────

def train(policy, optimizer, replay, epochs=30, device=DEVICE,
         output_dir="results/phase196_contact_jacobian_train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.train()
    losses = []

    print(f"\n[Phase 196 Training] {epochs} epochs on {replay.n} frames...")
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = max(1, replay.n // args.batch_size)  # At least 1 batch per epoch
        n_batches = min(n_batches, 200)  # Cap at 200 batches

        for _ in range(n_batches):
            batch_img, batch_state, batch_action, batch_weights = replay.sample()
            batch_img = batch_img.to(device)
            batch_state = batch_state.to(device)
            batch_action = batch_action.to(device)
            batch_weights = batch_weights.to(device)

            t_batch = (torch.rand(batch_img.shape[0], 1, device=device) ** 1.5) * 0.999
            noise = torch.randn_like(batch_action)
            alpha = 1 - t_batch.squeeze(-1)
            x_t = alpha.unsqueeze(-1) * batch_action + t_batch.squeeze(-1).unsqueeze(-1) * noise

            v_pred = policy(batch_img, batch_state, x_t, t_batch)
            v_target = batch_action - noise

            loss = ((v_pred - v_target) ** 2).mean(dim=-1)
            loss = (loss * batch_weights / batch_weights.mean()).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        elapsed = time.time() - t_start

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={lr:.6f}, elapsed={elapsed:.0f}s")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'loss': avg_loss,
                'policy_config': {
                    'state_dim': 11, 'action_dim': 9, 'hidden': 512
                }
            }, output_dir / f'epoch_{epoch}.pt')

    # Save best
    best_epoch = np.argmin(losses)
    torch.save({
        'epoch': best_epoch,
        'policy_state_dict': policy.state_dict(),
        'losses': losses,
        'policy_config': {
            'state_dim': 11, 'action_dim': 9, 'hidden': 512
        }
    }, output_dir / 'best_policy.pt')

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Phase 196 — GoalConditioned VLA (Contact-Jacobian Data)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(output_dir / 'training_loss.png', dpi=100)
    plt.close()

    import json
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump([{'epoch': i, 'loss': losses[i]} for i in range(len(losses))], f)

    print(f"\n✓ Best epoch: {best_epoch+1}, loss={losses[best_epoch]:.4f}")
    print(f"✓ Saved: {output_dir / 'best_policy.pt'}")
    return losses


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_policy(policy_path, n_goals=20, seed=42):
    """Evaluate VLA policy using CORRECT Contact-Jacobian P-controller as baseline."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, _CONTACT_JACOBIAN_PSEUDO_INV

    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    state_dict = torch.load(policy_path, map_location=DEVICE)

    # Handle different checkpoint formats
    if 'policy_state_dict' in state_dict:
        policy.load_state_dict(state_dict['policy_state_dict'], strict=False)
    else:
        policy.load_state_dict(state_dict, strict=False)
    policy.to(DEVICE)
    policy.eval()

    np.random.seed(seed)
    goals = [(np.random.uniform(-0.3, 0.4), np.random.uniform(-0.25, 0.25)) for _ in range(n_goals)]

    # Baseline: Contact-Jacobian P-controller (kP=2.0)
    pctrl_successes = 0
    for g in goals:
        g = np.array(g)
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id
        arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
        for step in range(200):
            base_xy = sim.data.xpos[base_body_id, :2]
            err = g - base_xy
            if np.linalg.norm(err) < 0.10:
                pctrl_successes += 1
                break
            v_desired = 2.0 * err
            wheel_speeds = np.clip(_CONTACT_JACOBIAN_PSEUDO_INV @ v_desired, -0.5, 0.5)
            action = np.concatenate([arm, wheel_speeds])
            sim.step(action)

    # VLA policy evaluation
    vla_successes = 0
    for g in goals:
        g = np.array(g)
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id
        arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
        for step in range(200):
            base_xy = sim.data.xpos[base_body_id, :2]
            err = g - base_xy
            if np.linalg.norm(err) < 0.10:
                vla_successes += 1
                break

            # Build state
            # CORRECT (Phase 222): wheel_vel from qvel[6:9], NOT qvel[9:12]=ARM velocities
            wheel_vel = sim.data.qvel[6:9].copy()
            state_vec = np.concatenate([
                arm, wheel_vel, g / 0.525
            ]).astype(np.float32)

            # Get image
            img = sim.render().astype(np.uint8)
            from PIL import Image
            pil_img = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - img_mean) / img_std
            arr = arr.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = policy.infer(img_tensor,
                                     torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE),
                                     num_steps=4)
                action = action.squeeze(0).cpu().numpy()

            sim.step(action)

    print(f"\n[Phase 196 Eval] {n_goals} random goals (seed={seed})")
    print(f"  P-ctrl (Contact-Jacobian kP=2.0): {pctrl_successes}/{n_goals} = {pctrl_successes/n_goals*100:.0f}% SR")
    print(f"  VLA policy:                     {vla_successes}/{n_goals} = {vla_successes/n_goals*100:.0f}% SR")

    return pctrl_successes/n_goals, vla_successes/n_goals


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/phase196_clean_50ep.h5')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default='results/phase196_contact_jacobian_train')
    parser.add_argument('--eval_only', type=str, default=None,
                       help='Path to policy.pt to eval only')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.data) or '.', exist_ok=True)

    # Verify data
    with h5py.File(args.data, 'r') as f:
        images_shape = f['images'].shape
        states_shape = f['states'].shape
    print(f"[Data Check] images={images_shape}, states={states_shape}")
    if images_shape[0] != states_shape[0]:
        raise ValueError(f"CRITICAL: images={images_shape[0]} != states={states_shape[0]}")
    if images_shape[0] < 1000:
        raise ValueError(f"WARNING: only {images_shape[0]} images")

    if args.eval_only:
        evaluate_policy(args.eval_only)
        sys.exit(0)

    replay = Phase196Replay(args.data, batch_size=args.batch_size)
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train(policy, optimizer, replay, epochs=args.epochs, device=DEVICE, output_dir=args.output)
    scheduler.step()

    # Save final
    output_dir = Path(args.output)
    torch.save(policy.state_dict(), output_dir / 'final_policy.pt')
    print(f"✓ Final policy: {output_dir / 'final_policy.pt'}")

    # Quick eval
    if (output_dir / 'best_policy.pt').exists():
        print("\n[Running evaluation on best policy...]")
        evaluate_policy(output_dir / 'best_policy.pt', n_goals=20)
