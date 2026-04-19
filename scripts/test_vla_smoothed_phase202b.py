#!/usr/bin/env python3
"""
Phase 202b: VLA + Action Smoother Closed-Loop Test
==================================================
Tests whether ActionSmoother (EMA + delta clip) improves VLA robustness.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
import numpy as np

# Action Smoother from vla_policy_node (copied for standalone use)
class ActionSmoother:
    def __init__(self, wheel_alpha=0.25, arm_alpha=0.70,
                 wheel_max_delta=0.8, arm_max_delta=0.5, warmup_steps=10):
        self.wheel_alpha = wheel_alpha
        self.arm_alpha = arm_alpha
        self.wheel_max_delta = wheel_max_delta
        self.arm_max_delta = arm_max_delta
        self.warmup_steps = warmup_steps
        self._wheel_smoothed = None
        self._arm_smoothed = None
        self._step = 0

    def smooth(self, action):
        arm = action[:6]
        wheel = action[6:9]
        if self._arm_smoothed is None:
            self._arm_smoothed = arm.copy()
        if self._wheel_smoothed is None:
            self._wheel_smoothed = wheel.copy()

        if self._step < self.warmup_steps:
            self._arm_smoothed = arm.copy()
            self._wheel_smoothed = wheel.copy()
            self._step += 1
            return action.copy()

        delta_arm = arm - self._arm_smoothed
        delta_arm_clamped = np.clip(delta_arm, -self.arm_max_delta, self.arm_max_delta)
        self._arm_smoothed = self._arm_smoothed + self.arm_alpha * delta_arm_clamped

        delta_wheel = wheel - self._wheel_smoothed
        delta_wheel_clamped = np.clip(delta_wheel, -self.wheel_max_delta, self.wheel_max_delta)
        self._wheel_smoothed = self._wheel_smoothed + self.wheel_alpha * delta_wheel_clamped

        result = np.concatenate([self._arm_smoothed, self._wheel_smoothed])
        return result

ARM_LIMITS = np.array([
    [-3.14, 3.14], [-1.57, 1.57], [-1.57, 1.57],
    [-1.57, 1.57], [-3.14, 3.14], [0.00, 0.04],
], dtype=np.float32)

def test_smoothed_vla(n_episodes=3, max_steps=200, wheel_alpha=0.25):
    print(f"\n{'='*60}")
    print(f"Phase 202b: VLA + ActionSmoother (wheel_alpha={wheel_alpha})")
    print(f"{'='*60}")

    from sim_lekiwi_urdf import LeKiWiSimURDF
    from scripts.train_phase196 import GoalConditionedPolicy

    DEVICE = "cpu"
    sim = LeKiWiSimURDF()

    print("  Loading Phase196 policy...")
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    import torch
    ckpt_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/phase196_contact_jacobian_train/epoch_14.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    policy.load_state_dict(ckpt.get("policy_state_dict", ckpt), strict=False)
    policy.eval()
    print(f"  ✓ Loaded epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?')}")

    img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(raw_img):
        from PIL import Image
        img = Image.fromarray(raw_img)
        img = img.resize((224, 224), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - img_mean) / img_std
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    def vla_infer(image_raw, state_11d):
        img_t = preprocess_image(image_raw)
        st_t = torch.from_numpy(state_11d[np.newaxis]).to(DEVICE).float()
        with torch.no_grad():
            x = torch.zeros(1, 9, device=DEVICE)
            dt = 1.0 / 4
            for i in range(4):
                t = torch.full((1, 1), i * dt, device=DEVICE)
                v = policy.forward(img_t, st_t, x, t)
                x = x + v * dt
        return x[0].cpu().numpy()

    WHEEL_LIMITS = np.array([[-5.0, 5.0]]*3, dtype=np.float32)

    def normalize(raw_action):
        arm  = raw_action[:6]
        wheel = raw_action[6:9]
        arm_n = ARM_LIMITS[:,0] + (arm+1)/2*(ARM_LIMITS[:,1]-ARM_LIMITS[:,0])
        wh_n  = WHEEL_LIMITS[:,0] + (wheel+1)/2*(WHEEL_LIMITS[:,1]-WHEEL_LIMITS[:,0])
        return np.concatenate([arm_n, wh_n]).astype(np.float32)

    def clip_action(native_action):
        arm = np.clip(native_action[:6], ARM_LIMITS[:,0], ARM_LIMITS[:,1])
        wh  = np.clip(native_action[6:9], -0.5, 0.5)
        return np.concatenate([arm, wh])

    goals_reached = []
    final_distances = []

    for ep in range(n_episodes):
        sim.reset()
        smoother = ActionSmoother(wheel_alpha=wheel_alpha, arm_alpha=0.70)
        np.random.seed(ep * 42)

        goal_angle = np.random.uniform(0, 2*np.pi)
        goal_dist = np.random.uniform(0.15, 0.35)
        goal_xy = np.array([np.cos(goal_angle)*goal_dist, np.sin(goal_angle)*goal_dist])
        sim.goal_xy = goal_xy

        print(f"\n  Episode {ep+1}/{n_episodes}: goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")

        for step in range(max_steps):
            obs = sim._obs()
            base_xy = obs['base_position'][:2]

            image_raw = sim.render()
            arm_pos = obs['arm_positions']
            wheel_vel = obs['wheel_velocities']
            goal_norm = np.clip(goal_xy / 0.525, -1.0, 1.0)
            state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

            raw_action = vla_infer(image_raw, state_11d)
            native_action = normalize(raw_action)
            clipped_action = clip_action(native_action)
            smoothed_action = smoother.smooth(clipped_action)

            sim.step(smoothed_action)

            dist = np.linalg.norm(base_xy - goal_xy)
            if step % 50 == 0:
                print(f"    step {step}: dist={dist:.3f}")

            if dist < 0.05:
                print(f"    ✓ GOAL REACHED at step {step}!")
                break

        final_obs = sim._obs()
        final_dist = np.linalg.norm(final_obs['base_position'][:2] - goal_xy)
        goals_reached.append(dist < 0.05)
        final_distances.append(final_dist)
        print(f"    final_dist={final_dist:.3f} {'✓' if dist < 0.05 else '✗'}")

    sr = sum(goals_reached) / len(goals_reached)
    mean_dist = np.mean(final_distances)
    print(f"\n  SR: {sum(goals_reached)}/{len(goals_reached)} = {sr:.1%}")
    print(f"  Mean Final Distance: {mean_dist:.3f}m")
    return sr

# Test different smoothing strengths
print("Testing with wheel_alpha=0.25 (default from bridge)...")
sr_025 = test_smoothed_vla(n_episodes=3, max_steps=200, wheel_alpha=0.25)

print("\nTesting with wheel_alpha=0.10 (more smoothing)...")
sr_010 = test_smoothed_vla(n_episodes=3, max_steps=200, wheel_alpha=0.10)

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"  wheel_alpha=0.25: SR={sr_025:.1%}")
print(f"  wheel_alpha=0.10: SR={sr_010:.1%}")
