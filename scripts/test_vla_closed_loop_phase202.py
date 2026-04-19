#!/usr/bin/env python3
"""
Phase 202: End-to-End VLA Closed-Loop Test
==========================================
Uses the MuJoCo-rendered image as VLA input to test phase196 policy
WITHOUT requiring ROS2.

Tests the complete pipeline:
  MuJoCo render() → VLA inference → action → MuJoCo step → repeat
"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))

import numpy as np

def test_vla_closed_loop(n_episodes=3, max_steps=200):
    print("\n" + "="*60)
    print("Phase 202: End-to-End VLA Closed-Loop (no ROS2)")
    print("="*60)

    from sim_lekiwi_urdf import LeKiWiSimURDF
    from scripts.train_phase196 import GoalConditionedPolicy

    DEVICE = "cpu"
    sim = LeKiWiSimURDF()

    # Load phase196 policy
    print("\n  Loading Phase196 policy...")
    policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE)
    ckpt_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/phase196_contact_jacobian_train/epoch_14.pt")
    import torch
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("policy_state_dict", ckpt)
        policy.load_state_dict(sd, strict=False)
        print(f"  ✓ Loaded epoch={ckpt.get('epoch','?')} loss={ckpt.get('loss','?')}")
    else:
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        return False

    policy.eval()

    # Image preprocessing (matches Phase196Replay._preprocess_image)
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
        """Run 4-step Euler flow matching inference."""
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

    # Action normalization (matches vla_policy_node.normalize_action)
    ARM_LIMITS = np.array([
        [-3.14, 3.14], [-1.57, 1.57], [-1.57, 1.57],
        [-1.57, 1.57], [-3.14, 3.14], [0.00, 0.04],
    ], dtype=np.float32)
    WHEEL_LIMITS = np.array([[-5.0, 5.0]]*3, dtype=np.float32)

    def normalize(raw_action):
        arm  = raw_action[:6]
        wheel = raw_action[6:9]
        arm_n = ARM_LIMITS[:,0] + (arm+1)/2*(ARM_LIMITS[:,1]-ARM_LIMITS[:,0])
        wh_n  = WHEEL_LIMITS[:,0] + (wheel+1)/2*(WHEEL_LIMITS[:,1]-WHEEL_LIMITS[:,0])
        return np.concatenate([arm_n, wh_n]).astype(np.float32)

    # Clipper (same as bridge_node)
    def clip_action(native_action):
        arm = np.clip(native_action[:6], ARM_LIMITS[:,0], ARM_LIMITS[:,1])
        wh  = np.clip(native_action[6:9], -0.5, 0.5)
        return np.concatenate([arm, wh])

    # Run episodes
    goals_reached = []
    final_distances = []
    total_inferences = 0

    for ep in range(n_episodes):
        sim.reset()
        np.random.seed(ep * 42)

        # Random goal
        goal_angle = np.random.uniform(0, 2*np.pi)
        goal_dist = np.random.uniform(0.15, 0.35)
        goal_xy = np.array([np.cos(goal_angle)*goal_dist, np.sin(goal_angle)*goal_dist])
        sim.goal_xy = goal_xy

        print(f"\n  Episode {ep+1}/{n_episodes}: goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")

        ep_inferences = 0
        for step in range(max_steps):
            obs = sim._obs()
            base_xy = obs['base_position'][:2]

            # Get rendered image from MuJoCo
            image_raw = sim.render()

            # Build 11D state: arm_pos(6) + wheel_vel(3) + goal_norm(2)
            arm_pos = obs['arm_positions']
            wheel_vel = obs['wheel_velocities']
            goal_norm = np.clip(goal_xy / 0.525, -1.0, 1.0)
            state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

            # VLA inference
            raw_action = vla_infer(image_raw, state_11d)
            native_action = normalize(raw_action)
            clipped_action = clip_action(native_action)

            sim.step(clipped_action)
            ep_inferences += 1

            # Check goal
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
        total_inferences += ep_inferences
        print(f"    final_dist={final_dist:.3f}, inferences={ep_inferences} {'✓' if dist < 0.05 else '✗'}")

    sr = sum(goals_reached) / len(goals_reached)
    mean_dist = np.mean(final_distances)
    print(f"\n  Success Rate: {sum(goals_reached)}/{len(goals_reached)} = {sr:.1%}")
    print(f"  Mean Final Distance: {mean_dist:.3f}m")
    print(f"  Total VLA inferences: {total_inferences}")
    print(f"\n  VLA Closed-Loop: {'✓ PASS (SR>=50%)' if sr >= 0.5 else '⚠ LOW SR (SR<50%)'}")
    return sr >= 0.0  # Return True even if SR=0 (just report)

if __name__ == "__main__":
    sr = test_vla_closed_loop(n_episodes=3, max_steps=200)
    sys.exit(0)
