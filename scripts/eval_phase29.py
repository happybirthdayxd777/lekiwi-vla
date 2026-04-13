#!/usr/bin/env python3
"""
Phase 29 — Quick Policy Evaluation
===================================
Evaluate the phase28 goal_aware policy (checkpoint_epoch_20.pt)
on LeKiwiSim (primitive) backend.

Tests: 5 goals x 150 steps each
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from scripts.train_task_oriented import CLIPFlowMatchingPolicy
from sim_lekiwi import LeKiwiSim


def resize(img_pil):
    pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    img_chw = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_chw).unsqueeze(0).cpu()


def make_state(obs, goal_x, goal_y):
    arm_pos = obs['arm_positions']
    wheel_vel = obs['wheel_velocities']
    goal_norm = np.array([goal_x / 1.0, goal_y / 1.0])
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)


def evaluate_goal(goal_x, goal_y, max_steps=150, threshold=0.15):
    sim = LeKiwiSim()
    sim.set_target(np.array([goal_x, goal_y, 0.02]))
    
    # Warmup
    for _ in range(40):
        sim.step(np.zeros(9))
    
    img_pil = sim.render()
    img_t = resize(img_pil)
    
    for step in range(max_steps):
        obs = sim._obs()
        state = make_state(obs, goal_x, goal_y)
        state_t = torch.from_numpy(state).unsqueeze(0).cpu()
        
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        
        action_np = np.clip(action.cpu().numpy()[0], -1, 1).astype(np.float32)
        obs = sim.step(action_np)
        img_t = resize(sim.render())
        
        base_xy = sim.data.qpos[:2]
        dist = np.linalg.norm(base_xy - np.array([goal_x, goal_y]))
        
        if step % 30 == 0:
            print(f"    step={step:3d} dist={dist:.3f}m")
        
        if dist < threshold:
            print(f"    => SUCCESS at step {step}, dist={dist:.3f}m")
            return True, dist, step
    
    return False, dist, max_steps


if __name__ == "__main__":
    print("Loading policy...")
    policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device='cpu')
    ckpt = torch.load('results/phase28_goal_aware/checkpoint_epoch_20.pt', map_location='cpu', weights_only=False)
    loaded = ckpt.get('policy_state_dict', ckpt)
    policy.load_state_dict(loaded, strict=False)
    policy.to('cpu').eval()
    print("Policy loaded OK\n")
    
    goals = [(0.3, 0.2), (0.5, 0.0), (0.4, 0.4), (0.2, -0.3), (0.45, 0.15)]
    results = []
    
    print("Running evaluation...\n")
    for i, (gx, gy) in enumerate(goals):
        print(f"[{i+1}/{len(goals)}] Goal ({gx}, {gy}):")
        success, dist, steps = evaluate_goal(gx, gy)
        results.append((gx, gy, success, dist, steps))
        print()
    
    sr = sum(s for _, _, s, _, _ in results) / len(results)
    mean_dist = np.mean([d for _, _, _, d, _ in results])
    mean_steps = np.mean([st for _, _, _, _, st in results])
    
    print("=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for gx, gy, s, d, st in results:
        print(f"  goal=({gx:+.1f},{gy:+.1f}): {'SUCCESS' if s else 'FAIL'} dist={d:.3f}m steps={st}")
    print(f"\nSuccess Rate: {sr*100:.0f}%")
    print(f"Mean distance: {mean_dist:.3f}m")
    print(f"Mean steps: {mean_steps:.0f}")
