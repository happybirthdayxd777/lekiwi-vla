#!/usr/bin/env python3
"""Phase 261: Quick eval of Stage2 curriculum checkpoint"""
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from PIL import Image
from sim_lekiwi_urdf import LeKiWiSimURDF

DEVICE = 'cpu'

def preprocess_image(img):
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img.transpose(2, 0, 1)

def build_state(sim, goal):
    arm_pos = sim.data.qpos[9:15]
    wheel_vel = sim.data.qvel[6:9]
    goal_norm = np.array([goal[0]/0.40, goal[1]/0.34])
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

# Load Stage2 checkpoint
print("[Stage2 Curriculum Policy — 10-goal eval]")
ckpt = torch.load('results/phase260_curriculum_train/stage2_r045.pt', map_location=DEVICE, weights_only=False)
print(f"  Checkpoint: stage={ckpt['stage']}, epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")

from scripts.train_phase227 import GoalConditionedPolicy
policy = GoalConditionedPolicy().to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'], strict=False)
policy.eval()
print("  Policy loaded OK\n")

def eval_policy(policy, n_goals, max_steps, success_radius, seed):
    np.random.seed(seed)
    successes = 0
    results = []
    for ep_i in range(n_goals):
        goal = np.array([np.random.uniform(-0.40, 0.40), np.random.uniform(-0.34, 0.34)])
        sim = LeKiWiSimURDF()
        sim.reset()
        base_body_id = sim.model.body('base').id
        arm = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0])
        if ep_i == 0:
            _ = sim.render()
        sim.step(np.zeros(9))
        steps = 0
        for step in range(max_steps):
            base_xy = sim.data.xpos[base_body_id, :2]
            dist = np.linalg.norm(goal - base_xy)
            if dist < success_radius:
                break
            img = sim.render().astype(np.uint8)
            state = build_state(sim, goal)
            img_t = torch.from_numpy(preprocess_image(img)).unsqueeze(0).float().to(DEVICE)
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4).cpu().numpy()[0]
            action = np.clip(action, -0.5, 0.5)
            sim.step(action)
            steps += 1
        final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        success = final_dist < success_radius
        successes += int(success)
        results.append({'goal': goal.tolist(), 'final_dist': float(final_dist), 'steps': steps, 'success': success})
        status = 'OK' if success else 'FAIL'
        print(f"  ep{ep_i+1}: goal=({goal[0]:.2f},{goal[1]:.2f}) dist={final_dist:.3f}m steps={steps} {status}")
    sr = successes / n_goals
    mean_steps = np.mean([e['steps'] for e in results])
    print(f"  SR: {successes}/{n_goals} = {sr*100:.1f}%, mean_steps={mean_steps:.1f}")
    return {'success_rate': sr, 'successes': successes, 'mean_steps': mean_steps}

# 10-goal eval
r = eval_policy(policy, 10, 200, 0.10, 42)
print(f"\nRESULT: {r['successes']}/10 = {r['success_rate']*100:.0f}% SR")