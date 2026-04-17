#!/usr/bin/env python3
"""
Phase 130: Test flow matching inference with MORE STEPS (fast version).
Goal: Determine if 8-step or 16-step Euler produces bounded outputs.

Usage:
    python3 scripts/test_inference_steps.py
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from PIL import Image

# ─── Load policy ──────────────────────────────────────────────────────────────

def load_policy():
    """Load CLIP-FM policy from latest checkpoint."""
    import os
    import json
    
    policy_dir = 'results/task_oriented_50ep'
    
    if not os.path.exists(policy_dir):
        print(f"[ERROR] Policy dir not found: {policy_dir}")
        return None
    
    state_dim, action_dim, hidden = 9, 9, 512
    config_path = os.path.join(policy_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        state_dim = cfg.get('state_dim', 9)
        action_dim = cfg.get('action_dim', 9)
        hidden = cfg.get('hidden', 512)
    
    from scripts.train_clip_fm import CLIPVisionEncoder, CLIPFlowMatchingPolicy
    
    policy = CLIPFlowMatchingPolicy(state_dim=state_dim, action_dim=action_dim, 
                                     hidden=hidden, device='cpu')
    
    possible_paths = [
        os.path.join(policy_dir, 'policy.pt'),
        os.path.join(policy_dir, 'final_policy.pt'),
    ]
    ckpt_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        policy.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded policy from {ckpt_path}")
    else:
        print(f"[WARN] No checkpoint found")
    
    policy.eval()
    return policy


def resize_for_clip(img):
    """Convert numpy image [H,W,3] to CLIP tensor [1,3,224,224]."""
    transform = __import__('torchvision.transforms', fromlist=['Compose']).Compose([
        __import__('torchvision.transforms', fromlist=['Resize']).Resize(224, interpolation=Image.BICUBIC),
        __import__('torchvision.transforms', fromlist=['CenterCrop']).CenterCrop(224),
        __import__('torchvision.transforms', fromlist=['ToTensor']).ToTensor(),
        __import__('torchvision.transforms', fromlist=['Normalize']).Normalize(
            mean=[0.481, 0.457, 0.408], std=[0.269, 0.261, 0.275]),
    ])
    return transform(Image.fromarray(img)).unsqueeze(0)


# ─── Fast test: one sim instance, many inferences ──────────────────────────────

def fast_inference_test(policy, num_runs=40):
    """Run inference test with ONE sim instance, reuse for speed."""
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    print("\n" + "="*60)
    print("FAST INFERENCE STEP COUNT TEST")
    print("="*60)
    
    # Create ONE sim
    sim = LeKiWiSimURDF()
    sim.reset()
    for _ in range(30):
        sim.step(np.zeros(9))
    
    # Get image + state ONCE
    img = sim.render()
    img_t = resize_for_clip(img)
    
    base_id = sim.model.body('base').id
    
    results = {4: [], 8: [], 16: []}
    
    def make_state():
        return np.concatenate([
            sim.data.qpos[:2],
            sim.data.qpos[2:6],
            sim.data.qpos[6:9],
        ]).astype(np.float32)
    
    def normalize_action_tanh(raw_action):
        raw = np.asarray(raw_action, dtype=np.float32)
        arm = raw[:6]
        wheel = raw[6:9]
        arm_n = 3.14 * np.tanh(arm / 3.14)
        wheel_n = 0.5 * np.tanh(wheel / 0.5)
        return np.concatenate([arm_n, wheel_n])
    
    for step_count in [4, 8, 16]:
        print(f"\n--- {step_count}-step Euler ---")
        
        for run in range(num_runs):
            # Resimulate robot forward a bit (to get different states) - reduced steps
            for _ in range(2):
                sim.step(np.zeros(9))
            
            state_9d = make_state()
            state_t = torch.from_numpy(state_9d).unsqueeze(0).cpu()
            
            with torch.no_grad():
                raw = policy.infer(img_t, state_t, num_steps=step_count).numpy().squeeze()
            
            action = normalize_action_tanh(raw)
            wheel = action[6:9]
            
            results[step_count].append({
                'wheel_raw': raw[6:9].copy(),
                'wheel_denorm': wheel.copy(),
                'raw_max_abs': np.abs(raw[6:9]).max(),
                'denorm_max_abs': np.abs(wheel).max(),
            })
        
        raw_max = [r['raw_max_abs'] for r in results[step_count]]
        denorm_max = [r['denorm_max_abs'] for r in results[step_count]]
        print(f"  Raw outputs:      mean_max={np.mean(raw_max):.3f}, max={np.max(raw_max):.3f}, std={np.std(raw_max):.3f}")
        print(f"  Denorm wheel spd:  mean_max={np.mean(denorm_max):.4f}, max={np.max(denorm_max):.4f}, >0.3 count={sum(1 for x in denorm_max if x > 0.3)}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    for n_steps in [4, 8, 16]:
        raw_max = [r['raw_max_abs'] for r in results[n_steps]]
        denorm_max = [r['denorm_max_abs'] for r in results[n_steps]]
        print(f"  {n_steps}-step: raw_max mean={np.mean(raw_max):.3f}, denorm_max mean={np.mean(denorm_max):.4f}")
    
    return results


def quick_sr_test(policy, step_count, num_episodes=8, max_steps=150):
    """Quick success rate test."""
    from sim_lekiwi_urdf import LeKiWiSimURDF
    
    print(f"\n--- {step_count}-step SR test ({num_episodes} eps) ---")
    
    def normalize_action_tanh(raw_action):
        raw = np.asarray(raw_action, dtype=np.float32)
        arm = raw[:6]
        wheel = raw[6:9]
        arm_n = 3.14 * np.tanh(arm / 3.14)
        wheel_n = 0.5 * np.tanh(wheel / 0.5)
        return np.concatenate([arm_n, wheel_n])
    
    goals = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5),
             (0.4, 0.3), (-0.4, 0.3), (0.3, -0.4), (-0.3, -0.4)][:num_episodes]
    
    successes = 0
    for gi, (gx, gy) in enumerate(goals):
        sim = LeKiWiSimURDF()
        sim.reset()
        sim.set_target([gx, gy])
        
        for _ in range(30):
            sim.step(np.zeros(9))
        
        for step in range(max_steps):
            img = sim.render()
            img_t = resize_for_clip(img)
            state_9d = np.concatenate([
                sim.data.qpos[:2], sim.data.qpos[2:6], sim.data.qpos[6:9],
            ]).astype(np.float32)
            state_t = torch.from_numpy(state_9d).unsqueeze(0).cpu()
            
            with torch.no_grad():
                raw = policy.infer(img_t, state_t, num_steps=step_count).numpy().squeeze()
            action = normalize_action_tanh(raw)
            
            sim.step(action)
            
            base_id = sim.model.body('base').id
            pos = sim.data.xpos[base_id, :2]
            dist = np.linalg.norm(pos - np.array([gx, gy]))
            
            if dist < 0.15:
                successes += 1
                break
    
    sr = successes / num_episodes
    print(f"  → SR: {successes}/{num_episodes} ({sr*100:.0f}%)")
    return sr


def main():
    print("Phase 130: Flow Matching Inference Step Count Test")
    print("="*60)
    
    policy = load_policy()
    if policy is None:
        print("[ABORT] No policy found")
        sys.exit(1)
    
    # Part 1: Raw + denorm output range test (fast, 40 runs per step count)
    results = fast_inference_test(policy, num_runs=40)
    
    # Part 2: Quick success rate
    print("\n" + "="*60)
    print("SUCCESS RATE TEST")
    print("="*60)
    for step_count in [4, 8]:
        sr = quick_sr_test(policy, step_count, num_episodes=8, max_steps=150)
    
    print("\n[DONE] Phase 130 complete")


if __name__ == '__main__':
    main()