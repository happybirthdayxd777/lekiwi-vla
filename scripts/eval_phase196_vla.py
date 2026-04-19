#!/usr/bin/env python3
"""Phase 197: Evaluate Phase 196 VLA policy (epoch_4) - QUICK version"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_phase196 import GoalConditionedPolicy, DEVICE
from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS
from PIL import Image

ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_4.pt', map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(raw_img)
    img = img.resize((224, 224), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)

def build_state(sim, goal):
    arm_pos = np.array([sim.data.qpos[sim.model.joint(n).qposadr[0]] for n in ARM_JOINTS])
    wheel_vel = np.array([sim.data.qvel[sim.model.joint(n).dofadr[0]] for n in WHEEL_JOINTS])
    goal_norm = np.clip(goal / 0.4, -1, 1)
    return np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

print("=" * 60)
print("Phase 197: VLA Policy Evaluation (epoch 4) - QUICK")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Checkpoint loss: {ckpt['loss']:.4f}")

n_goals = 10
max_steps = 200
successes = 0
dists = []
np.random.seed(42)

for g_i in range(n_goals):
    goal = np.random.uniform(-0.3, 0.4, 2)
    sim = LeKiWiSimURDF()
    sim.reset()
    for _ in range(max_steps):
        state = build_state(sim, goal)
        image = sim.render()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            img = preprocess_image(image).unsqueeze(0).to(DEVICE)
            action = policy.infer(img, s, num_steps=4).cpu().numpy()[0]
        action = np.clip(action, -0.5, 0.5)
        sim.step(action)
    final_dist = np.linalg.norm(sim.data.qpos[:2] - goal)
    successes += int(final_dist < 0.3)
    dists.append(final_dist)
    print(f"  Goal {g_i+1}: dist={final_dist:.3f}m {'✓' if final_dist < 0.3 else '✗'}")

print(f"\nPhase 196 VLA (epoch 4, {max_steps} steps): {successes}/{n_goals} = {successes/n_goals*100:.0f}% SR")
print(f"Baseline Contact-Jacobian P-ctrl (200 steps): ~94% SR")
print(f"Mean final distance: {np.mean(dists):.3f}m")
