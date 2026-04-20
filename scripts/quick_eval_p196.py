#!/usr/bin/env python3
"""Quick 5-goal eval of Phase196 VLA with fixed methodology."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, '.')
os.chdir('.')

from scripts.train_phase196 import GoalConditionedPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS
from PIL import Image

DEVICE = 'cpu'
ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_14.pt', map_location='cpu', weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img):
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

n_goals = 5
max_steps = 200
success_radius = 0.1
successes = 0
dists = []
steps_list = []

np.random.seed(42)
t0 = time.time()
for g_i in range(n_goals):
    goal = np.random.uniform(-0.3, 0.4, 2)
    sim = LeKiWiSimURDF()
    sim.reset()
    base_body_id = sim.model.body('base').id
    reached = False
    for step in range(max_steps):
        state = build_state(sim, goal)
        image = sim.render()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            img = preprocess_image(image).unsqueeze(0).to(DEVICE)
            action = policy.infer(img, s, num_steps=4).cpu().numpy()[0]
        action = np.clip(action, -0.5, 0.5)
        sim.step(action)
        dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        if dist < success_radius:
            successes += 1
            dists.append(dist)
            steps_list.append(step + 1)
            reached = True
            break
    if not reached:
        dists.append(dist)
        steps_list.append(max_steps)

elapsed = time.time() - t0
print(f"Quick 5-goal test: {successes}/{n_goals} = {100*successes/n_goals:.0f}% SR")
print(f"Mean dist: {np.mean(dists):.3f}m, Mean steps: {np.mean(steps_list):.1f}")
print(f"Time: {elapsed:.1f}s")
