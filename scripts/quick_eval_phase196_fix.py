#!/usr/bin/env python3
"""Quick 10-goal eval with detailed image diagnostic."""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_phase196 import GoalConditionedPolicy
from sim_lekiwi_urdf import LeKiWiSimURDF, ARM_JOINTS, WHEEL_JOINTS
from PIL import Image

DEVICE = 'cpu'
ckpt = torch.load('results/phase196_contact_jacobian_train/epoch_14.pt', map_location=DEVICE, weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9).to(DEVICE)
policy.load_state_dict(ckpt['policy_state_dict'])
policy.eval()

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(raw_img):
    if raw_img is None or raw_img.mean() < 1.0:
        raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
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

n_goals = 10
max_steps = 200
success_radius = 0.1
np.random.seed(42)

successes = 0
dists = []
steps_list = []
black_count = 0

t0 = time.time()
for g_i in range(n_goals):
    goal = np.random.uniform(-0.3, 0.4, 2)
    sim = LeKiWiSimURDF()
    sim.reset()
    base_body_id = sim.model.body('base').id

    reached = False
    for step in range(max_steps):
        if step == 0:
            sim.step(np.zeros(9))
        state = build_state(sim, goal)
        image = sim.render()
        img_mean = image.mean() if image is not None else 0.0
        if img_mean < 1.0:
            black_count += 1
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            img = preprocess_image(image).unsqueeze(0).to(DEVICE)
            action = policy.infer(img, s, num_steps=4).cpu().numpy()[0]
        action = np.clip(action, -0.5, 0.5)
        sim.step(action)

        dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
        if dist < success_radius:
            reached = True
            steps_list.append(step + 1)
            break

    final_dist = np.linalg.norm(sim.data.xpos[base_body_id, :2] - goal)
    successes += int(reached)
    dists.append(final_dist)
    mark = 'Y' if reached else 'N'
    print(f'  [{g_i+1:2d}/10] g=({goal[0]:+.2f},{goal[1]:+.2f}) d={final_dist:.3f}m {mark}')

elapsed = time.time() - t0
sr = successes / n_goals * 100
mean_dist = np.mean(dists)
mean_steps = np.mean(steps_list) if steps_list else max_steps

print(f'VLA Phase196: {successes}/{n_goals} = {sr:.1f}% SR ({n_goals}-goal)')
print(f'Mean dist: {mean_dist:.3f}m, Mean steps: {mean_steps:.1f}')
print(f'All-black frames encountered: {black_count}')
print(f'Total time: {elapsed:.0f}s')