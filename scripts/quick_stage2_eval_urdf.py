#!/usr/bin/env python3
"""
Phase 273: Stage2 policy eval on URDF sim — 10 goals, 200 steps each.
Purpose: Verify Stage2 policy works on URDF sim (not just primitive sim).

Stage2 checkpoint: results/phase260_curriculum_train/stage2_r045.pt
Goal-radius constraint: |r| < 0.45m (should be safe for Stage2)
"""
import sys, numpy as np, os, torch
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/hermes_research/lekiwi_vla"))
from sim_lekiwi_urdf import LeKiWiSimURDF
from scripts.train_curriculum import GoalConditionedPolicy

DEVICE = 'cpu'
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img):
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    arr = (arr - IMG_MEAN) / IMG_STD
    return arr.transpose(2, 0, 1)

ckpt_path = os.path.expanduser("~/hermes_research/lekiwi_vla/results/phase260_curriculum_train/stage2_r045.pt")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
policy = GoalConditionedPolicy(state_dim=11, action_dim=9, hidden=512, device=DEVICE).to(DEVICE)
policy.load_state_dict(ckpt.get('policy_state_dict', ckpt), strict=False)
policy.eval()
print(f"Stage2: epoch={ckpt.get('epoch','?')}, loss={ckpt.get('loss','?')}")

sim = LeKiWiSimURDF()
np.random.seed(42)

goals = [
    np.array([0.20,  0.15]),
    np.array([-0.15,  0.25]),
    np.array([0.30, -0.10]),
    np.array([-0.25, -0.20]),
    np.array([0.10, -0.30]),
    np.array([0.35,  0.20]),
    np.array([-0.30,  0.15]),
    np.array([0.15,  0.35]),
    np.array([-0.20, -0.35]),
    np.array([0.25,  0.00]),
]

MAX_STEPS = 200
SUCCESS_RADIUS = 0.10

results = []
for g_idx, goal in enumerate(goals):
    sim.reset()
    sim._goal_xy = goal.copy()

    for step in range(MAX_STEPS):
        arm_pos = sim.data.qpos[9:15]
        wheel_vel = sim.data.qvel[6:9]
        goal_norm = np.array([goal[0]/0.40, goal[1]/0.34])
        state = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)

        img = sim.render()
        img_t = torch.from_numpy(preprocess(img)[None]).float().to(DEVICE)
        st_t = torch.from_numpy(state[np.newaxis]).float().to(DEVICE)

        with torch.no_grad():
            action = policy.infer(img_t, st_t, num_steps=4)[0].cpu().numpy()

        # Stage2 action: arm_torque(6) + wheel_speed(3)
        # Convert to ctrl: wheel speeds -> wheel torques
        wheel_speeds = np.clip(action[6:9], -0.5, 0.5)  # already rad/s, clip
        ctrl_action = np.concatenate([action[:6], wheel_speeds])
        sim.step(ctrl_action)

        dist = np.linalg.norm(sim.data.xpos[sim.model.body('base').id, :2] - goal)
        if dist < SUCCESS_RADIUS:
            print(f"  Goal {g_idx}: SUCCESS @{step}, dist={dist:.4f}m")
            results.append((g_idx, 'success', step, dist))
            break
    else:
        final_dist = np.linalg.norm(sim.data.xpos[sim.model.body('base').id, :2] - goal)
        print(f"  Goal {g_idx}: FAILED, final_dist={final_dist:.3f}m, goal={goal}")
        results.append((g_idx, 'fail', MAX_STEPS, final_dist))

sr = sum(1 for r in results if r[1] == 'success') / len(results)
print(f"\nStage2 on URDF sim: {sr*100:.0f}% SR ({sum(1 for r in results if r[1]=='success')}/{len(results)})")
print(f"Mean steps to success: {np.mean([r[2] for r in results if r[1]=='success']):.1f}")
