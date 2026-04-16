#!/usr/bin/env python3
"""Quick policy inference test — inspect raw outputs."""
import sys, os, torch, numpy as np, h5py
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.expanduser('~/hermes_research/lekiwi_vla'))

# Load policy classes from train_clip_fm.py source
train_path = Path(os.path.expanduser('~/hermes_research/lekiwi_vla/scripts/train_clip_fm.py'))
src = train_path.read_text()

# Extract classes needed (CLIPVisionEncoder, FlowMatchingHead, CLIPFlowMatchingPolicy)
start = src.find('class CLIPVisionEncoder')
replay_start = src.find('class ReplayBuffer')
policy_src = src[start:replay_start]

# Add torch imports
policy_src = "import torch\nimport torch.nn as nn\n" + policy_src

# Create module namespace
namespace = {'nn': torch.nn, 'torch': torch}
exec(policy_src, namespace)

CLIPVisionEncoder = namespace['CLIPVisionEncoder']
FlowMatchingHead = namespace['FlowMatchingHead']
CLIPFlowMatchingPolicy = namespace['CLIPFlowMatchingPolicy']

# Load policy
ckpt = torch.load(
    os.path.expanduser('~/hermes_research/lekiwi_vla/results/task_oriented_50ep/final_policy.pt'),
    map_location='cpu', weights_only=False
)
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
policy.load_state_dict(ckpt, strict=False)
policy.eval()
print("Policy loaded OK")

# Load test image from P126
f = h5py.File(os.path.expanduser('~/hermes_research/lekiwi_vla/data/lekiwi_goal_p126_20ep.h5'), 'r')
img_np = f['images'][0]
f.close()

img_pil = Image.fromarray(img_np, 'RGB')
img_resized = img_pil.resize((224, 224), Image.BILINEAR)
img_np_f = np.array(img_resized).astype(np.float32) / 255.0
img_chw = img_np_f.transpose(2, 0, 1)
img_t = torch.from_numpy(img_chw).unsqueeze(0).cpu()

state = np.zeros(9, dtype=np.float32)
state_t = torch.from_numpy(state).unsqueeze(0).cpu()

# Run inference 5 times
print("\nRaw policy outputs (policy-space [-1, 1]):")
for i in range(5):
    with torch.no_grad():
        raw = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()
    print(f"  Trial {i+1}: arm=[{raw[0]:.3f}, {raw[1]:.3f}], wheel=[{raw[6]:.3f}, {raw[7]:.3f}, {raw[8]:.3f}]")

# Normalize
LEKIWI_ARM_LIMITS = np.array([[-3.14,3.14],[-1.57,1.57],[-1.57,1.57],[-1.57,1.57],[-3.14,3.14],[0.0,0.04]], dtype=np.float32)
LEKIWI_WHEEL_LIMITS = np.array([[-5.0,5.0]]*3, dtype=np.float32)

with torch.no_grad():
    raw = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()

arm_n = LEKIWI_ARM_LIMITS[:,0] + (raw[:6]+1)/2*(LEKIWI_ARM_LIMITS[:,1]-LEKIWI_ARM_LIMITS[:,0])
wheel_n = LEKIWI_WHEEL_LIMITS[:,0] + (raw[6:9]+1)/2*(LEKIWI_WHEEL_LIMITS[:,1]-LEKIWI_WHEEL_LIMITS[:,0])
action = np.concatenate([arm_n, wheel_n])
print(f"\nNormalized action (native units):")
print(f"  arm: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
print(f"  wheel: [{action[6]:.4f}, {action[7]:.4f}, {action[8]:.4f}]")

# Test with URDF sim
from sim_lekiwi_urdf import LeKiWiSimURDF
sim = LeKiWiSimURDF()
sim.reset()
base_id = sim.model.body('base').id
initial = sim.data.xpos[base_id, :2]
print(f"\nInitial: {initial}")

# Step the sim with this action
for _ in range(50):
    sim.step(action)

final = sim.data.xpos[base_id, :2]
print(f"After 50 steps with VLA action: {final}")
print(f"Movement: {np.linalg.norm(final - initial):.4f}m")

# Also test with P-controller action for comparison
print("\n--- P-controller comparison ---")
sim2 = LeKiWiSimURDF()
sim2.reset()
base_id2 = sim2.model.body('base').id
initial2 = sim2.data.xpos[base_id2, :2]

from sim_lekiwi_urdf import twist_to_contact_wheel_speeds
for step in range(50):
    pos = sim2.data.xpos[base_id2, :2]
    # Move toward (0.5, 0.5)
    dx, dy = 0.5 - pos[0], 0.5 - pos[1]
    d = np.linalg.norm([dx, dy])
    if d > 0.01:
        v_mag = min(1.5 * d, 0.3)
        vx, vy = v_mag * dx / d, v_mag * dy / d
    else:
        vx, vy = 0.0, 0.0
    ws = twist_to_contact_wheel_speeds(vx, vy)
    a = np.zeros(9)
    a[6:9] = np.clip(ws, -0.5, 0.5)
    sim2.step(a)

final2 = sim2.data.xpos[base_id2, :2]
print(f"P-controller after 50 steps: {final2}, moved {np.linalg.norm(final2 - initial2):.4f}m")