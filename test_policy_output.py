#!/usr/bin/env python3
"""Standalone flow matching test — verify policy inference works correctly."""
import torch
import torch.nn as nn
import numpy as np

# Replicate FlowMatchingHead exactly as in train_clip_fm.py
class FlowMatchingHead(nn.Module):
    def __init__(self, vision_dim=512, state_dim=9, action_dim=9, hidden=512):
        super().__init__()
        self.action_dim = action_dim
        total_dim = vision_dim + state_dim + action_dim + 256  # 786
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
        )
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, action_dim),
        )
        self.skip = nn.Linear(action_dim, action_dim, bias=False)

    def forward(self, vis, state, noisy_action, timestep):
        t_feat = self.time_mlp(timestep)
        x = torch.cat([vis, state, noisy_action, t_feat], dim=-1)
        return self.net(x) + self.skip(noisy_action)


class CLIPVisionEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            torch_dtype=torch.float32,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        for p in self.clip.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(768, 512).to(device)

    def forward(self, images):
        with torch.no_grad():
            outputs = self.clip.get_image_features(pixel_values=images)
            vis = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0].mean(dim=1)
        return self.proj(vis)


class CLIPFlowMatchingPolicy(nn.Module):
    def __init__(self, state_dim=9, action_dim=9, hidden=512, device="cpu"):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead(vision_dim=hidden, state_dim=state_dim,
                                          action_dim=action_dim, hidden=hidden)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

    @torch.no_grad()
    def infer(self, image, state, num_steps=4):
        action = torch.randn(image.shape[0], self.action_dim, device=self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full([image.shape[0], 1], 1.0 - i * dt, device=self.device)
            vis = self.vision_encoder(image)
            velocity = self.flow_head(vis, state, action, t)
            action = action - dt * velocity
        return action


# Test 1: Check flow head with fixed inputs (no CLIP dependency)
print("=" * 50)
print("Test 1: FlowMatchingHead standalone (no CLIP)")
print("=" * 50)
fh = FlowMatchingHead(vision_dim=512, state_dim=9, action_dim=9, hidden=512)
vis = torch.randn(1, 512) * 0.1
state = torch.randn(1, 9) * 0.1
noisy = torch.randn(1, 9)
t = torch.tensor([[0.5]])
out = fh(vis, state, noisy, t)
print(f"Forward pass: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

# Test 2: Full policy inference with random image
print("\n" + "=" * 50)
print("Test 2: Full policy inference")
print("=" * 50)
policy = CLIPFlowMatchingPolicy(state_dim=9, action_dim=9, hidden=512, device='cpu')
ckpt = torch.load('/Users/i_am_ai/hermes_research/lekiwi_vla/results/task_oriented_50ep/final_policy.pt',
                  map_location='cpu', weights_only=False)
policy.load_state_dict(ckpt, strict=False)
policy.eval()
print("Policy loaded")

img_t = torch.randn(1, 3, 224, 224) * 0.5 + 0.5  # Random image [0,1]
state_t = torch.randn(1, 9) * 0.1  # Near-zero state

print("\nPolicy inference with random inputs:")
for i in range(5):
    with torch.no_grad():
        raw = policy.infer(img_t, state_t, num_steps=4).numpy().squeeze()
    print(f"  Trial {i+1}: {raw[:3]}... (wheel: {raw[6]:.3f}, {raw[7]:.3f}, {raw[8]:.3f})")

# Test 3: Check with image from P126 dataset
print("\n" + "=" * 50)
print("Test 3: With real P126 image and zero state")
print("=" * 50)
import h5py
from PIL import Image

f = h5py.File('/Users/i_am_ai/hermes_research/lekiwi_vla/data/lekiwi_goal_p126_20ep.h5', 'r')
img_np = f['images'][0]
f.close()

img_pil = Image.fromarray(img_np, 'RGB')
img_resized = img_pil.resize((224, 224), Image.BILINEAR)
img_np_f = np.array(img_resized).astype(np.float32) / 255.0
img_chw = img_np_f.transpose(2, 0, 1)
img_t_real = torch.from_numpy(img_chw).unsqueeze(0).cpu()
state_zero = torch.zeros(1, 9)

print("With real image, zero state:")
for i in range(5):
    with torch.no_grad():
        raw = policy.infer(img_t_real, state_zero, num_steps=4).numpy().squeeze()
    print(f"  Trial {i+1}: wheel=[{raw[6]:.3f}, {raw[7]:.3f}, {raw[8]:.3f}]")
    print(f"           arm=[{raw[0]:.3f}, {raw[1]:.3f}]")

# Test 4: Check action output distribution
print("\n" + "=" * 50)
print("Test 4: Action output distribution (100 samples)")
print("=" * 50)
wheel_outputs = []
arm_outputs = []
for _ in range(100):
    with torch.no_grad():
        raw = policy.infer(img_t_real, state_zero, num_steps=4).numpy().squeeze()
    wheel_outputs.extend(raw[6:9].tolist())
    arm_outputs.extend(raw[:6].tolist())

wheel_outputs = np.array(wheel_outputs)
arm_outputs = np.array(arm_outputs)
print(f"Wheel outputs: mean={wheel_outputs.mean():.3f}, std={wheel_outputs.std():.3f}, min={wheel_outputs.min():.3f}, max={wheel_outputs.max():.3f}")
print(f"Arm outputs:   mean={arm_outputs.mean():.3f}, std={arm_outputs.std():.3f}, min={arm_outputs.min():.3f}, max={arm_outputs.max():.3f}")

# For reference: the normalize_action function
LEKIWI_ARM_LIMITS = np.array([[-3.14,3.14],[-1.57,1.57],[-1.57,1.57],[-1.57,1.57],[-3.14,3.14],[0.0,0.04]], dtype=np.float32)
LEKIWI_WHEEL_LIMITS = np.array([[-5.0,5.0]]*3, dtype=np.float32)
print(f"\nNormalized wheel range (if policy outputs [-1,1]): {LEKIWI_WHEEL_LIMITS}")
print(f"Normalized arm range (if policy outputs [-1,1]): {LEKIWI_ARM_LIMITS}")