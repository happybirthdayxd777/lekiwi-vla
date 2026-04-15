#!/usr/bin/env python3
"""Phase 70: Isolate URDF NaN — test action magnitude effect"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from sim_lekiwi_urdf import LeKiWiSimURDF

device = 'cpu'

# Load policy
import torch.nn as nn
class CLIPVisionEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Linear(768, 512).to(device)
        for p in self.clip.parameters(): p.requires_grad = False
    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.uint8)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
        return self.proj(pooled)

class FlowMatchingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 256))
        total_dim = 512 + 9 + 9 + 256
        self.net = nn.Sequential(
            nn.Linear(total_dim, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, 9),
        )
    def forward(self, vision, state, action, t):
        t_emb = self.time_mlp(t)
        x = torch.cat([vision, state, action, t_emb], dim=-1)
        return self.net(x)

class CLIPFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(device=device)
        self.flow_head = FlowMatchingHead()
    def forward(self, image, state, action, t):
        vis = self.vision_encoder(image)
        return self.flow_head(vis, state, action, t)
    def infer(self, image, state, num_steps=4):
        x = torch.zeros(1, 9).to(state.device)
        dt = 1.0 / num_steps
        for s in range(num_steps):
            t = torch.full((1,1), s*dt, device=state.device)
            with torch.no_grad():
                residual = self.forward(image, state, x, t)
            x = x + dt * residual
        return x

def resize224(img):
    if img is None: return np.zeros((224,224,3), dtype=np.uint8)
    from PIL import Image
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    return np.array(pil)

print("Loading policy...")
policy = CLIPFM().to(device)
sd = torch.load('results/phase63_reachable_train/final_policy.pt', map_location=device, weights_only=False)
policy.load_state_dict(sd, strict=False)
policy.eval()

# Test 1: raw policy actions (no clamping)
print("\n=== Test 1: RAW policy actions (no action clamp) ===")
sim = LeKiWiSimURDF()
sim.reset(target=(0.3, 0.2), seed=42)
nan_at_step = None

for step in range(200):
    arm_pos = sim.data.qpos[10:16]
    wheel_v = sim.data.qvel[6:9]
    state = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
    
    img = resize224(sim.render())
    img_t = (torch.from_numpy(img).permute(2,0,1).float() / 255.0).unsqueeze(0).to(device)
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        action = policy.infer(img_t, state_t, num_steps=4)
    action_np = action.cpu().numpy()[0]
    sim.step(action_np)
    
    if np.any(np.isnan(sim.data.qvel)) or np.any(np.isinf(sim.data.qvel)):
        nan_at_step = step
        print(f"  NaN at step {step}! action_wheels={action_np[6:9]}")
        break
    
    if step % 20 == 0:
        dist = np.linalg.norm(sim.data.qpos[:2] - np.array([0.3, 0.2]))
        print(f"  Step {step}: dist={dist:.3f}, action_wheels=[{action_np[6]:.2f},{action_np[7]:.2f},{action_np[8]:.2f}]")

if nan_at_step is None:
    print("  No NaN in 200 steps (raw policy)")

# Test 2: CLAMPED actions (wheel magnitude <= 0.5)
print("\n=== Test 2: CLAMPED wheel actions (|wheel| <= 0.5) ===")
sim2 = LeKiWiSimURDF()
sim2.reset(target=(0.3, 0.2), seed=42)
nan_at_step2 = None

for step in range(200):
    arm_pos = sim2.data.qpos[10:16]
    wheel_v = sim2.data.qvel[6:9]
    state = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
    
    img = resize224(sim2.render())
    img_t = (torch.from_numpy(img).permute(2,0,1).float() / 255.0).unsqueeze(0).to(device)
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        action = policy.infer(img_t, state_t, num_steps=4)
    action_np = action.cpu().numpy()[0]
    
    # CLAMP: limit wheel magnitude
    action_clamped = action_np.copy()
    action_clamped[6:9] = np.clip(action_clamped[6:9], -0.5, 0.5)
    
    sim2.step(action_clamped)
    
    if np.any(np.isnan(sim2.data.qvel)) or np.any(np.isinf(sim2.data.qvel)):
        nan_at_step2 = step
        print(f"  NaN at step {step}! action_wheels={action_np[6:9]}")
        break
    
    if step % 20 == 0:
        dist = np.linalg.norm(sim2.data.qpos[:2] - np.array([0.3, 0.2]))
        print(f"  Step {step}: dist={dist:.3f}, action_wheels=[{action_np[6]:.2f},{action_np[7]:.2f},{action_np[8]:.2f}]")

if nan_at_step2 is None:
    print("  No NaN in 200 steps (clamped policy)")

# Test 3: ZERO actions (verify sim is stable with no motion)
print("\n=== Test 3: ZERO actions ===")
sim3 = LeKiWiSimURDF()
sim3.reset(target=(0.3, 0.2), seed=42)
for step in range(200):
    sim3.step(np.zeros(9, dtype=np.float32))
    if np.any(np.isnan(sim3.data.qvel)) or np.any(np.isinf(sim3.data.qvel)):
        print(f"  NaN at step {step} with ZERO actions!")
        break
else:
    print("  Zero actions: 200 steps stable ✓")
