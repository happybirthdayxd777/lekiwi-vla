#!/usr/bin/env python3
"""Phase 70b: Multi-seed URDF stability + SR test"""
import torch, numpy as np, sys, os
from PIL import Image
sys.path.insert(0, '.')
from sim_lekiwi_urdf import LeKiWiSimURDF
import torch.nn as nn

device = 'cpu'

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
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    return np.array(pil)

print("Loading policy...")
policy = CLIPFM().to(device)
sd = torch.load('results/phase63_reachable_train/final_policy.pt', map_location=device, weights_only=False)
policy.load_state_dict(sd, strict=False)
policy.eval()

# Goals from Phase 63 training data
GOALS = [(0.3, 0.2), (0.2, -0.2), (0.3, 0.4), (0.15, 0.1), (0.4, 0.1)]
MAX_STEPS = 200
SEEDS = [42, 123, 456, 789, 999]

results = []
for goal_idx, goal in enumerate(GOALS):
    goal_str = f"({goal[0]},{goal[1]})"
    success_count = 0
    nan_count = 0
    all_dists = []
    all_steps = []
    
    for ep, seed in enumerate(SEEDS):
        sim = LeKiWiSimURDF()
        sim.reset(target=goal, seed=seed)
        
        ep_dists = []
        reached = False
        nan_occurred = False
        
        for step in range(MAX_STEPS):
            arm_pos = sim.data.qpos[10:16]
            wheel_v = sim.data.qvel[6:9]
            state = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
            
            img = resize224(sim.render())
            img_t = (torch.from_numpy(img).permute(2,0,1).float() / 255.0).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = policy.infer(img_t, state_t, num_steps=4)
            action_np = action.cpu().numpy()[0]
            
            # CLAMP wheel actions for stability
            action_clamped = action_np.copy()
            action_clamped[6:9] = np.clip(action_clamped[6:9], -0.5, 0.5)
            
            sim.step(action_clamped)
            
            if np.any(np.isnan(sim.data.qvel)) or np.any(np.isinf(sim.data.qvel)):
                nan_occurred = True
                break
            
            dist = np.linalg.norm(sim.data.qpos[:2] - np.array(goal))
            ep_dists.append(dist)
            if dist < 0.1 and not reached:
                reached = True
            
            if reached and step > 20:
                break
        
        if nan_occurred:
            nan_count += 1
            all_dists.append(ep_dists[-1] if ep_dists else 999)
            all_steps.append(MAX_STEPS)
        else:
            all_dists.append(min(ep_dists) if ep_dists else 999)
            all_steps.append(len(ep_dists))
            if reached:
                success_count += 1
    
    sr = success_count / len(SEEDS) * 100
    mean_dist = np.mean(all_dists)
    mean_steps = np.mean(all_steps)
    
    print(f"Goal {goal_str}: SR={sr:.0f}% ({success_count}/{len(SEEDS)}), mean_dist={mean_dist:.3f}m, mean_steps={mean_steps:.0f}, nan={nan_count}")
    results.append((goal, sr, mean_dist, mean_steps, nan_count))

print("\n=== SUMMARY ===")
total_sr = np.mean([r[1] for r in results])
total_nan = sum([r[4] for r in results])
print(f"Overall SR: {total_sr:.1f}%")
print(f"Total NaN events: {total_nan} / {len(GOALS)*len(SEEDS)}")
