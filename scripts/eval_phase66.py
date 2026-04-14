#!/usr/bin/env python3
"""Phase 66: Quick eval of phase63_reachable_train policy"""
import torch, numpy as np, sys, time
sys.path.insert(0, '.')
from sim_lekiwi_urdf import LeKiWiSimURDF
import torch.nn as nn
from PIL import Image

print('=== Phase 66: Eval phase63_reachable_train policy ===')
device = 'cpu'

# Policy matching train_clip_fm.py architecture
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

print('Loading policy...')
policy = CLIPFM().to(device)
sd = torch.load('results/phase63_reachable_train/final_policy.pt', map_location=device, weights_only=False)
missing, unexpected = policy.load_state_dict(sd, strict=False)
print(f'Loaded: missing={len(missing)}, unexpected={len(unexpected)}')
policy.eval()

def resize224(img):
    if img is None: return np.zeros((224,224,3), dtype=np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    return np.array(pil)

print('Running 3 episodes x 100 steps...')
results = []
for ep in range(3):
    sim = LeKiWiSimURDF()
    np.random.seed(ep * 100)
    
    ep_reward = 0.0
    reached = False
    
    for step in range(100):
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
        
        r = sim.get_reward()
        ep_reward += r
        if r > 0 and not reached:
            reached = True
    
    goal = getattr(sim, 'goal_xy', np.array([0.3, 0.0]))
    base_xy = sim.data.qpos[:2]
    final_dist = np.linalg.norm(base_xy - goal)
    
    results.append({'ep': ep+1, 'reward': ep_reward, 'reached': reached, 'dist': final_dist})
    print(f'  Episode {ep+1}: reward={ep_reward:+.3f}, reached={reached}, final_dist={final_dist:.3f}m')

print(f'\nMean reward: {np.mean([r["reward"] for r in results]):.3f}')
print(f'Success rate: {sum([r["reached"] for r in results])/len(results)*100:.0f}%')
print(f'Mean distance: {np.mean([r["dist"] for r in results]):.3f}m')