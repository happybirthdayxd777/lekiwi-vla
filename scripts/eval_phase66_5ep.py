#!/usr/bin/env python3
"""Phase 66d: Run 5 episodes, 200 steps — match Phase 34 benchmark settings"""
import torch, numpy as np, sys
sys.path.insert(0, '.')
from sim_lekiwi_urdf import LeKiWiSimURDF
import torch.nn as nn
from PIL import Image
import json

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

print('Loading policy...')
policy = CLIPFM().to(device)
sd = torch.load('results/phase63_reachable_train/final_policy.pt', map_location=device, weights_only=False)
policy.load_state_dict(sd, strict=False)
policy.eval()

def resize224(img):
    if img is None: return np.zeros((224,224,3), dtype=np.uint8)
    pil = Image.fromarray(img)
    pil = pil.resize((224, 224), Image.BILINEAR)
    return np.array(pil)

# Run 5 episodes, 200 steps each (matching Phase 34 benchmark)
print('Running 5 episodes x 200 steps...')
all_rewards = []
all_dists = []
all_success = []

for ep in range(5):
    sim = LeKiWiSimURDF()
    np.random.seed(ep * 100)
    
    target = sim._target[:2]
    ep_reward = 0.0
    reached = False
    min_dist = 999.0
    
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
        
        r = sim.get_reward()
        ep_reward += r
        dist = np.linalg.norm(sim.data.qpos[:2] - target)
        min_dist = min(min_dist, dist)
        if dist < 0.1 and not reached:
            reached = True
    
    final_dist = np.linalg.norm(sim.data.qpos[:2] - target)
    all_rewards.append(ep_reward)
    all_dists.append(final_dist)
    all_success.append(1.0 if reached else 0.0)
    
    status = "✓ REACHED" if reached else f"dist={final_dist:.3f}m"
    print(f'  Episode {ep+1}: reward={ep_reward:+.3f}, min_dist={min_dist:.3f}m, {status}')

print(f'\n=== phase63_reachable_train EVAL ===')
print(f'Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}')
print(f'Mean final dist: {np.mean(all_dists):.3f} ± {np.std(all_dists):.3f}m')
print(f'Success rate: {np.mean(all_success)*100:.0f}%')
print(f'Episodes: 5, Steps: 200, Sim: URDF')

# Save result
result = {
    'policy': 'phase63_reachable_train',
    'episodes': 5,
    'steps': 200,
    'mean_reward': float(np.mean(all_rewards)),
    'std_reward': float(np.std(all_rewards)),
    'mean_distance': float(np.mean(all_dists)),
    'std_distance': float(np.std(all_dists)),
    'success_rate': float(np.mean(all_success)),
    'all_rewards': [float(r) for r in all_rewards],
    'all_success': [float(s) for s in all_success],
}
with open('data/phase66_eval_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved to data/phase66_eval_results.json')