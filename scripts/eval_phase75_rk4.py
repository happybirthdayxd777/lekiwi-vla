#!/usr/bin/env python3
"""Phase 75: Test RK4 + wheel damping + velocity clamp for NaN stability."""
import torch, numpy as np, sys, cv2
sys.path.insert(0, '.')
from sim_lekiwi_urdf import LeKiWiSimURDF
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

device = 'cpu'

class CLIPVisionEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
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
        action = torch.zeros_like(state)
        for s in range(num_steps):
            t = torch.full((state.shape[0], 1), s / num_steps, dtype=torch.float32, device=device)
            pred = self.flow_head(self.vision_encoder(image), state, action, t)
            if s < num_steps - 1:
                action = action + pred / num_steps
            else:
                action = pred
        return action

def resize224(img):
    return cv2.resize(img, (224, 224))

print("Loading policy...")
policy = CLIPFM().to(device)
sd = torch.load('results/phase63_reachable_train/final_policy.pt', map_location=device, weights_only=False)
policy.load_state_dict(sd, strict=False)
policy.eval()

# ── Test 1: Verify RK4 + damping parameters ──────────────────────────────
print("\n=== Config verification ===")
sim = LeKiWiSimURDF()
print(f"  Integrator: {'RK4' if sim.model.opt.integrator == 2 else 'Euler/Other'}")
print(f"  Timestep: {sim.model.opt.timestep}")
w1_adr = sim.model.joint('w1').dofadr
print(f"  Wheel damping: w1={sim.model.dof_damping[w1_adr][0]:.1f}, w2={sim.model.dof_damping[w1_adr+1][0]:.1f}, w3={sim.model.dof_damping[w1_adr+2][0]:.1f}")

# ── Test 2: NaN stability — 5 episodes × 300 steps ─────────────────────
print("\n=== NaN Stability Test (5 eps × 300 steps, wheel clamp=0.5) ===")
results = []
for ep in range(5):
    sim = LeKiWiSimURDF()
    seed = 42 + ep * 100
    sim.reset(target=(0.3, 0.2), seed=seed)
    nan = False
    nan_step = None
    for step in range(300):
        arm_pos = sim.data.qpos[10:16]
        wheel_v = sim.data.qvel[6:9]
        state = np.concatenate([arm_pos, wheel_v]).astype(np.float32)
        img = resize224(sim.render())
        img_t = (torch.from_numpy(img).permute(2,0,1).float() / 255.0).unsqueeze(0).to(device)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.infer(img_t, state_t, num_steps=4)
        action_np = action.cpu().numpy()[0]
        action_np[6:9] = np.clip(action_np[6:9], -0.5, 0.5)
        obs, reward, done, _ = sim.step(action_np)
        if np.any(np.isnan(sim.data.qvel)) or np.any(np.isinf(sim.data.qvel)):
            nan = True
            nan_step = step
            break
    if nan:
        print(f"  Ep{ep+1} (seed={seed}): NaN at step {nan_step}")
    else:
        dist = float(np.linalg.norm(sim._target[:2] - sim.data.qpos[:2]))
        print(f"  Ep{ep+1} (seed={seed}): OK — dist={dist:.3f}m")
    results.append((nan, nan_step))

nan_count = sum(1 for r in results if r[0])
print(f"\nNaN rate: {nan_count}/5 episodes")
