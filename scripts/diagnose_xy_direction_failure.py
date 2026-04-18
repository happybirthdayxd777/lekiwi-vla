#!/usr/bin/env python3
"""
Phase 175: Diagnose +X-Y Direction Failure Root Cause
======================================================
Uses the SAME GoalConditionedPolicy class and load_policy() as eval_phase174_wheel_fix.py
(which works correctly and gets 53.3% SR).

This traces VLA wheel actions for +X-Y FAILED vs +X+Y SUCCESS episodes.
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[Phase 175] Device: {DEVICE}")


class CLIPSpatialEncoder(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        from transformers import CLIPModel
        print("[INFO] Loading CLIP ViT-B/32 (frozen)...")
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", torch_dtype=torch.float32,
        ).to(device)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images):
        pixel_values = (images * 255).clamp(0, 255).round().to(torch.float32)
        pixel_values = pixel_values.to(self.clip.device)
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = outputs.last_hidden_state
        return hidden


class GoalConditionedPolicy(nn.Module):
    """SAME ARCHITECTURE as eval_phase174_wheel_fix.py - this is the working version."""
    def __init__(self, state_dim=11, goal_dim=2, action_dim=9,
                 cross_heads=8, hidden=512, device=DEVICE):
        super().__init__()
        self.device = device
        self.clip_encoder = CLIPSpatialEncoder(device)
        self.vision_proj = nn.Linear(768, hidden).to(device)
        self.goal_mlp = nn.Sequential(
            nn.Linear(goal_dim, 256), nn.ReLU(), nn.Linear(256, 128),
        ).to(device)
        self.goal_proj = nn.Linear(128, 256).to(device)
        self.q_proj = nn.Linear(256, hidden).to(device)
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        ).to(device)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=cross_heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.time_net = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 256),
        ).to(device)
        self.action_head = nn.Sequential(
            nn.Linear(256 + 128 + hidden + 256, hidden),
            nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, action_dim),
        ).to(device)
        self.skip = nn.Linear(action_dim, action_dim, bias=False).to(device)

    def forward(self, image, state, noisy_action, t):
        clip_feat = self.clip_encoder(image)
        clip_proj = self.vision_proj(clip_feat)
        goal_emb = self.goal_mlp(state[:, 9:11])  # last 2 dims = goal
        goal_q = self.goal_proj(goal_emb)
        state_feat = self.state_net(state)
        q = self.q_proj(state_feat + goal_q)
        q = q.unsqueeze(1)
        cross_out, _ = self.cross_attn(q, clip_proj, clip_proj)
        cross_out = self.cross_norm(cross_out + q)
        t_feat = self.time_net(t)
        combined = torch.cat([state_feat, goal_emb, cross_out.squeeze(1), t_feat], dim=-1)
        v_pred = self.action_head(combined)
        v_pred = v_pred + self.skip(noisy_action)
        return v_pred

    def infer(self, image, state, num_steps=4):
        self.eval()
        x = torch.zeros_like(state[:, :9]).to(self.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(state.shape[0], 1).to(self.device) * (i * dt)
            v = self.forward(image, state, x, t)
            x = x + v * dt
        return torch.clamp(x, -0.5, 0.5)


def load_policy(ckpt_path, device=DEVICE):
    policy = GoalConditionedPolicy(state_dim=11, goal_dim=2, action_dim=9,
                                   cross_heads=8, hidden=512, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('policy_state_dict', ckpt)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device).eval()
    policy.device = device
    print(f"[LOAD] epoch={ckpt.get('epoch','?')}, eval_sr={ckpt.get('eval_sr','?')}")
    return policy


def resize_for_clip(img):
    pil = Image.fromarray(img)
    pil_resized = pil.resize((224, 224), Image.BILINEAR)
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    return img_np.transpose(2, 0, 1)


WHEEL_SCALE = 0.0834


def run_diagnostic_episode(sim, goal, goal_norm, policy, max_steps=50):
    """Run first 50 steps of VLA episode, collecting wheel actions."""
    from sim_lekiwi_urdf import LeKiWiSimURDF, twist_to_contact_wheel_speeds
    
    base_id = sim.model.body('base').id
    threshold = 0.15
    
    arm_joints = ["j0","j1","j2","j3","j4","j5"]
    joint_pos = [sim._jpos_idx[n] for n in arm_joints]
    wheel_joints = ["w1","w2","w3"]
    wheel_qvel = [sim._jvel_idx[n] for n in wheel_joints]
    
    wheel_actions = []
    base_positions = []
    pctrl_wheel_actions = []
    
    for step in range(max_steps):
        base_pos = sim.data.xpos[base_id, :2].copy()
        base_positions.append(base_pos.copy())
        
        dist = np.linalg.norm(base_pos - goal)
        if dist < threshold:
            break
        
        img = sim.render()
        img_t = resize_for_clip(img)
        
        arm_pos = np.array([sim.data.qpos[j] for j in joint_pos], dtype=np.float32)
        wheel_vel = np.array([sim.data.qvel[j] for j in wheel_qvel], dtype=np.float32)
        
        # Build 11D state: [arm(6) + wheel(3) + base_lin(2)] + goal(2)
        state_11d = np.concatenate([arm_pos, wheel_vel, goal_norm]).astype(np.float32)
        
        img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(DEVICE)
        state_tensor = torch.from_numpy(state_11d).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            raw_action = policy.infer(img_tensor, state_tensor, num_steps=4)
        raw_np = raw_action.squeeze(0).cpu().numpy()
        
        # Wheel action (CORRECT scale)
        raw_wheel = raw_np[6:9]
        wheel_action = raw_wheel * WHEEL_SCALE
        wheel_action = np.clip(wheel_action, -0.0417, 0.0417)
        wheel_actions.append(wheel_action.copy())
        
        # P-ctrl wheel action for comparison
        dx = goal[0] - base_pos[0]
        dy = goal[1] - base_pos[1]
        d = np.linalg.norm([dx, dy])
        if d < 0.05:
            pctrl_ws = np.zeros(3)
        else:
            v_mag = min(0.1 * d, 0.25)
            ws = twist_to_contact_wheel_speeds(v_mag * dx / d, v_mag * dy / d)
            pctrl_ws = np.clip(ws, -0.5, 0.5)
        pctrl_wheel_actions.append(pctrl_ws.copy())
        
        # Step
        arm_action = np.clip(raw_np[:6], -0.5, 0.5)
        action = np.concatenate([arm_action, wheel_action])
        sim.step(action)
    
    return {
        'wheel_actions': np.array(wheel_actions),
        'base_positions': np.array(base_positions),
        'pctrl_wheel_actions': np.array(pctrl_wheel_actions),
        'n_steps': len(wheel_actions),
        'final_dist': dist,
    }


def main():
    from sim_lekiwi_urdf import LeKiWiSimURDF

    print("\n" + "=*("*30)
    print("Phase 175: +X-Y Direction Failure Diagnostic")
    print("=*("*30)

    # Load Phase 174 eval results
    with open('results/phase174_eval_20260419_0536.json') as f:
        eval_data = json.load(f)

    results = eval_data['results']
    
    # Categorize
    x_pos_y_pos = [e for e in results if e['goal'][0] >= 0 and e['goal'][1] >= 0]
    x_pos_y_neg = [e for e in results if e['goal'][0] >= 0 and e['goal'][1] < 0]
    
    print(f"\n[Dataset] +X+Y: {sum(ep['vla_success'] for ep in x_pos_y_pos)}/{len(x_pos_y_pos)} SR")
    print(f"[Dataset] +X-Y: {sum(ep['vla_success'] for ep in x_pos_y_neg)}/{len(x_pos_y_neg)} SR")
    
    neg_pos = [e for e in results if e['goal'][0] < 0 and e['goal'][1] >= 0]
    neg_neg = [e for e in results if e['goal'][0] < 0 and e['goal'][1] < 0]
    print(f"[Dataset] -X+Y: {sum(ep['vla_success'] for ep in neg_pos)}/{len(neg_pos)} SR")
    print(f"[Dataset] -X-Y: {sum(ep['vla_success'] for ep in neg_neg)}/{len(neg_neg)} SR")

    # Load policy (same as eval_phase174_wheel_fix.py)
    print("\n[INFO] Loading cross-attention policy...")
    policy = load_policy("results/phase158_merged_jacobian_lr2e-05_ep7_20260419_0136/best_policy.pt", DEVICE)

    # Run diagnostic on FAILED +X-Y episodes
    print("\n--- Tracing +X-Y FAILED Episodes ---")
    failed_xpy = [e for e in x_pos_y_neg if not e['vla_success']]
    
    all_traces = {}
    
    for ep in failed_xpy[:5]:
        goal = ep['goal']
        goal_norm = np.clip(np.array(goal) / 1.0, -1.0, 1.0)  # normalize with same scale as eval
        
        sim = LeKiWiSimURDF()
        trace = run_diagnostic_episode(sim, goal, goal_norm, policy, max_steps=50)
        
        # Compute displacement
        if len(trace['base_positions']) > 1:
            total_disp = np.linalg.norm(trace['base_positions'][-1] - trace['base_positions'][0])
        else:
            total_disp = 0.0
        
        print(f"\n  Ep {ep['episode']:02d}: goal=({goal[0]:.3f},{goal[1]:.3f}) +X-Y")
        print(f"    VLA: FAIL (final_dist={trace['final_dist']:.3f}, disp={total_disp:.3f}m)")
        print(f"    P-ctrl: {'SUCC' if ep['pctrl_success'] else 'FAIL'} (steps={ep['pctrl_steps']})")
        
        wa = trace['wheel_actions']
        if len(wa) > 0:
            print(f"    VLA wheel actions: w1={wa[-1,0]:.4f}, w2={wa[-1,1]:.4f}, w3={wa[-1,2]:.4f}")
            print(f"    VLA wheel mean:    w1={wa[:,0].mean():.4f}, w2={wa[:,1].mean():.4f}, w3={wa[:,2].mean():.4f}")
            print(f"    P-ctrl wheel:      w1={trace['pctrl_wheel_actions'][-1,0]:.4f}, w2={trace['pctrl_wheel_actions'][-1,1]:.4f}, w3={trace['pctrl_wheel_actions'][-1,2]:.4f}")
        
        # Direction analysis: is VLA sending positive or negative wheel speeds?
        if len(wa) > 0:
            for wi in range(3):
                sign = '+' if wa[-1, wi] >= 0 else '-'
                psign = '+' if trace['pctrl_wheel_actions'][-1, wi] >= 0 else '-'
                match = 'MATCH' if sign == psign else 'DIFF'
                print(f"      w{wi+1}: VLA={sign}{abs(wa[-1,wi]):.4f}, P={psign}{abs(trace['pctrl_wheel_actions'][-1,wi]):.4f} [{match}]")
        
        all_traces[f"ep{ep['episode']:02d}_fail"] = {
            'goal': goal,
            'quadrant': '+X-Y',
            'vla_wheel_last': wa[-1].tolist() if len(wa) > 0 else [0,0,0],
            'pctrl_wheel_last': trace['pctrl_wheel_actions'][-1].tolist() if len(trace['pctrl_wheel_actions']) > 0 else [0,0,0],
            'vla_disp': float(total_disp),
            'final_dist': float(trace['final_dist']),
        }

    # Run diagnostic on SUCCESS +X+Y episodes
    print("\n--- Tracing +X+Y SUCCESS Episodes ---")
    success_xpy = [e for e in x_pos_y_pos if e['vla_success']]
    
    for ep in success_xpy[:4]:
        goal = ep['goal']
        goal_norm = np.clip(np.array(goal) / 1.0, -1.0, 1.0)
        
        sim = LeKiWiSimURDF()
        trace = run_diagnostic_episode(sim, goal, goal_norm, policy, max_steps=50)
        
        if len(trace['base_positions']) > 1:
            total_disp = np.linalg.norm(trace['base_positions'][-1] - trace['base_positions'][0])
        else:
            total_disp = 0.0
        
        print(f"\n  Ep {ep['episode']:02d}: goal=({goal[0]:.3f},{goal[1]:.3f}) +X+Y")
        print(f"    VLA: SUCC (steps={ep['vla_steps']}, disp={total_disp:.3f}m)")
        
        wa = trace['wheel_actions']
        if len(wa) > 0:
            print(f"    VLA wheel actions: w1={wa[-1,0]:.4f}, w2={wa[-1,1]:.4f}, w3={wa[-1,2]:.4f}")
            print(f"    VLA wheel mean:    w1={wa[:,0].mean():.4f}, w2={wa[:,1].mean():.4f}, w3={wa[:,2].mean():.4f}")
            print(f"    P-ctrl wheel:      w1={trace['pctrl_wheel_actions'][-1,0]:.4f}, w2={trace['pctrl_wheel_actions'][-1,1]:.4f}, w3={trace['pctrl_wheel_actions'][-1,2]:.4f}")
        
        all_traces[f"ep{ep['episode']:02d}_succ"] = {
            'goal': goal,
            'quadrant': '+X+Y',
            'vla_wheel_last': wa[-1].tolist() if len(wa) > 0 else [0,0,0],
            'pctrl_wheel_last': trace['pctrl_wheel_actions'][-1].tolist() if len(trace['pctrl_wheel_actions']) > 0 else [0,0,0],
            'vla_disp': float(total_disp),
            'final_dist': float(trace['final_dist']),
        }

    # Summary analysis
    print("\n--- Summary: Wheel Action Sign Patterns ---")
    print(f"{'Episode':<10} {'Quadrant':<10} {'VLA w1':>10} {'VLA w2':>10} {'VLA w3':>10} {'P-ctrl w1':>10} {'P-ctrl w2':>10} {'P-ctrl w3':>10}")
    for ep_id, data in all_traces.items():
        vla_w = data['vla_wheel_last']
        p_w = data['pctrl_wheel_last']
        print(f"{ep_id:<10} {data['quadrant']:<10} {vla_w[0]:>10.4f} {vla_w[1]:>10.4f} {vla_w[2]:>10.4f} {p_w[0]:>10.4f} {p_w[1]:>10.4f} {p_w[2]:>10.4f}")

    # Save results
    with open('results/phase175_diagnostic_20260419_0600.json', 'w') as f:
        json.dump({
            'summary': eval_data['summary'],
            'traces': all_traces,
            'phase': 175,
        }, f, indent=2)
    
    print(f"\n[INFO] Results saved to results/phase175_diagnostic_20260419_0600.json")
    print("\nPhase 175 diagnostic complete.")


if __name__ == "__main__":
    main()
