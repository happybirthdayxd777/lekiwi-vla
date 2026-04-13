#!/usr/bin/env python3
"""
Phase 29 — Quick Training + Evaluation on URDF P-controller data
================================================================
10 epochs, batch checkpointing, ends with evaluation.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path
import time

from scripts.train_task_oriented import CLIPFlowMatchingPolicy


def load_data(h5_path):
    f = h5py.File(h5_path, 'r')
    images = f['images'][:]
    states_9d = f['states'][:]
    actions = f['actions'][:]
    goals = f['goal_positions'][:]
    rewards = f['rewards'][:]
    f.close()

    goal_norm = np.clip(goals / 1.0, -1.0, 1.0).astype(np.float32)
    states_11d = np.concatenate([states_9d, goal_norm], axis=1)

    imgs = []
    for img in images:
        pil_resized = PILImage.fromarray(img).resize((224, 224))
        img_np = np.array(pil_resized).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(img_np.transpose(2, 0, 1)))

    print(f"Loaded {len(states_11d)} frames from {h5_path}")
    print(f"  Rewards: max={rewards.max():.3f}, positive={int((rewards>0).sum())}/{len(rewards)}")
    return torch.stack(imgs), torch.from_numpy(states_11d), torch.from_numpy(actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path("results/phase29_quick")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    imgs, states, actions = load_data("data/lekiwi_goal_urdf_10k.h5")

    print(f"\nPolicy init...")
    policy = CLIPFlowMatchingPolicy(state_dim=11, action_dim=9, hidden=512, device=args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    n = len(imgs)
    indices = np.arange(n)
    losses = []
    t0 = time.time()

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, 32):  # larger batch for speed
            batch_idx = indices[i:i+32]
            img_batch = imgs[batch_idx].to(args.device)
            state_batch = states[batch_idx].to(args.device)
            action_batch = actions[batch_idx].to(args.device)

            t = (torch.rand(batch_idx.shape[0], 1, device=args.device) ** 1.5) * 0.999
            noise = torch.randn_like(action_batch)
            x_t = (1 - t) * action_batch + t * noise

            optimizer.zero_grad()
            v_pred = policy(img_batch, state_batch, x_t, t)
            v_target = action_batch - noise
            loss = ((v_pred - v_target) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{args.epochs}: loss={avg_loss:.5f} [{elapsed:.0f}s]")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({'epoch': epoch, 'policy_state_dict': policy.state_dict()}, ckpt)
            print(f"  => Saved {ckpt.name}")

    # Save final
    torch.save({'epoch': args.epochs, 'policy_state_dict': policy.state_dict()},
                output_dir / "final_policy.pt")

    # Plot
    plt.plot(losses)
    plt.title('Phase 29 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig(output_dir / "training_curves.png", dpi=100)
    plt.close()

    # Quick evaluation
    print("\nQuick evaluation...")
    from sim_lekiwi_urdf import LeKiWiSimURDF

    def resize(img_pil):
        pil_r = PILImage.fromarray(img_pil).resize((224, 224))
        arr = np.array(pil_r, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).cpu()

    def make_state(obs, gx, gy):
        ap = obs['arm_positions']
        wv = obs['wheel_velocities']
        gn = np.array([gx / 1.0, gy / 1.0], dtype=np.float32)
        return np.concatenate([ap, wv, gn]).astype(np.float32)

    goals = [(0.3, 0.2), (0.4, 0.0), (0.35, 0.25)]
    results = []
    for gx, gy in goals:
        sim = LeKiWiSimURDF()
        sim.reset()
        sim.set_target(np.array([gx, gy, 0.02]))
        for _ in range(30):
            sim.step(np.zeros(9))
        img_t = resize(sim.render())
        success = False
        for step in range(120):
            obs = sim._obs()
            st = make_state(obs, gx, gy)
            with torch.no_grad():
                a = policy.infer(img_t, torch.from_numpy(st).unsqueeze(0).cpu(), num_steps=4)
            a_np = np.clip(a.cpu().numpy()[0], -1, 1).astype(np.float32)
            obs = sim.step(a_np)
            img_t = resize(sim.render())
            base_xy = sim.data.qpos[:2]
            dist = np.linalg.norm(base_xy - np.array([gx, gy]))
            if dist < 0.15:
                success = True
                print(f"  goal=({gx},{gy}): SUCCESS at step {step}, dist={dist:.3f}")
                break
        if not success:
            print(f"  goal=({gx},{gy}): FAIL dist={dist:.3f}")
        results.append((gx, gy, success, dist))

    sr = sum(s for _, _, s, _ in results) / len(results)
    print(f"\nSuccess Rate: {sr*100:.0f}%")

    with open(output_dir / "training_log.txt", "w") as log:
        log.write(f"Phase 29 quick train: {args.epochs} epochs\n")
        log.write(f"Final loss: {losses[-1]:.5f}\n")
        log.write(f"Success Rate: {sr*100:.0f}%\n")
        for i, l in enumerate(losses):
            log.write(f"epoch {i}: {l:.5f}\n")

    print("\n✓ Phase 29 complete")
