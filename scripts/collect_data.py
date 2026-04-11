#!/usr/bin/env python3
"""
Collect training data from LeKiwi simulation.
Saves (image, state, action) tuples as HDF5 for later training.

Usage:
  python3 collect_data.py --episodes 10 --steps 200 --output data/lekiwi_demo.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

from sim_lekiwi import LeKiwiSim


from PIL import Image

# Resize images to 224x224 for CNN compatibility
TARGET_SIZE = (224, 224)


def collect_episode(sim, max_steps=200):
    """Collect one episode: returns dicts of images, states, actions."""
    sim.reset()
    imgs, states, actions = [], [], []

    for _ in range(max_steps):
        # Render image and resize to CNN-friendly size
        img_pil = sim.render()
        if img_pil is None:
            img_pil = Image.new('RGB', (640, 480))
        img_pil = img_pil.resize(TARGET_SIZE, Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.uint8)

        # State: arm positions (qpos[0:6]) + wheel velocities (qvel[0:3])
        arm_pos = sim.data.qpos[0:6] if hasattr(sim.data, 'qpos') else np.zeros(6)
        wheel_vel = sim.data.qvel[0:3] if hasattr(sim.data, 'qvel') else np.zeros(3)
        state = np.concatenate([arm_pos, wheel_vel]).astype(np.float32)

        # Random exploratory action (arm + wheels, normalized [-1,1])
        action = np.random.uniform(-1, 1, size=9).astype(np.float32)

        imgs.append(img_arr)
        states.append(state)
        actions.append(action)

        # Step
        result = sim.step(action)
        if isinstance(result, tuple):
            obs, reward, term, trunc, info = result
            if term or trunc:
                break

    return {
        "image": np.stack(imgs),
        "state": np.stack(states),
        "action": np.stack(actions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=10)
    parser.add_argument("--steps",     type=int,   default=200)
    parser.add_argument("--output",    type=str,   default="data/lekiwi_demo.h5")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.episodes} episodes × {args.steps} steps")
    print(f"  Image size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (resized from 640x480)")

    sim = LeKiwiSim()
    all_images = []
    all_states = []
    all_actions = []

    for ep in range(args.episodes):
        ep_data = collect_episode(sim, max_steps=args.steps)
        all_images.append(ep_data["image"])
        all_states.append(ep_data["state"])
        all_actions.append(ep_data["action"])
        print(f"  Episode {ep+1}/{args.episodes}: {len(ep_data['image'])} frames")

    # Save as HDF5
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images",   data=np.concatenate(all_images))
        f.create_dataset("states",   data=np.concatenate(all_states))
        f.create_dataset("actions",  data=np.concatenate(all_actions))
        f.attrs["episodes"]  = args.episodes
        f.attrs["steps"]     = args.steps
        f.attrs["img_shape"] = all_images[0][0].shape  # (H, W, C)

    print(f"✓ Saved {sum(len(x) for x in all_images)} total frames")
    print(f"  Images:  {output_path}['images']  shape={all_images[0].shape}")
    print(f"  States:  {output_path}['states']  shape={all_states[0].shape}")
    print(f"  Actions: {output_path}['actions'] shape={all_actions[0].shape}")

if __name__ == "__main__":
    main()