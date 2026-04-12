#!/usr/bin/env python3
"""
Collect training data from LeKiWi simulation.
Saves (image, state, action) tuples as HDF5 for later training.

Supports two simulation backends:
  - primitive : LeKiwiSim (fast cylinders) from sim_lekiwi.py
  - urdf      : LeKiWiSimURDF (STL mesh geometry) from sim_lekiwi_urdf.py

Usage:
  python3 collect_data.py --episodes 10 --steps 200 --output data/lekiwi_demo.h5
  python3 collect_data.py --sim_type urdf --episodes 10 --steps 200 --output data/lekiwi_urdf_demo.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

# Resize images to 224x224 for CLIP ViT-B/32 compatibility
TARGET_SIZE = (224, 224)


def make_sim(sim_type: str):
    """Create simulation backend by type."""
    if sim_type == "urdf":
        from sim_lekiwi_urdf import LeKiWiSimURDF
        return LeKiWiSimURDF()
    else:
        from sim_lekiwi import LeKiwiSim
        return LeKiwiSim()


def collect_episode(sim, max_steps=200, record_wrist=False):
    """
    Collect one episode: returns dicts of images, states, actions.
    
    State: arm positions (qpos[0:6]) + wheel velocities (qvel[0:3])
          = [9] native units (matches CLIP-FM training format)
    Action: arm (6) + wheel (3), normalized [-1,1]
    
    Uses random-walk exploration for smooth, physically plausible trajectories.
    """
    try:
        sim.reset()
    except AttributeError:
        pass  # LeKiWiSim has no reset()

    imgs, states, actions = [], [], []
    wrist_imgs = [] if record_wrist else None

    # Smooth random-walk action: Brownian motion for arm + wheels
    action = np.zeros(9, dtype=np.float32)

    for step in range(max_steps):
        # Render image and resize to CLIP-compatible size
        img_arr = sim.render()
        if img_arr is None:
            img_arr = np.zeros((640, 480, 3), dtype=np.uint8)
        elif isinstance(img_arr, np.ndarray):
            img_arr = img_arr  # already numpy
        else:
            img_arr = np.array(img_arr)
        img_pil = Image.fromarray(img_arr).resize(TARGET_SIZE, Image.BILINEAR)
        img_arr = np.array(img_pil, dtype=np.uint8)

        # State: arm positions + wheel velocities (matches training format)
        # CRITICAL FIX (2026-04-13): Use sim._obs() for correct joint-level extraction.
        # LeKiWiSim (primitive):  qpos[0:6]=arm joints, qpos[6:9]=wheel (coincident by design)
        # LeKiWiSimURDF (mesh):  qpos[0:7]=base_free(xyz+quat), qpos[7:13]=arm joints
        # The old code used qpos[0:6] + qvel[0:3] which gave WRONG base pos + base vel!
        obs = sim._obs()
        arm_pos = obs["arm_positions"]
        wheel_vel = obs["wheel_velocities"]
        state = np.concatenate([arm_pos, wheel_vel]).astype(np.float32)

        # Random-walk action: small random delta, clamped to [-1,1]
        delta = np.random.uniform(-0.15, 0.15, size=9).astype(np.float32)
        action = np.clip(action + delta, -1.0, 1.0).astype(np.float32)

        imgs.append(img_arr)
        states.append(state)
        actions.append(action.copy())

        if record_wrist and hasattr(sim, 'render_wrist'):
            wimg = sim.render_wrist()
            if wimg is not None:
                if isinstance(wimg, np.ndarray):
                    wimg_pil = Image.fromarray(wimg).resize(TARGET_SIZE, Image.BILINEAR)
                else:
                    wimg_pil = wimg.resize(TARGET_SIZE, Image.BILINEAR)
                wrist_imgs.append(np.array(wimg_pil, dtype=np.uint8))

        # Step (LeKiWiSim returns 4-value tuple: obs, reward, terminated, info)
        result = sim.step(action)
        if isinstance(result, tuple) and len(result) >= 3:
            _, _, term, *_ = result
            if term:
                break

    result = {
        "image": np.stack(imgs),
        "state": np.stack(states),
        "action": np.stack(actions),
    }
    if record_wrist and wrist_imgs:
        result["wrist_image"] = np.stack(wrist_imgs)
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect LeKiWi training data")
    parser.add_argument("--episodes",   type=int,   default=10)
    parser.add_argument("--steps",     type=int,   default=200)
    parser.add_argument("--output",    type=str,   default="data/lekiwi_demo.h5")
    parser.add_argument("--sim_type",   type=str,   default="urdf",
                        choices=["primitive", "urdf"],
                        help="primitive=fast cylinders, urdf=STL mesh geometry")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--wrist",      action="store_true",
                        help="Also capture wrist camera images")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[collect_data] sim_type={args.sim_type}, episodes={args.episodes}, steps={args.steps}")
    print(f"  Image size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (resized from 640x480)")
    print(f"  Output: {output_path}")

    sim = make_sim(args.sim_type)
    print(f"[collect_data] Simulator: {type(sim).__name__}")

    all_images = []
    all_states = []
    all_actions = []
    all_wrist = []

    for ep in range(args.episodes):
        ep_data = collect_episode(sim, max_steps=args.steps, record_wrist=args.wrist)
        all_images.append(ep_data["image"])
        all_states.append(ep_data["state"])
        all_actions.append(ep_data["action"])
        if "wrist_image" in ep_data:
            all_wrist.append(ep_data["wrist_image"])
        n = len(ep_data["image"])
        print(f"  Episode {ep+1}/{args.episodes}: {n} frames")

    # Save as HDF5
    all_images_nd = np.concatenate(all_images)
    all_states_nd  = np.concatenate(all_states)
    all_actions_nd = np.concatenate(all_actions)

    print(f"\nSaving {len(all_images_nd)} frames to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images",   data=all_images_nd)
        f.create_dataset("states",   data=all_states_nd)
        f.create_dataset("actions",  data=all_actions_nd)
        if all_wrist:
            f.create_dataset("wrist_images", data=np.concatenate(all_wrist))
        f.attrs["episodes"]   = args.episodes
        f.attrs["steps"]      = args.steps
        f.attrs["sim_type"]   = args.sim_type
        f.attrs["img_shape"]  = all_images[0][0].shape   # (H, W, C)
        f.attrs["state_dim"]  = all_states[0][0].shape   # (9,)
        f.attrs["action_dim"] = all_actions[0][0].shape # (9,)

    total = sum(len(x) for x in all_images)
    print(f"✓ Saved {total} frames")
    print(f"  Images:   {output_path}['images']   {all_images_nd.shape}")
    print(f"  States:   {output_path}['states']   {all_states_nd.shape}")
    print(f"  Actions:  {output_path}['actions']  {all_actions_nd.shape}")
    if all_wrist:
        print(f"  Wrist:    {output_path}['wrist_images'] {all_wrist[0].shape}")

if __name__ == "__main__":
    main()