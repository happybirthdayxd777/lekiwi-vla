#!/usr/bin/env python3
"""
Convert Real Robot Docking Data → HDF5 Training Format
========================================================
Converts lekiwi_modular JSON trajectories to the same HDF5 format
used by train_flow_matching_real.py.

Real robot data has fields: t, phase, pitch1, pitch_smooth, pitch_ref, error, mode
We need: image (placeholder), state (9-D), action (9-D)

For the real data, state = [pitch1, pitch_smooth, pitch_ref, error, phase_encoded]
Action = commanded velocity (based on phase transitions)

Usage:
  python3 scripts/convert_docking_data.py \
    --input ~/hermes_research/lekiwi_modular/src/scripts/ \
    --output data/docking_real.h5
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# Phase encoding: one-hot for forward/sideways
PHASE_MAP = {"forward": 0, "sideways": 1, "unknown": 2}


def parse_trajectory(json_path):
    """Load a single JSON trajectory file."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def trajectory_to_state_action(traj):
    """
    Convert real robot trajectory to state/action pairs.

    Real data fields: t, phase, pitch1, pitch_smooth, pitch_ref, error, mode

    We construct:
    - state[9]: [pitch1, pitch_smooth, pitch_ref, error, phase_onehot(3), reserved(1)]
    - action[9]: [vx, vy, wz] derived from phase transitions + velocity commands

    For docking: action is the velocity command that drives the phase transition.
    We'll use pitch_ref - pitch_smooth as the control error signal,
    and derive approximate vx/vy from phase + pitch dynamics.
    """
    states = []
    actions = []

    for i, record in enumerate(traj):
        # State: pitch observations + phase encoding
        pitch1 = record.get("pitch1", 0.0)
        pitch_smooth = record.get("pitch_smooth", 0.0)
        pitch_ref = record.get("pitch_ref", 0.0)
        error = record.get("error", 0.0)
        phase = PHASE_MAP.get(record.get("phase", "unknown"), 2)

        # Normalize pitch values to roughly [-1, 1] range (pitch ~ -135 to -136 deg)
        # That's a range of ~1 degree across all data
        pitch1_n   = (pitch1   + 140) / 5.0    # roughly [-1, 1]
        pitch_sm_n = (pitch_smooth + 140) / 5.0
        pitch_rf_n = (pitch_ref + 140) / 5.0
        error_n    = error / 0.05                # error ~ 0.04 max → [-1, 1]

        # Phase one-hot (3 features)
        phase_oh = [1.0 if i == phase else 0.0 for i in range(3)]

        # State vector [9]
        state = np.array([pitch1_n, pitch_sm_n, pitch_rf_n, error_n] + phase_oh, dtype=np.float32)

        # Action: derive from phase transitions
        # forward phase → mainly vx command
        # sideways phase → mainly vy command
        # Use mode as a proxy for commanded velocity
        mode = record.get("mode", "forward")
        if mode == "forward":
            # Forward docking: vx ~ -0.1, vz ~ 0
            action = np.array([-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif mode == "sideways":
            # Sideways: vy ~ 0.05
            action = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            action = np.zeros(9, dtype=np.float32)

        states.append(state)
        actions.append(action)

    return np.stack(states), np.stack(actions)


def convert_directory(input_dir, output_path, max_files=None):
    """Convert all JSON trajectories in a directory to HDF5."""
    input_path = Path(input_dir)

    # Find all traj_log*.json files
    json_files = sorted(input_path.glob("traj_log*.json"))
    if max_files:
        json_files = json_files[:max_files]

    print(f"Found {len(json_files)} trajectory files in {input_dir}")

    all_states = []
    all_actions = []

    for f in json_files:
        traj = parse_trajectory(f)
        states, actions = trajectory_to_state_action(traj)
        all_states.append(states)
        all_actions.append(actions)
        print(f"  {f.name}: {len(traj)} records, phases={[r.get('phase') for r in traj[:3]]}")

    all_states   = np.concatenate(all_states, axis=0)
    all_actions  = np.concatenate(all_actions, axis=0)

    print(f"\nTotal: {len(all_states)} state-action pairs")
    print(f"  States shape:  {all_states.shape}")
    print(f"  Actions shape: {all_actions.shape}")

    # Save HDF5
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("states",  data=all_states)
        f.create_dataset("actions", data=all_actions)
        f.attrs["num_files"]   = len(json_files)
        f.attrs["state_dim"]   = all_states.shape[1]
        f.attrs["action_dim"]  = all_actions.shape[1]
        f.attrs["note"]        = "Real robot docking data (lekiwi_modular)"

    print(f"\n✓ Saved to {output_path}")

    # Quick analysis
    print("\n=== State statistics ===")
    for i, name in enumerate(["pitch1_n", "pitch_sm_n", "pitch_rf_n", "error_n", "phase_F", "phase_S", "phase_U"]):
        vals = all_states[:, i]
        print(f"  {name:12s}: mean={vals.mean():+.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")

    print("\n=== Action statistics ===")
    for i in range(all_actions.shape[1]):
        vals = all_actions[:, i]
        if vals.std() > 0.001:
            print(f"  action[{i}]: mean={vals.mean():+.3f} std={vals.std():.3f}")

    return all_states, all_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True, help="Directory with traj_log*.json files")
    parser.add_argument("--output", type=str, default="data/docking_real.h5")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files")
    args = parser.parse_args()

    convert_directory(args.input, args.output, args.max_files)

if __name__ == "__main__":
    main()