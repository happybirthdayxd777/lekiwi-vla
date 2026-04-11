#!/usr/bin/env python3
"""
GR00T-N1.5 Inference — NVIDIA's Open-Source VLA for LeKiwi
===========================================================
nvidia/GR00T-N1.5-3B: Flow Matching + DiT + Qwen2.5-VL
4-step inference, ~8GB VRAM

Usage:
  python3 scripts/infer_groot.py --task "move arm forward" --steps 50
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from PIL import Image
import numpy as np

# Check if transformers is available, if not note it
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] transformers not installed. Run: pip install transformers")

from sim_lekiwi import LeKiwiSim


def load_groot_model():
    """Load GR00T-N1.5 from HuggingFace."""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers not installed: pip install transformers")

    print("[INFO] Loading nvidia/GR00T-N1.5-3B from HuggingFace...")
    model_name = "nvidia/GR00T-N1.5-3B"

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    device = "mps" if torch.backends.mps.is_available() else "cuda"
    model.to(device)
    model.eval()

    print(f"[INFO] GR00T-N1.5 loaded on {device}")
    return model, processor, device


def infer_groot(model, processor, device, image, state, task_description, num_inference_steps=4):
    """
    Run GR00T-N1.5 inference.

    Args:
        model: loaded GR00T model
        processor: AutoProcessor
        device: mps/cuda/cpu
        image: PIL Image (camera frame)
        state: 9-dim state vector
        task_description: text task
        num_inference_steps: flow matching steps (default 4)
    Returns:
        action: 9-dim numpy array
    """
    # Prepare inputs
    inputs = processor(
        text=task_description,
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # State is appended as a special token (in a real integration)
    # For now we concatenate state to the text embedding
    state_str = f" state={','.join(f'{s:.2f}' for s in state)}"
    full_task = task_description + state_str

    inputs = processor(
        text=full_task,
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # GR00T uses generate() with flow matching steps
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_inference_steps=num_inference_steps,
            do_sample=True,
        )

    # Decode action tokens
    action = processor.decode(outputs[0], skip_special_tokens=True)
    return action


def demo_groot(task="move arm forward", num_steps=30):
    """Run GR00T inference demo on LeKiwi sim."""
    print(f"\n{'='*60}")
    print(f"  GR00T-N1.5 Inference Demo")
    print(f"  Task: {task}")
    print(f"{'='*60}\n")

    model, processor, device = load_groot_model()

    sim = LeKiwiSim()
    sim.reset()

    print(f"\n[Running {num_steps} steps on LeKiwi sim]\n")
    for step in range(num_steps):
        # Get observation
        img_pil = sim.render()
        arm_pos  = sim.data.qpos[0:6]
        wheel_v  = sim.data.qvel[0:3]
        state = np.concatenate([arm_pos, wheel_v])

        # Run GR00T inference
        action_text = infer_groot(model, processor, device, img_pil, state, task, num_inference_steps=4)
        print(f"  Step {step:3d}: action_text={action_text[:80]}")

        # Parse action (GR00T outputs structured text, parse numeric values)
        # In a real integration we'd parse the action tokens
        # For demo: use zero action or parse text
        try:
            # Try to extract numbers from the output
            import re
            nums = re.findall(r'-?\d+\.?\d*', action_text)
            if len(nums) >= 9:
                action = np.array([float(n) for n in nums[:9]])
                action = np.clip(action, -1, 1).astype(np.float32)
            else:
                action = np.zeros(9, dtype=np.float32)
        except:
            action = np.zeros(9, dtype=np.float32)

        sim.step(action)

    print("\n✓ Demo complete")

    # Cleanup
    del model
    torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="GR00T-N1.5 Inference on LeKiwi")
    parser.add_argument("--task",    type=str, default="move the robot arm forward")
    parser.add_argument("--steps",   type=int, default=30)
    parser.add_argument("--device",  type=str, default="mps")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers required")
        print("  pip install transformers")
        print("\nTo use GR00T, first install: pip install transformers")
        print("Then run this script again.")
        return

    demo_groot(task=args.task, num_steps=args.steps)

if __name__ == "__main__":
    main()