#!/usr/bin/env python3
"""
LeRobot Multi-Task DiT (Flow Matching) Training on LeKiwi Data
==============================================================
Trains a Flow Matching policy using LeRobot's multi_task_dit on LeKiwi data.

Usage:
  python3 scripts/train_flow_matching.py \
    --dataset <hf_repo_id> \
    --output results/lekiwi_flow_matching \
    --epochs 100

The trained policy can then be used with:
  python3 scripts/eval_policy.py --checkpoint results/lekiwi_flow_matching/checkpoints/latest
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.factory import make_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task DiT (Flow Matching) on LeKiwi")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset repo ID or local path")
    parser.add_argument("--output", type=str, default="results/lekiwi_fm",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run evaluation only")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LeRobot Multi-Task DiT (Flow Matching) Training")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {device}")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # ── Dataset ──────────────────────────────────────────────────────────
    print("\n[1] Loading dataset metadata...")
    dataset_meta = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_meta.features)

    # Separate input/output features
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features  = {k: ft for k, ft in features.items() if k not in output_features}

    print(f"  Input features : {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")
    print(f"  Total episodes : {dataset_meta.num_episodes}")
    print(f"  FPS            : {dataset_meta.fps}")

    # ── Policy Config ────────────────────────────────────────────────────
    print("\n[2] Creating Multi-Task DiT config (Flow Matching)...")
    cfg = MultiTaskDiTConfig(
        # Temporal
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,

        # Flow Matching objective ⚡
        objective="flow_matching",
        sigma_min=0.0,
        num_integration_steps=4,      # UnifoLM trick: only 4 inference steps!
        integration_method="euler",
        timestep_sampling_strategy="beta",
        timestep_sampling_s=0.999,
        timestep_sampling_alpha=1.5,
        timestep_sampling_beta=1.0,

        # Architecture
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_rope=True,
        rope_base=10000.0,
        timestep_embed_dim=256,

        # Vision encoder
        vision_encoder_name="openai/clip-vit-base-patch16",
        image_crop_shape=[224, 224],
        image_crop_is_random=True,

        # Text encoder
        text_encoder_name="openai/clip-vit-base-patch16",
        tokenizer_max_length=77,
        tokenizer_padding="max_length",
        tokenizer_padding_side="right",

        # Normalization
        normalization_mapping={
            "VISUAL": "mean_std",
            "STATE":  "min_max",
            "ACTION": "min_max",
        },

        # Training
        optimizer_lr=args.lr,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=0.0,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
        do_mask_loss_for_padding=True,
        max_grad_norm=1.0,
    )

    # Override computed field
    cfg.drop_n_last_frames = cfg.horizon - cfg.n_action_steps - cfg.n_obs_steps + 1  # = 7

    # Attach features to config
    cfg.input_features = input_features
    cfg.output_features = output_features

    print(f"  Objective   : Flow Matching (4-step inference)")
    print(f"  Horizon     : {cfg.horizon}")
    print(f"  Action steps: {cfg.n_action_steps}")
    print(f"  Hidden dim  : {cfg.hidden_dim} | Layers: {cfg.num_layers} | Heads: {cfg.num_heads}")

    # ── Policy ─────────────────────────────────────────────────────────
    print("\n[3] Creating policy...")
    policy = MultiTaskDiTPolicy(cfg)
    policy.train()
    policy.to(device)
    print(f"  Policy loaded on {device}")

    # ── Dataset Normalization Stats ──────────────────────────────────────
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_meta.stats)

    # ── Delta Timestamps ────────────────────────────────────────────────
    fps = dataset_meta.fps
    delta_timestamps = {
        "observation.image":  [i / fps for i in cfg.observation_delta_indices],
        "observation.state":  [i / fps for i in cfg.observation_delta_indices],
        "action":             [i / fps for i in cfg.action_delta_indices],
    }

    # ── Dataloader ──────────────────────────────────────────────────────
    print("\n[4] Setting up dataloader...")
    dataset = LeRobotDataset(args.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    print(f"  batches: {len(dataloader)} per epoch")

    # ── Training Loop ──────────────────────────────────────────────────
    if args.eval_only:
        print("\n[EVAL] Running evaluation...")
        eval_policy(policy, dataloader, device, delta_timestamps, preprocessor)
        return

    print(f"\n[5] Training for {args.epochs} epochs...")
    global_step = 0
    optimizer = policy.get_optimizer()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward pass
            output = policy(batch)

            # Backward
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            if global_step % 50 == 0:
                print(f"  epoch {epoch:3d} | step {global_step:6d} | loss: {loss.item():.4f}")

        print(f"  Epoch {epoch} done. Loss: {loss.item():.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            ckpt_path = output_dir / "checkpoints" / f"epoch_{epoch}"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            policy.save(ckpt_path)
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = output_dir / "checkpoints" / "latest"
    policy.save(final_path)
    print(f"\n✓ Training complete. Final checkpoint: {final_path}")


def eval_policy(policy, dataloader, device, delta_timestamps, preprocessor, num_batches=20):
    """Evaluate policy on a batch of data."""
    policy.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            output = policy(batch)
            total_loss += output["loss"].item()

    avg_loss = total_loss / num_batches
    print(f"  Eval loss: {avg_loss:.4f} (over {num_batches} batches)")
    policy.train()


if __name__ == "__main__":
    main()