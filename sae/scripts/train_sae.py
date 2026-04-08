#!/usr/bin/env python3
"""Train a TopK Sparse Autoencoder on collected DiT activations.

Usage:
    python scripts/train_sae.py \
        --activations_dir ./activations \
        --layer 24 \
        --expansion_factor 8 \
        --k 32 \
        --num_steps 100000 \
        --output_dir ./sae_output/layer_24
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace_step_sae.config import SAEConfig
from ace_step_sae.hooks import ActivationDataset
from ace_step_sae.trainer import SAETrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train a TopK SAE on DiT activations"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory with activation shards from collect_activations.py",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to train the SAE for",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=2048,
        help="Model hidden dimension (turbo=2048, base=4096)",
    )
    parser.add_argument(
        "--expansion_factor",
        type=int,
        default=8,
        help="SAE expansion factor",
    )
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument(
        "--num_steps", type=int, default=100_000, help="Training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--aux_loss_coeff",
        type=float,
        default=1 / 32,
        help="Auxiliary dead-feature loss coefficient",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./sae_output", help="Output dir"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load activations for this layer
    act_dir = Path(args.activations_dir)
    shards = sorted(act_dir.glob(f"layer_{args.layer:03d}_shard_*.pt"))
    if not shards:
        # Try flat file
        flat = act_dir / f"layer_{args.layer}.pt"
        if flat.exists():
            data = torch.load(flat, weights_only=True)
        else:
            logger.error(
                f"No activation data found for layer {args.layer} in {act_dir}"
            )
            sys.exit(1)
    else:
        logger.info(f"Loading {len(shards)} shards for layer {args.layer}...")
        data = torch.cat(
            [torch.load(s, weights_only=True) for s in shards], dim=0
        )

    logger.info(f"Loaded {data.shape[0]:,} activation vectors of dim {data.shape[1]}")

    # Config
    cfg = SAEConfig(
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        aux_loss_coeff=args.aux_loss_coeff,
        lr=args.lr,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        output_dir=args.output_dir,
    )

    torch.manual_seed(cfg.seed)

    # Optionally resume
    sae = None
    if args.resume:
        from ace_step_sae.model import TopKSparseAutoencoder
        logger.info(f"Resuming from {args.resume}")
        sae = TopKSparseAutoencoder.load(args.resume, device=args.device)

    # Train
    trainer = SAETrainer(cfg, data, sae=sae)
    trained_sae = trainer.train()

    logger.info("Training complete!")
    logger.info(f"  Alive features: {trained_sae.num_alive}/{cfg.d_sae}")
    logger.info(f"  Dead features:  {trained_sae.num_dead}/{cfg.d_sae}")

    # Quick eval
    eval_metrics = trainer.evaluate()
    logger.info(f"  Eval MSE: {eval_metrics['eval_mse']:.6f}")
    logger.info(f"  Eval Loss: {eval_metrics['eval_loss']:.6f}")


if __name__ == "__main__":
    main()
