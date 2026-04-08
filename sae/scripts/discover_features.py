#!/usr/bin/env python3
"""Analyze and discover interpretable features in a trained SAE.

Usage:
    python scripts/discover_features.py \
        --sae_path ./sae_output/layer_24/sae_final.pt \
        --activations_dir ./activations \
        --layer 24 \
        --output features_layer_24.json \
        --top_k_examples 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace_step_sae.model import TopKSparseAutoencoder
from ace_step_sae.features import FeatureAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Discover and analyze SAE features"
    )
    parser.add_argument(
        "--sae_path", type=str, required=True, help="Path to trained SAE"
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory with activation shards",
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer index"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--top_k_examples",
        type=int,
        default=10,
        help="Number of top-activating examples per feature",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size for analysis"
    )
    parser.add_argument("--device", type=str, default="cuda")
    # CLAP labeling options
    parser.add_argument(
        "--clap_label",
        action="store_true",
        help="Enable CLAP-based auto-labeling",
    )
    parser.add_argument(
        "--audio_manifest",
        type=str,
        default=None,
        help="Text file listing audio paths (one per line), indexed to match activation vectors",
    )
    parser.add_argument(
        "--label_vocabulary",
        type=str,
        default=None,
        help="Text file with candidate labels (one per line)",
    )
    args = parser.parse_args()

    # Load SAE
    logger.info(f"Loading SAE from {args.sae_path}...")
    sae = TopKSparseAutoencoder.load(args.sae_path, device=args.device)
    logger.info(
        f"SAE: d_model={sae.cfg.d_model}, d_sae={sae.cfg.d_sae}, k={sae.k}"
    )

    # Load activations
    act_dir = Path(args.activations_dir)
    shards = sorted(act_dir.glob(f"layer_{args.layer:03d}_shard_*.pt"))
    if shards:
        logger.info(f"Loading {len(shards)} shards...")
        data = torch.cat(
            [torch.load(s, weights_only=True) for s in shards], dim=0
        )
    else:
        flat = act_dir / f"layer_{args.layer}.pt"
        if flat.exists():
            data = torch.load(flat, weights_only=True)
        else:
            logger.error(f"No data for layer {args.layer}")
            sys.exit(1)

    logger.info(f"Analyzing {data.shape[0]:,} activation vectors...")

    # Analyze
    analyzer = FeatureAnalyzer(sae)
    records = analyzer.analyze(
        data,
        top_k_examples=args.top_k_examples,
        batch_size=args.batch_size,
    )

    # Optional CLAP labeling
    if args.clap_label:
        if args.audio_manifest is None or args.label_vocabulary is None:
            logger.error(
                "CLAP labeling requires --audio_manifest and --label_vocabulary"
            )
            sys.exit(1)

        with open(args.audio_manifest) as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        with open(args.label_vocabulary) as f:
            labels = [line.strip() for line in f if line.strip()]

        try:
            from transformers import ClapModel, ClapProcessor

            clap_model = ClapModel.from_pretrained(
                "laion/larger_clap_music"
            ).to(args.device)
            clap_processor = ClapProcessor.from_pretrained(
                "laion/larger_clap_music"
            )
            records = analyzer.label_with_clap(
                records,
                audio_paths,
                labels,
                clap_model=clap_model,
                clap_processor=clap_processor,
                device=args.device,
            )
        except ImportError:
            logger.error(
                "CLAP labeling requires: pip install transformers torchaudio"
            )

    # Save
    FeatureAnalyzer.save_records(records, args.output)

    # Print summary
    alive = [r for r in records if r.is_alive]
    labeled = [r for r in records if r.label]
    logger.info(f"Results: {len(alive)} alive, {len(labeled)} labeled")

    # Top features by activation frequency
    alive_sorted = sorted(alive, key=lambda r: r.activation_frequency, reverse=True)
    logger.info("\nTop 20 features by activation frequency:")
    for r in alive_sorted[:20]:
        label_str = f" [{r.label}]" if r.label else ""
        logger.info(
            f"  Feature {r.feature_idx:>6d}: freq={r.activation_frequency:.4f}  "
            f"mean_act={r.mean_activation:.4f}  max={r.max_activation:.4f}"
            f"{label_str}"
        )


if __name__ == "__main__":
    main()
