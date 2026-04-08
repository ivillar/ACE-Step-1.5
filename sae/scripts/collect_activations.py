#!/usr/bin/env python3
"""Collect residual-stream activations from ACE-Step DiT during generation.

Runs one or more generations through the ACE-Step pipeline and saves
the intermediate DiT activations to disk.  These are later used to
train Sparse Autoencoders.

Usage:
    python scripts/collect_activations.py \
        --model_dir ./ACE-Step \
        --output_dir ./activations \
        --layers 16 20 24 28 31 \
        --num_generations 100 \
        --captions captions.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace_step_sae.hooks import ActivationCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_acestep_model(model_dir: str, device: str = "cuda"):
    """Load the ACE-Step model.

    This expects the ACE-Step submodule at ``model_dir`` to be a working
    installation.  Adjust the import path to match your setup.
    """
    sys.path.insert(0, model_dir)
    try:
        from acestep.handler import AceStepHandler
    except ImportError:
        raise ImportError(
            f"Could not import ACE-Step from {model_dir}. "
            "Make sure the ACE-Step submodule is checked out and "
            "dependencies are installed."
        )

    handler = AceStepHandler()
    handler.init_service(device=device)
    return handler


def load_captions(path: str) -> list[str]:
    """Load captions from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def generate_random_captions(n: int) -> list[str]:
    """Generate diverse captions for activation collection."""
    genres = [
        "pop", "rock", "jazz", "classical", "electronic", "hip-hop",
        "R&B", "country", "metal", "folk", "blues", "reggae",
        "ambient", "lo-fi", "punk", "soul", "funk",
    ]
    moods = [
        "upbeat", "melancholic", "energetic", "calm", "dark",
        "dreamy", "aggressive", "romantic", "mysterious", "joyful",
    ]
    instruments = [
        "piano", "guitar", "drums", "synthesizer", "violin",
        "bass", "trumpet", "flute", "cello", "saxophone",
    ]
    import random

    captions = []
    for _ in range(n):
        genre = random.choice(genres)
        mood = random.choice(moods)
        inst = random.choice(instruments)
        bpm = random.randint(60, 180)
        caption = f"A {mood} {genre} track featuring {inst}, {bpm} BPM"
        captions.append(caption)
    return captions


def main():
    parser = argparse.ArgumentParser(
        description="Collect DiT activations from ACE-Step generations"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./ACE-Step",
        help="Path to ACE-Step installation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./activations",
        help="Directory to save activations",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="DiT layer indices to collect from (default: all)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=100,
        help="Number of generations to run",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default=None,
        help="Path to captions file (one per line). If not set, random captions are generated.",
    )
    parser.add_argument(
        "--max_samples_per_layer",
        type=int,
        default=None,
        help="Max activation vectors per layer",
    )
    parser.add_argument(
        "--timestep_filter",
        type=float,
        nargs=2,
        default=None,
        metavar=("T_MIN", "T_MAX"),
        help="Only collect activations at timesteps in [T_MIN, T_MAX]",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of each generated audio (seconds)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load captions
    if args.captions:
        captions = load_captions(args.captions)
    else:
        captions = generate_random_captions(args.num_generations)
    captions = captions[: args.num_generations]

    logger.info(f"Will run {len(captions)} generations")
    logger.info(f"Collecting from layers: {args.layers or 'all'}")
    logger.info(f"Output: {output_dir}")

    # Load model
    logger.info("Loading ACE-Step model...")
    handler = load_acestep_model(args.model_dir, args.device)

    # Get the DiT model
    dit_model = handler.model.decoder

    # Set up collector
    timestep_filter = (
        tuple(args.timestep_filter) if args.timestep_filter else None
    )
    collector = ActivationCollector(
        dit_model=dit_model,
        layers=args.layers,
        max_samples=args.max_samples_per_layer,
        timestep_filter=timestep_filter,
        stream_dir=str(output_dir),
    )

    logger.info("Installing hooks...")
    collector.install()

    # Run generations
    for i, caption in enumerate(captions):
        logger.info(f"Generation {i + 1}/{len(captions)}: {caption[:80]}...")
        try:
            handler.generate(
                caption=caption,
                duration=args.duration,
                seed=args.seed + i,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            continue

    collector.remove()

    # Report
    stats = collector.stats()
    logger.info("Collection complete:")
    for layer_idx, count in sorted(stats.items()):
        logger.info(f"  Layer {layer_idx}: {count:,} vectors")

    # Save metadata
    import json

    meta = {
        "layers": args.layers,
        "num_generations": len(captions),
        "timestep_filter": args.timestep_filter,
        "stats": {str(k): v for k, v in stats.items()},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Activations saved to {output_dir}")


if __name__ == "__main__":
    main()
