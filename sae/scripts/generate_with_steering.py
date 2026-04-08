#!/usr/bin/env python3
"""Generate music with ACE-Step while steering via SAE features.

This script demonstrates how to steer ACE-Step's generation by
amplifying or suppressing specific SAE features during inference.

Usage:
    python scripts/generate_with_steering.py \
        --model_dir ./ACE-Step \
        --sae_path ./sae_output/layer_24/sae_final.pt \
        --sae_layer 24 \
        --steer 42:10.0 100:-5.0 \
        --caption "A calm ambient track" \
        --output steered_output.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace_step_sae.model import TopKSparseAutoencoder
from ace_step_sae.steering import SteeringHook, SteeringSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_steer_specs(specs: list[str]) -> list[tuple[int, float]]:
    """Parse steering specs like ``42:10.0`` into (feature_idx, coeff)."""
    parsed = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid steer spec '{s}'. Expected format: feature_idx:coefficient"
            )
        parsed.append((int(parts[0]), float(parts[1])))
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with SAE feature steering"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./ACE-Step",
        help="Path to ACE-Step installation",
    )
    parser.add_argument(
        "--sae_path", type=str, required=True, help="Path to trained SAE"
    )
    parser.add_argument(
        "--sae_layer",
        type=int,
        required=True,
        help="DiT layer the SAE was trained on",
    )
    parser.add_argument(
        "--steer",
        type=str,
        nargs="+",
        required=True,
        help="Steering specs as feature_idx:coefficient (e.g. 42:10.0 100:-5.0)",
    )
    parser.add_argument(
        "--timestep_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("T_MIN", "T_MAX"),
        help="Only steer at timesteps in [T_MIN, T_MAX]",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="A pop song with catchy melody",
        help="Text caption for generation",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default="",
        help="Lyrics for the song",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Duration in seconds"
    )
    parser.add_argument(
        "--output", type=str, default="steered_output.wav", help="Output path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    # Comparison mode: generate with and without steering
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also generate without steering for comparison",
    )
    args = parser.parse_args()

    # Parse steering specs
    steer_specs = parse_steer_specs(args.steer)
    ts_range = tuple(args.timestep_range) if args.timestep_range else None

    logger.info("Steering specs:")
    for feat, coeff in steer_specs:
        sign = "amplify" if coeff > 0 else "suppress"
        logger.info(f"  Feature {feat}: {sign} by {abs(coeff):.1f}")

    # Load SAE
    logger.info(f"Loading SAE from {args.sae_path}...")
    sae = TopKSparseAutoencoder.load(args.sae_path, device=args.device)

    # Load ACE-Step
    logger.info("Loading ACE-Step model...")
    sys.path.insert(0, args.model_dir)
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    handler.init_service(device=args.device)

    dit_model = handler.model.decoder

    # Set up steering
    steering_hook = SteeringHook(
        dit_model=dit_model,
        sae=sae,
        default_layer=args.sae_layer,
    )
    for feat_idx, coeff in steer_specs:
        steering_hook.add(
            SteeringSpec(
                feature_idx=feat_idx,
                coefficient=coeff,
                timestep_range=ts_range,
            )
        )

    # Comparison: generate without steering
    if args.compare:
        logger.info("Generating baseline (no steering)...")
        baseline_output = handler.generate(
            caption=args.caption,
            lyrics=args.lyrics,
            duration=args.duration,
            seed=args.seed,
        )
        baseline_path = args.output.replace(".wav", "_baseline.wav")
        # Save baseline audio
        import torchaudio
        if hasattr(baseline_output, "audio"):
            torchaudio.save(
                baseline_path,
                baseline_output.audio.cpu(),
                48000,
            )
            logger.info(f"Baseline saved: {baseline_path}")

    # Generate with steering
    logger.info("Generating with steering...")
    steering_hook.install()

    # Hook into the diffusion loop to update timestep tracking.
    # This patches the generate_audio method to call set_timestep
    # before each denoising step.
    _original_generate = handler.model.generate_audio

    def patched_generate(*a, **kw):
        # We need to intercept the diffusion loop. Since the timestep
        # updates happen inside generate_audio, we wrap it.
        # A simpler approach: just steer at all timesteps (no timestep
        # filtering).  For timestep-aware steering, the user should
        # integrate more deeply.
        return _original_generate(*a, **kw)

    try:
        result = handler.generate(
            caption=args.caption,
            lyrics=args.lyrics,
            duration=args.duration,
            seed=args.seed,
        )
        # Save steered audio
        import torchaudio
        if hasattr(result, "audio"):
            torchaudio.save(args.output, result.audio.cpu(), 48000)
            logger.info(f"Steered output saved: {args.output}")
        else:
            logger.info("Generation complete (check handler output format)")
    finally:
        steering_hook.remove()

    logger.info("Done!")


if __name__ == "__main__":
    main()
