"""Wizard sub-step for advanced parameter configuration."""

from acestep.cli.prompt_helpers import (
    prompt_bool,
    prompt_float,
    prompt_int,
    prompt_with_default,
)


def collect_advanced_parameters(args) -> None:
    """Interactively collect advanced DiT, LM, and output parameters."""
    if args.task_type == "text2music" and not args.sample_mode:
        args.use_format = prompt_bool(
            "Use format_sample to enhance caption/lyrics", args.use_format,
        )

    _collect_metadata(args)
    _collect_dit_settings(args)
    _collect_lm_settings(args)
    _collect_output_settings(args)


def _collect_metadata(args) -> None:
    """Optional metadata: duration, BPM, keyscale, time-signature, language."""
    print("\n--- Optional Metadata ---")
    args.duration = prompt_float(
        "Duration in seconds (10-600)", args.duration, min_value=10, max_value=600,
    )
    args.bpm = prompt_int("BPM (30-300, empty for auto)", args.bpm, min_value=30, max_value=300)
    args.keyscale = prompt_with_default(
        "Keyscale (e.g., 'C Major', empty for auto)", args.keyscale,
    )
    args.timesignature = prompt_with_default(
        "Time signature (e.g., '4/4', empty for auto)", args.timesignature,
    )
    args.vocal_language = prompt_with_default(
        "Vocal language (e.g., 'en', 'zh', 'unknown')", args.vocal_language,
    )


def _collect_dit_settings(args) -> None:
    """Advanced DiT settings: seed, steps, guidance, shift, timesteps, cover strength."""
    print("\n--- Advanced DiT Settings ---")
    args.seed = prompt_int("Random seed (-1 for random)", args.seed)
    args.inference_steps = prompt_int("Inference steps", args.inference_steps, min_value=1)
    if args.config_path and "base" in args.config_path:
        args.guidance_scale = prompt_float(
            "Guidance scale (for base models)", args.guidance_scale,
        )
        args.use_adg = prompt_bool("Enable Adaptive Dual Guidance (ADG)", args.use_adg)
        args.cfg_interval_start = prompt_float(
            "CFG interval start (0.0-1.0)", args.cfg_interval_start, 0.0, 1.0,
        )
        args.cfg_interval_end = prompt_float(
            "CFG interval end (0.0-1.0)", args.cfg_interval_end, 0.0, 1.0,
        )
    args.shift = prompt_float("Timestep shift (1.0-5.0)", args.shift, 1.0, 5.0)
    args.infer_method = prompt_with_default(
        "Inference method (ode/sde)", args.infer_method,
    )
    timesteps_input = prompt_with_default(
        "Custom timesteps list (e.g., [0.97, 0.5, 0])", args.timesteps, required=False,
    )
    if timesteps_input:
        args.timesteps = timesteps_input

    if args.task_type == "cover":
        args.audio_cover_strength = prompt_float(
            "Audio cover strength (0.0-1.0)", args.audio_cover_strength, 0.0, 1.0,
        )


def _collect_lm_settings(args) -> None:
    """Advanced LM settings."""
    print("\n--- Advanced LM Settings ---")
    args.thinking = prompt_bool("Enable LM 'thinking'", args.thinking)
    args.lm_temperature = prompt_float(
        "LM temperature (0.0-2.0)", args.lm_temperature, 0.0, 2.0,
    )
    args.lm_cfg_scale = prompt_float("LM CFG scale", args.lm_cfg_scale)
    args.lm_top_k = prompt_int("LM top-k (0 disables)", args.lm_top_k, min_value=0)
    args.lm_top_p = prompt_float("LM top-p (0.0-1.0)", args.lm_top_p, 0.0, 1.0)
    args.lm_negative_prompt = prompt_with_default(
        "LM negative prompt", args.lm_negative_prompt,
    )
    args.use_cot_metas = prompt_bool("Use CoT for metadata", args.use_cot_metas)
    args.use_cot_caption = prompt_bool(
        "Use CoT for caption refinement", args.use_cot_caption,
    )
    args.use_cot_lyrics = prompt_bool("Use CoT for lyrics generation", args.use_cot_lyrics)
    args.use_cot_language = prompt_bool(
        "Use CoT for language detection", args.use_cot_language,
    )
    args.use_constrained_decoding = prompt_bool(
        "Use constrained decoding", args.use_constrained_decoding,
    )


def _collect_output_settings(args) -> None:
    """Output settings: save directory, format, seeds, batch options."""
    print("\n--- Output Settings ---")
    args.save_dir = prompt_with_default("Save directory", args.save_dir)
    args.audio_format = prompt_with_default("Audio format (mp3/wav/flac)", args.audio_format)
    args.use_random_seed = prompt_bool("Use random seed per batch", args.use_random_seed)
    seeds_input = prompt_with_default(
        "Custom seeds (comma/space separated, leave empty for random)",
        "", required=False,
    )
    if seeds_input:
        seeds = [s for s in seeds_input.replace(",", " ").split() if s.strip()]
        try:
            args.seeds = [int(s) for s in seeds]
        except ValueError:
            print("Invalid seeds input. Ignoring custom seeds.")
    args.allow_lm_batch = prompt_bool("Allow LM batch processing", args.allow_lm_batch)
    args.lm_batch_chunk_size = prompt_int(
        "LM batch chunk size", args.lm_batch_chunk_size, min_value=1,
    )
    args.constrained_decoding_debug = prompt_bool(
        "Constrained decoding debug", args.constrained_decoding_debug,
    )
