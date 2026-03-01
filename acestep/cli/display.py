"""Display and summary helpers for CLI output."""

import os
from typing import Optional

from acestep.inference import GenerationConfig, GenerationParams


def summarize_lyrics(lyrics: Optional[str]) -> str:
    """Return a short human-readable summary of the lyrics value."""
    if not lyrics:
        return "none"
    if isinstance(lyrics, str):
        stripped = lyrics.strip()
        if not stripped:
            return "none"
        if os.path.isfile(stripped):
            return f"file: {os.path.basename(stripped)}"
        if len(stripped) <= 60:
            return stripped.replace("\n", " ")
        return f"text ({len(stripped)} chars)"
    return "provided"


def build_meta_dict(params: GenerationParams) -> Optional[dict]:
    """Build a metadata dict from *params* for DiT prompt construction."""
    meta: dict = {}
    if params.bpm is not None:
        meta["bpm"] = params.bpm
    if params.timesignature:
        meta["timesignature"] = params.timesignature
    if params.keyscale:
        meta["keyscale"] = params.keyscale
    if params.duration is not None:
        meta["duration"] = params.duration
    return meta or None


def print_final_parameters(
    args,
    params: GenerationParams,
    config: GenerationConfig,
    params_defaults: GenerationParams,
    config_defaults: GenerationConfig,
    compact: bool,
    resolved_device: Optional[str] = None,
) -> None:
    """Print a summary (compact) or full dump of generation parameters."""
    if not compact:
        print("\n--- Final Parameters (Args) ---")
        for k in sorted(vars(args).keys()):
            print(f"{k}: {getattr(args, k)}")
        print("------------------------------")
        print("\n--- Final Parameters (GenerationParams) ---")
        for k in sorted(vars(params).keys()):
            print(f"{k}: {getattr(params, k)}")
        print("-------------------------------------------")
        print("\n--- Final Parameters (GenerationConfig) ---")
        for k in sorted(vars(config).keys()):
            print(f"{k}: {getattr(config, k)}")
        print("-------------------------------------------\n")
        return

    device_display = args.device
    if resolved_device and resolved_device != args.device:
        device_display = f"{args.device} -> {resolved_device}"

    print("\n--- Final Parameters (Summary) ---")
    print(f"task_type: {params.task_type}")
    print(f"caption: {params.caption or 'none'}")
    print(f"lyrics: {summarize_lyrics(params.lyrics)}")
    print(f"duration: {params.duration}s")
    print(f"outputs: {config.batch_size}")
    if params.bpm not in (None, params_defaults.bpm):
        print(f"bpm: {params.bpm}")
    if params.keyscale not in (None, params_defaults.keyscale):
        print(f"keyscale: {params.keyscale}")
    if params.timesignature not in (None, params_defaults.timesignature):
        print(f"timesignature: {params.timesignature}")
    print(f"instrumental: {params.instrumental}")
    print(f"thinking: {params.thinking}")
    print(f"lm_model: {args.lm_model_path or 'auto'}")
    print(f"dit_model: {args.config_path or 'auto'}")
    print(f"backend: {args.backend}")
    print(f"device: {device_display}")
    print(f"audio_format: {config.audio_format}")
    print(f"save_dir: {args.save_dir}")
    if config.seeds:
        print(f"seeds: {config.seeds}")
    else:
        print(f"seed: {params.seed} (random={config.use_random_seed})")
    print("-------------------------------\n")


def print_dit_prompt(dit_handler, params: GenerationParams) -> None:
    """Display the fully-constructed DiT caption and lyrics branches."""
    meta = build_meta_dict(params)
    caption_input, lyrics_input = dit_handler.build_dit_inputs(
        task=params.task_type,
        instruction=params.instruction,
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        metas=meta,
        vocal_language=params.vocal_language or "unknown",
    )
    print("\n--- Final DiT Prompt (Caption Branch) ---")
    print(caption_input)
    print("\n--- Final DiT Prompt (Lyrics Branch) ---")
    print(lyrics_input)
    print("----------------------------------------\n")
