"""Display and formatting helpers for CLI output."""

import os


def summarize_lyrics(lyrics) -> str:
    """Return a short human-readable summary of lyrics content."""
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


def print_final_parameters(
    sys_cfg, params, config, compact, resolved_device=None,
) -> None:
    """Print a summary (compact) or full dump (debug) of generation parameters."""
    if not compact:
        print("\n--- Final Parameters (GenerationParams) ---")
        for k in sorted(vars(params).keys()):
            print(f"{k}: {getattr(params, k)}")
        print("-------------------------------------------")
        print("\n--- Final Parameters (GenerationConfig) ---")
        for k in sorted(vars(config).keys()):
            print(f"{k}: {getattr(config, k)}")
        print("-------------------------------------------\n")
        return

    device_display = str(sys_cfg["device"])
    if resolved_device and resolved_device != str(sys_cfg["device"]):
        device_display = f"{sys_cfg['device']} -> {resolved_device}"

    print("\n--- Final Parameters (Summary) ---")
    print(f"task_type: {params.task_type}")
    print(f"caption: {params.caption or 'none'}")
    print(f"lyrics: {summarize_lyrics(params.lyrics)}")
    print(f"duration: {params.duration}s")
    print(f"outputs: {config.batch_size}")
    if params.bpm is not None:
        print(f"bpm: {params.bpm}")
    if params.keyscale:
        print(f"keyscale: {params.keyscale}")
    if params.timesignature:
        print(f"timesignature: {params.timesignature}")
    print(f"instrumental: {params.instrumental}")
    print(f"thinking: {params.thinking}")
    print(f"lm_model: {sys_cfg['lm_model_path'] or 'auto'}")
    print(f"dit_model: {sys_cfg['config_path'] or 'auto'}")
    print(f"backend: {sys_cfg['backend']}")
    print(f"device: {device_display}")
    print(f"audio_format: {config.audio_format}")
    print(f"save_dir: {sys_cfg['save_dir']}")
    if config.seeds:
        print(f"seeds: {config.seeds}")
    else:
        print(f"seed: {params.seed} (random={config.use_random_seed})")
    print("-------------------------------\n")


def build_meta_dict(params):
    """Build a metadata dict from params for DiT input building."""
    meta = {}
    if params.bpm is not None:
        meta["bpm"] = params.bpm
    if params.timesignature:
        meta["timesignature"] = params.timesignature
    if params.keyscale:
        meta["keyscale"] = params.keyscale
    if params.duration is not None:
        meta["duration"] = params.duration
    return meta or None


def print_dit_prompt(dit_handler, params) -> None:
    """Print the final DiT prompt for both caption and lyrics branches."""
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


def merge_time_costs(lm_time_costs, result) -> dict:
    """Merge LM phase time costs into the generation result's time_costs dict."""
    time_costs = result.extra_outputs.get("time_costs", {})
    if lm_time_costs and time_costs is not None:
        if not isinstance(time_costs, dict):
            time_costs = {}
            result.extra_outputs["time_costs"] = time_costs
        if lm_time_costs["total_time"] > 0.0:
            time_costs["lm_phase1_time"] = lm_time_costs["phase1_time"]
            time_costs["lm_phase2_time"] = lm_time_costs["phase2_time"]
            time_costs["lm_total_time"] = lm_time_costs["total_time"]
            dit_total = float(time_costs.get("dit_total_time_cost", 0.0) or 0.0)
            time_costs["pipeline_total_time"] = lm_time_costs["total_time"] + dit_total
    return time_costs
