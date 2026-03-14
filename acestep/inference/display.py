"""Display and formatting helpers for CLI output."""

import os

from loguru import logger


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


def log_parameters(
    sys_cfg, params, config, compact, resolved_device=None,
) -> None:
    """Log a summary (compact) or full dump (debug) of generation parameters."""
    if not compact:
        logger.debug("Final Parameters (GenerationParams):")
        for k in sorted(vars(params).keys()):
            logger.debug(f"  {k}: {getattr(params, k)}")
        logger.debug("Final Parameters (GenerationConfig):")
        for k in sorted(vars(config).keys()):
            logger.debug(f"  {k}: {getattr(config, k)}")
        return

    device_display = str(sys_cfg["device"])
    if resolved_device and resolved_device != str(sys_cfg["device"]):
        device_display = f"{sys_cfg['device']} -> {resolved_device}"

    lines = [
        f"task_type={params.task_type}",
        f"caption={params.caption or 'none'}",
        f"lyrics={summarize_lyrics(params.lyrics)}",
        f"duration={params.duration}s",
        f"outputs={config.batch_size}",
    ]
    if params.bpm is not None:
        lines.append(f"bpm={params.bpm}")
    if params.keyscale:
        lines.append(f"keyscale={params.keyscale}")
    if params.timesignature:
        lines.append(f"timesignature={params.timesignature}")
    lines += [
        f"instrumental={params.instrumental}",
        f"thinking={params.thinking}",
        f"lm_model={sys_cfg['lm_model_path'] or 'auto'}",
        f"dit_model={sys_cfg['config_path'] or 'auto'}",
        f"backend={sys_cfg['backend']}",
        f"device={device_display}",
        f"audio_format={config.audio_format}",
        f"save_dir={sys_cfg['save_dir']}",
    ]
    if config.seeds:
        lines.append(f"seeds={config.seeds}")
    else:
        lines.append(f"seed={params.seed} (random={config.use_random_seed})")
    logger.info("Parameters: " + ", ".join(lines))


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


def log_dit_prompt(dit_handler, params) -> None:
    """Log the final DiT prompt for both caption and lyrics branches."""
    meta = build_meta_dict(params)
    caption_input, lyrics_input = dit_handler.build_dit_inputs(
        task=params.task_type,
        instruction=params.instruction,
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        metas=meta,
        vocal_language=params.vocal_language or "unknown",
    )
    logger.info(f"DiT prompt (caption): {caption_input}")
    logger.info(f"DiT prompt (lyrics): {lyrics_input}")


def log_performance(lm_time_costs, result, used_thinking) -> None:
    """Merge LM time costs into result and log a performance summary.

    Args:
        lm_time_costs: LM phase timing dict (may be None if LM was not used).
        result: The GenerationResult from generate_music.
        used_thinking: Whether the LM thinking path was active.
    """
    time_costs = result.extra_outputs.get("time_costs", {})

    # Merge LM phase times into the result dict
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

    if not time_costs:
        return

    total = time_costs.get("pipeline_total_time", 0)
    parts = [f"total={total:.2f}s"]
    if used_thinking:
        lm1 = time_costs.get("lm_phase1_time", 0)
        lm2 = time_costs.get("lm_phase2_time", 0)
        parts.append(f"LM={lm1 + lm2:.2f}s")
    parts.append(f"DiT={time_costs.get('dit_total_time_cost', 0):.2f}s")
    logger.info("Performance: " + ", ".join(parts))
