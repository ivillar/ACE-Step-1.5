"""Constants, task-instruction helpers, and default-value application for the CLI."""

from typing import Any, Dict, List, Optional

from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS
from acestep.inference import GenerationConfig, GenerationParams


TRACK_CHOICES = [
    "vocals",
    "backing_vocals",
    "drums",
    "bass",
    "guitar",
    "keyboard",
    "percussion",
    "strings",
    "synth",
    "fx",
    "brass",
    "woodwinds",
]

BASE_ONLY_TASKS = {"lego", "extract", "complete"}
SKIP_LM_TASKS = {"cover", "repaint"}


def default_instruction_for_task(
    task_type: str,
    tracks: Optional[List[str]] = None,
) -> str:
    """Return the default DiT instruction string for *task_type*."""
    if task_type == "lego":
        track = tracks[0] if tracks else "guitar"
        return TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track.upper())
    if task_type == "extract":
        track = tracks[0] if tracks else "vocals"
        return TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track.upper())
    if task_type == "complete":
        tracks_list = ", ".join(tracks) if tracks else "drums, bass, guitar"
        return TASK_INSTRUCTIONS["complete"].format(TRACK_CLASSES=tracks_list)
    return DEFAULT_DIT_INSTRUCTION


def build_all_defaults(
    params: GenerationParams,
    config: GenerationConfig,
) -> Dict[str, Any]:
    """Canonical mapping of all CLI-settable generation fields to their defaults.

    This is the single source of truth used by both the initial namespace
    construction in ``cli.py`` and by ``apply_optional_defaults``.
    """
    return {
        "duration": params.duration,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "timesignature": params.timesignature,
        "vocal_language": params.vocal_language,
        "inference_steps": params.inference_steps,
        "seed": params.seed,
        "guidance_scale": params.guidance_scale,
        "use_adg": params.use_adg,
        "cfg_interval_start": params.cfg_interval_start,
        "cfg_interval_end": params.cfg_interval_end,
        "shift": 3.0,
        "infer_method": params.infer_method,
        "timesteps": None,
        "repainting_start": params.repainting_start,
        "repainting_end": params.repainting_end,
        "audio_cover_strength": params.audio_cover_strength,
        "thinking": params.thinking,
        "lm_temperature": params.lm_temperature,
        "lm_cfg_scale": params.lm_cfg_scale,
        "lm_top_k": params.lm_top_k,
        "lm_top_p": params.lm_top_p,
        "lm_negative_prompt": params.lm_negative_prompt,
        "use_cot_metas": params.use_cot_metas,
        "use_cot_caption": params.use_cot_caption,
        "use_cot_lyrics": params.use_cot_lyrics,
        "use_cot_language": params.use_cot_language,
        "use_constrained_decoding": params.use_constrained_decoding,
        "batch_size": config.batch_size,
        "allow_lm_batch": config.allow_lm_batch,
        "use_random_seed": config.use_random_seed,
        "seeds": config.seeds,
        "lm_batch_chunk_size": config.lm_batch_chunk_size,
        "constrained_decoding_debug": config.constrained_decoding_debug,
        "audio_format": config.audio_format,
        "sample_mode": False,
        "sample_query": "",
        "use_format": False,
    }


def apply_optional_defaults(
    args,
    params_defaults: GenerationParams,
    config_defaults: GenerationConfig,
) -> None:
    """Fill missing attributes on *args* from the dataclass defaults."""
    for key, default_value in build_all_defaults(params_defaults, config_defaults).items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_value)
