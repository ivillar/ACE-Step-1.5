"""Constants, task-instruction helpers, and default-value application for the CLI."""

from typing import List, Optional

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


def apply_optional_defaults(
    args,
    params_defaults: GenerationParams,
    config_defaults: GenerationConfig,
) -> None:
    """Fill missing attributes on *args* from the dataclass defaults."""
    optional_defaults = {
        "duration": params_defaults.duration,
        "bpm": params_defaults.bpm,
        "keyscale": params_defaults.keyscale,
        "timesignature": params_defaults.timesignature,
        "vocal_language": params_defaults.vocal_language,
        "inference_steps": params_defaults.inference_steps,
        "seed": params_defaults.seed,
        "guidance_scale": params_defaults.guidance_scale,
        "use_adg": params_defaults.use_adg,
        "cfg_interval_start": params_defaults.cfg_interval_start,
        "cfg_interval_end": params_defaults.cfg_interval_end,
        "shift": 3.0,
        "infer_method": params_defaults.infer_method,
        "timesteps": None,
        "repainting_start": params_defaults.repainting_start,
        "repainting_end": params_defaults.repainting_end,
        "audio_cover_strength": params_defaults.audio_cover_strength,
        "thinking": params_defaults.thinking,
        "lm_temperature": params_defaults.lm_temperature,
        "lm_cfg_scale": params_defaults.lm_cfg_scale,
        "lm_top_k": params_defaults.lm_top_k,
        "lm_top_p": params_defaults.lm_top_p,
        "lm_negative_prompt": params_defaults.lm_negative_prompt,
        "use_cot_metas": params_defaults.use_cot_metas,
        "use_cot_caption": params_defaults.use_cot_caption,
        "use_cot_lyrics": params_defaults.use_cot_lyrics,
        "use_cot_language": params_defaults.use_cot_language,
        "use_constrained_decoding": params_defaults.use_constrained_decoding,
        "batch_size": config_defaults.batch_size,
        "allow_lm_batch": config_defaults.allow_lm_batch,
        "use_random_seed": config_defaults.use_random_seed,
        "seeds": config_defaults.seeds,
        "lm_batch_chunk_size": config_defaults.lm_batch_chunk_size,
        "constrained_decoding_debug": config_defaults.constrained_decoding_debug,
        "audio_format": config_defaults.audio_format,
        "sample_mode": False,
        "sample_query": "",
        "use_format": False,
    }
    for key, default_value in optional_defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default_value)
