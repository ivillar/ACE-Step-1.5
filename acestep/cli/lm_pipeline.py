"""LM manual-edit pipeline: generate audio codes with optional metadata re-generation."""

from typing import Any, Dict, Optional

from acestep.generation_helpers import (
    accumulate_lm_time_costs,
    build_user_metadata,
    format_seed_string,
    safe_parse_metadata_value,
)
from acestep.inference import GenerationConfig, GenerationParams


def run_lm_generation(
    llm_handler,
    dit_handler,
    params: GenerationParams,
    config: GenerationConfig,
) -> Dict[str, Any]:
    """Execute the LM generation loop (up to two attempts for metadata edits).

    Returns:
        dict with keys: ``success``, ``lm_time_costs``, ``audio_codes``,
        ``lm_metadata``, ``edited_caption``, ``edited_lyrics``,
        ``edited_instruction``, ``edited_metas``.
    """
    top_k_value = (
        None if not params.lm_top_k or params.lm_top_k == 0
        else int(params.lm_top_k)
    )
    top_p_value = (
        None if not params.lm_top_p or params.lm_top_p >= 1.0
        else params.lm_top_p
    )
    actual_batch_size = config.batch_size if config.batch_size is not None else 1
    seed_str = format_seed_string(config.seeds)
    actual_seed_list, _ = dit_handler.prepare_seeds(
        actual_batch_size, seed_str, config.use_random_seed,
    )

    originals = snapshot_originals(params)
    lm_time_costs = {"phase1_time": 0.0, "phase2_time": 0.0, "total_time": 0.0}
    lm_metadata: dict = {}
    audio_codes: Any = ""

    for attempt in range(2):
        user_metadata = _build_user_metadata(params, attempt)
        lm_result = llm_handler.generate_with_stop_condition(
            caption=params.caption or "",
            lyrics=params.lyrics or "",
            infer_type="llm_dit",
            temperature=params.lm_temperature,
            cfg_scale=params.lm_cfg_scale,
            negative_prompt=params.lm_negative_prompt,
            top_k=top_k_value,
            top_p=top_p_value,
            target_duration=params.duration,
            user_metadata=user_metadata,
            use_cot_caption=params.use_cot_caption,
            use_cot_language=params.use_cot_language,
            use_cot_metas=params.use_cot_metas,
            use_constrained_decoding=params.use_constrained_decoding,
            constrained_decoding_debug=config.constrained_decoding_debug,
            batch_size=actual_batch_size,
            seeds=actual_seed_list,
        )
        accumulate_lm_time_costs(lm_time_costs, lm_result)

        if not lm_result.get("success", False):
            error_msg = lm_result.get("error", "Unknown LM error")
            print(f"\nGeneration failed: {error_msg}")
            print(f"   Status: {lm_result.get('error', '')}")
            return {"lm_time_costs": lm_time_costs, "success": False}

        if actual_batch_size > 1:
            lm_metadata = (lm_result.get("metadata") or [{}])[0]
            audio_codes = lm_result.get("audio_codes", [])
        else:
            lm_metadata = lm_result.get("metadata", {}) or {}
            audio_codes = lm_result.get("audio_codes", "")

        if not audio_codes:
            print("WARNING: LM did not return audio codes; proceeding without codes.")

        if attempt == 0 and _should_regenerate(llm_handler, params, originals):
            llm_handler._skip_prompt_edit = True
            continue
        break

    return {
        "success": True,
        "lm_time_costs": lm_time_costs,
        "audio_codes": audio_codes,
        "lm_metadata": lm_metadata,
        "edited_caption": getattr(llm_handler, "_edited_caption", None),
        "edited_lyrics": getattr(llm_handler, "_edited_lyrics", None),
        "edited_instruction": getattr(llm_handler, "_edited_instruction", None),
        "edited_metas": getattr(llm_handler, "_edited_metas", {}),
    }


def snapshot_originals(params: GenerationParams) -> Dict[str, Any]:
    """Capture current param values before LM modifies them."""
    return {
        "duration": params.duration,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "timesignature": params.timesignature,
        "vocal_language": params.vocal_language,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_metadata(params: GenerationParams, attempt: int) -> Optional[dict]:
    """Thin wrapper around ``build_user_metadata`` that adds retry-specific extras."""
    extras: Optional[dict] = None
    if attempt > 0:
        extras = {}
        if params.caption and params.caption.strip():
            extras["caption"] = params.caption.strip()
        if params.vocal_language and params.vocal_language not in ("", "unknown"):
            extras["language"] = params.vocal_language
    return build_user_metadata(
        params.bpm, params.keyscale, params.timesignature,
        params.duration, extras=extras,
    )


def _should_regenerate(llm_handler, params, originals) -> bool:
    """Check edited metas for changes that warrant a second LM attempt."""
    edited_metas = getattr(llm_handler, "_edited_metas", {})
    if not edited_metas:
        return False

    changes = _detect_meta_changes(edited_metas, originals)
    if not any(changes.values()):
        return False

    if changes["duration"]:
        params.duration = safe_parse_metadata_value(
            "duration", edited_metas, as_float=True,
        )
    if changes["bpm"]:
        params.bpm = safe_parse_metadata_value(
            "bpm", edited_metas, as_int=True,
        )
    if changes["keyscale"]:
        params.keyscale = edited_metas.get("keyscale")
    if changes["timesignature"]:
        params.timesignature = edited_metas.get("timesignature")
    if changes["language"]:
        params.vocal_language = (
            edited_metas.get("language") or edited_metas.get("vocal_language")
        )

    expanded_caption = edited_metas.get("caption")
    if expanded_caption and expanded_caption.strip():
        params.caption = expanded_caption

    print("INFO: Edited metadata detected. Regenerating audio codes with updated values.")
    return True


def _detect_meta_changes(edited_metas: dict, originals: dict) -> Dict[str, bool]:
    parsed_dur = safe_parse_metadata_value("duration", edited_metas, as_float=True)
    parsed_bpm = safe_parse_metadata_value("bpm", edited_metas, as_int=True)
    orig_dur = originals["duration"]
    return {
        "duration": parsed_dur is not None and (
            orig_dur is None or float(orig_dur) <= 0
            or abs(float(orig_dur) - parsed_dur) > 1e-6
        ),
        "bpm": parsed_bpm is not None and parsed_bpm != originals["bpm"],
        "keyscale": (
            edited_metas.get("keyscale") is not None
            and edited_metas["keyscale"] != originals["keyscale"]
        ),
        "timesignature": (
            edited_metas.get("timesignature") is not None
            and edited_metas["timesignature"] != originals["timesignature"]
        ),
        "language": (
            (edited_metas.get("language") or edited_metas.get("vocal_language")) is not None
            and (edited_metas.get("language") or edited_metas.get("vocal_language"))
            != originals["vocal_language"]
        ),
    }
