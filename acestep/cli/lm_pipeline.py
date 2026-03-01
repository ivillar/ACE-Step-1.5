"""LM manual-edit pipeline: generate audio codes with optional metadata re-generation."""

from typing import Any, Dict, Optional

from acestep.inference import GenerationConfig, GenerationParams

from acestep.cli.parsing import parse_number


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
    seed_str = _format_seed_string(config)
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
        _accumulate_time_costs(lm_time_costs, lm_result)

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

def _format_seed_string(config: GenerationConfig) -> str:
    if config.seeds is None:
        return ""
    if isinstance(config.seeds, list) and len(config.seeds) > 0:
        return ",".join(str(s) for s in config.seeds)
    if isinstance(config.seeds, int):
        return str(config.seeds)
    return ""


def _build_user_metadata(params: GenerationParams, attempt: int) -> Optional[dict]:
    meta: dict = {}
    if params.bpm is not None:
        try:
            bpm_value = float(params.bpm)
            if bpm_value > 0:
                meta["bpm"] = int(bpm_value)
        except (ValueError, TypeError):
            pass
    if (
        params.keyscale and params.keyscale.strip()
        and params.keyscale.strip().lower() not in {"n/a", ""}
    ):
        meta["keyscale"] = params.keyscale.strip()
    if (
        params.timesignature and params.timesignature.strip()
        and params.timesignature.strip().lower() not in {"n/a", ""}
    ):
        meta["timesignature"] = params.timesignature.strip()
    if params.duration is not None:
        try:
            duration_value = float(params.duration)
            if duration_value > 0:
                meta["duration"] = int(duration_value)
        except (ValueError, TypeError):
            pass
    if attempt > 0:
        if params.caption and params.caption.strip():
            meta["caption"] = params.caption.strip()
        if params.vocal_language and params.vocal_language not in ("", "unknown"):
            meta["language"] = params.vocal_language
    return meta or None


def _accumulate_time_costs(lm_time_costs: Dict[str, float], lm_result: dict) -> None:
    extra = (lm_result.get("extra_outputs") or {}).get("time_costs", {})
    if not extra:
        return
    lm_time_costs["phase1_time"] += float(extra.get("phase1_time", 0.0) or 0.0)
    lm_time_costs["phase2_time"] += float(extra.get("phase2_time", 0.0) or 0.0)
    lm_time_costs["total_time"] += float(
        extra.get(
            "total_time",
            (extra.get("phase1_time", 0.0) or 0.0)
            + (extra.get("phase2_time", 0.0) or 0.0),
        ) or 0.0
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
        params.duration = _safe_parse("duration", edited_metas, as_float=True)
    if changes["bpm"]:
        params.bpm = _safe_parse("bpm", edited_metas, as_int=True)
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
    parsed_dur = _safe_parse("duration", edited_metas, as_float=True)
    parsed_bpm = _safe_parse("bpm", edited_metas, as_int=True)
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


def _safe_parse(
    key: str, metas: dict,
    as_int: bool = False, as_float: bool = False,
) -> Optional[float]:
    raw = metas.get(key)
    if not raw:
        return None
    parsed = parse_number(raw)
    if parsed is None or parsed <= 0:
        return None
    if as_int:
        return int(parsed)
    return float(parsed) if as_float else parsed
