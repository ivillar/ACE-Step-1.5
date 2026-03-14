"""Consolidated LM pipeline: pre-generation steps, LM generation loop, result merging, and prompt editing."""

import re
from typing import Any

from acestep.inference.parsing import parse_description_hints, parse_number
from acestep.inference.helpers import (
    accumulate_lm_time_costs,
    build_user_metadata,
    format_seed_string,
    safe_parse_metadata_value,
)
from acestep.inference.params import GenerationConfig, GenerationParams, create_sample, format_sample

# ---------------------------------------------------------------------------
# Pre-generation steps
# ---------------------------------------------------------------------------

def run_pre_generation_steps(
    params: GenerationParams, llm_handler, *,
    sample_mode: bool, sample_query: str, use_format: bool,
) -> None:
    """Execute sample_mode, format_sample, and use_cot_lyrics before generation."""
    format_has_duration = False

    if sample_mode or (sample_query and str(sample_query).strip()):
        _run_sample_mode(params, llm_handler, sample_query)

    if use_format and (params.caption or params.lyrics):
        format_has_duration = _run_format_sample(params, llm_handler)

    if params.use_cot_lyrics:
        _run_cot_lyrics_generation(params, llm_handler)

    if sample_mode or format_has_duration:
        params.use_cot_metas = False


def _lm_top_k(params: GenerationParams):
    """Convert lm_top_k=0 to None (disabled) for vLLM compatibility."""
    return None if not params.lm_top_k or params.lm_top_k == 0 else int(params.lm_top_k)


def _lm_top_p(params: GenerationParams):
    """Convert lm_top_p>=1.0 to None (disabled) for vLLM compatibility."""
    return None if not params.lm_top_p or params.lm_top_p >= 1.0 else params.lm_top_p


def _apply_sample_result(params: GenerationParams, result) -> None:
    """Apply a create_sample result to params."""
    params.caption = result.caption
    params.lyrics = result.lyrics
    if params.bpm is None:
        params.bpm = result.bpm
    if not params.keyscale:
        params.keyscale = result.keyscale
    if not params.timesignature:
        params.timesignature = result.timesignature
    if params.duration <= 0:
        params.duration = result.duration


def _run_sample_mode(params: GenerationParams, llm_handler, sample_query: str) -> None:
    if not llm_handler.llm_initialized:
        raise RuntimeError(
            "sample_mode/sample_query requires the LM handler, "
            "but it's not initialized."
        )

    query = (
        sample_query
        if sample_query and str(sample_query).strip()
        else "NO USER INPUT"
    )
    parsed_language, parsed_instrumental = parse_description_hints(query)
    if params.vocal_language and params.vocal_language not in ("en", "unknown", ""):
        sample_language = params.vocal_language
    else:
        sample_language = parsed_language

    print("\nINFO: Creating sample via 'create_sample'...")
    result = create_sample(
        llm_handler=llm_handler, query=query,
        instrumental=parsed_instrumental, vocal_language=sample_language,
        temperature=params.lm_temperature, top_k=_lm_top_k(params), top_p=_lm_top_p(params),
    )
    if result.success:
        _apply_sample_result(params, result)
        params.instrumental = bool(result.instrumental)
        if params.vocal_language in ("unknown", "", None):
            params.vocal_language = result.language
        print("Sample created. Using generated parameters.")
    else:
        raise RuntimeError(
            f"create_sample failed: {result.error or result.status_message}"
        )


def _run_format_sample(params: GenerationParams, llm_handler) -> bool:
    """Returns True if format_sample provided a duration."""
    if not llm_handler.llm_initialized:
        raise RuntimeError(
            "use_format requires the LM handler, but it's not initialized."
        )

    user_metadata: dict = {}
    if params.bpm is not None:
        user_metadata["bpm"] = params.bpm
    if params.duration is not None and float(params.duration) > 0:
        user_metadata["duration"] = float(params.duration)
    if params.keyscale:
        user_metadata["keyscale"] = params.keyscale
    if params.timesignature:
        user_metadata["timesignature"] = params.timesignature
    if params.vocal_language and params.vocal_language != "unknown":
        user_metadata["language"] = params.vocal_language

    print("\nINFO: Formatting caption/lyrics via 'format_sample'...")
    result = format_sample(
        llm_handler=llm_handler,
        caption=params.caption or "", lyrics=params.lyrics or "",
        user_metadata=user_metadata or None,
        temperature=params.lm_temperature,
        top_k=_lm_top_k(params), top_p=_lm_top_p(params),
    )
    if result.success:
        params.caption = result.caption or params.caption
        params.lyrics = result.lyrics or params.lyrics
        fmt_has_duration = False
        if result.duration:
            params.duration = result.duration
            fmt_has_duration = True
        if result.bpm:
            params.bpm = result.bpm
        if result.keyscale:
            params.keyscale = result.keyscale
        if result.timesignature:
            params.timesignature = result.timesignature
        print("Format complete.")
        return fmt_has_duration
    else:
        raise RuntimeError(
            f"format_sample failed: {result.error or result.status_message}"
        )


def _run_cot_lyrics_generation(params: GenerationParams, llm_handler) -> None:
    if not llm_handler.llm_initialized:
        raise RuntimeError(
            "use_cot_lyrics requires the LM handler, but it's not initialized. "
            "Ensure thinking is enabled."
        )

    print("\nINFO: Generating lyrics and metadata via 'create_sample'...")
    result = create_sample(
        llm_handler=llm_handler, query=params.caption,
        instrumental=False,
        vocal_language=(
            params.vocal_language if params.vocal_language != "unknown" else None
        ),
        temperature=params.lm_temperature,
        top_k=_lm_top_k(params), top_p=_lm_top_p(params),
    )
    if result.success:
        print("Automatic sample creation successful. Using generated parameters:")
        _apply_sample_result(params, result)
        if params.vocal_language == "unknown":
            params.vocal_language = result.language
        lyrics_preview = params.lyrics[:150].strip().replace("\n", " ")
        print(f"  - Caption: {params.caption}")
        print(f"  - Lyrics: '{lyrics_preview}...'")
        print(
            f"  - Metadata: BPM={params.bpm}, Key='{params.keyscale}', "
            f"Lang='{params.vocal_language}'"
        )
        params.use_cot_metas = False
        params.use_cot_caption = False
    else:
        print(f"WARNING: Automatic lyric generation failed: {result.error}")
        print("         Proceeding with an instrumental track instead.")
        params.lyrics = "[Instrumental]"
        params.instrumental = True

    params.use_cot_lyrics = False


# ---------------------------------------------------------------------------
# Prompt editing (monkey-patch)
# ---------------------------------------------------------------------------

def _edit_formatted_prompt_via_file(formatted_prompt: str, instruction_path: str) -> str:
    """Write *formatted_prompt* to *instruction_path*, wait for user edits, then read back."""
    try:
        with open(instruction_path, "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    except Exception as e:
        print(f"WARNING: Failed to write {instruction_path}: {e}")
        return formatted_prompt

    print("\n--- Final Draft Saved ---")
    print(f"Saved to {instruction_path}")
    print("Edit the file now. Press Enter when ready to continue.")
    input()

    try:
        with open(instruction_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"WARNING: Failed to read {instruction_path}: {e}")
        return formatted_prompt


def _extract_caption_lyrics(formatted_prompt: str) -> tuple[str | None, str | None]:
    """Best-effort extraction of caption and lyrics from a formatted prompt string."""
    matches = list(
        re.finditer(r"# Caption\n(.*?)\n+# Lyric\n(.*)", formatted_prompt, re.DOTALL)
    )
    if not matches:
        return None, None

    caption = matches[-1].group(1).strip()
    lyrics = matches[-1].group(2)

    cut_markers = [
        "<|eot_id|>", "<|start_header_id|>", "<|assistant|>",
        "<|user|>", "<|system|>", "<|im_end|>", "<|im_start|>",
    ]
    cut_at = len(lyrics)
    for marker in cut_markers:
        pos = lyrics.find(marker)
        if pos != -1:
            cut_at = min(cut_at, pos)
    lyrics = lyrics[:cut_at].rstrip()

    return caption or None, lyrics or None


def _extract_instruction(formatted_prompt: str) -> str | None:
    """Best-effort extraction of instruction text from a formatted prompt string."""
    match = re.search(r"# Instruction\n(.*?)\n\n", formatted_prompt, re.DOTALL)
    if not match:
        return None
    instruction = match.group(1).strip()
    return instruction or None


def _extract_cot_metadata(formatted_prompt: str) -> dict[str, str]:
    """Best-effort extraction of COT metadata (supports multi-line values)."""
    matches = list(
        re.finditer(r"<think>\n(.*?)\n</think>", formatted_prompt, re.DOTALL)
    )
    if not matches:
        return {}
    block = matches[-1].group(1)
    metadata: dict[str, str] = {}
    current_key: str | None = None
    current_value_lines: list[str] = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        key_match = re.match(r"^(\w+):\s*(.*)", line)
        if key_match:
            if current_key:
                metadata[current_key] = " ".join(current_value_lines).strip()
            current_key = key_match.group(1).strip().lower()
            current_value_lines = [key_match.group(2).strip()]
        elif current_key:
            current_value_lines.append(line)

    if current_key and current_value_lines:
        metadata[current_key] = " ".join(current_value_lines).strip()

    return metadata


def install_prompt_edit_hook(
    llm_handler,
    instruction_path: str,
    preloaded_prompt: str | None = None,
) -> None:
    """Monkey-patch *llm_handler.build_formatted_prompt_with_cot* to allow user editing."""
    original = llm_handler.build_formatted_prompt_with_cot
    cache: dict = {}

    def wrapped(
        caption, lyrics, cot_text, is_negative_prompt=False, negative_prompt="NO USER INPUT"
    ):
        prompt = original(
            caption, lyrics, cot_text,
            is_negative_prompt=is_negative_prompt,
            negative_prompt=negative_prompt,
        )
        if is_negative_prompt:
            conditional_prompt = original(
                caption, lyrics, cot_text,
                is_negative_prompt=False,
                negative_prompt=negative_prompt,
            )
            cached = cache.get(conditional_prompt)
            if cached and (cached.get("edited_caption") or cached.get("edited_lyrics")):
                return original(
                    cached.get("edited_caption") or caption,
                    cached.get("edited_lyrics") or lyrics,
                    cot_text,
                    is_negative_prompt=True,
                    negative_prompt=negative_prompt,
                )
            return prompt

        cached = cache.get(prompt)
        if cached:
            return cached["edited_prompt"]

        if getattr(llm_handler, "_skip_prompt_edit", False):
            cache[prompt] = {
                "edited_prompt": prompt,
                "edited_caption": None,
                "edited_lyrics": None,
            }
            return prompt

        if preloaded_prompt is not None:
            edited = preloaded_prompt
        else:
            edited = _edit_formatted_prompt_via_file(prompt, instruction_path)

        edited_caption, edited_lyrics = _extract_caption_lyrics(edited)
        if edited != prompt:
            print("INFO: Using edited draft for audio-token prompt.")
            if edited_caption or edited_lyrics:
                llm_handler._edited_caption = edited_caption
                llm_handler._edited_lyrics = edited_lyrics
            edited_instruction = _extract_instruction(edited)
            if edited_instruction:
                llm_handler._edited_instruction = edited_instruction
            edited_metas = _extract_cot_metadata(edited)
            if edited_metas:
                llm_handler._edited_metas = edited_metas

        cache[prompt] = {
            "edited_prompt": edited,
            "edited_caption": edited_caption,
            "edited_lyrics": edited_lyrics,
        }
        return edited

    llm_handler.build_formatted_prompt_with_cot = wrapped


# ---------------------------------------------------------------------------
# LM generation loop
# ---------------------------------------------------------------------------

def snapshot_originals(params: GenerationParams) -> dict[str, Any]:
    """Capture current param values before LM modifies them."""
    return {
        "duration": params.duration,
        "bpm": params.bpm,
        "keyscale": params.keyscale,
        "timesignature": params.timesignature,
        "vocal_language": params.vocal_language,
    }


def run_lm_generation(
    llm_handler,
    dit_handler,
    params: GenerationParams,
    config: GenerationConfig,
    originals: dict[str, Any],
) -> dict[str, Any]:
    """Execute the LM generation loop (up to two attempts for metadata edits)."""
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

    regenerated = attempt > 0
    if regenerated:
        edited_caption_out = None
        edited_lyrics_out = None
    else:
        edited_caption_out = getattr(llm_handler, "_edited_caption", None)
        edited_lyrics_out = getattr(llm_handler, "_edited_lyrics", None)

    return {
        "success": True,
        "lm_time_costs": lm_time_costs,
        "audio_codes": audio_codes,
        "lm_metadata": lm_metadata,
        "edited_caption": edited_caption_out,
        "edited_lyrics": edited_lyrics_out,
        "edited_instruction": getattr(llm_handler, "_edited_instruction", None),
        "edited_metas": getattr(llm_handler, "_edited_metas", {}),
        "regenerated": regenerated,
    }


def _build_user_metadata(params: GenerationParams, attempt: int) -> dict | None:
    extras: dict | None = None
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


def _detect_meta_changes(edited_metas: dict, originals: dict) -> dict[str, bool]:
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


# ---------------------------------------------------------------------------
# LM result merging
# ---------------------------------------------------------------------------

def apply_lm_results(
    params: GenerationParams,
    lm_result: dict[str, Any],
    originals: dict[str, Any],
) -> None:
    """Merge LM outputs (metadata, edits) back into *params*."""
    edited_metas = lm_result.get("edited_metas") or {}
    edited_caption = lm_result.get("edited_caption")
    edited_lyrics = lm_result.get("edited_lyrics")
    edited_instruction = lm_result.get("edited_instruction")
    lm_metadata = lm_result.get("lm_metadata") or {}
    regenerated = lm_result.get("regenerated", False)
    if regenerated:
        edited_metas_no_text = {k: v for k, v in edited_metas.items() if k != "caption"}
        _merge_caption(params, edited_metas_no_text, None, lm_metadata)
        _merge_lyrics(params, None, lm_metadata)
    else:
        _merge_caption(params, edited_metas, edited_caption, lm_metadata)
        _merge_lyrics(params, edited_lyrics, lm_metadata)

    if edited_instruction:
        params.instruction = edited_instruction

    if edited_metas:
        _apply_edited_metas(params, edited_metas)
    else:
        _apply_lm_metadata_fallback(params, lm_metadata)

    if params.use_cot_language:
        edited_lang = (
            (edited_metas.get("language") or edited_metas.get("vocal_language"))
            if edited_metas else None
        )
        if not edited_lang:
            lm_lang = lm_metadata.get("vocal_language") or lm_metadata.get("language")
            if lm_lang:
                params.vocal_language = lm_lang

    _set_cot_fields(params, lm_metadata, originals)

    params.thinking = False
    params.use_cot_caption = False
    params.use_cot_lyrics = False
    params.use_cot_language = False
    params.use_cot_metas = False

    if audio_codes := lm_result.get("audio_codes"):
        params.audio_codes = audio_codes


def _merge_caption(params, edited_metas, edited_caption, lm_metadata) -> None:
    meta_caption = edited_metas.get("caption") if edited_metas else None
    if meta_caption and meta_caption.strip():
        params.caption = meta_caption
    elif edited_caption:
        params.caption = edited_caption
    elif params.use_cot_caption and lm_metadata.get("caption"):
        params.caption = lm_metadata["caption"]


def _merge_lyrics(params, edited_lyrics, lm_metadata) -> None:
    if edited_lyrics:
        params.lyrics = edited_lyrics
    elif not params.lyrics and lm_metadata.get("lyrics"):
        params.lyrics = lm_metadata["lyrics"]


def _apply_edited_metas(params: GenerationParams, metas: dict) -> None:
    bpm = safe_parse_metadata_value("bpm", metas, as_int=True)
    if bpm is not None:
        params.bpm = bpm
    duration = safe_parse_metadata_value("duration", metas, as_float=True)
    if duration is not None:
        params.duration = duration
    if metas.get("keyscale"):
        params.keyscale = metas["keyscale"]
    if metas.get("timesignature"):
        params.timesignature = metas["timesignature"]
    lang = metas.get("language") or metas.get("vocal_language")
    if lang:
        params.vocal_language = lang


def _apply_lm_metadata_fallback(params: GenerationParams, lm_metadata: dict) -> None:
    if params.bpm is None and lm_metadata.get("bpm") not in (None, "N/A", ""):
        parsed = parse_number(str(lm_metadata["bpm"]))
        if parsed is not None:
            params.bpm = int(parsed)
    if not params.keyscale and lm_metadata.get("keyscale"):
        params.keyscale = lm_metadata["keyscale"]
    if not params.timesignature and lm_metadata.get("timesignature"):
        params.timesignature = lm_metadata["timesignature"]
    if params.duration is None and lm_metadata.get("duration") not in (None, "N/A", ""):
        parsed = parse_number(str(lm_metadata["duration"]))
        if parsed is not None:
            params.duration = float(parsed)
    if params.vocal_language in (None, "", "unknown"):
        lang = lm_metadata.get("vocal_language") or lm_metadata.get("language")
        if lang:
            params.vocal_language = lang


def _set_cot_fields(
    params: GenerationParams, lm_metadata: dict, originals: dict[str, Any],
) -> None:
    if not lm_metadata:
        return
    if originals["bpm"] is None:
        params.cot_bpm = params.bpm
    if not originals["keyscale"]:
        params.cot_keyscale = params.keyscale
    if not originals["timesignature"]:
        params.cot_timesignature = params.timesignature
    orig_dur = originals["duration"]
    if orig_dur is None or float(orig_dur) <= 0:
        params.cot_duration = params.duration
    if originals["vocal_language"] in (None, "", "unknown"):
        params.cot_vocal_language = params.vocal_language
    if not params.caption:
        params.cot_caption = lm_metadata.get("caption", "")
    if not params.lyrics:
        params.cot_lyrics = lm_metadata.get("lyrics", "")
