"""Merge LM generation results back into GenerationParams."""

from typing import Any, Dict

from acestep.cli.parsing import parse_number
from acestep.generation_helpers import safe_parse_metadata_value
from acestep.inference import GenerationParams


def apply_lm_results_to_params(
    params: GenerationParams,
    lm_result: Dict[str, Any],
    originals: Dict[str, Any],
) -> None:
    """Merge LM outputs (metadata, edits) back into *params*."""
    edited_metas = lm_result.get("edited_metas") or {}
    edited_caption = lm_result.get("edited_caption")
    edited_lyrics = lm_result.get("edited_lyrics")
    edited_instruction = lm_result.get("edited_instruction")
    lm_metadata = lm_result.get("lm_metadata") or {}

    _apply_caption(params, edited_metas, edited_caption, lm_metadata)
    _apply_lyrics(params, edited_lyrics, lm_metadata)

    if edited_instruction:
        params.instruction = edited_instruction

    if edited_metas:
        _apply_edited_metas(params, edited_metas)
    else:
        _apply_lm_metadata_fallback(params, lm_metadata)

    if params.use_cot_language:
        _apply_cot_language(params, edited_metas, lm_metadata)

    _populate_cot_fields(params, lm_metadata, originals)

    params.thinking = False
    params.use_cot_caption = False
    params.use_cot_language = False
    params.use_cot_metas = False

    if audio_codes := lm_result.get("audio_codes"):
        params.audio_codes = audio_codes


def _apply_caption(params, edited_metas, edited_caption, lm_metadata) -> None:
    meta_caption = edited_metas.get("caption") if edited_metas else None
    if meta_caption and meta_caption.strip():
        params.caption = meta_caption
    elif edited_caption:
        params.caption = edited_caption
    elif params.use_cot_caption and lm_metadata.get("caption"):
        params.caption = lm_metadata["caption"]


def _apply_lyrics(params, edited_lyrics, lm_metadata) -> None:
    if edited_lyrics:
        params.lyrics = edited_lyrics
    elif not params.lyrics and lm_metadata.get("lyrics"):
        params.lyrics = lm_metadata["lyrics"]


def _apply_cot_language(params, edited_metas, lm_metadata) -> None:
    edited_lang = (
        (edited_metas.get("language") or edited_metas.get("vocal_language"))
        if edited_metas else None
    )
    if not edited_lang:
        lm_lang = lm_metadata.get("vocal_language") or lm_metadata.get("language")
        if lm_lang:
            params.vocal_language = lm_lang


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


def _populate_cot_fields(
    params: GenerationParams, lm_metadata: dict, originals: Dict[str, Any],
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
