"""Shared helpers for generation input processing.

Consolidates duplicated logic used by both the CLI LM pipeline
(``acestep/cli/lm_pipeline.py``, ``acestep/cli/lm_result_merge.py``)
and the inference API (``acestep/inference.py``).
"""

from typing import Any

from acestep.inference.parsing import parse_number


def build_user_metadata(
    bpm: Any,
    keyscale: Any,
    timesignature: Any,
    duration: Any,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a user-metadata dict from generation parameters.

    Only includes fields that carry a meaningful, positive value.
    Pass *extras* to merge additional key-value pairs (e.g. caption or
    language on CLI retry attempts).

    Returns:
        A metadata dict, or ``None`` when every field is empty/invalid.
    """
    meta: dict[str, Any] = {}

    if bpm is not None:
        try:
            bpm_value = float(bpm)
            if bpm_value > 0:
                meta["bpm"] = int(bpm_value)
        except (ValueError, TypeError):
            pass

    if keyscale and str(keyscale).strip():
        clean = str(keyscale).strip()
        if clean.lower() not in {"n/a", ""}:
            meta["keyscale"] = clean

    if timesignature and str(timesignature).strip():
        clean = str(timesignature).strip()
        if clean.lower() not in {"n/a", ""}:
            meta["timesignature"] = clean

    if duration is not None:
        try:
            duration_value = float(duration)
            if duration_value > 0:
                meta["duration"] = int(duration_value)
        except (ValueError, TypeError):
            pass

    if extras:
        meta.update(extras)

    return meta or None


def format_seed_string(seeds: int | list[int] | None) -> str:
    """Convert a seeds value (None, int, or list[int]) to a comma-separated string.

    Returns:
        A string suitable for ``dit_handler.prepare_seeds``, or ``""``
        when *seeds* is ``None`` or an empty list.
    """
    if seeds is None:
        return ""
    if isinstance(seeds, list) and len(seeds) > 0:
        return ",".join(str(s) for s in seeds)
    if isinstance(seeds, int):
        return str(seeds)
    return ""


def accumulate_lm_time_costs(
    totals: dict[str, float],
    chunk_result: dict[str, Any],
) -> None:
    """Add LM time-cost fields from *chunk_result* into running *totals*.

    Expects *chunk_result* to be the raw dict returned by
    ``llm_handler.generate_with_stop_condition``, with an optional
    ``extra_outputs.time_costs`` sub-dict.
    """
    extra = (chunk_result.get("extra_outputs") or {}).get("time_costs", {})
    if not extra:
        return
    totals["phase1_time"] += float(extra.get("phase1_time", 0.0) or 0.0)
    totals["phase2_time"] += float(extra.get("phase2_time", 0.0) or 0.0)
    totals["total_time"] += float(
        extra.get(
            "total_time",
            (extra.get("phase1_time", 0.0) or 0.0)
            + (extra.get("phase2_time", 0.0) or 0.0),
        ) or 0.0
    )


def safe_parse_metadata_value(
    key: str,
    metas: dict[str, Any],
    *,
    as_int: bool = False,
    as_float: bool = False,
) -> float | None:
    """Parse a positive numeric value from ``metas[key]``.

    Returns ``None`` when the key is missing, falsy, or not a positive
    number.  When *as_int* is True the result is cast to ``int``; when
    *as_float* is True it is cast to ``float``.
    """
    raw = metas.get(key)
    if not raw:
        return None
    parsed = parse_number(str(raw))
    if parsed is None or parsed <= 0:
        return None
    if as_int:
        return int(parsed)
    return float(parsed) if as_float else parsed
