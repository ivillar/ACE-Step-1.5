"""Parsing utilities for CLI arguments and user input."""

import ast
import re
from typing import List, Optional

import torch


def parse_description_hints(description: str) -> tuple[Optional[str], bool]:
    """Extract language and instrumental hints from a free-text description.

    Returns:
        (detected_language_code | None, is_instrumental)
    """
    if not description:
        return None, False

    description_lower = description.lower().strip()

    language_mapping = {
        "english": "en", "en": "en",
        "chinese": "zh", "中文": "zh", "zh": "zh", "mandarin": "zh",
        "japanese": "ja", "日本語": "ja", "ja": "ja",
        "korean": "ko", "한국어": "ko", "ko": "ko",
        "spanish": "es", "español": "es", "es": "es",
        "french": "fr", "français": "fr", "fr": "fr",
        "german": "de", "deutsch": "de", "de": "de",
        "italian": "it", "italiano": "it", "it": "it",
        "portuguese": "pt", "português": "pt", "pt": "pt",
        "russian": "ru", "русский": "ru", "ru": "ru",
        "bengali": "bn", "bn": "bn",
        "hindi": "hi", "hi": "hi",
        "arabic": "ar", "ar": "ar",
        "thai": "th", "th": "th",
        "vietnamese": "vi", "vi": "vi",
        "indonesian": "id", "id": "id",
        "turkish": "tr", "tr": "tr",
        "dutch": "nl", "nl": "nl",
        "polish": "pl", "pl": "pl",
    }

    detected_language = None
    for lang_name, lang_code in language_mapping.items():
        if len(lang_name) <= 2:
            pattern = (
                r"(?:^|\s|[.,;:!?])" + re.escape(lang_name) + r"(?:$|\s|[.,;:!?])"
            )
        else:
            pattern = r"\b" + re.escape(lang_name) + r"\b"
        if re.search(pattern, description_lower):
            detected_language = lang_code
            break

    is_instrumental = False
    if "instrumental" in description_lower:
        is_instrumental = True
    elif "pure music" in description_lower or "pure instrument" in description_lower:
        is_instrumental = True
    elif description_lower.endswith(" solo") or description_lower == "solo":
        is_instrumental = True

    return detected_language, is_instrumental


def parse_number(value: str) -> Optional[float]:
    """Extract the first numeric substring from *value*."""
    try:
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if not match:
            return None
        return float(match.group(0))
    except Exception:
        return None


def parse_timesteps_input(value) -> Optional[List[float]]:
    """Parse a timesteps value from CLI/TOML into a list of floats."""
    if value is None:
        return None
    if isinstance(value, list):
        if all(isinstance(t, (int, float)) for t in value):
            return [float(t) for t in value]
        return None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("[") or raw.startswith("("):
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return None
        if isinstance(parsed, list) and all(isinstance(t, (int, float)) for t in parsed):
            return [float(t) for t in parsed]
        return None
    try:
        return [float(t.strip()) for t in raw.split(",") if t.strip()]
    except Exception:
        return None


def parse_bool(value: str) -> bool:
    """Truthy string check (``true``, ``1``, ``yes``, ``y``)."""
    return str(value).lower() in {"true", "1", "yes", "y"}


def resolve_device(device: str) -> str:
    """Map ``'auto'`` to the best available accelerator."""
    if device == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
