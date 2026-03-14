"""Parsing utilities for CLI arguments and user input."""

import re


def parse_description_hints(description: str) -> tuple[str | None, bool]:
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


def parse_number(value: str) -> float | None:
    """Extract the first numeric substring from *value*."""
    try:
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if not match:
            return None
        return float(match.group(0))
    except Exception:
        return None
