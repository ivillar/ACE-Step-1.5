"""Interactive terminal prompt utilities for the CLI wizard."""

import os
from pathlib import Path
from typing import Callable, List, Optional


def expand_audio_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve and normalize an audio file path to a POSIX string."""
    if not path_str or not isinstance(path_str, str):
        return path_str
    try:
        return Path(path_str).expanduser().resolve(strict=False).as_posix()
    except Exception:
        return Path(path_str).expanduser().absolute().as_posix()


def prompt_non_empty(prompt: str) -> str:
    """Prompt until a non-empty value is entered."""
    value = input(prompt).strip()
    while not value:
        value = input(prompt).strip()
    return value


def prompt_with_default(
    prompt: str,
    default: Optional[str] = None,
    required: bool = False,
) -> str:
    """Prompt with an optional default; loop if *required* and blank."""
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        if not required:
            return ""
        print("This value is required. Please try again.")


def prompt_bool(prompt: str, default: bool) -> bool:
    """Prompt for a yes/no answer with a sensible default."""
    default_str = "y" if default else "n"
    while True:
        value = input(f"{prompt} (y/n) [default: {default_str}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "1", "true"}:
            return True
        if value in {"n", "no", "0", "false"}:
            return False
        print("Please enter 'y' or 'n'.")


def prompt_choice_from_list(
    prompt: str,
    options: List[str],
    default: Optional[str] = None,
    allow_custom: bool = True,
    custom_validator: Optional[Callable[[str], bool]] = None,
    custom_error: Optional[str] = None,
) -> Optional[str]:
    """Display numbered options and return the user's selection."""
    if not options:
        return default
    print("\n" + prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    default_display = default if default not in (None, "") else "auto"
    while True:
        choice = input(
            f"Choose a model (number or name) [default: {default_display}]: "
        ).strip()
        if not choice:
            return None if default_display == "auto" else default
        if choice.lower() == "auto":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            print("Invalid selection. Please choose a valid number.")
            continue
        if allow_custom:
            if custom_validator and not custom_validator(choice):
                print(custom_error or "Invalid selection. Please try again.")
                continue
            if choice not in options:
                print("Unknown model. Using as-is.")
            return choice
        print("Please choose a valid option.")


def _prompt_numeric(prompt, default, min_value, max_value, cast_fn, type_name):
    """Prompt for a numeric value, casting with *cast_fn* and validating bounds."""
    default_display = "auto" if default is None else default
    while True:
        value = input(f"{prompt} [{default_display}]: ").strip()
        if not value:
            return default
        try:
            parsed = cast_fn(value)
        except ValueError:
            print(f"Invalid input. Please enter {type_name}.")
            continue
        if min_value is not None and parsed < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and parsed > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return parsed


def prompt_int(
    prompt: str,
    default: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    """Prompt for an integer within optional bounds."""
    return _prompt_numeric(prompt, default, min_value, max_value, int, "an integer")


def prompt_float(
    prompt: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[float]:
    """Prompt for a float within optional bounds."""
    return _prompt_numeric(prompt, default, min_value, max_value, float, "a number")


def prompt_existing_file(prompt: str, default: Optional[str] = None) -> str:
    """Prompt until an existing file path is entered."""
    while True:
        suffix = f" [{default}]" if default else ""
        path = input(f"{prompt}{suffix}: ").strip()
        if not path and default:
            path = default
        if os.path.isfile(path):
            return expand_audio_path(path)
        print("Invalid file path. Please try again.")
