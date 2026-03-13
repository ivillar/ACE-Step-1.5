"""Helpers for reading environment variables."""

import os


def env_is_truthy(name: str, default: str = "") -> bool:
    """Return True if env var is set to '1', 'true', or 'yes' (case-insensitive)."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes")
