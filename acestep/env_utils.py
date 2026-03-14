"""Helpers for reading environment variables."""

import os


def env_is_truthy(name: str, default: str = "") -> bool:
    """Return True if env var is set to '1', 'true', or 'yes' (case-insensitive)."""
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


_mlx_available_cache: bool | None = None


def mlx_available() -> bool:
    """Return True if the MLX framework is importable (Apple Silicon).

    The result is cached after the first call to avoid repeated import
    attempts (which can crash with nanobind duplicate-enum errors).
    """
    global _mlx_available_cache
    if _mlx_available_cache is not None:
        return _mlx_available_cache
    try:
        import mlx.core  # noqa: F401
        _mlx_available_cache = True
    except Exception:
        _mlx_available_cache = False
    return _mlx_available_cache
