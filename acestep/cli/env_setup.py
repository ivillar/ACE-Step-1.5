"""Environment bootstrap: dotenv loading, proxy clearing, and logging configuration."""

import os
import sys
from typing import Optional


def load_dotenv_config() -> None:
    """Load .env or .env.example from the project root, if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    env_example_path = os.path.join(project_root, ".env.example")

    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded configuration from {env_path}")
    elif os.path.exists(env_example_path):
        load_dotenv(env_example_path)
        print(f"Loaded configuration from {env_example_path} (fallback)")


def clear_proxy_env() -> None:
    """Remove proxy environment variables that may interfere with network behaviour."""
    for var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        os.environ.pop(var, None)


def configure_logging(
    level: Optional[str] = None,
    suppress_audio_tokens: Optional[bool] = None,
) -> None:
    """Configure loguru with the given level and optional audio-token suppression."""
    try:
        from loguru import logger
    except Exception:
        return

    if suppress_audio_tokens is None:
        suppress_audio_tokens = os.environ.get(
            "ACE_STEP_SUPPRESS_AUDIO_TOKENS", "1"
        ) not in {"0", "false", "False"}
    if level is None:
        level = "INFO"
    level = str(level).upper()

    def _log_filter(record) -> bool:
        message = record.get("message", "")
        if (
            "DiT TEXT ENCODER INPUT" in message
            or "text_prompt:" in message
            or (message.strip() and set(message.strip()) == {"="})
        ):
            return False
        if not suppress_audio_tokens:
            return True
        return "<|audio_code_" not in message

    logger.remove()
    logger.add(sys.stderr, level=level, filter=_log_filter)
