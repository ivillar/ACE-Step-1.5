"""Entry point for `uv run vizstep` / `python -m vizstep`."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import uvicorn
from loguru import logger


def _find_project_root() -> Path | None:
    """Walk up from this file or cwd to find the repo root (has conf/ dir)."""
    for candidate in [Path(__file__).parent.parent, Path.cwd()]:
        if (candidate / "conf").is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="VizStep — ACE-Step Visualization Engine")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Model checkpoint directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu/mps)")
    parser.add_argument("--no-lm", action="store_true", help="Skip loading the language model")
    parser.add_argument("--debug", type=str, default=None, help="Debug viz component (e.g. attn)")
    args = parser.parse_args()

    dit_wrapper = None
    lm_wrapper = None

    if args.debug:
        logger.info(f"[vizstep] Debug mode: {args.debug} — skipping model loading")
    else:
        # Load models
        logger.info("[vizstep] Loading models...")

        from acestep.models.dit.wrapper import AceStepDiTWrapper

        dit_wrapper = AceStepDiTWrapper(device=args.device)

        project_root = _find_project_root()
        if project_root is None:
            logger.warning("[vizstep] Could not find project root (no conf/ directory) — running in demo mode")
        else:
            # config_path is a model name (e.g. "acestep-v15-turbo"), not a file path.
            # It gets joined with checkpoint_dir to form the model path.
            load_kwargs: dict = {
                "project_root": str(project_root),
                "config_path": "acestep-v15-turbo",
                "device": args.device,
            }
            if args.checkpoint_dir:
                load_kwargs["checkpoint_dir"] = args.checkpoint_dir

            dit_wrapper.load_models(**load_kwargs)
            logger.info("[vizstep] DiT models loaded")

        if not args.no_lm:
            try:
                from acestep.models.lm.wrapper import AceStepLMWrapper

                lm_kwargs: dict = {"device": args.device}
                if args.checkpoint_dir:
                    lm_kwargs["checkpoint_dir"] = args.checkpoint_dir
                lm_wrapper = AceStepLMWrapper(**lm_kwargs)
                logger.info("[vizstep] LM loaded")
            except Exception as exc:
                logger.warning(f"[vizstep] Could not load LM: {exc}")

    # Configure server
    from vizstep.backend.server import configure

    configure(dit_wrapper, lm_wrapper, debug=args.debug)

    url = f"http://{args.host}:{args.port}"
    logger.info(f"[vizstep] Starting server at {url}")

    if not args.no_browser:
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "vizstep.backend.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
