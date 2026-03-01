"""DiT and LM handler initialization for the CLI pipeline."""

import os
from typing import Optional

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import (
    SUBMODEL_REGISTRY,
    check_main_model_exists,
    check_model_exists,
    ensure_dit_model,
    ensure_lm_model,
    ensure_main_model,
    get_checkpoints_dir,
)

from acestep.cli.defaults import BASE_ONLY_TASKS, SKIP_LM_TASKS


def resolve_config_path(args, parser, dit_handler: AceStepHandler) -> None:
    """Auto-select or validate `args.config_path`, downloading models if needed."""
    checkpoints_dir = get_checkpoints_dir()

    if args.config_path is None:
        _auto_select_config_path(args, parser, dit_handler, checkpoints_dir)

    if args.task_type in BASE_ONLY_TASKS and "base" not in str(args.config_path).lower():
        parser.error(
            f"task_type '{args.task_type}' requires a base model config "
            "(e.g., 'acestep-v15-base')."
        )

    _ensure_checkpoint_models(args, parser, checkpoints_dir)


def _auto_select_config_path(args, parser, dit_handler, checkpoints_dir) -> None:
    """Discover or download a suitable DiT config when none was specified."""
    available = dit_handler.get_available_acestep_v15_models()
    if args.task_type in BASE_ONLY_TASKS and available:
        available = [m for m in available if "base" in m.lower()]

    if not available:
        print("No DiT models found. Downloading main model (acestep-v15-turbo + core components)...")
        success, msg = ensure_main_model(checkpoints_dir)
        print(msg)
        if not success:
            parser.error(f"Failed to download main model: {msg}")
        available = dit_handler.get_available_acestep_v15_models()
        if args.task_type in BASE_ONLY_TASKS and available:
            available = [m for m in available if "base" in m.lower()]

    if args.task_type in BASE_ONLY_TASKS and not available:
        print("Base-only task selected. Downloading base DiT model (acestep-v15-base)...")
        success, msg = ensure_dit_model("acestep-v15-base", checkpoints_dir)
        print(msg)
        if not success:
            parser.error(f"Failed to download base DiT model: {msg}")
        available = dit_handler.get_available_acestep_v15_models()
        if available:
            available = [m for m in available if "base" in m.lower()]

    if available:
        if args.task_type in BASE_ONLY_TASKS:
            preferred = "acestep-v15-base"
        else:
            preferred = "acestep-v15-turbo"
        args.config_path = preferred if preferred in available else available[0]
        print(f"Auto-selected config_path: {args.config_path}")
    else:
        parser.error("No available DiT models found. Please specify --config_path.")


def _ensure_checkpoint_models(args, parser, checkpoints_dir) -> None:
    """Download main model and specific DiT model if missing."""
    if not check_main_model_exists(checkpoints_dir):
        print("Main model components not found. Downloading main model...")
        success, msg = ensure_main_model(checkpoints_dir)
        print(msg)
        if not success:
            parser.error(f"Failed to download main model: {msg}")

    if args.config_path:
        config_name = str(args.config_path)
        known_models = {"acestep-v15-turbo"} | set(SUBMODEL_REGISTRY.keys())
        if check_model_exists(config_name, checkpoints_dir):
            pass
        elif config_name in known_models:
            success, msg = ensure_dit_model(config_name, checkpoints_dir)
            if not success:
                parser.error(f"Failed to download DiT model '{config_name}': {msg}")
        else:
            print(
                f"Warning: DiT model '{config_name}' not found locally and "
                "not in registry. Skipping auto-download."
            )


def initialize_dit(
    args,
    dit_handler: AceStepHandler,
    device: str,
) -> None:
    """Load the DiT model into *dit_handler*."""
    use_flash_attention = args.use_flash_attention
    if use_flash_attention is None:
        use_flash_attention = dit_handler.is_flash_attention_available(device)

    compile_model = os.environ.get("ACESTEP_COMPILE_MODEL", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }

    print(f"Initializing DiT handler with model: {args.config_path}")
    dit_handler.initialize_service(
        project_root=args.project_root,
        config_path=args.config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=args.offload_dit_to_cpu,
    )


def requires_lm(args) -> bool:
    """Return True if the current args require the LM handler."""
    if args.task_type in SKIP_LM_TASKS:
        return False
    return (
        args.thinking
        or args.sample_mode
        or bool(args.sample_query and str(args.sample_query).strip())
        or args.use_format
        or args.use_cot_metas
        or args.use_cot_caption
        or args.use_cot_lyrics
        or args.use_cot_language
    )


def initialize_lm(
    args,
    parser,
    llm_handler: LLMHandler,
    device: str,
) -> None:
    """Resolve the LM model path, download if needed, and initialize the handler."""
    checkpoints_dir = get_checkpoints_dir()

    if args.lm_model_path is None:
        available = llm_handler.get_available_5hz_lm_models()
        if available:
            args.lm_model_path = available[0]
            print(f"Using default LM model: {args.lm_model_path}")
        else:
            success, msg = ensure_lm_model(checkpoints_dir=checkpoints_dir)
            print(msg)
            if not success:
                parser.error(
                    "No LM models available. Please specify --lm_model_path "
                    "or disable --thinking."
                )
            available = llm_handler.get_available_5hz_lm_models()
            if not available:
                parser.error(
                    "No LM models available after download. "
                    "Please specify --lm_model_path or disable --thinking."
                )
            args.lm_model_path = available[0]
            print(f"Using default LM model: {args.lm_model_path}")
    else:
        _validate_lm_model_path(args, parser, checkpoints_dir)

    print(f"Initializing LM handler with model: {args.lm_model_path}")
    llm_handler.initialize(
        checkpoint_dir=args.checkpoint_dir,
        lm_model_path=args.lm_model_path,
        backend=args.backend,
        device=device,
        offload_to_cpu=args.offload_to_cpu,
        dtype=None,
    )


def _validate_lm_model_path(args, parser, checkpoints_dir) -> None:
    """Ensure a user-specified LM model path is available or downloadable."""
    lm_model_path = str(args.lm_model_path)
    if os.path.isabs(lm_model_path) and os.path.exists(lm_model_path):
        return
    if check_model_exists(lm_model_path, checkpoints_dir):
        return
    if lm_model_path in SUBMODEL_REGISTRY:
        success, msg = ensure_lm_model(lm_model_path, checkpoints_dir=checkpoints_dir)
        print(msg)
        if not success:
            parser.error(f"Failed to download LM model '{lm_model_path}': {msg}")
        return
    parser.error(
        f"LM model '{lm_model_path}' not found locally and not in registry. "
        "Please provide a valid --lm_model_path."
    )
