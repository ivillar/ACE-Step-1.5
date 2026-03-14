"""ACE-Step 1.5 CLI — Hydra config-driven music generation."""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


# --- Environment bootstrap ---
def _bootstrap():
    """Load dotenv, clear proxies, configure logging."""
    try:
        from dotenv import load_dotenv
        root = os.path.dirname(os.path.abspath(__file__))
        for name in (".env", ".env.example"):
            path = os.path.join(root, name)
            if os.path.exists(path):
                load_dotenv(path)
                break
    except ImportError:
        pass
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                   filter=lambda r: "<|audio_code_" not in r.get("message", ""))
    except Exception:
        pass

_bootstrap()

from acestep.dit_wrapper import AceStepDiTWrapper  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.lm_wrapper import AceStepLMWrapper  # noqa: E402
from acestep.model_downloader import (  # noqa: E402
    SUBMODEL_REGISTRY, check_main_model_exists, check_model_exists,
    ensure_dit_model, ensure_lm_model, ensure_main_model, get_checkpoints_dir,
)
from acestep.cli.pipeline import (  # noqa: E402
    apply_lm_results, run_lm_generation,
    run_pre_generation_steps, snapshot_originals,
)

SKIP_LM_TASKS = {"cover", "repaint"}
BASE_ONLY_TASKS = {"lego", "extract", "complete"}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Entry point for the ACE-Step CLI."""
    # Unpack system config into a plain mutable dict (Hydra DictConfig is frozen)
    sys_cfg = OmegaConf.to_container(cfg.system, resolve=True)

    # Resolve device
    device = str(sys_cfg["device"])
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build params and config from Hydra config groups
    params_dict = OmegaConf.to_container(cfg.params, resolve=True)
    config_dict = OmegaConf.to_container(cfg.generation, resolve=True)
    params = GenerationParams(**params_dict)
    config = GenerationConfig(**config_dict)

    # Pipeline flags
    sample_mode = bool(cfg.sample_mode)
    sample_query = str(cfg.sample_query) if cfg.sample_query else ""
    use_format = bool(cfg.use_format)

    # Create handlers
    dit_handler = AceStepDiTWrapper()
    llm_handler = AceStepLMWrapper()

    # Resolve config path + download models
    _resolve_config_path(sys_cfg, dit_handler, params.task_type)

    # Initialize DiT
    _initialize_dit(sys_cfg, dit_handler, device)

    # Initialize LM if needed
    if _requires_lm(params, sample_mode, sample_query, use_format):
        _initialize_lm(sys_cfg, llm_handler, device)
    elif params.task_type in SKIP_LM_TASKS:
        print(f"LM is not required for task_type '{params.task_type}'. Skipping.")
    else:
        print("LM 'thinking' is disabled. Skipping LM handler initialization.")

    print("Handlers initialized.")

    # Pre-generation LM steps
    run_pre_generation_steps(
        params, llm_handler,
        sample_mode=sample_mode,
        sample_query=sample_query,
        use_format=use_format,
    )

    # Print summary
    log_level_upper = str(sys_cfg["log_level"]).upper()
    _print_final_parameters(
        sys_cfg, params, config,
        compact=(log_level_upper != "DEBUG"),
        resolved_device=device,
    )

    # Run generation
    _run_generation(sys_cfg, params, config, dit_handler, llm_handler, log_level_upper)


# ---------------------------------------------------------------------------
# Config resolution + model downloads
# ---------------------------------------------------------------------------

def _resolve_config_path(sys_cfg, dit_handler, task_type) -> None:
    """Auto-select or validate config_path, downloading models if needed."""
    checkpoints_dir = sys_cfg["checkpoint_dir"]

    if sys_cfg["config_path"] is None:
        available = dit_handler.get_available_acestep_v15_models(checkpoints_dir)
        if task_type in BASE_ONLY_TASKS and available:
            available = [m for m in available if "base" in m.lower()]

        if not available:
            print("No DiT models found. Downloading main model (acestep-v15-turbo + core components)...")
            success, msg = ensure_main_model(checkpoints_dir)
            print(msg)
            if not success:
                raise RuntimeError(f"Failed to download main model: {msg}")
            available = dit_handler.get_available_acestep_v15_models()
            if task_type in BASE_ONLY_TASKS and available:
                available = [m for m in available if "base" in m.lower()]

        if task_type in BASE_ONLY_TASKS and not available:
            print("Base-only task selected. Downloading base DiT model (acestep-v15-base)...")
            success, msg = ensure_dit_model("acestep-v15-base", checkpoints_dir)
            print(msg)
            if not success:
                raise RuntimeError(f"Failed to download base DiT model: {msg}")
            available = dit_handler.get_available_acestep_v15_models()
            if available:
                available = [m for m in available if "base" in m.lower()]

        if available:
            preferred = "acestep-v15-base" if task_type in BASE_ONLY_TASKS else "acestep-v15-turbo"
            sys_cfg["config_path"] = preferred if preferred in available else available[0]
            print(f"Auto-selected config_path: {sys_cfg['config_path']}")
        else:
            raise RuntimeError("No available DiT models found. Please specify system.config_path.")

    if task_type in BASE_ONLY_TASKS and "base" not in str(sys_cfg["config_path"]).lower():
        raise RuntimeError(
            f"task_type '{task_type}' requires a base model config "
            "(e.g., 'acestep-v15-base')."
        )

    # Ensure checkpoint models exist
    if not check_main_model_exists(checkpoints_dir):
        print("Main model components not found. Downloading main model...")
        success, msg = ensure_main_model(checkpoints_dir)
        print(msg)
        if not success:
            raise RuntimeError(f"Failed to download main model: {msg}")

    if sys_cfg["config_path"]:
        config_name = str(sys_cfg["config_path"])
        known_models = {"acestep-v15-turbo"} | set(SUBMODEL_REGISTRY.keys())
        if check_model_exists(config_name, checkpoints_dir):
            pass
        elif config_name in known_models:
            success, msg = ensure_dit_model(config_name, checkpoints_dir)
            if not success:
                raise RuntimeError(f"Failed to download DiT model '{config_name}': {msg}")
        else:
            print(
                f"Warning: DiT model '{config_name}' not found locally and "
                "not in registry. Skipping auto-download."
            )


# ---------------------------------------------------------------------------
# Handler initialization
# ---------------------------------------------------------------------------

def _initialize_dit(sys_cfg, dit_handler, device) -> None:
    use_flash_attention = sys_cfg["use_flash_attention"]
    if use_flash_attention is None:
        use_flash_attention = dit_handler.is_flash_attention_available(device)

    compile_model = os.environ.get("ACESTEP_COMPILE_MODEL", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }

    print(f"Initializing DiT handler with model: {sys_cfg['config_path']}")
    dit_handler.initialize_service(
        project_root=sys_cfg["project_root"],
        config_path=sys_cfg["config_path"],
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=compile_model,
        offload_to_cpu=sys_cfg["offload_to_cpu"],
        offload_dit_to_cpu=sys_cfg["offload_dit_to_cpu"],
        checkpoint_dir=sys_cfg["checkpoint_dir"],
    )


def _requires_lm(params, sample_mode, sample_query, use_format) -> bool:
    if params.task_type in SKIP_LM_TASKS:
        return False
    return (
        params.thinking
        or sample_mode
        or bool(sample_query and str(sample_query).strip())
        or use_format
        or params.use_cot_metas
        or params.use_cot_caption
        or params.use_cot_lyrics
        or params.use_cot_language
    )


def _initialize_lm(sys_cfg, llm_handler, device) -> None:
    checkpoints_dir = get_checkpoints_dir(sys_cfg["checkpoint_dir"])

    if sys_cfg["lm_model_path"] is None:
        available = llm_handler.get_available_5hz_lm_models(checkpoints_dir)
        if available:
            sys_cfg["lm_model_path"] = available[0]
            print(f"Using default LM model: {sys_cfg['lm_model_path']}")
        else:
            success, msg = ensure_lm_model(checkpoints_dir=checkpoints_dir)
            print(msg)
            if not success:
                raise RuntimeError(
                    "No LM models available. Please specify system.lm_model_path "
                    "or disable params.thinking."
                )
            available = llm_handler.get_available_5hz_lm_models()
            if not available:
                raise RuntimeError(
                    "No LM models available after download. "
                    "Please specify system.lm_model_path or disable params.thinking."
                )
            sys_cfg["lm_model_path"] = available[0]
            print(f"Using default LM model: {sys_cfg['lm_model_path']}")
    else:
        lm_model_path = str(sys_cfg["lm_model_path"])
        if not (os.path.isabs(lm_model_path) and os.path.exists(lm_model_path)):
            if not check_model_exists(lm_model_path, checkpoints_dir):
                if lm_model_path in SUBMODEL_REGISTRY:
                    success, msg = ensure_lm_model(lm_model_path, checkpoints_dir=checkpoints_dir)
                    print(msg)
                    if not success:
                        raise RuntimeError(f"Failed to download LM model '{lm_model_path}': {msg}")
                else:
                    raise RuntimeError(
                        f"LM model '{lm_model_path}' not found locally and not in registry. "
                        "Please provide a valid system.lm_model_path."
                    )

    print(f"Initializing LM handler with model: {sys_cfg['lm_model_path']}")
    llm_handler.initialize(
        checkpoint_dir=sys_cfg["checkpoint_dir"],
        lm_model_path=sys_cfg["lm_model_path"],
        backend=sys_cfg["backend"],
        device=device,
        offload_to_cpu=sys_cfg["offload_to_cpu"],
        dtype=None,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _summarize_lyrics(lyrics) -> str:
    if not lyrics:
        return "none"
    if isinstance(lyrics, str):
        stripped = lyrics.strip()
        if not stripped:
            return "none"
        if os.path.isfile(stripped):
            return f"file: {os.path.basename(stripped)}"
        if len(stripped) <= 60:
            return stripped.replace("\n", " ")
        return f"text ({len(stripped)} chars)"
    return "provided"


def _print_final_parameters(
    sys_cfg, params, config, compact, resolved_device=None,
) -> None:
    if not compact:
        print("\n--- Final Parameters (GenerationParams) ---")
        for k in sorted(vars(params).keys()):
            print(f"{k}: {getattr(params, k)}")
        print("-------------------------------------------")
        print("\n--- Final Parameters (GenerationConfig) ---")
        for k in sorted(vars(config).keys()):
            print(f"{k}: {getattr(config, k)}")
        print("-------------------------------------------\n")
        return

    device_display = str(sys_cfg["device"])
    if resolved_device and resolved_device != str(sys_cfg["device"]):
        device_display = f"{sys_cfg['device']} -> {resolved_device}"

    print("\n--- Final Parameters (Summary) ---")
    print(f"task_type: {params.task_type}")
    print(f"caption: {params.caption or 'none'}")
    print(f"lyrics: {_summarize_lyrics(params.lyrics)}")
    print(f"duration: {params.duration}s")
    print(f"outputs: {config.batch_size}")
    if params.bpm is not None:
        print(f"bpm: {params.bpm}")
    if params.keyscale:
        print(f"keyscale: {params.keyscale}")
    if params.timesignature:
        print(f"timesignature: {params.timesignature}")
    print(f"instrumental: {params.instrumental}")
    print(f"thinking: {params.thinking}")
    print(f"lm_model: {sys_cfg['lm_model_path'] or 'auto'}")
    print(f"dit_model: {sys_cfg['config_path'] or 'auto'}")
    print(f"backend: {sys_cfg['backend']}")
    print(f"device: {device_display}")
    print(f"audio_format: {config.audio_format}")
    print(f"save_dir: {sys_cfg['save_dir']}")
    if config.seeds:
        print(f"seeds: {config.seeds}")
    else:
        print(f"seed: {params.seed} (random={config.use_random_seed})")
    print("-------------------------------\n")


def _build_meta_dict(params):
    meta = {}
    if params.bpm is not None:
        meta["bpm"] = params.bpm
    if params.timesignature:
        meta["timesignature"] = params.timesignature
    if params.keyscale:
        meta["keyscale"] = params.keyscale
    if params.duration is not None:
        meta["duration"] = params.duration
    return meta or None


def _print_dit_prompt(dit_handler, params) -> None:
    meta = _build_meta_dict(params)
    caption_input, lyrics_input = dit_handler.build_dit_inputs(
        task=params.task_type,
        instruction=params.instruction,
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        metas=meta,
        vocal_language=params.vocal_language or "unknown",
    )
    print("\n--- Final DiT Prompt (Caption Branch) ---")
    print(caption_input)
    print("\n--- Final DiT Prompt (Lyrics Branch) ---")
    print(lyrics_input)
    print("----------------------------------------\n")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _run_generation(sys_cfg, params, config, dit_handler, llm_handler, log_level_upper) -> None:
    manual_edit = (
        params.thinking
        and params.task_type not in SKIP_LM_TASKS
        and not (params.audio_codes and str(params.audio_codes).strip())
    )

    print("\n--- Starting Generation ---")
    print(f'Caption: "{params.caption}"')
    print(f"Duration: {params.duration}s | Outputs: {config.batch_size}")
    if config.seeds:
        print(f"Custom Seeds: {config.seeds}")
    print("---------------------------\n")

    lm_time_costs = None
    if manual_edit:
        originals = snapshot_originals(params)
        lm_result = run_lm_generation(llm_handler, dit_handler, params, config, originals)
        lm_time_costs = lm_result.get("lm_time_costs")
        if not lm_result.get("success", False):
            return
        apply_lm_results(params, lm_result, originals)
        if log_level_upper in {"INFO", "DEBUG"}:
            _print_dit_prompt(dit_handler, params)
        print("Running DiT generation with edited prompt and cached audio codes...")
    else:
        if log_level_upper in {"INFO", "DEBUG"}:
            _print_dit_prompt(dit_handler, params)

    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=sys_cfg["save_dir"],
    )

    # Print results
    if not result.success:
        print(f"\nGeneration failed: {result.error}")
        print(f"   Status: {result.status_message}")
        return

    print(
        f"\nGeneration successful! {len(result.audios)} audio(s) "
        f"saved in '{sys_cfg['save_dir']}/'"
    )
    for i, audio in enumerate(result.audios):
        print(f"  [{i + 1}] Path: {audio['path']} | Seed: {audio['params']['seed']}")

    time_costs = result.extra_outputs.get("time_costs", {})
    if manual_edit and lm_time_costs and time_costs is not None:
        if not isinstance(time_costs, dict):
            time_costs = {}
            result.extra_outputs["time_costs"] = time_costs
        if lm_time_costs["total_time"] > 0.0:
            time_costs["lm_phase1_time"] = lm_time_costs["phase1_time"]
            time_costs["lm_phase2_time"] = lm_time_costs["phase2_time"]
            time_costs["lm_total_time"] = lm_time_costs["total_time"]
            dit_total = float(time_costs.get("dit_total_time_cost", 0.0) or 0.0)
            time_costs["pipeline_total_time"] = lm_time_costs["total_time"] + dit_total
    if time_costs:
        print("\n--- Performance ---")
        total = time_costs.get("pipeline_total_time", 0)
        print(f"Total time: {total:.2f}s")
        if params.thinking:
            lm1 = time_costs.get("lm_phase1_time", 0)
            lm2 = time_costs.get("lm_phase2_time", 0)
            print(f"  - LM time: {lm1 + lm2:.2f}s")
        print(f"  - DiT time: {time_costs.get('dit_total_time_cost', 0):.2f}s")
        print("-------------------\n")


if __name__ == "__main__":
    run()
