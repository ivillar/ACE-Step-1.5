"""ACE-Step 1.5 CLI — Hydra config-driven music generation."""

import os
import sys

# --- Loguru setup (must run before acestep imports that use loguru) ---
try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               filter=lambda r: "<|audio_code_" not in r.get("message", ""))
except Exception:
    pass

# --- dotenv loading ---
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.abspath(__file__))
    for _name in (".env", ".env.example"):
        _path = os.path.join(_root, _name)
        if os.path.exists(_path):
            load_dotenv(_path)
            break
except ImportError:
    pass

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from acestep.models.dit import AceStepDiTWrapper  # noqa: E402
from acestep.inference.params import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.models.lm import AceStepLMWrapper  # noqa: E402
from acestep.download_utils import (  # noqa: E402
    SUBMODEL_REGISTRY, check_main_model_exists, check_model_exists,
    ensure_dit_model, ensure_lm_model, ensure_main_model, get_checkpoints_dir,
)
from acestep.inference.pipeline import (  # noqa: E402
    apply_lm_results, run_lm_generation,
    run_pre_generation_steps, snapshot_originals,
)
from acestep.inference.display import (  # noqa: E402
    merge_time_costs, print_dit_prompt, print_final_parameters,
)

SKIP_LM_TASKS = {"cover", "repaint"}
BASE_ONLY_TASKS = {"lego", "extract", "complete"}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Entry point for the ACE-Step CLI."""
    sys_cfg = OmegaConf.to_container(cfg.system, resolve=True)

    device = str(sys_cfg["device"])
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    params = GenerationParams(**OmegaConf.to_container(cfg.params, resolve=True))
    config = GenerationConfig(**OmegaConf.to_container(cfg.generation, resolve=True))

    sample_mode = bool(cfg.sample_mode)
    sample_query = str(cfg.sample_query) if cfg.sample_query else ""
    use_format = bool(cfg.use_format)

    dit_handler = AceStepDiTWrapper()
    llm_handler = AceStepLMWrapper()

    _resolve_config_path(sys_cfg, dit_handler, params.task_type)
    _initialize_dit(sys_cfg, dit_handler, device)

    manual_edit = (
        _requires_lm(params, sample_mode, sample_query, use_format)
        and params.task_type not in SKIP_LM_TASKS
        and not (params.audio_codes and str(params.audio_codes).strip())
    )

    if _requires_lm(params, sample_mode, sample_query, use_format):
        _initialize_lm(sys_cfg, llm_handler, device)
    elif params.task_type in SKIP_LM_TASKS:
        print(f"LM is not required for task_type '{params.task_type}'. Skipping.")
    else:
        print("LM 'thinking' is disabled. Skipping LM handler initialization.")

    print("Handlers initialized.")

    run_pre_generation_steps(
        params, llm_handler,
        sample_mode=sample_mode,
        sample_query=sample_query,
        use_format=use_format,
    )

    log_level_upper = str(sys_cfg["log_level"]).upper()
    print_final_parameters(
        sys_cfg, params, config,
        compact=(log_level_upper != "DEBUG"),
        resolved_device=device,
    )

    _run_generation(sys_cfg, params, config, dit_handler, llm_handler, log_level_upper, manual_edit)


# ---------------------------------------------------------------------------
# Config resolution + model downloads
# ---------------------------------------------------------------------------

def _ensure_download(fn, *args) -> None:
    """Call a download function; raise on failure."""
    success, msg = fn(*args)
    print(msg)
    if not success:
        raise RuntimeError(msg)


def _filter_base_models(available, task_type):
    """Filter to base models only if the task requires it."""
    if task_type in BASE_ONLY_TASKS and available:
        return [m for m in available if "base" in m.lower()]
    return available


def _resolve_config_path(sys_cfg, dit_handler, task_type) -> None:
    """Auto-select or validate config_path, downloading models if needed."""
    checkpoints_dir = sys_cfg["checkpoint_dir"]

    if sys_cfg["config_path"] is None:
        available = _filter_base_models(
            dit_handler.get_available_acestep_v15_models(checkpoints_dir), task_type,
        )

        if not available:
            print("No DiT models found. Downloading main model...")
            _ensure_download(ensure_main_model, checkpoints_dir)
            available = _filter_base_models(
                dit_handler.get_available_acestep_v15_models(), task_type,
            )

        if task_type in BASE_ONLY_TASKS and not available:
            print("Base-only task selected. Downloading base DiT model...")
            _ensure_download(ensure_dit_model, "acestep-v15-base", checkpoints_dir)
            available = _filter_base_models(
                dit_handler.get_available_acestep_v15_models(), task_type,
            )

        if not available:
            raise RuntimeError("No available DiT models found. Please specify system.config_path.")

        preferred = "acestep-v15-base" if task_type in BASE_ONLY_TASKS else "acestep-v15-turbo"
        sys_cfg["config_path"] = preferred if preferred in available else available[0]
        print(f"Auto-selected config_path: {sys_cfg['config_path']}")

    if task_type in BASE_ONLY_TASKS and "base" not in str(sys_cfg["config_path"]).lower():
        raise RuntimeError(
            f"task_type '{task_type}' requires a base model config "
            "(e.g., 'acestep-v15-base')."
        )

    if not check_main_model_exists(checkpoints_dir):
        print("Main model components not found. Downloading main model...")
        _ensure_download(ensure_main_model, checkpoints_dir)

    if sys_cfg["config_path"]:
        config_name = str(sys_cfg["config_path"])
        known_models = {"acestep-v15-turbo"} | set(SUBMODEL_REGISTRY.keys())
        if check_model_exists(config_name, checkpoints_dir):
            pass
        elif config_name in known_models:
            _ensure_download(ensure_dit_model, config_name, checkpoints_dir)
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
    lm_path = sys_cfg["lm_model_path"]

    if lm_path is None:
        available = llm_handler.get_available_5hz_lm_models(checkpoints_dir)
        if not available:
            _ensure_download(ensure_lm_model, checkpoints_dir)
            available = llm_handler.get_available_5hz_lm_models()
        if not available:
            raise RuntimeError(
                "No LM models available. Please specify system.lm_model_path "
                "or disable params.thinking."
            )
        sys_cfg["lm_model_path"] = available[0]
        print(f"Using default LM model: {sys_cfg['lm_model_path']}")
    else:
        lm_path = str(lm_path)
        if not (os.path.isabs(lm_path) and os.path.exists(lm_path)):
            if not check_model_exists(lm_path, checkpoints_dir):
                if lm_path in SUBMODEL_REGISTRY:
                    _ensure_download(ensure_lm_model, lm_path, checkpoints_dir)
                else:
                    raise RuntimeError(
                        f"LM model '{lm_path}' not found locally and not in registry. "
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
# Generation
# ---------------------------------------------------------------------------

def _run_generation(sys_cfg, params, config, dit_handler, llm_handler, log_level_upper, manual_edit) -> None:
    lm_time_costs = None
    if manual_edit:
        originals = snapshot_originals(params)
        lm_result = run_lm_generation(llm_handler, dit_handler, params, config, originals)
        lm_time_costs = lm_result.get("lm_time_costs")
        if not lm_result.get("success", False):
            return
        apply_lm_results(params, lm_result, originals)
        if log_level_upper in {"INFO", "DEBUG"}:
            print_dit_prompt(dit_handler, params)
        print("Running DiT generation with edited prompt and cached audio codes...")
    else:
        if log_level_upper in {"INFO", "DEBUG"}:
            print_dit_prompt(dit_handler, params)

    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=sys_cfg["save_dir"],
    )

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

    time_costs = merge_time_costs(lm_time_costs, result) if manual_edit else result.extra_outputs.get("time_costs", {})
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
