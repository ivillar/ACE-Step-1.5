"""ACE-Step 1.5 Inference Pipeline — Hydra config-driven music generation."""

import os
import sys

from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf 
from acestep.models.dit import AceStepDiTWrapper
from acestep.inference.params import GenerationConfig, GenerationParams, generate_music
from acestep.models.lm import AceStepLMWrapper
import acestep.download_utils
import acestep.inference.pipeline
import acestep.inference.display

logger.remove()
logger.add(sys.stderr, level="INFO",
            filter=lambda r: "<|audio_code_" not in r.get("message", ""))

SKIP_LM_TASKS = {"cover", "repaint"}
BASE_ONLY_TASKS = {"lego", "extract", "complete"}

def resolve_config_path(sys_cfg, task_type) -> None:
    """Auto-select or validate config_path, downloading models if needed."""
    checkpoints_dir = sys_cfg["checkpoint_dir"]

    assert "config_path" in sys_cfg, "You must provide a config_path in your configuration."

    if task_type in BASE_ONLY_TASKS and "base" not in str(sys_cfg["config_path"]).lower():
        raise RuntimeError(
            f"task_type '{task_type}' requires a base model config "
            "(e.g., 'acestep-v15-base')."
        )

    if not acestep.download_utils.check_main_model_exists(checkpoints_dir):
        logger.info("Main model components not found, downloading...")
        acestep.download_utils.ensure_download(acestep.download_utils.ensure_main_model, checkpoints_dir)

    config_name = str(sys_cfg["config_path"])
    known_models = {"acestep-v15-turbo"} | set(acestep.download_utils.SUBMODEL_REGISTRY.keys())
    if acestep.download_utils.check_model_exists(config_name, checkpoints_dir):
        pass
    elif config_name in known_models:
        acestep.download_utils.ensure_download(acestep.download_utils.ensure_dit_model, config_name, checkpoints_dir)
    else:
        logger.warning(
            "DiT model '{}' not found locally and not in registry, skipping auto-download",
            config_name,
        )

def requires_lm(params, sample_mode, sample_query, use_format) -> bool:
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

def initialize_lm(sys_cfg, llm_wrapper, device) -> None:
    checkpoints_dir = acestep.download_utils.get_checkpoints_dir(sys_cfg["checkpoint_dir"])
    lm_path = sys_cfg["lm_model_path"]

    if lm_path is None:
        available = llm_wrapper.get_available_5hz_lm_models(checkpoints_dir)
        if not available:
            acestep.download_utils.ensure_download(acestep.download_utils.ensure_lm_model, checkpoints_dir)
            available = llm_wrapper.get_available_5hz_lm_models()
        if not available:
            raise RuntimeError(
                "No LM models available. Please specify system.lm_model_path "
                "or disable params.thinking."
            )
        sys_cfg["lm_model_path"] = available[0]
        logger.info("Using default LM model: {}", sys_cfg["lm_model_path"])
    else:
        lm_path = str(lm_path)
        if not (os.path.isabs(lm_path) and os.path.exists(lm_path)):
            if not acestep.download_utils.check_model_exists(lm_path, checkpoints_dir):
                if lm_path in acestep.download_utils.SUBMODEL_REGISTRY:
                    acestep.download_utils.ensure_download(acestep.download_utils.ensure_lm_model, lm_path, checkpoints_dir)
                else:
                    raise RuntimeError(
                        f"LM model '{lm_path}' not found locally and not in registry. "
                        "Please provide a valid system.lm_model_path."
                    )

    logger.info("Initializing LM wrapper with model: {}", sys_cfg["lm_model_path"])
    llm_wrapper.initialize(
        checkpoint_dir=sys_cfg["checkpoint_dir"],
        lm_model_path=sys_cfg["lm_model_path"],
        backend=sys_cfg["backend"],
        device=device,
        offload_to_cpu=sys_cfg["offload_to_cpu"],
        dtype=None,
    )

def run_generation(sys_cfg, params, config, dit_wrapper, llm_wrapper, log_level_upper, manual_edit) -> None:
    lm_time_costs = None
    if manual_edit:
        originals = acestep.inference.pipeline.snapshot_originals(params)
        lm_result = acestep.inference.pipeline.run_lm_generation(llm_wrapper, dit_wrapper, params, config, originals)
        lm_time_costs = lm_result.get("lm_time_costs")
        if not lm_result.get("success", False):
            return
        acestep.inference.pipeline.apply_lm_results(params, lm_result, originals)
        if log_level_upper in {"INFO", "DEBUG"}:
            acestep.inference.display.log_dit_prompt(dit_wrapper, params)
        logger.info("Running DiT generation with edited prompt and cached audio codes...")
    else:
        if log_level_upper in {"INFO", "DEBUG"}:
            acestep.inference.display.log_dit_prompt(dit_wrapper, params)

    result = generate_music(
        dit_wrapper, llm_wrapper, params, config, save_dir=sys_cfg["save_dir"],
    )

    if not result.success:
        logger.error("Generation failed: {} ({})", result.error, result.status_message)
        return

    logger.info(
        "Generation successful! {} audio(s) saved in '{}'",
        len(result.audios), sys_cfg["save_dir"],
    )
    for i, audio in enumerate(result.audios):
        logger.info("  [{}] path={} seed={}", i + 1, audio["path"], audio["params"]["seed"])

    acestep.inference.display.log_performance(lm_time_costs if manual_edit else None, result, params.thinking)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    """Entry point for the ACE-Step CLI."""
    sys_cfg = OmegaConf.to_container(cfg.system, resolve=True)
    assert isinstance(sys_cfg, dict)

    device = str(sys_cfg["device"])
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    params = GenerationParams(**OmegaConf.to_container(cfg.params, resolve=True))
    config = GenerationConfig(**OmegaConf.to_container(cfg.generation, resolve=True))

    sample_mode = bool(cfg.sample_mode)
    sample_query = str(cfg.sample_query) if cfg.sample_query else ""
    use_format = bool(cfg.use_format)

    llm_wrapper = AceStepLMWrapper()

    resolve_config_path(sys_cfg, params.task_type)
    logger.info("Initializing DiT wrapper with model: {}", sys_cfg["config_path"])
    dit_wrapper = AceStepDiTWrapper(
        project_root=sys_cfg["project_root"],
        config_path=sys_cfg["config_path"],
        device=device,
        use_flash_attention=sys_cfg["use_flash_attention"],
        offload_to_cpu=sys_cfg["offload_to_cpu"],
        offload_dit_to_cpu=sys_cfg["offload_dit_to_cpu"],
        checkpoint_dir=sys_cfg["checkpoint_dir"],
    )

    if requires_lm(params, sample_mode, sample_query, use_format):
        initialize_lm(sys_cfg, llm_wrapper, device)
    elif params.task_type in SKIP_LM_TASKS:
        logger.info("LM not required for task_type '{}', skipping", params.task_type)
    else:
        logger.info("LM thinking disabled, skipping LM wrapper initialization")

    logger.info("wrappers initialized")

    acestep.inference.pipeline.run_pre_generation_steps(
        params, llm_wrapper,
        sample_mode=sample_mode,
        sample_query=sample_query,
        use_format=use_format,
    )

    # Compute after pre-generation steps, which may clear params.thinking
    manual_edit = (
        params.thinking
        and params.task_type not in SKIP_LM_TASKS
        and not (params.audio_codes and str(params.audio_codes).strip())
    )

    log_level_upper = str(sys_cfg["log_level"]).upper()
    acestep.inference.display.log_parameters(
        sys_cfg, params, config,
        compact=(log_level_upper != "DEBUG"),
        resolved_device=device,
    )

    run_generation(sys_cfg, params, config, dit_wrapper, llm_wrapper, log_level_upper, manual_edit)


if __name__ == "__main__":
    run()
