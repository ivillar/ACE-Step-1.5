"""ACE-Step 1.5 CLI — interactive wizard and config-driven music generation."""

import argparse
import os
import sys

from acestep.cli.env_setup import clear_proxy_env, configure_logging, load_dotenv_config

load_dotenv_config()
clear_proxy_env()
configure_logging()

from acestep.gpu_config import get_gpu_config, is_mps_platform, set_global_gpu_config  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402

from acestep.cli.defaults import SKIP_LM_TASKS, build_all_defaults  # noqa: E402
from acestep.cli.display import print_dit_prompt, print_final_parameters  # noqa: E402
from acestep.cli.handler_init import (  # noqa: E402
    initialize_dit, initialize_lm, requires_lm, resolve_config_path,
)
from acestep.cli.lm_pipeline import run_lm_generation, snapshot_originals  # noqa: E402
from acestep.cli.lm_result_merge import apply_lm_results_to_params  # noqa: E402
from acestep.cli.parsing import resolve_device  # noqa: E402
from acestep.cli.pre_generation import run_pre_generation_lm_steps  # noqa: E402
from acestep.cli.prompt_editing import install_prompt_edit_hook  # noqa: E402
from acestep.cli.validation import postprocess_args  # noqa: E402
from acestep.cli.wizard import run_wizard  # noqa: E402


def _get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    """Entry point for the ACE-Step CLI."""
    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    mps_available = is_mps_platform()
    auto_offload = (
        (not mps_available) and gpu_config.gpu_memory_gb > 0 and gpu_config.gpu_memory_gb < 16
    )

    _print_gpu_banner(gpu_config, mps_available, auto_offload)

    params_defaults = GenerationParams()
    config_defaults = GenerationConfig()
    parser, cli_args = _parse_cli_args()
    configure_logging(level=cli_args.log_level)

    args = _build_initial_args(
        cli_args, parser, params_defaults, config_defaults,
        gpu_config, mps_available, auto_offload,
    )

    if cli_args.configure:
        args, _ = run_wizard(
            args, configure_only=True, default_config_path=cli_args.config,
            params_defaults=params_defaults, config_defaults=config_defaults,
        )
        print("Configuration complete. Exiting without generation.")
        sys.exit(0)

    if not cli_args.config:
        args, should_generate = run_wizard(
            args, configure_only=False,
            params_defaults=params_defaults, config_defaults=config_defaults,
        )
        if not should_generate:
            print("Configuration complete. Exiting without generation.")
            sys.exit(0)

    timesteps = postprocess_args(args, parser)
    device = resolve_device(args.device)

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    resolve_config_path(args, parser, dit_handler)
    initialize_dit(args, dit_handler, device)

    if requires_lm(args):
        initialize_lm(args, parser, llm_handler, device)
    elif args.task_type in SKIP_LM_TASKS:
        print(f"LM is not required for task_type '{args.task_type}'. Skipping.")
    else:
        print("LM 'thinking' is disabled. Skipping LM handler initialization.")

    print("Handlers initialized.")

    run_pre_generation_lm_steps(args, parser, llm_handler)
    _setup_prompt_edit_hook(args, llm_handler)

    params, config = _build_generation_objects(args, timesteps)

    log_level_upper = str(getattr(args, "log_level", "INFO")).upper()
    print_final_parameters(
        args, params, config, params_defaults, config_defaults,
        compact=(log_level_upper != "DEBUG"), resolved_device=device,
    )

    _run_generation(args, params, config, dit_handler, llm_handler, log_level_upper)


# ---------------------------------------------------------------------------
# Thin orchestration helpers
# ---------------------------------------------------------------------------

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="ACE-Step 1.5: Music generation (wizard/config only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, help="Path to a TOML config file.")
    parser.add_argument(
        "--configure", action="store_true",
        help="Run wizard to save configuration without generating.",
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["vllm", "pt", "mlx"],
        help="5Hz LM backend. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL).",
    )
    return parser, parser.parse_args()


def _print_gpu_banner(gpu_config, mps_available: bool, auto_offload: bool) -> None:
    print(f"\n{'=' * 60}")
    print("GPU Configuration Detected:")
    print(f"{'=' * 60}")
    print(f"  GPU Memory: {gpu_config.gpu_memory_gb:.2f} GiB")
    print(f"  Configuration Tier: {gpu_config.tier}")
    max_lm = gpu_config.max_duration_with_lm
    max_no_lm = gpu_config.max_duration_without_lm
    print(f"  Max Duration (with LM): {max_lm}s ({max_lm // 60} min)")
    print(f"  Max Duration (without LM): {max_no_lm}s ({max_no_lm // 60} min)")
    print(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    print(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    print(f"  Default LM Init: {gpu_config.init_lm_default}")
    print(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    print(f"{'=' * 60}\n")

    if auto_offload:
        print("Auto-enabling CPU offload (GPU < 16GB)")
    elif gpu_config.gpu_memory_gb > 0:
        print("CPU offload disabled by default (GPU >= 16GB)")
    elif mps_available:
        print("MPS detected, running on Apple GPU")
    else:
        print("No GPU detected, running on CPU")


def _build_initial_args(
    cli_args, parser, params_defaults, config_defaults,
    gpu_config, mps_available, auto_offload,
):
    """Construct the initial ``argparse.Namespace`` from defaults + TOML config."""
    import toml

    default_batch_size = 1 if not cli_args.config else config_defaults.batch_size

    if mps_available:
        try:
            import mlx.core  # noqa: F401
            default_backend = "mlx"
            print("Apple Silicon detected with MLX available. Using MLX backend.")
        except ImportError:
            default_backend = "vllm"
    else:
        default_backend = "vllm"

    defaults = _default_namespace_dict(
        params_defaults, config_defaults, gpu_config,
        default_backend, default_batch_size, auto_offload,
        cli_args.log_level,
    )

    args = argparse.Namespace(**defaults)
    args.config = None
    if cli_args.config:
        if not os.path.exists(cli_args.config):
            parser.error(f"Config file not found: {cli_args.config}")
        try:
            with open(cli_args.config, "r") as f:
                config_from_file = toml.load(f)
            print(f"Configuration loaded from {cli_args.config}")
        except Exception as e:
            parser.error(f"Error loading TOML config file {cli_args.config}: {e}")
        for key, value in config_from_file.items():
            setattr(args, key, value)
        args.config = cli_args.config

    if cli_args.backend is not None:
        args.backend = cli_args.backend
    return args


def _default_namespace_dict(
    params_defaults, config_defaults, gpu_config, backend, batch_size,
    auto_offload, log_level,
):
    """Return the full defaults dict for argparse.Namespace construction."""
    project_root = _get_project_root()
    defaults = build_all_defaults(params_defaults, config_defaults)
    defaults.update({
        "project_root": project_root,
        "config_path": None,
        "checkpoint_dir": os.path.join(project_root, "checkpoints"),
        "lm_model_path": None,
        "backend": backend,
        "device": "auto",
        "use_flash_attention": None,
        "offload_to_cpu": auto_offload,
        "offload_dit_to_cpu": False,
        "save_dir": "output",
        "caption": "",
        "prompt": "",
        "lyrics": None,
        "instrumental": False,
        "task_type": params_defaults.task_type,
        "instruction": params_defaults.instruction,
        "reference_audio": params_defaults.reference_audio,
        "src_audio": params_defaults.src_audio,
        "lego_track": "",
        "extract_track": "",
        "complete_tracks": "",
        "audio_codes": "",
        "thinking": gpu_config.init_lm_default,
        "batch_size": batch_size,
        "log_level": log_level,
    })
    return defaults


def _setup_prompt_edit_hook(args, llm_handler) -> None:
    if not (args.thinking and args.task_type not in SKIP_LM_TASKS):
        return
    instruction_path = os.path.join(
        os.path.abspath(args.project_root) if args.project_root else os.getcwd(),
        "instruction.txt",
    )
    preloaded_prompt = None
    if args.config and os.path.exists(instruction_path):
        try:
            with open(instruction_path, "r", encoding="utf-8") as f:
                preloaded_prompt = f.read()
            print(f"INFO: Found {instruction_path}. Using it without editing.")
        except Exception as e:
            print(f"WARNING: Failed to read {instruction_path}: {e}")
    if preloaded_prompt is not None and not preloaded_prompt.strip():
        preloaded_prompt = None
    install_prompt_edit_hook(
        llm_handler, instruction_path, preloaded_prompt=preloaded_prompt,
    )


def _build_generation_objects(args, timesteps):
    params = GenerationParams.from_namespace(args)
    if timesteps is not None:
        params.timesteps = timesteps
    config = GenerationConfig.from_namespace(args)
    return params, config


def _run_generation(
    args, params, config, dit_handler, llm_handler, log_level_upper,
) -> None:
    manual_edit = (
        args.thinking
        and args.task_type not in SKIP_LM_TASKS
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
        apply_lm_results_to_params(params, lm_result, originals)
        if hasattr(llm_handler, "_skip_prompt_edit"):
            llm_handler._skip_prompt_edit = False
        if log_level_upper in {"INFO", "DEBUG"}:
            print_dit_prompt(dit_handler, params)
        print("Running DiT generation with edited prompt and cached audio codes...")

    else:
        if log_level_upper in {"INFO", "DEBUG"}:
            print_dit_prompt(dit_handler, params)

    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=args.save_dir,
    )
    _print_results(result, args, manual_edit, lm_time_costs)


def _print_results(result, args, manual_edit, lm_time_costs) -> None:
    if not result.success:
        print(f"\nGeneration failed: {result.error}")
        print(f"   Status: {result.status_message}")
        return

    print(
        f"\nGeneration successful! {len(result.audios)} audio(s) "
        f"saved in '{args.save_dir}/'"
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
        if args.thinking:
            lm1 = time_costs.get("lm_phase1_time", 0)
            lm2 = time_costs.get("lm_phase2_time", 0)
            print(f"  - LM time: {lm1 + lm2:.2f}s")
        print(f"  - DiT time: {time_costs.get('dit_total_time_cost', 0):.2f}s")
        print("-------------------\n")


if __name__ == "__main__":
    main()
