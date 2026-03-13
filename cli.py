"""ACE-Step 1.5 CLI — interactive wizard and config-driven music generation."""

import argparse
import os
import sys

import pickle
import dill

from acestep.cli.env_setup import clear_proxy_env, configure_logging, load_dotenv_config

load_dotenv_config()
clear_proxy_env()
configure_logging()

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
from acestep.cli.pre_generation import run_pre_generation_lm_steps  # noqa: E402
from acestep.cli.prompt_editing import install_prompt_edit_hook  # noqa: E402

def main() -> None:
    """Entry point for the ACE-Step CLI."""
    with open('args.pkl', 'rb') as f:
        args = pickle.load(f)

    with open('parser.pkl', 'rb') as f:
        parser = dill.load(f)
    device = 'cuda'
    timesteps = None
    with open('params_defaults.pkl', 'rb') as f:
        params_defaults = pickle.load(f)

    with open('config_defaults.pkl', 'rb') as f:
        config_defaults = pickle.load(f)

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
