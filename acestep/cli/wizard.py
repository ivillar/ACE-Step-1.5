"""Interactive CLI wizard — top-level orchestrator."""

import os
import sys
from typing import Optional, Tuple

import toml

from acestep.inference import GenerationConfig, GenerationParams

from acestep.cli.defaults import apply_optional_defaults
from acestep.cli.prompt_helpers import prompt_int
from acestep.cli.wizard_advanced import collect_advanced_parameters
from acestep.cli.wizard_content import collect_content_inputs, collect_task_and_models


def run_wizard(
    args,
    configure_only: bool = False,
    default_config_path: Optional[str] = None,
    params_defaults: Optional[GenerationParams] = None,
    config_defaults: Optional[GenerationConfig] = None,
) -> Tuple:
    """Run the interactive generation wizard, returning ``(args, should_generate)``."""
    print("Welcome to the ACE-Step Music Generation Wizard!")
    print("This will guide you through creating your music.")
    print("Press Ctrl+C at any time to exit.")
    print("Note: Required models will be auto-downloaded if missing.")
    print("-" * 30)

    try:
        collect_task_and_models(args)
        collect_content_inputs(args)

        args.batch_size = prompt_int(
            "Number of outputs (audio clips) to generate",
            args.batch_size if args.batch_size is not None else 2,
            min_value=1,
        )

        advanced = input(
            "\nConfigure advanced parameters? (y/n) [default: n]: "
        ).lower()
        if advanced == "y":
            collect_advanced_parameters(args)
        elif params_defaults and config_defaults:
            apply_optional_defaults(args, params_defaults, config_defaults)

        _print_summary(args)

        if not configure_only:
            confirm = input(
                "Start generation with these settings? (y/n) [default: y]: "
            ).lower()
            if confirm == "n":
                print("Generation cancelled.")
                sys.exit(0)

        _save_config(args, default_config_path)

    except (KeyboardInterrupt, EOFError):
        print("\nWizard cancelled. Exiting.")
        sys.exit(0)

    return args, not configure_only


def _print_summary(args) -> None:
    """Print a short human-readable summary of the wizard selections."""
    print("\n--- Summary ---")
    print(f"Task: {args.task_type}")
    if args.caption:
        print(f"Description: {args.caption}")
    if args.task_type in {"lego", "extract", "complete"}:
        print(f"Instruction: {args.instruction}")
    if args.src_audio:
        print(f"Source audio: {args.src_audio}")
    print(f"Duration: {args.duration}s")
    print(f"Outputs: {args.batch_size}")
    if args.instrumental:
        print("Lyrics: Instrumental")
    elif args.use_cot_lyrics:
        print(f"Lyrics: Auto-generated ({args.vocal_language})")
    elif args.lyrics and os.path.isfile(args.lyrics):
        print(f"Lyrics: Provided from file ({args.lyrics})")
    elif args.lyrics:
        print("Lyrics: Provided as text")
    print("-" * 30)


def _save_config(args, default_config_path: Optional[str]) -> None:
    """Prompt for a filename and persist the current args as TOML."""
    default_filename = default_config_path or "config.toml"
    config_filename = input(
        f"\nEnter filename to save configuration [{default_filename}]: "
    )
    if not config_filename:
        config_filename = default_filename
    if not config_filename.endswith(".toml"):
        config_filename += ".toml"

    try:
        config_to_save = {
            k: v for k, v in vars(args).items()
            if k not in ["config"] and not k.startswith("_")
        }
        with open(config_filename, "w") as f:
            toml.dump(config_to_save, f)
        print(f"Configuration saved to {config_filename}")
        print(f"You can reuse it next time with: python cli.py -c {config_filename}")
    except Exception as e:
        print(f"Error saving configuration: {e}. Please try again.")
