"""Argument post-processing and validation for the CLI pipeline."""

import os
from typing import List, Optional

from acestep.constants import DEFAULT_DIT_INSTRUCTION, TASK_INSTRUCTIONS
from acestep.inference import GenerationParams

from acestep.cli.defaults import default_instruction_for_task
from acestep.cli.parsing import parse_timesteps_input
from acestep.cli.prompt_helpers import expand_audio_path


def postprocess_args(args, parser) -> Optional[List[float]]:
    """Normalize and validate CLI args after TOML merge.

    Returns:
        Parsed timesteps list (or None).
    """
    if args.use_cot_lyrics and not args.thinking:
        print(
            "INFO: Automatic lyric generation requires the LM handler. "
            "Forcing --thinking=True."
        )
        args.thinking = True

    if not args.project_root:
        args.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    else:
        args.project_root = os.path.abspath(os.path.expanduser(str(args.project_root)))

    if args.checkpoint_dir:
        args.checkpoint_dir = os.path.expanduser(str(args.checkpoint_dir))
        if not os.path.isabs(args.checkpoint_dir):
            args.checkpoint_dir = os.path.join(args.project_root, args.checkpoint_dir)

    if args.src_audio:
        args.src_audio = expand_audio_path(args.src_audio)
    if args.reference_audio:
        args.reference_audio = expand_audio_path(args.reference_audio)

    timesteps = _parse_and_validate_timesteps(args, parser)
    _normalize_seeds(args)
    _normalize_instrumental(args)
    _validate_task_requirements(args, parser)
    _resolve_lyrics_path(args, parser)

    if args.backend == "pyTorch":
        args.backend = "pt"
    if args.backend not in {"vllm", "pt", "mlx"}:
        args.backend = "vllm"

    return timesteps


def _parse_and_validate_timesteps(args, parser) -> Optional[List[float]]:
    """Parse timesteps from args and raise on invalid input."""
    try:
        timesteps = parse_timesteps_input(args.timesteps)
        if args.timesteps and timesteps is None:
            raise ValueError(
                "Timesteps must be a list of numbers or a comma-separated string."
            )
    except ValueError as e:
        parser.error(
            "Invalid format for timesteps. Expected a list of numbers "
            f"(e.g., '[1.0, 0.5, 0.0]' or '0.97,0.5,0'). Error: {e}"
        )
    return timesteps


def _normalize_seeds(args) -> None:
    """Derive batch_size from explicit seeds when provided."""
    if args.seeds:
        args.batch_size = len(args.seeds)
        args.use_random_seed = False
        args.seed = -1


def _normalize_instrumental(args) -> None:
    """Ensure instrumental flag and lyrics are consistent."""
    if args.instrumental and not args.lyrics:
        args.lyrics = "[Instrumental]"
    elif isinstance(args.lyrics, str) and args.lyrics.strip().lower() in {
        "[inst]", "[instrumental]",
    }:
        args.instrumental = True


def _validate_task_requirements(args, parser) -> None:
    """Enforce required fields and mutual constraints per task type."""
    params_defaults = GenerationParams()

    if args.task_type in {"cover", "repaint", "lego", "extract", "complete"}:
        if not args.src_audio:
            parser.error(
                f"--src_audio is required for task_type '{args.task_type}'."
            )
    if args.task_type in {"cover", "repaint", "lego", "complete"}:
        if not args.caption:
            parser.error(
                f"--caption is required for task_type '{args.task_type}'."
            )

    if args.task_type == "text2music":
        if not args.caption and not args.lyrics:
            if not args.sample_mode and not args.sample_query:
                parser.error("--caption or --lyrics is required for text2music.")
        if args.use_cot_lyrics and not args.caption:
            parser.error("--use_cot_lyrics requires --caption for lyric generation.")
        if args.sample_mode or args.sample_query:
            args.sample_mode = True
    else:
        if args.sample_mode or args.sample_query:
            parser.error(
                "--sample_mode/sample_query are only supported for task_type 'text2music'."
            )

    if args.sample_mode and args.use_cot_lyrics:
        print("INFO: sample_mode enabled. Disabling --use_cot_lyrics.")
        args.use_cot_lyrics = False

    if (
        args.instruction == DEFAULT_DIT_INSTRUCTION
        and args.task_type in TASK_INSTRUCTIONS
    ):
        if args.task_type in {"text2music", "cover", "repaint"}:
            args.instruction = TASK_INSTRUCTIONS[args.task_type]

    if args.task_type == "repaint":
        if args.repainting_end != -1 and args.repainting_end <= args.repainting_start:
            parser.error(
                "--repainting_end must be greater than --repainting_start (or -1)."
            )

    _resolve_task_instruction(args, parser, params_defaults)


def _resolve_task_instruction(args, parser, params_defaults) -> None:
    """Auto-generate instruction for lego/extract/complete when not provided."""
    if args.task_type not in {"lego", "extract", "complete"}:
        return

    has_custom = bool(
        args.instruction
        and args.instruction.strip()
        and args.instruction.strip() != params_defaults.instruction
    )
    if has_custom:
        return

    if args.task_type == "lego":
        if not args.lego_track:
            parser.error("--instruction or --lego_track is required for lego task.")
        args.instruction = default_instruction_for_task("lego", [args.lego_track.strip()])
    elif args.task_type == "extract":
        if not args.extract_track:
            parser.error("--instruction or --extract_track is required for extract task.")
        args.instruction = default_instruction_for_task(
            "extract", [args.extract_track.strip()],
        )
    elif args.task_type == "complete":
        if not args.complete_tracks:
            parser.error(
                "--instruction or --complete_tracks is required for complete task."
            )
        tracks = [t.strip() for t in args.complete_tracks.split(",") if t.strip()]
        if not tracks:
            parser.error("--complete_tracks must contain at least one track.")
        args.instruction = default_instruction_for_task("complete", tracks)


def _resolve_lyrics_path(args, parser) -> None:
    """Resolve lyrics: 'generate' keyword, file path, or inline text."""
    lyrics_arg = args.lyrics
    if not isinstance(lyrics_arg, str) or not lyrics_arg:
        return

    lyrics_arg = os.path.expanduser(lyrics_arg)
    if not os.path.isabs(lyrics_arg):
        resolved = None
        if args.config:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            candidate = os.path.join(config_dir, lyrics_arg)
            if os.path.isfile(candidate):
                resolved = candidate
        if resolved is None and args.project_root:
            candidate = os.path.join(os.path.abspath(args.project_root), lyrics_arg)
            if os.path.isfile(candidate):
                resolved = candidate
        if resolved is not None:
            lyrics_arg = resolved

    if lyrics_arg == "generate":
        args.use_cot_lyrics = True
        args.lyrics = ""
        print("Lyrics generation enabled.")
    elif os.path.isfile(lyrics_arg):
        print(f"INFO: Attempting to load lyrics from file: {lyrics_arg}")
        try:
            with open(lyrics_arg, "r", encoding="utf-8") as f:
                args.lyrics = f.read()
            print(f"Lyrics loaded from file: {lyrics_arg}")
        except Exception as e:
            parser.error(f"Could not read lyrics file {lyrics_arg}. Error: {e}")
