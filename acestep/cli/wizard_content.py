"""Wizard sub-steps for task selection, model selection, and content input."""

import os
import sys
from typing import Optional

from acestep.constants import DEFAULT_DIT_INSTRUCTION
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler

from acestep.cli.defaults import TRACK_CHOICES, default_instruction_for_task
from acestep.cli.prompt_helpers import (
    prompt_bool,
    prompt_choice_from_list,
    prompt_existing_file,
    prompt_float,
    prompt_non_empty,
    prompt_with_default,
)


def collect_task_and_models(args) -> None:
    """Prompt for task type and DiT/LM model selection, mutating *args* in place."""
    print("\n--- Task Type ---")
    print("1. text2music - generate music from text/lyrics.")
    print("2. cover     - transform existing audio into a new style.")
    print("3. repaint   - regenerate a specific time segment of audio.")
    print("4. lego      - generate a specific instrument track in context.")
    print("5. extract   - isolate a specific instrument track from a mix.")
    print("6. complete  - complete/extend partial tracks with new instruments.")

    task_map = {
        "1": "text2music", "2": "cover", "3": "repaint",
        "4": "lego", "5": "extract", "6": "complete",
    }
    current_task = args.task_type or "text2music"
    task_default = next((k for k, v in task_map.items() if v == current_task), "1")
    task_choice = input(f"Choose a task (1-6) [default: {task_default}]: ").strip()
    if not task_choice:
        task_choice = task_default
    args.task_type = task_map.get(task_choice, "text2music")
    if args.task_type in {"lego", "extract", "complete"}:
        print(
            "Note: This task requires a base DiT model (acestep-v15-base). "
            "It will be auto-downloaded if missing."
        )

    _select_dit_model(args)
    _select_lm_model(args)


def _select_dit_model(args) -> None:
    """DiT model selection sub-step."""
    dit_handler = AceStepHandler()
    available = dit_handler.get_available_acestep_v15_models()
    base_only = args.task_type in {"lego", "extract", "complete"}

    if base_only and available:
        available = [m for m in available if "base" in m.lower()]
    if base_only and args.config_path and "base" not in str(args.config_path).lower():
        args.config_path = None

    if base_only:
        if available:
            selected = args.config_path if args.config_path in available else available[0]
            args.config_path = selected
            print(f"\nNote: This task requires a base model. Using: {selected}")
        else:
            print(
                "\nNote: This task requires a base model (e.g., 'acestep-v15-base'). "
                "It will be auto-downloaded if missing."
            )
    elif available:
        selected = prompt_choice_from_list(
            "--- Available DiT Models ---", available,
            default=args.config_path, allow_custom=True,
        )
        if selected is not None:
            args.config_path = selected
    else:
        print(
            "\nNote: No local DiT models found. The main model will be "
            "auto-downloaded during initialization."
        )


def _select_lm_model(args) -> None:
    """LM model selection sub-step."""
    llm_handler = LLMHandler()
    available = llm_handler.get_available_5hz_lm_models()
    if available:
        selected = prompt_choice_from_list(
            "--- Available LM Models ---", available,
            default=args.lm_model_path, allow_custom=True,
        )
        if selected is not None:
            args.lm_model_path = selected
    else:
        print(
            "\nNote: No local LM models found. If LM features are enabled, "
            "a default LM will be auto-downloaded."
        )


def collect_content_inputs(args) -> None:
    """Prompt for task-specific audio sources, caption, and lyrics."""
    if args.task_type in {"cover", "repaint", "lego", "extract", "complete"}:
        args.src_audio = prompt_existing_file(
            "Enter path to source audio file", default=args.src_audio,
        )

    if args.task_type == "repaint":
        args.repainting_start = prompt_float(
            "Repaint start time in seconds", args.repainting_start,
        )
        args.repainting_end = prompt_float(
            "Repaint end time in seconds", args.repainting_end,
        )

    if args.task_type in {"lego", "extract"}:
        _collect_track_and_instruction(args)

    if args.task_type == "complete":
        _collect_complete_tracks_and_instruction(args)

    _collect_caption(args)
    _collect_lyrics(args)


def _collect_track_and_instruction(args) -> None:
    """Prompt for a single track and instruction (lego / extract)."""
    print("\nAvailable tracks:")
    print(", ".join(TRACK_CHOICES))
    track_default = args.lego_track if args.task_type == "lego" else args.extract_track
    track = prompt_with_default("Choose a track", track_default, required=True)
    if track not in TRACK_CHOICES:
        print("Unknown track. Using as-is.")
    if args.task_type == "lego":
        args.lego_track = track
    else:
        args.extract_track = track
    if not args.instruction or args.instruction == DEFAULT_DIT_INSTRUCTION:
        args.instruction = default_instruction_for_task(args.task_type, [track])
    args.instruction = prompt_with_default("Instruction", args.instruction, required=True)


def _collect_complete_tracks_and_instruction(args) -> None:
    """Prompt for multiple tracks and instruction (complete task)."""
    print("\nAvailable tracks:")
    print(", ".join(TRACK_CHOICES))
    tracks_raw = prompt_with_default(
        "Choose tracks (comma-separated)", args.complete_tracks, required=True,
    )
    tracks = [t.strip() for t in tracks_raw.split(",") if t.strip()]
    args.complete_tracks = ",".join(tracks)
    if not args.instruction or args.instruction == DEFAULT_DIT_INSTRUCTION:
        args.instruction = default_instruction_for_task(args.task_type, tracks)
    args.instruction = prompt_with_default("Instruction", args.instruction, required=True)


def _collect_caption(args) -> None:
    """Prompt for a music description / caption."""
    if args.task_type in {"cover", "repaint", "lego", "complete"}:
        args.caption = prompt_with_default(
            "Enter a music description (e.g., 'upbeat electronic dance music')",
            args.caption, required=True,
        )
    elif args.task_type == "text2music":
        args.sample_mode = prompt_bool(
            "Use Simple Mode (auto-generate caption/lyrics via LM)", args.sample_mode,
        )
        if args.sample_mode:
            args.sample_query = prompt_with_default(
                "Describe the music you want (for auto-generation)",
                args.sample_query, required=False,
            )
        else:
            caption = prompt_with_default(
                "Enter a music description (optional if you provide lyrics)",
                args.caption, required=False,
            )
            if caption:
                args.caption = caption


def _collect_lyrics(args) -> None:
    """Prompt for lyrics selection (instrumental / auto / file / paste)."""
    eligible = args.task_type in {"text2music", "cover", "repaint", "lego", "complete"}
    if not eligible or args.sample_mode:
        return

    print("\n--- Lyrics Options ---")
    print("1. Instrumental (no lyrics).")
    print("2. Generate lyrics automatically.")
    print("3. Provide path to a .txt file.")
    print("4. Paste lyrics directly.")

    if args.instrumental or args.lyrics == "[Instrumental]":
        default_choice = "1"
    elif args.use_cot_lyrics:
        default_choice = "2"
    elif args.lyrics and isinstance(args.lyrics, str) and os.path.isfile(args.lyrics):
        default_choice = "3"
    elif args.lyrics:
        default_choice = "4"
    else:
        default_choice = "1"

    choice = input(f"Your choice (1-4) [default: {default_choice}]: ").strip()
    if not choice:
        choice = default_choice

    if choice == "1":
        args.instrumental = True
        args.lyrics = "[Instrumental]"
        args.use_cot_lyrics = False
        print("Instrumental music will be generated.")
    elif choice == "2":
        args.use_cot_lyrics = True
        args.lyrics = ""
        args.instrumental = False
        print("Lyrics will be generated automatically.")
    elif choice == "3":
        args.instrumental = False
        args.use_cot_lyrics = False
        default_lyrics_path = (
            args.lyrics
            if isinstance(args.lyrics, str) and os.path.isfile(args.lyrics)
            else None
        )
        while True:
            lyrics_path = prompt_existing_file(
                "Please enter the path to your .txt lyrics file", default_lyrics_path,
            )
            if lyrics_path.endswith(".txt"):
                args.lyrics = lyrics_path
                print(f"Lyrics will be loaded from: {lyrics_path}")
                break
            print("Invalid file path or not a .txt file. Please try again.")
    elif choice == "4":
        args.instrumental = False
        args.use_cot_lyrics = False
        default_lyrics = (
            args.lyrics
            if isinstance(args.lyrics, str) and args.lyrics and not os.path.isfile(args.lyrics)
            else None
        )
        args.lyrics = prompt_with_default(
            "Paste lyrics (single line or use \\n)", default_lyrics, required=True,
        )

    if not args.instrumental:
        lang = prompt_with_default(
            "Vocal language (e.g., 'en', 'zh', 'unknown')",
            args.vocal_language, required=False,
        ).lower()
        if lang:
            args.vocal_language = lang

    if args.use_cot_lyrics:
        if not args.caption:
            args.caption = prompt_non_empty(
                "Enter a music description for lyric generation: "
            )
