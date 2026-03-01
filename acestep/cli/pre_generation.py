"""Pre-generation LM steps: sample mode, format_sample, and CoT lyrics generation."""

from acestep.inference import create_sample, format_sample

from acestep.cli.parsing import parse_description_hints


def run_pre_generation_lm_steps(args, parser, llm_handler) -> None:
    """Execute sample_mode, format_sample, and use_cot_lyrics before generation."""
    format_has_duration = False

    if args.sample_mode or (args.sample_query and str(args.sample_query).strip()):
        _run_sample_mode(args, parser, llm_handler)

    if args.use_format and (args.caption or args.lyrics):
        format_has_duration = _run_format_sample(args, parser, llm_handler)

    if args.use_cot_lyrics:
        _run_cot_lyrics_generation(args, parser, llm_handler)

    if args.sample_mode or format_has_duration:
        args.use_cot_metas = False


def _run_sample_mode(args, parser, llm_handler) -> None:
    if not llm_handler.llm_initialized:
        parser.error(
            "--sample_mode/sample_query requires the LM handler, "
            "but it's not initialized."
        )

    sample_query = (
        args.sample_query
        if args.sample_query and str(args.sample_query).strip()
        else "NO USER INPUT"
    )
    parsed_language, parsed_instrumental = parse_description_hints(sample_query)
    if args.vocal_language and args.vocal_language not in ("en", "unknown", ""):
        sample_language = args.vocal_language
    else:
        sample_language = parsed_language

    print("\nINFO: Creating sample via 'create_sample'...")
    result = create_sample(
        llm_handler=llm_handler, query=sample_query,
        instrumental=parsed_instrumental, vocal_language=sample_language,
        temperature=args.lm_temperature, top_k=args.lm_top_k, top_p=args.lm_top_p,
    )
    if result.success:
        args.caption = result.caption
        args.lyrics = result.lyrics
        args.instrumental = bool(result.instrumental)
        if args.bpm is None:
            args.bpm = result.bpm
        if not args.keyscale:
            args.keyscale = result.keyscale
        if not args.timesignature:
            args.timesignature = result.timesignature
        if args.duration <= 0:
            args.duration = result.duration
        if args.vocal_language in ("unknown", "", None):
            args.vocal_language = result.language
        args.sample_mode = True
        print("Sample created. Using generated parameters.")
    else:
        parser.error(
            f"create_sample failed: {result.error or result.status_message}"
        )


def _run_format_sample(args, parser, llm_handler) -> bool:
    """Returns True if format_sample provided a duration."""
    if not llm_handler.llm_initialized:
        parser.error(
            "--use_format requires the LM handler, but it's not initialized."
        )

    user_metadata: dict = {}
    if args.bpm is not None:
        user_metadata["bpm"] = args.bpm
    if args.duration is not None and float(args.duration) > 0:
        user_metadata["duration"] = float(args.duration)
    if args.keyscale:
        user_metadata["keyscale"] = args.keyscale
    if args.timesignature:
        user_metadata["timesignature"] = args.timesignature
    if args.vocal_language and args.vocal_language != "unknown":
        user_metadata["language"] = args.vocal_language

    print("\nINFO: Formatting caption/lyrics via 'format_sample'...")
    result = format_sample(
        llm_handler=llm_handler,
        caption=args.caption or "", lyrics=args.lyrics or "",
        user_metadata=user_metadata or None,
        temperature=args.lm_temperature,
        top_k=args.lm_top_k, top_p=args.lm_top_p,
    )
    if result.success:
        args.caption = result.caption or args.caption
        args.lyrics = result.lyrics or args.lyrics
        fmt_has_duration = False
        if result.duration:
            args.duration = result.duration
            fmt_has_duration = True
        if result.bpm:
            args.bpm = result.bpm
        if result.keyscale:
            args.keyscale = result.keyscale
        if result.timesignature:
            args.timesignature = result.timesignature
        print("Format complete.")
        return fmt_has_duration
    else:
        parser.error(
            f"format_sample failed: {result.error or result.status_message}"
        )


def _run_cot_lyrics_generation(args, parser, llm_handler) -> None:
    if not llm_handler.llm_initialized:
        parser.error(
            "--use_cot_lyrics requires the LM handler, but it's not initialized. "
            "Ensure --thinking is enabled."
        )

    print("\nINFO: Generating lyrics and metadata via 'create_sample'...")
    result = create_sample(
        llm_handler=llm_handler, query=args.caption,
        instrumental=False,
        vocal_language=(
            args.vocal_language if args.vocal_language != "unknown" else None
        ),
        temperature=args.lm_temperature,
        top_k=args.lm_top_k, top_p=args.lm_top_p,
    )
    if result.success:
        print("Automatic sample creation successful. Using generated parameters:")
        args.caption = result.caption
        args.lyrics = result.lyrics
        if args.bpm is None:
            args.bpm = result.bpm
        if not args.keyscale:
            args.keyscale = result.keyscale
        if not args.timesignature:
            args.timesignature = result.timesignature
        if args.duration <= 0:
            args.duration = result.duration
        if args.vocal_language == "unknown":
            args.vocal_language = result.language
        lyrics_preview = args.lyrics[:150].strip().replace("\n", " ")
        print(f"  - Caption: {args.caption}")
        print(f"  - Lyrics: '{lyrics_preview}...'")
        print(
            f"  - Metadata: BPM={args.bpm}, Key='{args.keyscale}', "
            f"Lang='{args.vocal_language}'"
        )
        args.use_cot_metas = False
        args.use_cot_caption = False
    else:
        print(f"WARNING: Automatic lyric generation failed: {result.error}")
        print("         Proceeding with an instrumental track instead.")
        args.lyrics = "[Instrumental]"
        args.instrumental = True

    args.use_cot_lyrics = False
