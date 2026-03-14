"""LM generation functions: prompt building, batch handling, and inference dispatch."""

import random
import re
import time
import traceback
from typing import Any

import torch
from loguru import logger

from acestep.constants import (
    DEFAULT_LM_INSTRUCTION,
    DURATION_MAX,
    DURATION_MIN,
)
from acestep.models.lm.constrained_logits_processor import MetadataConstrainedLogitsProcessor
from acestep.gpu_utils import get_global_gpu_config

__all__ = [
    "_compute_max_new_tokens",
    "_has_meaningful_negative_prompt",
    "_setup_constrained_processor",
    "_build_unconditional_prompt",
    "_normalize_batch_input",
    "generate_with_stop_condition",
    "generate_from_formatted_prompt",
]


def _compute_max_new_tokens(
    self,
    target_duration: float | None,
    generation_phase: str,
    fallback_max: int | None = None,
) -> int:
    """
    Compute max_new_tokens based on target duration and generation phase.

    In the two-phase architecture:
    - CoT phase: generates metadata (~50-200 tokens) + needs buffer for safety.
    - Codes phase: CoT is already in the prompt; only audio codes are generated.
      The constrained decoder forces EOS at exactly target_codes, so only a
      small buffer (10 tokens) is needed to avoid a misleading progress bar.

    Duration is clamped to ``[DURATION_MIN, max_dur]`` where *max_dur* is the
    GPU-config-dependent maximum (from ``get_global_gpu_config()``) capped at
    ``DURATION_MAX``.  This keeps the progress-bar total aligned with what the
    constrained decoder actually enforces.

    Args:
        target_duration: Target duration in seconds (5 codes = 1 second).
        generation_phase: "cot" or "codes".
        fallback_max: Fallback value when target_duration is not set.

    Returns:
        Computed max_new_tokens value, capped at model's max length.
    """
    if target_duration is not None and target_duration > 0:
        # Determine the effective upper bound from GPU config (if available)
        # so that max_new_tokens does not exceed what the constrained decoder
        # will actually enforce on lower-tier GPUs.
        gpu_max_dur = DURATION_MAX
        try:
            gpu_cfg = get_global_gpu_config()
            gpu_max_dur = min(gpu_cfg.max_duration_with_lm, DURATION_MAX)
        except Exception:
            pass  # Fallback to DURATION_MAX if GPU config unavailable

        effective_duration = max(DURATION_MIN, min(gpu_max_dur, target_duration))
        target_codes = int(effective_duration * 5)
        if generation_phase == "codes":
            # Codes phase: CoT already in prompt, only audio codes generated.
            # Constrained decoder forces EOS at target_codes, so small buffer suffices.
            max_new_tokens = target_codes + 10
        else:
            # CoT phase or mixed: add larger buffer for metadata overhead.
            max_new_tokens = target_codes + 500
    else:
        if fallback_max is not None:
            max_new_tokens = fallback_max
        else:
            max_new_tokens = getattr(self, "max_model_len", 4096) - 64

    # Cap at model's max length
    if hasattr(self, "max_model_len"):
        max_new_tokens = min(max_new_tokens, self.max_model_len - 64)

    return max_new_tokens


def _has_meaningful_negative_prompt(self, negative_prompt: str) -> bool:
    """Check if negative prompt is meaningful (not default/empty)"""
    return negative_prompt and negative_prompt.strip() and negative_prompt.strip() != "NO USER INPUT"


def _setup_constrained_processor(
    self,
    use_constrained_decoding: bool,
    constrained_decoding_debug: bool,
    target_duration: float | None,
    user_metadata: dict[str, str | None] | None,
    stop_at_reasoning: bool,
    skip_genres: bool,
    skip_caption: bool,
    skip_language: bool,
    generation_phase: str,
    is_batch: bool = False,
    metadata_temperature: float | None = None,
    codes_temperature: float | None = None,
) -> MetadataConstrainedLogitsProcessor | None:
    """Setup and configure constrained processor for generation"""
    use_phase_temperatures = not is_batch and (metadata_temperature is not None or codes_temperature is not None)

    if not use_constrained_decoding and not use_phase_temperatures:
        return None

    # Reset processor state for new generation
    self.constrained_processor.reset()

    # Use shared processor, just update settings
    self.constrained_processor.enabled = use_constrained_decoding
    self.constrained_processor.debug = constrained_decoding_debug

    # Phase temperatures only supported in single mode
    if use_phase_temperatures:
        self.constrained_processor.metadata_temperature = metadata_temperature
        self.constrained_processor.codes_temperature = codes_temperature
    else:
        self.constrained_processor.metadata_temperature = None
        self.constrained_processor.codes_temperature = None

    self.constrained_processor.set_target_duration(target_duration)

    # Batch mode uses default/disabled settings for these options
    if is_batch:
        self.constrained_processor.set_user_metadata(None)
        self.constrained_processor.set_stop_at_reasoning(False)
        self.constrained_processor.set_skip_genres(True)
        self.constrained_processor.set_skip_caption(True)
        self.constrained_processor.set_skip_language(True)
    else:
        # Single mode uses provided settings
        self.constrained_processor.set_user_metadata(user_metadata)
        self.constrained_processor.set_stop_at_reasoning(stop_at_reasoning)
        self.constrained_processor.set_skip_genres(skip_genres)
        self.constrained_processor.set_skip_caption(skip_caption)
        self.constrained_processor.set_skip_language(skip_language)

    # Set generation phase for phase-aware processing
    self.constrained_processor.set_generation_phase(generation_phase)

    return self.constrained_processor


def _build_unconditional_prompt(
    self,
    caption: str,
    lyrics: str,
    cot_text: str,
    negative_prompt: str,
    generation_phase: str,
    is_batch: bool = False,
) -> str:
    """Build unconditional prompt for CFG based on generation phase and batch mode"""
    if is_batch or generation_phase == "codes":
        # Codes phase or batch mode: use empty CoT in unconditional prompt
        return self.build_formatted_prompt_with_cot(
            caption, lyrics, cot_text, is_negative_prompt=True, negative_prompt=negative_prompt
        )
    else:
        # CoT phase (single mode only): unconditional prompt
        # If negative_prompt is provided, use it as caption; otherwise remove caption and keep only lyrics
        return self.build_formatted_prompt(
            caption, lyrics, is_negative_prompt=True, generation_phase="cot", negative_prompt=negative_prompt
        )


def _normalize_batch_input(self, formatted_prompts: str | list[str]) -> tuple[list[str], bool]:
    """Normalize batch input: convert single string to list and return (list, is_batch)"""
    is_batch = isinstance(formatted_prompts, list)
    if is_batch:
        return formatted_prompts, is_batch
    else:
        return [formatted_prompts], is_batch


def generate_with_stop_condition(
    self,
    caption: str,
    lyrics: str,
    infer_type: str,
    temperature: float = 0.85,
    cfg_scale: float = 1.0,
    negative_prompt: str = "NO USER INPUT",
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
    target_duration: float | None = None,
    user_metadata: dict[str, str | None] | None = None,
    use_cot_metas: bool = True,
    use_cot_caption: bool = True,
    use_cot_language: bool = True,
    use_cot_lyrics: bool = False,
    batch_size: int | None = None,
    seeds: list[int] | None = None,
    progress=None,
) -> dict[str, Any]:
    """Two-phase LM generation: CoT generation followed by audio codes generation.

    - infer_type='dit': Phase 1 only - generate CoT and return metas (no audio codes)
    - infer_type='llm_dit': Phase 1 + Phase 2 - generate CoT then audio codes

    Args:
        target_duration: Target duration in seconds for codes generation constraint.
                        5 codes = 1 second. If specified, blocks EOS until target reached.
        user_metadata: User-provided metadata fields (e.g. bpm/duration/keyscale/timesignature).
                       If specified, constrained decoding will inject these values directly.
        use_cot_caption: Whether to generate caption in CoT (default True).
        use_cot_language: Whether to generate language in CoT (default True).
        use_cot_lyrics: Whether to generate lyrics in CoT when lyrics are empty (default False).
        batch_size: Optional batch size for batch generation. If None or 1, returns single result.
                   If > 1, returns batch results (lists).
        seeds: Optional list of seeds for batch generation (for reproducibility).
              Only used when batch_size > 1. TODO: not used yet

    Returns:
        Dictionary containing:
            - metadata: Dict or List[Dict] - Generated metadata
            - audio_codes: str or List[str] - Generated audio codes
            - success: bool - Whether generation succeeded
            - error: Optional[str] - Error message if failed
            - extra_outputs: Dict with time_costs and other info
    """
    if progress is None:
        def progress(*args, **kwargs):
            pass

    infer_type = (infer_type or "").strip().lower()
    if infer_type not in {"dit", "llm_dit"}:
        error_msg = f"invalid infer_type: {infer_type!r} (expected 'dit' or 'llm_dit')"
        return {
            "metadata": [] if (batch_size and batch_size > 1) else {},
            "audio_codes": [] if (batch_size and batch_size > 1) else "",
            "success": False,
            "error": error_msg,
            "extra_outputs": {"time_costs": {}},
        }

    # Determine if batch mode
    is_batch = batch_size and batch_size > 1
    actual_batch_size = batch_size if is_batch else 1

    # Initialize variables
    metadata = {}
    audio_codes = ""
    has_all_metas = self.has_all_metas(user_metadata)
    phase1_time = 0.0
    phase2_time = 0.0

    # Handle seeds for batch mode
    if is_batch:
        if seeds is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size)]
        elif len(seeds) < actual_batch_size:
            seeds = list(seeds) + [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size - len(seeds))]
        else:
            seeds = seeds[:actual_batch_size]

    # ========== PHASE 1: CoT Generation ==========
    # Skip CoT if all metadata are user-provided OR caption is already formatted
    progress(0.1, "Phase 1: Generating CoT metadata (once for all items)...")
    if not has_all_metas and use_cot_metas:
        if is_batch:
            logger.info("Batch Phase 1: Generating CoT metadata (once for all items)...")
        else:
            logger.info("Phase 1: Generating CoT metadata...")
        phase1_start = time.time()

        # Build formatted prompt for CoT phase
        formatted_prompt = self.build_formatted_prompt(caption, lyrics, generation_phase="cot")

        logger.info(f"generate_with_stop_condition: formatted_prompt={formatted_prompt}")
        # Generate CoT (stop at </think>)
        cot_output_text, status = self.generate_from_formatted_prompt(
            formatted_prompt=formatted_prompt,
            cfg={
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "negative_prompt": negative_prompt,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "target_duration": None,  # No duration constraint for CoT phase
                "user_metadata": user_metadata,
                "skip_caption": not use_cot_caption,
                "skip_language": not use_cot_language,
                "skip_genres": True,  # Generate genres
                "generation_phase": "cot",
                # Pass context for building unconditional prompt in CoT phase
                "caption": caption,
                "lyrics": lyrics,
            },
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
            stop_at_reasoning=not (use_cot_lyrics and not (lyrics or "").strip()),
        )

        phase1_time = time.time() - phase1_start

        if not cot_output_text:
            return {
                "metadata": [] if is_batch else {},
                "audio_codes": [] if is_batch else "",
                "success": False,
                "error": status,
                "extra_outputs": {"time_costs": {"phase1_time": phase1_time}},
            }

        # Parse metadata from CoT output
        metadata, _ = self.parse_lm_output(cot_output_text)
        # When use_cot_lyrics and lyrics were empty, extract lyrics from after </think>
        if use_cot_lyrics and not (lyrics or "").strip():
            extracted = self._extract_lyrics_from_output(cot_output_text)
            if extracted:
                # Drop any trailing audio code tokens if the model continued past lyrics
                code_pattern = r'<\|audio_code_\d+\|>'
                if re.search(code_pattern, extracted):
                    extracted = re.split(code_pattern, extracted, maxsplit=1)[0]
                metadata["lyrics"] = extracted.strip()
        if is_batch:
            logger.info(f"Batch Phase 1 completed in {phase1_time:.2f}s. Generated metadata: {list(metadata.keys())}")
        else:
            logger.info(f"Phase 1 completed in {phase1_time:.2f}s. Generated metadata: {list(metadata.keys())}")
    else:
        # Use user-provided metadata
        if is_batch:
            logger.info("Batch Phase 1: Using user-provided metadata (skipping generation)")
        else:
            logger.info("Phase 1: Using user-provided metadata (skipping generation)")
        metadata = {k: v for k, v in user_metadata.items() if v is not None}

    # If infer_type is 'dit', stop here and return only metadata
    if infer_type == "dit":
        if is_batch:
            metadata_list = [metadata.copy() for _ in range(actual_batch_size)]
            return {
                "metadata": metadata_list,
                "audio_codes": [""] * actual_batch_size,
                "success": True,
                "error": None,
                "extra_outputs": {
                    "time_costs": {
                        "phase1_time": phase1_time,
                        "total_time": phase1_time,
                    }
                },
            }
        else:
            return {
                "metadata": metadata,
                "audio_codes": "",
                "success": True,
                "error": None,
                "extra_outputs": {
                    "time_costs": {
                        "phase1_time": phase1_time,
                        "total_time": phase1_time,
                    }
                },
            }

    # ========== PHASE 2: Audio Codes Generation ==========
    if is_batch:
        logger.info(f"Batch Phase 2: Generating audio codes for {actual_batch_size} items...")
    else:
        logger.info("Phase 2: Generating audio codes...")
    phase2_start = time.time()

    # Format metadata as CoT using YAML (matching training format)
    cot_text = self._format_metadata_as_cot(metadata)

    # Build formatted prompt with CoT for codes generation phase
    formatted_prompt_with_cot = self.build_formatted_prompt_with_cot(caption, lyrics, cot_text)
    logger.info(f"generate_with_stop_condition: formatted_prompt_with_cot={formatted_prompt_with_cot}")

    progress(0.5, f"Phase 2: Generating audio codes for {actual_batch_size} items...")
    if is_batch:
        # Batch mode: generate codes for all items
        formatted_prompts = [formatted_prompt_with_cot] * actual_batch_size

        # Call backend-specific batch generation
        try:
            if self.llm_backend == "vllm":
                codes_outputs = self._run_vllm(
                    formatted_prompts=formatted_prompts,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    constrained_decoding_debug=constrained_decoding_debug,
                    target_duration=target_duration,
                    generation_phase="codes",
                    caption=caption,
                    lyrics=lyrics,
                    cot_text=cot_text,
                    seeds=seeds,
                )
            elif self.llm_backend == "mlx":
                codes_outputs = self._run_mlx(
                    formatted_prompts=formatted_prompts,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    constrained_decoding_debug=constrained_decoding_debug,
                    target_duration=target_duration,
                    generation_phase="codes",
                    caption=caption,
                    lyrics=lyrics,
                    cot_text=cot_text,
                    seeds=seeds,
                )
            else:  # pt backend
                codes_outputs = self._run_pt(
                    formatted_prompts=formatted_prompts,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    constrained_decoding_debug=constrained_decoding_debug,
                    target_duration=target_duration,
                    generation_phase="codes",
                    caption=caption,
                    lyrics=lyrics,
                    cot_text=cot_text,
                    seeds=seeds,
                )
        except Exception as e:
            error_msg = f"Error in batch codes generation: {str(e)}"
            logger.error(error_msg)
            return {
                "metadata": [],
                "audio_codes": [],
                "success": False,
                "error": error_msg,
                "extra_outputs": {
                    "time_costs": {
                        "phase1_time": phase1_time,
                        "phase2_time": 0.0,
                        "total_time": phase1_time,
                    }
                },
            }

        # Parse audio codes from each output
        audio_codes_list = []
        metadata_list = []
        for output_text in codes_outputs:
            _, audio_codes_item = self.parse_lm_output(output_text)
            audio_codes_list.append(audio_codes_item)
            metadata_list.append(metadata.copy())  # Same metadata for all

        phase2_time = time.time() - phase2_start

        # Log results
        codes_counts = [len(codes.split('<|audio_code_')) - 1 if codes else 0 for codes in audio_codes_list]
        logger.info(f"Batch Phase 2 completed in {phase2_time:.2f}s. Generated codes: {codes_counts}")

        total_time = phase1_time + phase2_time
        return {
            "metadata": metadata_list,
            "audio_codes": audio_codes_list,
            "success": True,
            "error": None,
            "extra_outputs": {
                "time_costs": {
                    "phase1_time": phase1_time,
                    "phase2_time": phase2_time,
                    "total_time": total_time,
                },
                "codes_counts": codes_counts,
                "total_codes": sum(codes_counts),
            },
        }
    else:
        # Single mode: generate codes for one item
        codes_output_text, status = self.generate_from_formatted_prompt(
            formatted_prompt=formatted_prompt_with_cot,
            cfg={
                "temperature": temperature,
                "cfg_scale": cfg_scale,
                "negative_prompt": negative_prompt,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "target_duration": target_duration,
                "user_metadata": None,  # No user metadata injection in Phase 2
                "skip_caption": True,  # Skip caption since CoT is already included
                "skip_language": True,  # Skip language since CoT is already included
                "generation_phase": "codes",
                # Pass context for building unconditional prompt in codes phase
                "caption": caption,
                "lyrics": lyrics,
                "cot_text": cot_text,
            },
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
            stop_at_reasoning=False,  # Generate codes until EOS
        )

        if not codes_output_text:
            total_time = phase1_time + phase2_time
            return {
                "metadata": metadata,
                "audio_codes": "",
                "success": False,
                "error": status,
                "extra_outputs": {
                    "time_costs": {
                        "phase1_time": phase1_time,
                        "phase2_time": phase2_time,
                        "total_time": total_time,
                    }
                },
            }

        phase2_time = time.time() - phase2_start

        # Parse audio codes from output (metadata should be same as Phase 1)
        _, audio_codes = self.parse_lm_output(codes_output_text)

        codes_count = len(audio_codes.split('<|audio_code_')) - 1 if audio_codes else 0
        logger.info(f"Phase 2 completed in {phase2_time:.2f}s. Generated {codes_count} audio codes")

        total_time = phase1_time + phase2_time
        return {
            "metadata": metadata,
            "audio_codes": audio_codes,
            "success": True,
            "error": None,
            "extra_outputs": {
                "time_costs": {
                    "phase1_time": phase1_time,
                    "phase2_time": phase2_time,
                    "total_time": total_time,
                },
                "codes_count": codes_count,
            },
        }


def generate_from_formatted_prompt(
    self,
    formatted_prompt: str,
    cfg: dict[str, Any] | None = None,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
    stop_at_reasoning: bool = False,
) -> tuple[str, str]:
    """
    Generate raw LM text output from a pre-built formatted prompt.

    Args:
        formatted_prompt: Prompt that is already formatted by `build_formatted_prompt`.
        cfg: Optional dict supporting keys:
            - temperature (float)
            - cfg_scale (float)
            - negative_prompt (str) used when cfg_scale > 1
            - top_k (int), top_p (float), repetition_penalty (float)
            - target_duration (float): Target duration in seconds for codes generation
            - generation_phase (str): "cot" or "codes" for phase-aware CFG
        use_constrained_decoding: Whether to use FSM-based constrained decoding
        constrained_decoding_debug: Whether to enable debug logging for constrained decoding
        stop_at_reasoning: If True, stop generation immediately after </think> tag (no audio codes)

    Returns:
        (output_text, status_message)

    Example:
        prompt = handler.build_formatted_prompt(caption, lyric)
        text, status = handler.generate_from_formatted_prompt(prompt, {"temperature": 0.7})
    """
    if not getattr(self, "llm_initialized", False):
        return "", "❌ 5Hz LM not initialized. Please initialize it first."
    # Check that the appropriate model is loaded for the active backend
    if self.llm_backend == "mlx":
        if self._mlx_model is None or self.llm_tokenizer is None:
            return "", "❌ 5Hz LM is missing MLX model or tokenizer."
    elif self.llm is None or self.llm_tokenizer is None:
        return "", "❌ 5Hz LM is missing model or tokenizer."

    cfg = cfg or {}
    temperature = cfg.get("temperature", 0.6)
    cfg_scale = cfg.get("cfg_scale", 1.0)
    negative_prompt = cfg.get("negative_prompt", "NO USER INPUT")
    top_k = cfg.get("top_k")
    top_p = cfg.get("top_p")
    repetition_penalty = cfg.get("repetition_penalty", 1.0)
    target_duration = cfg.get("target_duration")
    user_metadata = cfg.get("user_metadata")  # User-provided metadata fields
    skip_caption = cfg.get("skip_caption", False)  # Skip caption generation in CoT
    skip_language = cfg.get("skip_language", False)  # Skip language generation in CoT
    skip_genres = cfg.get("skip_genres", False)  # Skip genres generation in CoT
    generation_phase = cfg.get("generation_phase", "cot")  # "cot" or "codes"
    # Additional context for codes phase unconditional prompt building
    caption = cfg.get("caption", "")
    lyrics = cfg.get("lyrics", "")
    cot_text = cfg.get("cot_text", "")

    try:
        if self.llm_backend == "vllm":
            output_text = self._run_vllm(
                formatted_prompts=formatted_prompt,
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
                constrained_decoding_debug=constrained_decoding_debug,
                target_duration=target_duration,
                user_metadata=user_metadata,
                stop_at_reasoning=stop_at_reasoning,
                skip_genres=skip_genres,
                skip_caption=skip_caption,
                skip_language=skip_language,
                generation_phase=generation_phase,
                caption=caption,
                lyrics=lyrics,
                cot_text=cot_text,
            )
            return output_text, f"✅ Generated successfully (vllm) | length={len(output_text)}"

        elif self.llm_backend == "mlx":
            # MLX backend (Apple Silicon native)
            output_text = self._run_mlx(
                formatted_prompts=formatted_prompt,
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=use_constrained_decoding,
                constrained_decoding_debug=constrained_decoding_debug,
                target_duration=target_duration,
                user_metadata=user_metadata,
                stop_at_reasoning=stop_at_reasoning,
                skip_genres=skip_genres,
                skip_caption=skip_caption,
                skip_language=skip_language,
                generation_phase=generation_phase,
                caption=caption,
                lyrics=lyrics,
                cot_text=cot_text,
            )
            return output_text, f"✅ Generated successfully (mlx) | length={len(output_text)}"

        # PyTorch backend (fallback)
        output_text = self._run_pt(
            formatted_prompts=formatted_prompt,
            temperature=temperature,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_constrained_decoding=use_constrained_decoding,
            constrained_decoding_debug=constrained_decoding_debug,
            target_duration=target_duration,
            user_metadata=user_metadata,
            stop_at_reasoning=stop_at_reasoning,
            skip_genres=skip_genres,
            skip_caption=skip_caption,
            skip_language=skip_language,
            generation_phase=generation_phase,
            caption=caption,
            lyrics=lyrics,
            cot_text=cot_text,
        )
        return output_text, f"✅ Generated successfully (pt) | length={len(output_text)}"

    except Exception as e:
        # Log full traceback for debugging
        error_detail = traceback.format_exc()
        logger.error(f"Error in generate_from_formatted_prompt: {type(e).__name__}: {e}\n{error_detail}")
        # Reset nano-vllm state on error to prevent stale context from causing
        # subsequent CUDA illegal memory access errors
        if self.llm_backend == "vllm":
            try:
                from nanovllm.utils.context import reset_context
                reset_context()
            except ImportError:
                pass
            # Also reset the LLM scheduler to release allocated KV cache blocks
            # This prevents 'deque index out of range' errors from block leaks
            try:
                if hasattr(self.llm, 'reset'):
                    self.llm.reset()
            except Exception:
                pass  # Ignore errors during cleanup
        # Clear accelerator cache to release any corrupted memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        return "", f"❌ Error generating from formatted prompt: {type(e).__name__}: {e or error_detail.splitlines()[-1]}"
