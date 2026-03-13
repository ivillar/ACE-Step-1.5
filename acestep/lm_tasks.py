"""LM task methods: understand, create_sample, format_sample."""

import os
import sys
import traceback
import time
import random
import warnings
from typing import Optional, Dict, Any, Tuple, List, Union
from contextlib import contextmanager
import yaml
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor
from acestep.constants import DEFAULT_LM_INSTRUCTION, DEFAULT_LM_UNDERSTAND_INSTRUCTION, DEFAULT_LM_INSPIRED_INSTRUCTION, DEFAULT_LM_REWRITE_INSTRUCTION, DURATION_MIN, DURATION_MAX
from acestep.gpu_config import get_lm_gpu_memory_ratio, get_gpu_memory_gb, get_lm_model_size, get_global_gpu_config


def build_formatted_prompt_for_understanding(
    self,
    audio_codes: str,
    is_negative_prompt: bool = False,
    negative_prompt: str = "NO USER INPUT"
) -> str:
    """
    Build the chat-formatted prompt for audio understanding from codes.

    This is the reverse of generation: given audio codes, generate metadata and lyrics.

    Args:
        audio_codes: Audio code string (e.g., "<|audio_code_123|><|audio_code_456|>...")
        is_negative_prompt: If True, builds unconditional prompt for CFG
        negative_prompt: Negative prompt for CFG (used when is_negative_prompt=True)

    Returns:
        Formatted prompt string

    Example:
        codes = "<|audio_code_18953|><|audio_code_13833|>..."
        prompt = handler.build_formatted_prompt_for_understanding(codes)
    """
    if self.llm_tokenizer is None:
        raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")

    # For understanding task, user provides audio codes
    # Unconditional prompt uses negative_prompt or empty string
    if is_negative_prompt:
        user_content = negative_prompt if negative_prompt and negative_prompt.strip() else ""
    else:
        user_content = audio_codes

    return self.llm_tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": f"# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n"
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def understand_audio_from_codes(
    self,
    audio_codes: str,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Understand audio codes and generate metadata + lyrics.

    This is the reverse of the normal generation flow:
    - Input: Audio codes
    - Output: Metadata (bpm, caption, duration, etc.) + Lyrics

    Note: cfg_scale and negative_prompt are not supported in understand mode.

    Args:
        audio_codes: String of audio code tokens (e.g., "<|audio_code_123|><|audio_code_456|>...")
        temperature: Sampling temperature for generation
        top_k: Top-K sampling (None = disabled)
        top_p: Top-P (nucleus) sampling (None = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding for metadata
        constrained_decoding_debug: Whether to enable debug logging for constrained decoding

    Returns:
        Tuple of (metadata_dict, status_message)
        metadata_dict contains:
            - bpm: int or str
            - caption: str
            - duration: int or str
            - keyscale: str
            - language: str
            - timesignature: str
            - lyrics: str (extracted from output after </think>)

    Example:
        codes = "<|audio_code_18953|><|audio_code_13833|>..."
        metadata, status = handler.understand_audio_from_codes(codes)
        print(metadata['caption'])  # "A cinematic orchestral piece..."
        print(metadata['lyrics'])   # "[Intro: ...]\\n..."
    """
    if not getattr(self, "llm_initialized", False):
        return {}, "❌ 5Hz LM not initialized. Please initialize it first."

    if not audio_codes or not audio_codes.strip():
        return {}, "❌ No audio codes provided. Please paste audio codes first."

    logger.info(f"Understanding audio codes (length: {len(audio_codes)} chars)")

    # Build formatted prompt for understanding
    formatted_prompt = self.build_formatted_prompt_for_understanding(audio_codes)
    print(f"formatted_prompt: {formatted_prompt}")
    # Generate using constrained decoding (understand phase)
    # We want to generate metadata first (CoT), then lyrics (natural text)
    # Note: cfg_scale and negative_prompt are not used in understand mode
    output_text, status = self.generate_from_formatted_prompt(
        formatted_prompt=formatted_prompt,
        cfg={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "target_duration": None,  # No duration constraint for understanding
            "user_metadata": None,  # No user metadata injection
            "skip_caption": False,  # Generate caption
            "skip_language": False,  # Generate language
            "skip_genres": False,  # Generate genres
            "generation_phase": "understand",  # Understanding phase: generate CoT metadata, then free-form lyrics
            # Context for building unconditional prompt
            "caption": "",
            "lyrics": "",
        },
        use_constrained_decoding=use_constrained_decoding,
        constrained_decoding_debug=constrained_decoding_debug,
        stop_at_reasoning=False,  # Continue after </think> to generate lyrics
    )

    if not output_text:
        return {}, status

    # Parse metadata and extract lyrics
    metadata, _ = self.parse_lm_output(output_text)

    # Extract lyrics section (everything after </think>)
    lyrics = self._extract_lyrics_from_output(output_text)
    if lyrics:
        metadata['lyrics'] = lyrics

    logger.info(f"Understanding completed. Generated {len(metadata)} metadata fields")
    if constrained_decoding_debug:
        logger.debug(f"Generated metadata: {list(metadata.keys())}")
        logger.debug(f"Output text preview: {output_text[:200]}...")

    status_msg = f"✅ Understanding completed successfully\nGenerated fields: {', '.join(metadata.keys())}"
    return metadata, status_msg


def _extract_lyrics_from_output(self, output_text: str) -> str:
    """
    Extract lyrics section from LLM output.

    The lyrics appear after the </think> tag and typically start with "# Lyric"
    or directly with lyric content.

    Args:
        output_text: Full LLM output text

    Returns:
        Extracted lyrics string, or empty string if no lyrics found
    """
    import re

    # Find the </think> tag
    think_end_pattern = r'</think>'
    match = re.search(think_end_pattern, output_text)

    if not match:
        # No </think> tag found, no lyrics
        return ""

    # Extract everything after </think>
    after_think = output_text[match.end():].strip()

    if not after_think:
        return ""

    # Remove "# Lyric" header if present
    lyric_header_pattern = r'^#\s*Lyri[c|cs]?\s*\n'
    after_think = re.sub(lyric_header_pattern, '', after_think, flags=re.IGNORECASE)

    # Remove <|im_end|> tag at the end if present
    after_think = re.sub(r'<\|im_end\|>\s*$', '', after_think)

    return after_think.strip()


def build_formatted_prompt_for_inspiration(
    self,
    query: str,
    instrumental: bool = False,
    is_negative_prompt: bool = False,
    negative_prompt: str = "NO USER INPUT"
) -> str:
    """
    Build the chat-formatted prompt for inspiration/simple mode.

    This generates a complete sample (caption, lyrics, metadata) from a user's
    natural language music description query.

    Args:
        query: User's natural language music description
        instrumental: Whether to generate instrumental music (no vocals)
        is_negative_prompt: If True, builds unconditional prompt for CFG
        negative_prompt: Negative prompt for CFG (used when is_negative_prompt=True)

    Returns:
        Formatted prompt string

    Example:
        query = "a soft Bengali love song for a quiet evening"
        prompt = handler.build_formatted_prompt_for_inspiration(query, instrumental=False)
    """
    if self.llm_tokenizer is None:
        raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")

    # Build user content with query and instrumental flag
    instrumental_str = "true" if instrumental else "false"

    if is_negative_prompt:
        # For CFG unconditional prompt
        user_content = negative_prompt if negative_prompt and negative_prompt.strip() else ""
    else:
        # Normal prompt: query + instrumental flag
        user_content = f"{query}\n\ninstrumental: {instrumental_str}"

    return self.llm_tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": f"# Instruction\n{DEFAULT_LM_INSPIRED_INSTRUCTION}\n\n"
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def create_sample_from_query(
    self,
    query: str,
    instrumental: bool = False,
    vocal_language: Optional[str] = None,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Create a complete music sample from a user's natural language query.

    This is the "Simple Mode" / "Inspiration Mode" feature that generates:
    - Metadata (bpm, caption, duration, keyscale, language, timesignature)
    - Lyrics (unless instrumental=True)

    Args:
        query: User's natural language music description
        instrumental: Whether to generate instrumental music (no vocals)
        vocal_language: Allowed vocal language for constrained decoding (e.g., "en", "zh").
                       If provided and not "unknown", it will be used.
        temperature: Sampling temperature for generation (0.0-2.0)
        top_k: Top-K sampling (None = disabled)
        top_p: Top-P (nucleus) sampling (None = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding
        constrained_decoding_debug: Whether to enable debug logging

    Returns:
        Tuple of (metadata_dict, status_message)
        metadata_dict contains:
            - bpm: int or str
            - caption: str
            - duration: int or str
            - keyscale: str
            - language: str
            - timesignature: str
            - lyrics: str (extracted from output after </think>)
            - instrumental: bool (echoed back)

    Example:
        query = "a soft Bengali love song for a quiet evening"
        metadata, status = handler.create_sample_from_query(query, instrumental=False, vocal_language="bn")
        print(metadata['caption'])  # "A gentle romantic acoustic pop ballad..."
        print(metadata['lyrics'])   # "[Intro: ...]\\n..."
    """
    if not getattr(self, "llm_initialized", False):
        return {}, "❌ 5Hz LM not initialized. Please initialize it first."

    if not query or not query.strip():
        query = "NO USER INPUT"

    logger.info(f"Creating sample from query: {query[:100]}... (instrumental={instrumental}, vocal_language={vocal_language})")

    # Build formatted prompt for inspiration
    formatted_prompt = self.build_formatted_prompt_for_inspiration(
        query=query,
        instrumental=instrumental,
    )
    logger.debug(f"Formatted prompt for inspiration: {formatted_prompt}")

    # Build user_metadata if vocal_language is specified and is not "unknown"
    user_metadata = None
    skip_language = False
    if vocal_language and vocal_language.strip() and vocal_language.strip().lower() != "unknown":
        # Use the specified language for constrained decoding
        user_metadata = {"language": vocal_language.strip()}
        # skip_language = True  # Skip language generation since we're injecting it
        logger.info(f"Using user-specified language: {vocal_language.strip()}")

    # Generate using constrained decoding (inspiration phase)
    # Similar to understand mode - generate metadata first (CoT), then lyrics
    # Note: cfg_scale and negative_prompt are not used in create_sample mode
    output_text, status = self.generate_from_formatted_prompt(
        formatted_prompt=formatted_prompt,
        cfg={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "target_duration": None,  # No duration constraint
            "user_metadata": user_metadata,  # Inject language if specified
            "skip_caption": False,  # Generate caption
            "skip_language": False,
            "skip_genres": False,  # Generate genres
            "generation_phase": "understand",  # Use understand phase for metadata + free-form lyrics
            "caption": "",
            "lyrics": "",
        },
        use_constrained_decoding=use_constrained_decoding,
        constrained_decoding_debug=constrained_decoding_debug,
        stop_at_reasoning=False,  # Continue after </think> to generate lyrics
    )

    if not output_text:
        return {}, status

    # Parse metadata and extract lyrics
    metadata, _ = self.parse_lm_output(output_text)

    # Extract lyrics section (everything after </think>)
    lyrics = self._extract_lyrics_from_output(output_text)
    if lyrics:
        metadata['lyrics'] = lyrics
    elif instrumental:
        # For instrumental, set empty lyrics or placeholder
        metadata['lyrics'] = "[Instrumental]"

    # Echo back the instrumental flag
    metadata['instrumental'] = instrumental

    logger.info(f"Sample created successfully. Generated {metadata} fields")
    if constrained_decoding_debug:
        logger.debug(f"Generated metadata: {list(metadata.keys())}")
        logger.debug(f"Output text preview: {output_text[:300]}...")

    status_msg = f"✅ Sample created successfully\nGenerated fields: {metadata}"
    return metadata, status_msg


def build_formatted_prompt_for_format(
    self,
    caption: str,
    lyrics: str,
    is_negative_prompt: bool = False,
    negative_prompt: str = "NO USER INPUT"
) -> str:
    """
    Build the chat-formatted prompt for format/rewrite mode.

    This formats user-provided caption and lyrics into a more detailed and specific
    musical description with metadata.

    Args:
        caption: User's caption/description of the music
        lyrics: User's lyrics
        is_negative_prompt: If True, builds unconditional prompt for CFG
        negative_prompt: Negative prompt for CFG (used when is_negative_prompt=True)

    Returns:
        Formatted prompt string

    Example:
        caption = "Latin pop, reggaeton, flamenco-pop"
        lyrics = "[Verse 1]\\nTengo un nudo..."
        prompt = handler.build_formatted_prompt_for_format(caption, lyrics)
    """
    if self.llm_tokenizer is None:
        raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")

    if is_negative_prompt:
        # For CFG unconditional prompt
        user_content = negative_prompt if negative_prompt and negative_prompt.strip() else ""
    else:
        # Normal prompt: caption + lyrics
        user_content = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}"

    return self.llm_tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": f"# Instruction\n{DEFAULT_LM_REWRITE_INSTRUCTION}\n\n"
            },
            {
                "role": "user",
                "content": user_content
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_sample_from_input(
    self,
    caption: str,
    lyrics: str,
    user_metadata: Optional[Dict[str, Any]] = None,
    temperature: float = 0.85,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Format user-provided caption and lyrics into structured music metadata.

    This is the "Format" feature that takes user input and generates:
    - Enhanced caption with detailed music description
    - Metadata (bpm, duration, keyscale, language, timesignature)
    - Formatted lyrics (preserved from input)

    Note: cfg_scale and negative_prompt are not supported in format mode.

    Args:
        caption: User's caption/description (e.g., "Latin pop, reggaeton")
        lyrics: User's lyrics with structure tags
        user_metadata: Optional dict with user-provided metadata to constrain decoding.
                      Supported keys: bpm, duration, keyscale, timesignature, language
        temperature: Sampling temperature for generation (0.0-2.0)
        top_k: Top-K sampling (None = disabled)
        top_p: Top-P (nucleus) sampling (None = disabled)
        repetition_penalty: Repetition penalty (1.0 = no penalty)
        use_constrained_decoding: Whether to use FSM-based constrained decoding
        constrained_decoding_debug: Whether to enable debug logging

    Returns:
        Tuple of (metadata_dict, status_message)
        metadata_dict contains:
            - bpm: int or str
            - caption: str (enhanced)
            - duration: int or str
            - keyscale: str
            - language: str
            - timesignature: str
            - lyrics: str (from input, possibly formatted)

    Example:
        caption = "Latin pop, reggaeton, flamenco-pop"
        lyrics = "[Verse 1]\\nTengo un nudo en la garganta..."
        metadata, status = handler.format_sample_from_input(caption, lyrics)
        print(metadata['caption'])  # "A dramatic and powerful Latin pop track..."
        print(metadata['bpm'])      # 100
    """
    if not getattr(self, "llm_initialized", False):
        return {}, "❌ 5Hz LM not initialized. Please initialize it first."

    if not caption or not caption.strip():
        caption = "NO USER INPUT"
    if not lyrics or not lyrics.strip():
        lyrics = "[Instrumental]"

    logger.info(f"Formatting sample from input: caption={caption[:50]}..., lyrics length={len(lyrics)}")

    # Build formatted prompt for format task
    formatted_prompt = self.build_formatted_prompt_for_format(
        caption=caption,
        lyrics=lyrics,
    )
    logger.debug(f"Formatted prompt for format: {formatted_prompt}")

    # Build constrained decoding metadata from user_metadata
    constrained_metadata = None
    if user_metadata:
        constrained_metadata = {}
        if user_metadata.get('bpm') is not None:
            try:
                bpm_val = int(user_metadata['bpm'])
                if bpm_val > 0:
                    constrained_metadata['bpm'] = bpm_val
            except (ValueError, TypeError):
                pass
        if user_metadata.get('duration') is not None:
            try:
                dur_val = int(user_metadata['duration'])
                if dur_val > 0:
                    constrained_metadata['duration'] = dur_val
            except (ValueError, TypeError):
                pass
        if user_metadata.get('keyscale'):
            constrained_metadata['keyscale'] = user_metadata['keyscale']
        if user_metadata.get('timesignature'):
            constrained_metadata['timesignature'] = user_metadata['timesignature']
        if user_metadata.get('language'):
            constrained_metadata['language'] = user_metadata['language']

        # Only use if we have at least one field
        if not constrained_metadata:
            constrained_metadata = None
        else:
            logger.info(f"Using user-provided metadata constraints: {constrained_metadata}")

    # Generate using constrained decoding (format phase)
    # Similar to understand/inspiration mode - generate metadata first (CoT), then formatted lyrics
    # Note: cfg_scale and negative_prompt are not used in format mode
    output_text, status = self.generate_from_formatted_prompt(
        formatted_prompt=formatted_prompt,
        cfg={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "target_duration": None,  # No duration constraint for generation length
            "user_metadata": constrained_metadata,  # Inject user-provided metadata
            "skip_caption": False,  # Generate caption
            "skip_language": constrained_metadata.get('language') is not None if constrained_metadata else False,
            "skip_genres": False,  # Generate genres
            "generation_phase": "understand",  # Use understand phase for metadata + free-form lyrics
            "caption": "",
            "lyrics": "",
        },
        use_constrained_decoding=use_constrained_decoding,
        constrained_decoding_debug=constrained_decoding_debug,
        stop_at_reasoning=False,  # Continue after </think> to get formatted lyrics
    )

    if not output_text:
        return {}, status

    # Parse metadata and extract lyrics
    metadata, _ = self.parse_lm_output(output_text)

    # Extract formatted lyrics section (everything after </think>)
    formatted_lyrics = self._extract_lyrics_from_output(output_text)
    if formatted_lyrics:
        metadata['lyrics'] = formatted_lyrics
    else:
        # If no lyrics generated, keep original input
        metadata['lyrics'] = lyrics

    logger.info(f"Format completed successfully. Generated {metadata} fields")
    if constrained_decoding_debug:
        logger.debug(f"Generated metadata: {list(metadata.keys())}")
        logger.debug(f"Output text preview: {output_text[:300]}...")

    status_msg = f"✅ Format completed successfully\nGenerated fields: {', '.join(metadata.keys())}"
    return metadata, status_msg


def get_hf_model_for_scoring(self):
    """
    Get HuggingFace model for perplexity scoring.

    For vllm backend, loads HuggingFace model from disk (weights are cached by transformers).
    For pt backend, returns the existing model.
    For mlx backend, loads HuggingFace model from disk (MLX model can't be used for torch scoring).

    Returns:
        HuggingFace model instance
    """
    if self.llm_backend == "pt":
        # For PyTorch backend, directly return the model
        return self.llm

    elif self.llm_backend == "vllm":
        # For vllm backend, load HuggingFace model from disk
        # Note: transformers caches model weights, so this doesn't duplicate disk I/O
        if self._hf_model_for_scoring is None:
            logger.info("Loading HuggingFace model for scoring (from checkpoint)")

            # Get model path from vllm config
            model_runner = self.llm.model_runner
            model_path = model_runner.config.model

            # Load HuggingFace model from the same checkpoint
            # This will load the original unfused weights
            import time
            start_time = time.time()
            self._hf_model_for_scoring = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )
            load_time = time.time() - start_time
            logger.info(f"HuggingFace model loaded in {load_time:.2f}s")

            # When offload_to_cpu is enabled, keep the model on CPU to save
            # VRAM.  The caller (_load_scoring_model_context in
            # core/scoring/lm_score.py) will move it to the accelerator only
            # for the duration of the forward pass.
            if self.offload_to_cpu:
                self._hf_model_for_scoring.eval()
                logger.info("HuggingFace model for scoring kept on CPU (offload_to_cpu=True)")
            else:
                device = next(model_runner.model.parameters()).device
                self._hf_model_for_scoring = self._hf_model_for_scoring.to(device)
                self._hf_model_for_scoring.eval()
                logger.info(f"HuggingFace model for scoring ready on {device}")

        return self._hf_model_for_scoring

    elif self.llm_backend == "mlx":
        # For MLX backend, load HuggingFace model from disk for PyTorch scoring
        if self._hf_model_for_scoring is None:
            logger.info("Loading HuggingFace model for scoring (MLX backend, need PyTorch model)")

            # Get model path from stored path
            model_path = getattr(self, '_mlx_model_path', None)
            if model_path is None:
                raise ValueError("MLX model path not stored. Cannot load HuggingFace model for scoring.")

            import time
            start_time = time.time()
            self._hf_model_for_scoring = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=self.dtype
            )
            load_time = time.time() - start_time
            logger.info(f"HuggingFace model loaded in {load_time:.2f}s")

            # When offload_to_cpu is enabled, keep on CPU; the scoring
            # context manager will move it to the accelerator as needed.
            if self.offload_to_cpu:
                self._hf_model_for_scoring.eval()
                logger.info("HuggingFace model for scoring kept on CPU (offload_to_cpu=True)")
            else:
                device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
                self._hf_model_for_scoring = self._hf_model_for_scoring.to(device)
                self._hf_model_for_scoring.eval()
                logger.info(f"HuggingFace model for scoring ready on {device}")

        return self._hf_model_for_scoring

    else:
        raise ValueError(f"Unknown backend: {self.llm_backend}")
