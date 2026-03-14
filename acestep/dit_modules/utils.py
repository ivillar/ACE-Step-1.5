"""Consolidated handler functions – utils module."""

import json
import os
import random
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION, SAMPLE_RATE, SFT_GEN_PROMPT, TASK_INSTRUCTIONS
from acestep.gpu_config import get_effective_free_vram_gb, get_global_gpu_config
from acestep.lora.manager import (
    add_lora,
    add_voice_lora,
    apply_scale_to_adapter,
    collect_adapter_names,
    debug_lora_registry_snapshot,
    ensure_lora_registry,
    get_lora_status,
    load_lora,
    rebuild_lora_registry,
    remove_lora,
    set_active_lora_adapter,
    set_lora_scale,
    set_use_lora,
    unload_lora,
)
from acestep.lora.manager import (
    sync_lora_state as sync_lora_state_from_service,
)

# --- From padding_utils.py ---

def prepare_padding_info(
    self,
    actual_batch_size,
    processed_src_audio,
    audio_duration,
    repainting_start,
    repainting_end,
    is_repaint_task,
    is_lego_task,
    is_cover_task,
    can_use_repainting,
):
    """Prepare padded target wavs and repaint coordinates for each batch item."""
    try:
        target_wavs_batch = []
        # Store padding info for each batch item to adjust repainting coordinates
        padding_info_batch = []
        for _i in range(actual_batch_size):
            if processed_src_audio is not None:
                if is_cover_task:
                    # Cover task: Use src_audio directly without padding
                    batch_target_wavs = processed_src_audio
                    padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
                elif is_repaint_task or is_lego_task:
                    # Repaint/lego task: May need padding for outpainting
                    src_audio_duration = processed_src_audio.shape[-1] / SAMPLE_RATE

                    # Determine actual end time
                    if repainting_end is None or repainting_end < 0:
                        actual_end = src_audio_duration
                    else:
                        actual_end = repainting_end

                    left_padding_duration = max(0, -repainting_start) if repainting_start is not None else 0
                    right_padding_duration = max(0, actual_end - src_audio_duration)

                    # Create padded audio
                    left_padding_frames = int(left_padding_duration * SAMPLE_RATE)
                    right_padding_frames = int(right_padding_duration * SAMPLE_RATE)

                    if left_padding_frames > 0 or right_padding_frames > 0:
                        # Pad the src audio
                        batch_target_wavs = torch.nn.functional.pad(
                            processed_src_audio, (left_padding_frames, right_padding_frames), "constant", 0
                        )
                    else:
                        batch_target_wavs = processed_src_audio

                    # Store padding info for coordinate adjustment
                    padding_info_batch.append(
                        {
                            "left_padding_duration": left_padding_duration,
                            "right_padding_duration": right_padding_duration,
                        }
                    )
                else:
                    # Other tasks: Use src_audio directly without padding
                    batch_target_wavs = processed_src_audio
                    padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
            else:
                padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
                if audio_duration is not None and float(audio_duration) > 0:
                    batch_target_wavs = self.create_target_wavs(float(audio_duration))
                else:
                    import random

                    random_duration = random.uniform(10.0, 120.0)
                    batch_target_wavs = self.create_target_wavs(random_duration)
            target_wavs_batch.append(batch_target_wavs)

        # Stack target_wavs into batch tensor
        # Ensure all tensors have the same shape by padding to max length
        max_frames = max(wav.shape[-1] for wav in target_wavs_batch)
        padded_target_wavs = []
        for wav in target_wavs_batch:
            if wav.shape[-1] < max_frames:
                pad_frames = max_frames - wav.shape[-1]
                padded_wav = torch.nn.functional.pad(wav, (0, pad_frames), "constant", 0)
                padded_target_wavs.append(padded_wav)
            else:
                padded_target_wavs.append(wav)

        target_wavs_tensor = torch.stack(padded_target_wavs, dim=0)  # [batch_size, 2, frames]

        if can_use_repainting:
            # Repaint task: Set repainting parameters
            if repainting_start is None:
                repainting_start_batch = None
            elif isinstance(repainting_start, (int, float)):
                if processed_src_audio is not None:
                    adjusted_start = repainting_start + padding_info_batch[0]["left_padding_duration"]
                    repainting_start_batch = [adjusted_start] * actual_batch_size
                else:
                    repainting_start_batch = [repainting_start] * actual_batch_size
            else:
                # List input - adjust each item
                repainting_start_batch = []
                for i in range(actual_batch_size):
                    if processed_src_audio is not None:
                        adjusted_start = repainting_start[i] + padding_info_batch[i]["left_padding_duration"]
                        repainting_start_batch.append(adjusted_start)
                    else:
                        repainting_start_batch.append(repainting_start[i])

            # Handle repainting_end - use src audio duration if not specified or negative
            if processed_src_audio is not None:
                # If src audio is provided, use its duration as default end
                src_audio_duration = processed_src_audio.shape[-1] / SAMPLE_RATE
                if repainting_end is None or repainting_end < 0:
                    # Use src audio duration (before padding), then adjust for padding
                    adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
                    repainting_end_batch = [adjusted_end] * actual_batch_size
                else:
                    # Adjust repainting_end to be relative to padded audio
                    adjusted_end = repainting_end + padding_info_batch[0]["left_padding_duration"]
                    repainting_end_batch = [adjusted_end] * actual_batch_size
            else:
                # No src audio - repainting doesn't make sense without it
                if repainting_end is None or repainting_end < 0:
                    repainting_end_batch = None
                elif isinstance(repainting_end, (int, float)):
                    repainting_end_batch = [repainting_end] * actual_batch_size
                else:
                    # List input - adjust each item
                    repainting_end_batch = []
                    for i in range(actual_batch_size):
                        repainting_end_batch.append(repainting_end[i])
        else:
            # All other tasks (cover, text2music, extract, complete): No repainting
            # Only repaint and lego tasks should have repainting parameters
            repainting_start_batch = None
            repainting_end_batch = None

        return repainting_start_batch, repainting_end_batch, target_wavs_tensor
    except (TypeError, ValueError, RuntimeError, IndexError):
        logger.exception("[prepare_padding_info] Error preparing padding information")
        fallback = torch.stack([self.create_target_wavs(30.0) for _ in range(actual_batch_size)], dim=0)
        return None, None, fallback

# --- From memory_utils.py ---

def is_silence(self, audio: torch.Tensor) -> bool:
    """Return True when audio is effectively silent."""
    return bool(torch.all(audio.abs() < 1e-6))

def _get_system_memory_gb(self) -> float | None:
    """Return total system RAM in GB when available."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if page_size and page_count:
            return (page_size * page_count) / (1024**3)
    except (ValueError, OSError, AttributeError):
        return None
    return None

def _get_effective_mps_memory_gb(self) -> float | None:
    """Best-effort MPS memory estimate (recommended max or system RAM)."""
    if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
        try:
            return torch.mps.recommended_max_memory() / (1024**3)
        except Exception:
            pass
    system_gb = self._get_system_memory_gb()
    if system_gb is None:
        return None
    return system_gb * 0.75

VAE_DECODE_MAX_CHUNK_SIZE = 512

def _get_auto_decode_chunk_size(self) -> int:
    """Choose a conservative VAE decode chunk size based on available memory."""
    override = os.environ.get("ACESTEP_VAE_DECODE_CHUNK_SIZE")
    if override:
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass

    max_chunk = self.VAE_DECODE_MAX_CHUNK_SIZE

    if self.device == "mps":
        mem_gb = self._get_effective_mps_memory_gb()
        if mem_gb is not None:
            if mem_gb >= 48:
                return min(1536, max_chunk)
            if mem_gb >= 24:
                return min(1024, max_chunk)
        return min(512, max_chunk)

    if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
        try:
            free_gb = get_effective_free_vram_gb()
        except Exception:
            free_gb = 0
        logger.debug(f"[_get_auto_decode_chunk_size] Effective free VRAM: {free_gb:.2f} GB")
        if free_gb >= 24.0:
            return min(512, max_chunk)
        if free_gb >= 16.0:
            return min(384, max_chunk)
        if free_gb >= 12.0:
            return min(256, max_chunk)
        return min(128, max_chunk)
    return min(256, max_chunk)

def _should_offload_wav_to_cpu(self) -> bool:
    """Decide whether to offload decoded wavs to CPU for memory safety."""
    override = os.environ.get("ACESTEP_MPS_DECODE_OFFLOAD")
    if override:
        return override.lower() in ("1", "true", "yes")
    if self.device == "mps":
        mem_gb = self._get_effective_mps_memory_gb()
        if mem_gb is not None and mem_gb >= 32:
            return False
        return True
    if self.device == "cuda" or (isinstance(self.device, str) and self.device.startswith("cuda")):
        try:
            free_gb = get_effective_free_vram_gb()
            logger.debug(f"[_should_offload_wav_to_cpu] Effective free VRAM: {free_gb:.2f} GB")
            if free_gb >= 24.0:
                return False
        except Exception:
            pass
    return True

def _vram_guard_reduce_batch(
    self,
    batch_size: int,
    audio_duration: float | None = None,
    use_lm: bool = False,
) -> int:
    """Auto-reduce batch_size when free VRAM is too tight."""
    if batch_size <= 1:
        return batch_size

    device = self.device
    if device == "cpu" or device == "mps":
        return batch_size

    if self.offload_to_cpu:
        gpu_config = get_global_gpu_config()
        if gpu_config is not None:
            tier_max = gpu_config.max_batch_size_with_lm
            if batch_size <= tier_max:
                logger.debug(
                    f"[VRAM guard] offload_to_cpu=True, batch_size={batch_size} <= "
                    f"tier limit {tier_max} — skipping dynamic VRAM check"
                )
                return batch_size

    try:
        free_gb = get_effective_free_vram_gb()
    except Exception:
        return batch_size

    duration_sec = float(audio_duration) if audio_duration and float(audio_duration) > 0 else 60.0
    per_sample_gb = 0.5 + max(0.0, 0.15 * (duration_sec - 60.0) / 60.0)
    if hasattr(self, "model") and self.model is not None:
        model_name = getattr(self, "config_path", "") or ""
        if "base" in model_name.lower():
            per_sample_gb *= 2.0

    safety_margin_gb = 1.5
    available_for_batch = free_gb - safety_margin_gb
    if available_for_batch <= 0:
        logger.warning(f"[VRAM guard] Only {free_gb:.1f} GB free — reducing batch_size to 1")
        return 1

    max_safe_batch = max(1, int(available_for_batch / per_sample_gb))
    if max_safe_batch < batch_size:
        logger.warning(
            f"[VRAM guard] Free VRAM {free_gb:.1f} GB can safely fit ~{max_safe_batch} samples "
            f"(requested {batch_size}). Reducing batch_size to {max_safe_batch}."
        )
        return max_safe_batch
    return batch_size

def _get_vae_dtype(self, device: str | None = None) -> torch.dtype:
    """Get VAE dtype based on target device and GPU tier."""
    target_device = device or self.device
    if target_device in ["cuda", "xpu"]:
        return torch.bfloat16
    if target_device == "mps":
        return torch.float16
    if target_device == "cpu":
        return torch.float32
    return self.dtype

# --- From metadata_utils.py ---

def _create_default_meta(self) -> str:
    """Create default metadata string."""
    return (
        "- bpm: N/A\n"
        "- timesignature: N/A\n"
        "- keyscale: N/A\n"
        "- duration: 30 seconds\n"
    )

def _dict_to_meta_string(self, meta_dict: dict[str, Any]) -> str:
    """Convert metadata dict to formatted string."""
    bpm = meta_dict.get("bpm", meta_dict.get("tempo", "N/A"))
    timesignature = meta_dict.get("timesignature", meta_dict.get("time_signature", "N/A"))
    keyscale = meta_dict.get("keyscale", meta_dict.get("key", meta_dict.get("scale", "N/A")))
    duration = meta_dict.get("duration", meta_dict.get("length", 30))

    if isinstance(duration, (int, float)):
        duration = f"{int(duration)} seconds"
    elif not isinstance(duration, str):
        duration = "30 seconds"

    return (
        f"- bpm: {bpm}\n"
        f"- timesignature: {timesignature}\n"
        f"- keyscale: {keyscale}\n"
        f"- duration: {duration}\n"
    )

def _parse_metas(self, metas: list[str | dict[str, Any]]) -> list[str]:
    """Parse and normalize metadata values with safe fallbacks."""
    parsed_metas = []
    for meta in metas:
        if meta is None:
            parsed_meta = self._create_default_meta()
        elif isinstance(meta, str):
            parsed_meta = meta
        elif isinstance(meta, dict):
            parsed_meta = self._dict_to_meta_string(meta)
        else:
            parsed_meta = self._create_default_meta()
        parsed_metas.append(parsed_meta)
    return parsed_metas

def prepare_metadata(
    self, bpm: int | str | None, key_scale: str, time_signature: str
) -> dict[str, Any]:
    """Build metadata dict for generation."""
    return self._build_metadata_dict(bpm, key_scale, time_signature)

def _build_metadata_dict(
    self,
    bpm: int | str | None,
    key_scale: str,
    time_signature: str,
    duration: float | None = None,
) -> dict[str, Any]:
    """Build metadata dictionary with defaults for missing fields."""
    metadata_dict: dict[str, Any] = {}
    metadata_dict["bpm"] = bpm if bpm else "N/A"
    metadata_dict["keyscale"] = key_scale if key_scale.strip() else "N/A"
    if time_signature.strip() and time_signature != "N/A" and time_signature:
        metadata_dict["timesignature"] = time_signature
    else:
        metadata_dict["timesignature"] = "N/A"
    if duration is not None:
        metadata_dict["duration"] = f"{int(duration)} seconds"
    return metadata_dict

# --- From progress.py ---

def _get_project_root(self) -> str:
    """Get project root directory path."""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(current_file)
                )
            )
        )
    )

def _load_progress_estimates(self) -> None:
    """Load persisted diffusion progress estimates if available."""
    try:
        if os.path.exists(self._progress_estimates_path):
            with open(self._progress_estimates_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("records"), list):
                    self._progress_estimates = data
    except Exception:
        # Ignore corrupted cache; it will be overwritten on next save.
        self._progress_estimates = {"records": []}

def _save_progress_estimates(self) -> None:
    """Persist diffusion progress estimates."""
    try:
        os.makedirs(os.path.dirname(self._progress_estimates_path), exist_ok=True)
        with open(self._progress_estimates_path, "w", encoding="utf-8") as f:
            json.dump(self._progress_estimates, f)
    except Exception:
        pass

def _duration_bucket(self, duration_sec: float | None) -> str:
    if duration_sec is None or duration_sec <= 0:
        return "unknown"
    if duration_sec <= 60:
        return "short"
    if duration_sec <= 180:
        return "medium"
    if duration_sec <= 360:
        return "long"
    return "xlong"

def _update_progress_estimate(
    self,
    per_step_sec: float,
    infer_steps: int,
    batch_size: int,
    duration_sec: float | None,
) -> None:
    if per_step_sec <= 0 or infer_steps <= 0:
        return
    record = {
        "device": self.device,
        "infer_steps": int(infer_steps),
        "batch_size": int(batch_size),
        "duration_sec": float(duration_sec) if duration_sec and duration_sec > 0 else None,
        "duration_bucket": self._duration_bucket(duration_sec),
        "per_step_sec": float(per_step_sec),
        "updated_at": time.time(),
    }
    with self._progress_estimates_lock:
        records = self._progress_estimates.get("records", [])
        records.append(record)
        # Keep recent 100 records
        records = records[-100:]
        self._progress_estimates["records"] = records
        self._progress_estimates["updated_at"] = time.time()
        self._save_progress_estimates()

def _estimate_diffusion_per_step(
    self,
    infer_steps: int,
    batch_size: int,
    duration_sec: float | None,
) -> float | None:
    # Prefer most recent exact-ish record
    target_bucket = self._duration_bucket(duration_sec)
    with self._progress_estimates_lock:
        records = list(self._progress_estimates.get("records", []))
    if not records:
        return None

    # Filter by device first
    device_records = [r for r in records if r.get("device") == self.device] or records

    # Exact match by steps/batch/bucket
    for r in reversed(device_records):
        if (
            r.get("infer_steps") == infer_steps
            and r.get("batch_size") == batch_size
            and r.get("duration_bucket") == target_bucket
        ):
            return r.get("per_step_sec")

    # Same steps + bucket, scale by batch and duration when possible
    for r in reversed(device_records):
        if r.get("infer_steps") == infer_steps and r.get("duration_bucket") == target_bucket:
            base = r.get("per_step_sec")
            base_batch = r.get("batch_size", batch_size)
            base_dur = r.get("duration_sec")
            if base and base_batch:
                est = base * (batch_size / base_batch)
                if duration_sec and base_dur:
                    est *= (duration_sec / base_dur)
                return est

    # Same steps, scale by batch and duration ratio if available
    for r in reversed(device_records):
        if r.get("infer_steps") == infer_steps:
            base = r.get("per_step_sec")
            base_batch = r.get("batch_size", batch_size)
            base_dur = r.get("duration_sec")
            if base and base_batch:
                est = base * (batch_size / base_batch)
                if duration_sec and base_dur:
                    est *= (duration_sec / base_dur)
                return est

    # Fallback to global median
    per_steps = [r.get("per_step_sec") for r in device_records if r.get("per_step_sec")]
    if per_steps:
        per_steps.sort()
        return per_steps[len(per_steps) // 2]
    return None

def _start_diffusion_progress_estimator(
    self,
    progress,
    start: float,
    end: float,
    infer_steps: int,
    batch_size: int,
    duration_sec: float | None,
    desc: str,
):
    """Best-effort progress updates during diffusion using previous step timing."""
    if progress is None or infer_steps <= 0:
        return None, None
    per_step = self._estimate_diffusion_per_step(
        infer_steps=infer_steps,
        batch_size=batch_size,
        duration_sec=duration_sec,
    ) or self._last_diffusion_per_step_sec
    if not per_step or per_step <= 0:
        return None, None
    expected = per_step * infer_steps
    if expected <= 0:
        return None, None
    stop_event = threading.Event()

    def _runner():
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            frac = min(0.999, elapsed / expected)
            value = start + (end - start) * frac
            try:
                progress(value, desc=desc)
            except Exception:
                pass
            stop_event.wait(0.5)

    thread = threading.Thread(target=_runner, name="diffusion-progress", daemon=True)
    thread.start()
    return stop_event, thread

# --- From prompt_utils.py ---

def _format_instruction(self, instruction: str) -> str:
    """Ensure instruction ends with a colon."""
    if not instruction.endswith(":"):
        instruction = instruction + ":"
    return instruction

def _format_lyrics(self, lyrics: str, language: str) -> str:
    """Format lyrics text with language header."""
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

def _pad_sequences(
    self, sequences: list[torch.Tensor], max_length: int, pad_value: int = 0
) -> torch.Tensor:
    """Pad sequence tensors to the same length."""
    return torch.stack(
        [
            torch.nn.functional.pad(seq, (0, max_length - len(seq)), "constant", pad_value)
            for seq in sequences
        ]
    )

def extract_caption_from_sft_format(self, caption: str) -> str:
    """Extract caption body from SFT-formatted prompt when present."""
    try:
        if "# Instruction" in caption and "# Caption" in caption:
            pattern = r"#\s*Caption\s*\n(.*?)(?:\n\s*#\s*Metas|$)"
            match = re.search(pattern, caption, re.DOTALL)
            if match:
                return match.group(1).strip()
        return caption
    except (AttributeError, TypeError, re.error):
        logger.exception("[extract_caption_from_sft_format] Error extracting caption")
        return caption

def build_dit_inputs(
    self,
    task: str,
    instruction: str | None,
    caption: str,
    lyrics: str,
    metas: str | dict[str, Any] | None = None,
    vocal_language: str = "en",
) -> tuple[str, str]:
    """Build caption and lyric input text for DiT branches.

    Args:
        task: Task name (currently informational; reserved for task-specific formatting).
        instruction: Instruction text; falls back to default when empty.
        caption: Caption fallback value.
        lyrics: Raw lyric text.
        metas: Optional metadata (string or dict) that may include caption/language.
        vocal_language: Fallback lyric language when not present in metadata.

    Returns:
        Tuple of ``(caption_input, lyrics_input)`` for caption and lyric encoder branches.
    """
    final_instruction = self._format_instruction(instruction or DEFAULT_DIT_INSTRUCTION)
    actual_caption = caption
    actual_language = vocal_language

    if metas is not None:
        try:
            if isinstance(metas, str):
                parsed_metas = self._parse_metas([metas])
                meta_dict = parsed_metas[0] if parsed_metas and isinstance(parsed_metas[0], dict) else {}
            elif isinstance(metas, dict):
                meta_dict = metas
            else:
                meta_dict = {}
        except (TypeError, ValueError, KeyError, IndexError):
            logger.exception("[build_dit_inputs] Error parsing metas")
            meta_dict = {}
        if "caption" in meta_dict and meta_dict["caption"]:
            actual_caption = str(meta_dict["caption"])
        if "language" in meta_dict and meta_dict["language"]:
            actual_language = str(meta_dict["language"])

    parsed_meta = self._parse_metas([metas])[0]
    caption_input = SFT_GEN_PROMPT.format(final_instruction, actual_caption, parsed_meta)
    lyrics_input = self._format_lyrics(lyrics, actual_language)
    return caption_input, lyrics_input

def _get_text_hidden_states(self, text_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get hidden states and attention mask from text encoder."""
    if self.text_tokenizer is None or self.text_encoder is None:
        raise ValueError("Text encoder not initialized")

    try:
        with self._load_model_context("text_encoder"):
            text_inputs = self.text_tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            text_attention_mask = text_inputs.attention_mask.to(self.device).bool()

            with torch.inference_mode():
                text_outputs = self.text_encoder(text_input_ids)
                if hasattr(text_outputs, "last_hidden_state"):
                    text_hidden_states = text_outputs.last_hidden_state
                elif isinstance(text_outputs, tuple):
                    text_hidden_states = text_outputs[0]
                else:
                    text_hidden_states = text_outputs

            text_hidden_states = text_hidden_states.to(self.dtype)
            return text_hidden_states, text_attention_mask
    except (AttributeError, RuntimeError, TypeError, ValueError):
        logger.exception("[_get_text_hidden_states] Failed to encode text prompt")
        raise

def _extract_caption_and_language(
    self,
    metas: list[str | dict[str, Any]],
    captions: list[str],
    vocal_languages: list[str],
) -> tuple[list[str], list[str]]:
    """Extract caption/language values from metas with fallback values."""
    actual_captions = list(captions)
    actual_languages = list(vocal_languages)

    for i, meta in enumerate(metas):
        if i >= len(actual_captions):
            break

        meta_dict = None
        if isinstance(meta, str):
            parsed = self._parse_metas([meta])
            if parsed and isinstance(parsed[0], dict):
                meta_dict = parsed[0]
        elif isinstance(meta, dict):
            meta_dict = meta

        if meta_dict:
            if "caption" in meta_dict and meta_dict["caption"]:
                actual_captions[i] = str(meta_dict["caption"])
            if "language" in meta_dict and meta_dict["language"]:
                actual_languages[i] = str(meta_dict["language"])
    return actual_captions, actual_languages

# --- From task_utils.py ---

def prepare_seeds(
    self, actual_batch_size: int, seed, use_random_seed: bool
) -> tuple[list[int], str]:
    """Prepare per-item seeds and UI seed string."""
    actual_seed_list: list[int] = []
    seed_value_for_ui = ""
    try:
        if use_random_seed:
            actual_seed_list = [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size)]
            seed_value_for_ui = ", ".join(str(s) for s in actual_seed_list)
        else:
            seed_list: list[int] = []
            if isinstance(seed, str):
                for s in [s.strip() for s in seed.split(",")]:
                    if s == "-1" or s == "":
                        seed_list.append(-1)
                    else:
                        try:
                            seed_list.append(int(float(s)))
                        except (ValueError, TypeError) as exc:
                            logger.debug(f"[prepare_seeds] Failed to parse seed value '{s}': {exc}")
                            seed_list.append(-1)
            elif seed is None or (isinstance(seed, (int, float)) and seed < 0):
                seed_list = [-1] * actual_batch_size
            elif isinstance(seed, (int, float)):
                seed_list = [int(seed)]
            else:
                seed_list = [-1] * actual_batch_size

            has_single_non_negative_seed = len(seed_list) == 1 and seed_list[0] != -1
            for i in range(actual_batch_size):
                seed_val = seed_list[i] if i < len(seed_list) else -1
                if has_single_non_negative_seed and actual_batch_size > 1 and i > 0:
                    actual_seed_list.append(random.randint(0, 2**32 - 1))
                elif seed_val == -1:
                    actual_seed_list.append(random.randint(0, 2**32 - 1))
                else:
                    actual_seed_list.append(int(seed_val))
            seed_value_for_ui = ", ".join(str(s) for s in actual_seed_list)
    except (TypeError, ValueError, OverflowError):
        logger.exception("[prepare_seeds] Failed to prepare seeds")
        actual_seed_list = [random.randint(0, 2**32 - 1) for _ in range(actual_batch_size)]
        seed_value_for_ui = ", ".join(str(s) for s in actual_seed_list)

    return actual_seed_list, seed_value_for_ui

def generate_instruction(
    self,
    task_type: str,
    track_name: str | None = None,
    complete_track_classes: list[str] | None = None,
) -> str:
    """Generate task instruction text from task type and track context."""
    if task_type == "text2music":
        return TASK_INSTRUCTIONS["text2music"]
    if task_type == "repaint":
        return TASK_INSTRUCTIONS["repaint"]
    if task_type == "cover":
        return TASK_INSTRUCTIONS["cover"]
    if task_type == "extract":
        return (
            TASK_INSTRUCTIONS["extract"].format(TRACK_NAME=track_name.upper())
            if track_name
            else TASK_INSTRUCTIONS["extract_default"]
        )
    if task_type == "lego":
        return (
            TASK_INSTRUCTIONS["lego"].format(TRACK_NAME=track_name.upper())
            if track_name
            else TASK_INSTRUCTIONS["lego_default"]
        )
    if task_type == "complete":
        if complete_track_classes and len(complete_track_classes) > 0:
            track_classes_upper = [t.upper() for t in complete_track_classes]
            return TASK_INSTRUCTIONS["complete"].format(
                TRACK_CLASSES=" | ".join(track_classes_upper)
            )
        return TASK_INSTRUCTIONS["complete_default"]
    return TASK_INSTRUCTIONS["text2music"]

def determine_task_type(self, task_type, audio_code_string):
    """Compute task-mode booleans for downstream generation logic."""
    is_repaint_task = task_type == "repaint"
    is_lego_task = task_type == "lego"
    is_cover_task = task_type == "cover"

    if isinstance(audio_code_string, list):
        has_codes = any((c or "").strip() for c in audio_code_string)
    else:
        has_codes = bool(audio_code_string and str(audio_code_string).strip())

    if has_codes:
        is_cover_task = True
    can_use_repainting = is_repaint_task or is_lego_task
    return is_repaint_task, is_lego_task, is_cover_task, can_use_repainting

def create_target_wavs(self, duration_seconds: float) -> torch.Tensor:
    """Create silent stereo target audio with safe duration handling."""
    try:
        duration_seconds = max(0.1, round(duration_seconds, 1))
        frames = int(duration_seconds * SAMPLE_RATE)
        return torch.zeros(2, frames)
    except (TypeError, ValueError, OverflowError):
        logger.exception("[create_target_wavs] Error creating target audio")
        return torch.zeros(2, 30 * SAMPLE_RATE)

# --- From training_preset.py ---

def switch_to_training_preset(self) -> tuple[str, bool]:
    """Reinitialize with quantization disabled using the last successful init parameters.

    Returns:
        Tuple[str, bool]:
            - A human-readable status message for UI/API consumers.
            - ``True`` when the preset is already safe or reinitialization succeeds,
              otherwise ``False``.
    """
    if self.quantization is None:
        return "Already in training-safe preset (quantization disabled).", True

    if not self.last_init_params:
        return "Cannot switch preset automatically: no previous init parameters found.", False

    params = dict(self.last_init_params)
    params["quantization"] = None

    status, ok = self.initialize_service(
        project_root=params["project_root"],
        config_path=params["config_path"],
        device=params["device"],
        use_flash_attention=params["use_flash_attention"],
        compile_model=params["compile_model"],
        offload_to_cpu=params["offload_to_cpu"],
        offload_dit_to_cpu=params["offload_dit_to_cpu"],
        quantization=None,
        prefer_source=params.get("prefer_source"),
        use_mlx_dit=params.get("use_mlx_dit", True),
    )
    if ok:
        return f"Switched to training preset (quantization disabled).\n{status}", True
    return f"Failed to switch to training preset.\n{status}", False

# --- From lyric_alignment_common.py ---

def _resolve_custom_layers_config(
    self, custom_layers_config: dict[int, list[int]] | None
) -> dict[int, list[int]]:
    """Return caller config when provided, otherwise host default config."""
    if custom_layers_config is not None:
        return custom_layers_config
    return self.custom_layers_config

def _move_alignment_inputs_to_runtime(
    self,
    pred_latent: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move alignment tensors to the handler runtime device and dtype."""
    device = self.device
    dtype = self.dtype
    return (
        pred_latent.to(device=device, dtype=dtype),
        encoder_hidden_states.to(device=device, dtype=dtype),
        encoder_attention_mask.to(device=device, dtype=dtype),
        context_latents.to(device=device, dtype=dtype),
    )

def _sample_noise_like(self, reference: torch.Tensor, seed: int | None) -> torch.Tensor:
    """Sample deterministic noise for a tensor shape, including MPS-safe seeding."""
    if seed is None:
        return torch.randn_like(reference)

    device = reference.device
    dtype = reference.dtype
    is_mps = (isinstance(device, str) and device == "mps") or (
        hasattr(device, "type") and device.type == "mps"
    )
    gen_device = "cpu" if is_mps else device
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))
    return torch.randn(reference.shape, generator=generator, device=gen_device, dtype=dtype).to(device)

def _extract_lyric_segment(
    self,
    lyric_token_ids: torch.Tensor,
    vocal_language: str,
) -> tuple[Sequence[int], list[int], int, int]:
    """Split token ids into header and lyric ranges."""
    raw_lyric_ids: Sequence[int]
    if isinstance(lyric_token_ids, torch.Tensor):
        raw_lyric_ids = lyric_token_ids[0].tolist()
    else:
        raw_lyric_ids = lyric_token_ids

    header_str = f"# Languages\n{vocal_language}\n\n# Lyric\n"
    header_ids = self.text_tokenizer.encode(header_str, add_special_tokens=False)
    start_idx = len(header_ids)
    try:
        end_idx = raw_lyric_ids.index(151643)  # <|endoftext|>
    except ValueError:
        end_idx = len(raw_lyric_ids)

    pure_lyric_ids = list(raw_lyric_ids[start_idx:end_idx])
    return raw_lyric_ids, pure_lyric_ids, start_idx, end_idx

def _lyric_timestamp_error(self, message: str) -> dict[str, Any]:
    """Build the standard timestamp error payload."""
    return {
        "lrc_text": "",
        "sentence_timestamps": [],
        "token_timestamps": [],
        "success": False,
        "error": message,
    }

def _lyric_score_error(self, message: str) -> dict[str, Any]:
    """Build the standard lyric-score error payload."""
    return {
        "lm_score": 0.0,
        "dit_score": 0.0,
        "success": False,
        "error": message,
    }

# --- From lyric_score.py ---

@torch.inference_mode()
def get_lyric_score(
    self,
    pred_latent: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    lyric_token_ids: torch.Tensor,
    vocal_language: str = "en",
    inference_steps: int = 8,
    seed: int = 42,
    custom_layers_config: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    """Calculate lyric alignment scores for pure-noise and regressed-latent inputs.

    Args:
        pred_latent (torch.Tensor): Generated latent tensor shaped ``[B, T, D]``.
        encoder_hidden_states (torch.Tensor): Decoder conditioning states.
        encoder_attention_mask (torch.Tensor): Conditioning attention mask.
        context_latents (torch.Tensor): Context latents aligned to ``pred_latent``.
        lyric_token_ids (torch.Tensor): Tokenized lyric ids for alignment slicing.
        vocal_language (str): Language tag used to parse lyric header tokens.
        inference_steps (int): Positive diffusion step count for ``t_last``.
        seed (int): Noise seed used for deterministic score sampling.
        custom_layers_config (Optional[Dict[int, List[int]]]): Optional attention layer/head map.

    Returns:
        Dict[str, Any]: Score payload with ``lm_score``, ``dit_score``, ``success``, and ``error``.

    Raises:
        Exception: Unexpected runtime failures are re-raised after logging.
    """
    if self.model is None:
        return self._lyric_score_error("Model not initialized")

    custom_layers_config = self._resolve_custom_layers_config(custom_layers_config)

    try:
        (
            pred_latent,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
        ) = self._move_alignment_inputs_to_runtime(
            pred_latent=pred_latent,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )

        bsz = pred_latent.shape[0]
        if not isinstance(inference_steps, int) or inference_steps <= 0:
            raise ValueError(
                f"inference_steps must be a positive non-zero integer, got {inference_steps!r}"
            )

        x0 = self._sample_noise_like(pred_latent, seed)
        t_last_val = 1.0 / inference_steps

        xt_lm = x0
        xt_dit = t_last_val * x0 + (1.0 - t_last_val) * pred_latent
        xt_in = torch.cat([xt_lm, xt_dit], dim=0)
        t_in = torch.cat(
            [
                torch.tensor([1.0] * bsz, device=pred_latent.device, dtype=pred_latent.dtype),
                torch.tensor([t_last_val] * bsz, device=pred_latent.device, dtype=pred_latent.dtype),
            ],
            dim=0,
        )
        encoder_hidden_states_in = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
        encoder_attention_mask_in = torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0)
        context_latents_in = torch.cat([context_latents, context_latents], dim=0)
        attention_mask_in = torch.ones(
            2 * bsz,
            xt_in.shape[1],
            device=pred_latent.device,
            dtype=pred_latent.dtype,
        )

        with self._load_model_context("model"):
            decoder = self.model.decoder
            if hasattr(decoder, "eval"):
                decoder.eval()
            decoder_outputs = decoder(
                hidden_states=xt_in,
                timestep=t_in,
                timestep_r=t_in,
                attention_mask=attention_mask_in,
                encoder_hidden_states=encoder_hidden_states_in,
                use_cache=False,
                past_key_values=None,
                encoder_attention_mask=encoder_attention_mask_in,
                context_latents=context_latents_in,
                output_attentions=True,
                custom_layers_config=custom_layers_config,
                enable_early_exit=True,
            )

        if decoder_outputs[2] is None:
            return self._lyric_score_error("Model did not return attentions")

        captured_layers = []
        for layer_attn in decoder_outputs[2]:
            if layer_attn is None:
                continue
            captured_layers.append(layer_attn.transpose(-1, -2))
        if not captured_layers:
            return self._lyric_score_error("No valid attention layers returned")

        stacked = torch.stack(captured_layers)
        all_layers_matrix_lm = stacked[:, :bsz, ...]
        all_layers_matrix_dit = stacked[:, bsz:, ...]
        if bsz == 1:
            all_layers_matrix_lm = all_layers_matrix_lm.squeeze(1)
            all_layers_matrix_dit = all_layers_matrix_dit.squeeze(1)

        _, pure_lyric_ids, start_idx, end_idx = self._extract_lyric_segment(
            lyric_token_ids=lyric_token_ids,
            vocal_language=vocal_language,
        )
        if start_idx >= all_layers_matrix_lm.shape[-2]:
            return self._lyric_score_error("Lyrics indices out of bounds")

        pure_matrix_lm = all_layers_matrix_lm[..., start_idx:end_idx, :]
        pure_matrix_dit = all_layers_matrix_dit[..., start_idx:end_idx, :]

        from acestep.core.scoring.dit_score import MusicLyricScorer

        aligner = MusicLyricScorer(self.text_tokenizer)
        lm_score = self._calculate_single_lyric_score(
            aligner=aligner,
            matrix=pure_matrix_lm,
            pure_lyric_ids=pure_lyric_ids,
            custom_layers_config=custom_layers_config,
        )
        dit_score = self._calculate_single_lyric_score(
            aligner=aligner,
            matrix=pure_matrix_dit,
            pure_lyric_ids=pure_lyric_ids,
            custom_layers_config=custom_layers_config,
        )
        return {
            "lm_score": lm_score,
            "dit_score": dit_score,
            "success": True,
            "error": None,
        }
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        logger.exception("[get_lyric_score] Failed")
        return self._lyric_score_error(f"Error generating score: {exc}")
    except Exception:
        logger.exception("[get_lyric_score] Unexpected failure")
        raise

def _calculate_single_lyric_score(
    self,
    aligner: Any,
    matrix: torch.Tensor,
    pure_lyric_ids: list[int],
    custom_layers_config: dict[int, list[int]],
) -> float:
    """Run one alignment-score evaluation against one attention matrix."""
    info = aligner.lyrics_alignment_info(
        attention_matrix=matrix,
        token_ids=pure_lyric_ids,
        custom_config=custom_layers_config,
        return_matrices=False,
        medfilt_width=1,
    )
    if info.get("energy_matrix") is None:
        return 0.0
    res = aligner.calculate_score(
        energy_matrix=info["energy_matrix"],
        type_mask=info["type_mask"],
        path_coords=info["path_coords"],
    )
    return float(res.get("lyrics_score", res.get("final_score", 0.0)))

# --- From lyric_timestamp.py ---

@torch.inference_mode()
def get_lyric_timestamp(
    self,
    pred_latent: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    lyric_token_ids: torch.Tensor,
    total_duration_seconds: float,
    vocal_language: str = "en",
    inference_steps: int = 8,
    seed: int = 42,
    custom_layers_config: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    """Generate LRC timestamps by aligning decoder cross-attention to lyric tokens.

    Args:
        pred_latent (torch.Tensor): Generated latent tensor shaped ``[B, T, D]``.
        encoder_hidden_states (torch.Tensor): Decoder conditioning states.
        encoder_attention_mask (torch.Tensor): Conditioning attention mask.
        context_latents (torch.Tensor): Context latents aligned to ``pred_latent``.
        lyric_token_ids (torch.Tensor): Tokenized lyric sequence including header tokens.
        total_duration_seconds (float): Audio duration used to scale timestamp output.
        vocal_language (str): Language tag used to locate lyric header boundary.
        inference_steps (int): Positive diffusion step count for ``t_last``.
        seed (int): Noise seed used for deterministic timestamp sampling.
        custom_layers_config (Optional[Dict[int, List[int]]]): Optional attention layer/head map.

    Returns:
        Dict[str, Any]: Timestamp payload with ``lrc_text``, token/sentence timestamps, ``success``, and ``error``.

    Raises:
        Exception: Unexpected runtime failures are re-raised after logging.
    """
    if self.model is None:
        return self._lyric_timestamp_error("Model not initialized")

    custom_layers_config = self._resolve_custom_layers_config(custom_layers_config)

    try:
        (
            pred_latent,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
        ) = self._move_alignment_inputs_to_runtime(
            pred_latent=pred_latent,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
        bsz = pred_latent.shape[0]
        if not isinstance(inference_steps, int) or inference_steps <= 0:
            return self._lyric_timestamp_error(
                f"inference_steps must be a positive non-zero integer, got {inference_steps!r}"
            )

        t_last_val = 1.0 / inference_steps
        t_tensor = torch.tensor([t_last_val] * bsz, device=pred_latent.device, dtype=pred_latent.dtype)
        noise = self._sample_noise_like(pred_latent, seed)
        xt = t_last_val * noise + (1.0 - t_last_val) * pred_latent
        attention_mask = torch.ones(bsz, pred_latent.shape[1], device=pred_latent.device, dtype=pred_latent.dtype)

        with self._load_model_context("model"):
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t_tensor,
                timestep_r=t_tensor,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
                past_key_values=None,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                output_attentions=True,
                custom_layers_config=custom_layers_config,
                enable_early_exit=True,
            )

        if decoder_outputs[2] is None:
            return self._lyric_timestamp_error("Model did not return attentions")

        captured_layers = []
        for layer_attn in decoder_outputs[2]:
            if layer_attn is None:
                continue
            captured_layers.append(layer_attn[:bsz].transpose(-1, -2))
        if not captured_layers:
            return self._lyric_timestamp_error("No valid attention layers returned")

        stacked = torch.stack(captured_layers)
        all_layers_matrix = stacked.squeeze(1) if bsz == 1 else stacked

        _, pure_lyric_ids, start_idx, end_idx = self._extract_lyric_segment(
            lyric_token_ids=lyric_token_ids,
            vocal_language=vocal_language,
        )
        pure_lyric_matrix = all_layers_matrix[:, :, start_idx:end_idx, :]

        from acestep.core.scoring.dit_alignment import MusicStampsAligner

        aligner = MusicStampsAligner(self.text_tokenizer)
        align_info = aligner.stamps_align_info(
            attention_matrix=pure_lyric_matrix,
            lyrics_tokens=pure_lyric_ids,
            total_duration_seconds=total_duration_seconds,
            custom_config=custom_layers_config,
            return_matrices=False,
            violence_level=2.0,
            medfilt_width=1,
        )
        if align_info.get("calc_matrix") is None:
            return self._lyric_timestamp_error(
                align_info.get("error", "Failed to process attention matrix")
            )

        result = aligner.get_timestamps_and_lrc(
            calc_matrix=align_info["calc_matrix"],
            lyrics_tokens=pure_lyric_ids,
            total_duration_seconds=total_duration_seconds,
        )
        return {
            "lrc_text": result["lrc_text"],
            "sentence_timestamps": result["sentence_timestamps"],
            "token_timestamps": result["token_timestamps"],
            "success": True,
            "error": None,
        }
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        logger.exception("[get_lyric_timestamp] Failed")
        return self._lyric_timestamp_error(f"Error generating timestamps: {exc}")
    except Exception:
        logger.exception("[get_lyric_timestamp] Unexpected failure")
        raise

# --- From lora_manager.py ---

_ensure_lora_registry = ensure_lora_registry
_sync_lora_state_from_service = sync_lora_state_from_service
_debug_lora_registry_snapshot = debug_lora_registry_snapshot
_collect_adapter_names = collect_adapter_names

_rebuild_lora_registry = rebuild_lora_registry
_apply_scale_to_adapter = apply_scale_to_adapter

add_lora = add_lora
add_voice_lora = add_voice_lora
load_lora = load_lora
remove_lora = remove_lora
unload_lora = unload_lora
set_use_lora = set_use_lora
set_lora_scale = set_lora_scale
set_active_lora_adapter = set_active_lora_adapter
get_lora_status = get_lora_status
