"""Consolidated handler functions – generation module."""

import traceback
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from acestep.constants import DEFAULT_DIT_INSTRUCTION
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from acestep.constants import TASK_INSTRUCTIONS
from typing import Any, Dict, List, Optional, Sequence
import os
import time
from typing import Any, Dict, Optional, Tuple
from acestep.gpu_config import get_effective_free_vram_gb
from typing import Any, Dict
import random
from typing import Any, Dict, List, Optional, Tuple
from typing import Any, Dict, Optional
from acestep.models.mlx.dit_generate import mlx_generate_diffusion

# Module-level code from service_generate_request.py
MAX_BATCH_SIZE = 8


# --- From generate_music.py ---

def generate_music(
    self,
    captions: str,
    lyrics: str,
    bpm: Optional[int] = None,
    key_scale: str = "",
    time_signature: str = "",
    vocal_language: str = "en",
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    use_random_seed: bool = True,
    seed: Optional[Union[str, float, int]] = -1,
    reference_audio=None,
    audio_duration: Optional[float] = None,
    batch_size: Optional[int] = None,
    src_audio=None,
    audio_code_string: Union[str, List[str]] = "",
    repainting_start: float = 0.0,
    repainting_end: Optional[float] = None,
    instruction: str = DEFAULT_DIT_INSTRUCTION,
    audio_cover_strength: float = 1.0,
    cover_noise_strength: float = 0.0,
    task_type: str = "text2music",
    use_adg: bool = False,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 1.0,
    infer_method: str = "ode",
    noise_schedule: str = "linear",
    use_tiled_decode: bool = True,
    timesteps: Optional[List[float]] = None,
    latent_shift: float = 0.0,
    latent_rescale: float = 1.0,
    progress=None,
) -> Dict[str, Any]:
    """Generate audio from text/reference inputs and return response payload.

    Args:
        captions: Text prompt describing requested music.
        lyrics: Lyric text used for conditioning.
        reference_audio: Optional reference-audio payload.
        src_audio: Optional source audio for repaint/cover.
        inference_steps: Diffusion step count.
        guidance_scale: CFG guidance value.
        seed: Optional explicit seed from caller/UI.
        infer_method: Diffusion method name.
        timesteps: Optional custom timestep schedule.
        use_tiled_decode: Whether tiled VAE decode is used.
        latent_shift: Additive latent post-processing value.
        latent_rescale: Multiplicative latent post-processing value.
        progress: Optional callback taking ``(ratio, desc=...)``.

    Returns:
        Dict[str, Any]: Standard payload with generated audio tensors, status,
        intermediate outputs, success flag, and optional error text.

    Raises:
        No exceptions are re-raised. Runtime failures are converted into the
        returned error payload.
    """
    progress = self._resolve_generate_music_progress(progress)
    if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
        readiness_error = self._validate_generate_music_readiness()
        return readiness_error

    task_type, instruction = self._resolve_generate_music_task(
        task_type=task_type,
        audio_code_string=audio_code_string,
        instruction=instruction,
    )

    logger.info("[generate_music] Starting generation...")
    if progress:
        progress(0.51, desc="Preparing inputs...")
    logger.info("[generate_music] Preparing inputs...")

    runtime = self._prepare_generate_music_runtime(
        batch_size=batch_size,
        audio_duration=audio_duration,
        repainting_end=repainting_end,
        seed=seed,
        use_random_seed=use_random_seed,
    )
    actual_batch_size = runtime["actual_batch_size"]
    actual_seed_list = runtime["actual_seed_list"]
    seed_value_for_ui = runtime["seed_value_for_ui"]
    audio_duration = runtime["audio_duration"]
    repainting_end = runtime["repainting_end"]

    try:
        refer_audios, processed_src_audio, audio_error = self._prepare_reference_and_source_audio(
            reference_audio=reference_audio,
            src_audio=src_audio,
            audio_code_string=audio_code_string,
            actual_batch_size=actual_batch_size,
            task_type=task_type,
        )
        if audio_error is not None:
            return audio_error

        service_inputs = self._prepare_generate_music_service_inputs(
            actual_batch_size=actual_batch_size,
            processed_src_audio=processed_src_audio,
            audio_duration=audio_duration,
            captions=captions,
            lyrics=lyrics,
            vocal_language=vocal_language,
            instruction=instruction,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=time_signature,
            task_type=task_type,
            audio_code_string=audio_code_string,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
        )
        service_run = self._run_generate_music_service_with_progress(
            progress=progress,
            actual_batch_size=actual_batch_size,
            audio_duration=audio_duration,
            inference_steps=inference_steps,
            timesteps=timesteps,
            service_inputs=service_inputs,
            refer_audios=refer_audios,
            guidance_scale=guidance_scale,
            actual_seed_list=actual_seed_list,
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            use_adg=use_adg,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=shift,
            infer_method=infer_method,
            noise_schedule=noise_schedule,
        )
        outputs = service_run["outputs"]
        infer_steps_for_progress = service_run["infer_steps_for_progress"]

        pred_latents, time_costs = self._prepare_generate_music_decode_state(
            outputs=outputs,
            infer_steps_for_progress=infer_steps_for_progress,
            actual_batch_size=actual_batch_size,
            audio_duration=audio_duration,
            latent_shift=latent_shift,
            latent_rescale=latent_rescale,
        )
        pred_wavs, pred_latents_cpu, time_costs = self._decode_generate_music_pred_latents(
            pred_latents=pred_latents,
            progress=progress,
            use_tiled_decode=use_tiled_decode,
            time_costs=time_costs,
        )
        return self._build_generate_music_success_payload(
            outputs=outputs,
            pred_wavs=pred_wavs,
            pred_latents_cpu=pred_latents_cpu,
            time_costs=time_costs,
            seed_value_for_ui=seed_value_for_ui,
            actual_batch_size=actual_batch_size,
            progress=progress,
        )
    except Exception as exc:
        error_msg = f"Error: {exc!s}\n{traceback.format_exc()}"
        logger.exception("[generate_music] Generation failed")
        return {
            "audios": [],
            "status_message": error_msg,
            "extra_outputs": {},
            "success": False,
            "error": f"{exc!s}",
        }

# --- From generate_music_request.py ---

def _resolve_generate_music_progress(
    self,
    progress: Optional[Callable[..., Any]],
) -> Callable[..., Any]:
    """Return a callable progress callback, defaulting to no-op."""
    if progress is not None:
        return progress

    def _progress(*args: Any, **kwargs: Any) -> Any:
        """No-op callback for non-UI call sites."""
        _ = args, kwargs
        return None

    return _progress

def _validate_generate_music_readiness(self) -> Optional[Dict[str, Any]]:
    """Return standardized error payload when model components are unavailable."""
    if self.model is None or self.vae is None or self.text_tokenizer is None or self.text_encoder is None:
        return {
            "audios": [],
            "status_message": "\u274c Model not fully initialized. Please initialize all components first.",
            "extra_outputs": {},
            "success": False,
            "error": "Model not fully initialized",
        }
    return None

def _has_non_empty_audio_codes(self, value: Union[str, List[str]]) -> bool:
    """Return ``True`` when at least one non-empty audio-code string is present."""
    if isinstance(value, list):
        return any((x or "").strip() for x in value)
    return bool(value and str(value).strip())

def _resolve_generate_music_task(
    self,
    task_type: str,
    audio_code_string: Union[str, List[str]],
    instruction: str,
) -> Tuple[str, str]:
    """Auto-switch text2music to cover task when audio codes are provided."""
    if task_type == "text2music" and self._has_non_empty_audio_codes(audio_code_string):
        return "cover", TASK_INSTRUCTIONS["cover"]
    return task_type, instruction

def _prepare_generate_music_runtime(
    self,
    batch_size: Optional[int],
    audio_duration: Optional[float],
    repainting_end: Optional[float],
    seed: Optional[Union[str, float, int]],
    use_random_seed: bool,
) -> Dict[str, Any]:
    """Prepare runtime batch/seed/duration values for generation."""
    self.current_offload_cost = 0.0
    actual_batch_size = batch_size if batch_size is not None else self.batch_size
    actual_batch_size = max(1, actual_batch_size)
    actual_batch_size = self._vram_guard_reduce_batch(actual_batch_size, audio_duration=audio_duration)
    actual_seed_list, seed_value_for_ui = self.prepare_seeds(actual_batch_size, seed, use_random_seed)

    if audio_duration is not None and float(audio_duration) <= 0:
        audio_duration = None
    if repainting_end is not None and float(repainting_end) < 0:
        repainting_end = None

    return {
        "actual_batch_size": actual_batch_size,
        "actual_seed_list": actual_seed_list,
        "seed_value_for_ui": seed_value_for_ui,
        "audio_duration": audio_duration,
        "repainting_end": repainting_end,
    }

def _prepare_reference_and_source_audio(
    self,
    reference_audio: Optional[str],
    src_audio: Optional[str],
    audio_code_string: Union[str, List[str]],
    actual_batch_size: int,
    task_type: str,
) -> Tuple[Optional[List[List[torch.Tensor]]], Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """Prepare reference/source audio tensors and return early error payload when invalid."""
    if reference_audio is not None:
        logger.info("[generate_music] Processing reference audio...")
        processed_ref_audio = self.process_reference_audio(reference_audio)
        if processed_ref_audio is None:
            return None, None, {
                "audios": [],
                "status_message": (
                    "Reference audio is invalid, unreadable, or silent. "
                    "Please upload a valid audible audio file."
                ),
                "extra_outputs": {},
                "success": False,
                "error": "Invalid reference audio",
            }
        refer_audios = [[processed_ref_audio] for _ in range(actual_batch_size)]
    else:
        refer_audios = [[torch.zeros(2, 30 * self.sample_rate)] for _ in range(actual_batch_size)]

    processed_src_audio = None
    if task_type == "text2music":
        if src_audio is not None:
            logger.info("[generate_music] text2music task does not use src_audio, ignoring")
    elif src_audio is not None:
        if self._has_non_empty_audio_codes(audio_code_string):
            logger.info("[generate_music] Audio codes provided, ignoring src_audio and using codes instead")
        else:
            logger.info("[generate_music] Processing source audio...")
            processed_src_audio = self.process_src_audio(src_audio)
            if processed_src_audio is None:
                logger.error("[generate_music] Source audio is invalid after processing")
                return None, None, {
                    "audios": [],
                    "status_message": (
                        "Source audio is invalid, unreadable, or silent. "
                        "Please upload a valid audible audio file."
                    ),
                    "extra_outputs": {},
                    "success": False,
                    "error": "Invalid source audio",
                }

    return refer_audios, processed_src_audio, None

def _prepare_generate_music_service_inputs(
    self,
    actual_batch_size: int,
    processed_src_audio: Optional[torch.Tensor],
    audio_duration: Optional[float],
    captions: str,
    lyrics: str,
    vocal_language: str,
    instruction: str,
    bpm: Optional[int],
    key_scale: str,
    time_signature: str,
    task_type: str,
    audio_code_string: Union[str, List[str]],
    repainting_start: float,
    repainting_end: Optional[float],
) -> Dict[str, Any]:
    """Prepare service inputs (batch text, repaint spans, and optional code hints)."""
    captions_batch, instructions_batch, lyrics_batch, vocal_languages_batch, metas_batch = self.prepare_batch_data(
        actual_batch_size,
        processed_src_audio,
        audio_duration,
        captions,
        lyrics,
        vocal_language,
        instruction,
        bpm,
        key_scale,
        time_signature,
    )

    is_repaint_task, is_lego_task, is_cover_task, can_use_repainting = self.determine_task_type(task_type, audio_code_string)
    repainting_start_batch, repainting_end_batch, target_wavs_tensor = self.prepare_padding_info(
        actual_batch_size,
        processed_src_audio,
        audio_duration,
        repainting_start,
        repainting_end,
        is_repaint_task,
        is_lego_task,
        is_cover_task,
        can_use_repainting,
    )
    audio_code_hints_batch = None
    if self._has_non_empty_audio_codes(audio_code_string):
        if isinstance(audio_code_string, list):
            audio_code_hints_batch = audio_code_string
        else:
            audio_code_hints_batch = [audio_code_string] * actual_batch_size

    return {
        "captions_batch": captions_batch,
        "instructions_batch": instructions_batch,
        "lyrics_batch": lyrics_batch,
        "vocal_languages_batch": vocal_languages_batch,
        "metas_batch": metas_batch,
        "repainting_start_batch": repainting_start_batch,
        "repainting_end_batch": repainting_end_batch,
        "target_wavs_tensor": target_wavs_tensor,
        "audio_code_hints_batch": audio_code_hints_batch,
        "should_return_intermediate": task_type == "text2music",
    }

# --- From generate_music_execute.py ---

def _run_generate_music_service_with_progress(
    self,
    progress: Any,
    actual_batch_size: int,
    audio_duration: Optional[float],
    inference_steps: int,
    timesteps: Optional[Sequence[float]],
    service_inputs: Dict[str, Any],
    refer_audios: Optional[List[Any]],
    guidance_scale: float,
    actual_seed_list: Optional[List[int]],
    audio_cover_strength: float,
    cover_noise_strength: float,
    use_adg: bool,
    cfg_interval_start: float,
    cfg_interval_end: float,
    shift: float,
    infer_method: str,
    noise_schedule: str = "linear",
) -> Dict[str, Any]:
    """Invoke ``service_generate`` while maintaining background progress estimation."""
    infer_steps_for_progress = len(timesteps) if timesteps else inference_steps
    progress_desc = f"Generating music (batch size: {actual_batch_size})..."
    progress(0.52, desc=progress_desc)
    stop_event = None
    progress_thread = None
    try:
        stop_event, progress_thread = self._start_diffusion_progress_estimator(
            progress=progress,
            start=0.52,
            end=0.79,
            infer_steps=infer_steps_for_progress,
            batch_size=actual_batch_size,
            duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
            desc=progress_desc,
        )
        outputs = self.service_generate(
            captions=service_inputs["captions_batch"],
            lyrics=service_inputs["lyrics_batch"],
            metas=service_inputs["metas_batch"],
            vocal_languages=service_inputs["vocal_languages_batch"],
            refer_audios=refer_audios,
            target_wavs=service_inputs["target_wavs_tensor"],
            infer_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed_list,
            repainting_start=service_inputs["repainting_start_batch"],
            repainting_end=service_inputs["repainting_end_batch"],
            instructions=service_inputs["instructions_batch"],
            audio_cover_strength=audio_cover_strength,
            cover_noise_strength=cover_noise_strength,
            use_adg=use_adg,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_end=cfg_interval_end,
            shift=shift,
            infer_method=infer_method,
            audio_code_hints=service_inputs["audio_code_hints_batch"],
            return_intermediate=service_inputs["should_return_intermediate"],
            timesteps=timesteps,
            noise_schedule=noise_schedule,
        )
    finally:
        if stop_event is not None:
            stop_event.set()
        if progress_thread is not None:
            progress_thread.join(timeout=1.0)
    return {"outputs": outputs, "infer_steps_for_progress": infer_steps_for_progress}

# --- From generate_music_decode.py ---

def _prepare_generate_music_decode_state(
    self,
    outputs: Dict[str, Any],
    infer_steps_for_progress: int,
    actual_batch_size: int,
    audio_duration: Optional[float],
    latent_shift: float,
    latent_rescale: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Collect decode inputs and validate raw diffusion latents.

    Args:
        outputs: ``service_generate`` output payload containing target latents and timings.
        infer_steps_for_progress: Effective diffusion step count for estimates.
        actual_batch_size: Effective generation batch size.
        audio_duration: Optional generation duration in seconds.
        latent_shift: Additive latent post-processing shift.
        latent_rescale: Multiplicative latent post-processing scale.

    Returns:
        Tuple containing validated ``pred_latents`` and mutable ``time_costs``.

    Raises:
        RuntimeError: If latents contain NaN/Inf values or collapse to all zeros.
    """
    logger.info("[generate_music] Model generation completed. Decoding latents...")
    pred_latents = outputs["target_latents"]
    time_costs = outputs["time_costs"]
    time_costs["offload_time_cost"] = self.current_offload_cost

    per_step = time_costs.get("diffusion_per_step_time_cost")
    if isinstance(per_step, (int, float)) and per_step > 0:
        self._last_diffusion_per_step_sec = float(per_step)
        self._update_progress_estimate(
            per_step_sec=float(per_step),
            infer_steps=infer_steps_for_progress,
            batch_size=actual_batch_size,
            duration_sec=audio_duration if audio_duration and audio_duration > 0 else None,
        )

    if self.debug_stats:
        logger.debug(
            f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype} "
            f"{pred_latents.min()=}, {pred_latents.max()=}, {pred_latents.mean()=} "
            f"{pred_latents.std()=}"
        )
    else:
        logger.debug(f"[generate_music] pred_latents: {pred_latents.shape}, dtype={pred_latents.dtype}")
    logger.debug(f"[generate_music] time_costs: {time_costs}")

    if torch.isnan(pred_latents).any() or torch.isinf(pred_latents).any():
        raise RuntimeError(
            "Generation produced NaN or Inf latents. "
            "This usually indicates a checkpoint/config mismatch "
            "or unsupported quantization/backend combination. "
            "Try running with --backend pt or verify your model checkpoints match this release."
        )
    if pred_latents.numel() > 0 and pred_latents.abs().sum() == 0:
        raise RuntimeError(
            "Generation produced zero latents. "
            "This usually indicates a checkpoint/config mismatch or unsupported setup."
        )
    if latent_shift != 0.0 or latent_rescale != 1.0:
        logger.info(
            f"[generate_music] Applying latent post-processing: shift={latent_shift}, "
            f"rescale={latent_rescale}"
        )
        if self.debug_stats:
            logger.debug(
                f"[generate_music] Latent BEFORE shift/rescale: min={pred_latents.min():.4f}, "
                f"max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, "
                f"std={pred_latents.std():.4f}"
            )
        pred_latents = pred_latents * latent_rescale + latent_shift
        if self.debug_stats:
            logger.debug(
                f"[generate_music] Latent AFTER shift/rescale: min={pred_latents.min():.4f}, "
                f"max={pred_latents.max():.4f}, mean={pred_latents.mean():.4f}, "
                f"std={pred_latents.std():.4f}"
            )
    return pred_latents, time_costs

def _decode_generate_music_pred_latents(
    self,
    pred_latents: torch.Tensor,
    progress: Any,
    use_tiled_decode: bool,
    time_costs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Decode predicted latents and update decode timing metrics.

    Args:
        pred_latents: Predicted latent tensor shaped ``[batch, frames, dim]``.
        progress: Optional progress callback.
        use_tiled_decode: Whether tiled VAE decode should be used.
        time_costs: Mutable time-cost payload from service generation.

    Returns:
        Tuple of decoded waveforms, CPU latents, and updated time-cost payload.
    """
    if progress:
        progress(0.8, desc="Decoding audio...")
    logger.info("[generate_music] Decoding latents with VAE...")
    start_time = time.time()
    with torch.inference_mode():
        with self._load_model_context("vae"):
            pred_latents_cpu = pred_latents.detach().cpu()
            pred_latents_for_decode = pred_latents.transpose(1, 2).contiguous().to(self.vae.dtype)
            del pred_latents
            self._empty_cache()

            logger.debug(
                "[generate_music] Before VAE decode: "
                f"allocated={self._memory_allocated()/1024**3:.2f}GB, "
                f"max={self._max_memory_allocated()/1024**3:.2f}GB"
            )
            using_mlx_vae = self.use_mlx_vae and self.mlx_vae is not None
            vae_cpu = False
            vae_device = None
            if not using_mlx_vae:
                vae_cpu = os.environ.get("ACESTEP_VAE_ON_CPU", "0").lower() in ("1", "true", "yes")
                if not vae_cpu:
                    if self.device == "mps":
                        logger.info(
                            "[generate_music] MPS device: skipping VRAM check "
                            "(unified memory), keeping VAE on MPS"
                        )
                    else:
                        effective_free = get_effective_free_vram_gb()
                        logger.info(
                            "[generate_music] Effective free VRAM before VAE decode: "
                            f"{effective_free:.2f} GB"
                        )
                        if effective_free < 0.5:
                            logger.warning(
                                "[generate_music] Only "
                                f"{effective_free:.2f} GB free VRAM; auto-enabling CPU VAE decode"
                            )
                            vae_cpu = True
                if vae_cpu:
                    logger.info("[generate_music] Moving VAE to CPU for decode (ACESTEP_VAE_ON_CPU=1)...")
                    vae_device = next(self.vae.parameters()).device
                    self.vae = self.vae.cpu()
                    pred_latents_for_decode = pred_latents_for_decode.cpu()
                    self._empty_cache()
            try:
                if use_tiled_decode:
                    logger.info("[generate_music] Using tiled VAE decode to reduce VRAM usage...")
                    pred_wavs = self.tiled_decode(pred_latents_for_decode)
                elif using_mlx_vae:
                    try:
                        pred_wavs = self._mlx_vae_decode(pred_latents_for_decode)
                    except Exception as exc:
                        logger.warning(
                            f"[generate_music] MLX direct decode failed ({exc}), falling back to PyTorch"
                        )
                        decoder_output = self.vae.decode(pred_latents_for_decode)
                        pred_wavs = decoder_output.sample
                        del decoder_output
                else:
                    decoder_output = self.vae.decode(pred_latents_for_decode)
                    pred_wavs = decoder_output.sample
                    del decoder_output
            finally:
                if vae_cpu and vae_device is not None:
                    logger.info("[generate_music] Restoring VAE to original device after CPU decode path...")
                    self.vae = self.vae.to(vae_device)
                    pred_latents_for_decode = pred_latents_for_decode.to(vae_device)
                self._empty_cache()
            logger.debug(
                "[generate_music] After VAE decode: "
                f"allocated={self._memory_allocated()/1024**3:.2f}GB, "
                f"max={self._max_memory_allocated()/1024**3:.2f}GB"
            )
            del pred_latents_for_decode
            if pred_wavs.dtype != torch.float32:
                pred_wavs = pred_wavs.float()
            peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
            if torch.any(peak > 1.0):
                pred_wavs = pred_wavs / peak.clamp(min=1.0)
            self._empty_cache()
    end_time = time.time()
    time_costs["vae_decode_time_cost"] = end_time - start_time
    time_costs["total_time_cost"] = time_costs["total_time_cost"] + time_costs["vae_decode_time_cost"]
    time_costs["offload_time_cost"] = self.current_offload_cost
    return pred_wavs, pred_latents_cpu, time_costs

# --- From generate_music_payload.py ---

def _build_generate_music_success_payload(
    self,
    outputs: Dict[str, Any],
    pred_wavs,
    pred_latents_cpu,
    time_costs: Dict[str, Any],
    seed_value_for_ui: int,
    actual_batch_size: int,
    progress: Any,
) -> Dict[str, Any]:
    """Assemble final success response from decoded tensors and model outputs.

    Args:
        outputs: Service output payload containing intermediate generation tensors.
        pred_wavs: Decoded waveform tensor shaped ``[batch, channels, samples]``.
        pred_latents_cpu: CPU latent tensor preserved for extra outputs.
        time_costs: Updated time-cost payload including decode/offload timings.
        seed_value_for_ui: Seed value displayed in UI outputs.
        actual_batch_size: Effective generation batch size.
        progress: Optional progress callback.

    Returns:
        Dict[str, Any]: Standard success payload returned by ``generate_music``.
    """
    logger.info("[generate_music] VAE decode completed. Preparing audio tensors...")
    if progress:
        progress(0.99, desc="Preparing audio data...")

    audio_tensors = []
    for index in range(actual_batch_size):
        audio_tensor = pred_wavs[index].cpu()
        audio_tensors.append(audio_tensor)

    status_message = "Generation completed successfully!"
    logger.info(f"[generate_music] Done! Generated {len(audio_tensors)} audio tensors.")

    src_latents = outputs.get("src_latents")
    target_latents_input = outputs.get("target_latents_input")
    chunk_masks = outputs.get("chunk_masks")
    spans = outputs.get("spans", [])
    latent_masks = outputs.get("latent_masks")

    encoder_hidden_states = outputs.get("encoder_hidden_states")
    encoder_attention_mask = outputs.get("encoder_attention_mask")
    context_latents = outputs.get("context_latents")
    lyric_token_idss = outputs.get("lyric_token_idss")

    extra_outputs = {
        "pred_latents": pred_latents_cpu,
        "target_latents": target_latents_input.detach().cpu() if target_latents_input is not None else None,
        "src_latents": src_latents.detach().cpu() if src_latents is not None else None,
        "chunk_masks": chunk_masks.detach().cpu() if chunk_masks is not None else None,
        "latent_masks": latent_masks.detach().cpu() if latent_masks is not None else None,
        "spans": spans,
        "time_costs": time_costs,
        "seed_value": seed_value_for_ui,
        "encoder_hidden_states": (
            encoder_hidden_states.detach().cpu()
            if encoder_hidden_states is not None
            else None
        ),
        "encoder_attention_mask": (
            encoder_attention_mask.detach().cpu()
            if encoder_attention_mask is not None
            else None
        ),
        "context_latents": context_latents.detach().cpu() if context_latents is not None else None,
        "lyric_token_idss": lyric_token_idss.detach().cpu() if lyric_token_idss is not None else None,
    }

    audios = []
    for audio_tensor in audio_tensors:
        audios.append({"tensor": audio_tensor, "sample_rate": self.sample_rate})

    return {
        "audios": audios,
        "status_message": status_message,
        "extra_outputs": extra_outputs,
        "success": True,
        "error": None,
    }

# --- From service_generate.py ---

@torch.inference_mode()
def service_generate(
    self,
    captions: Union[str, List[str]],
    lyrics: Union[str, List[str]],
    keys: Optional[Union[str, List[str]]] = None,
    target_wavs: Optional[torch.Tensor] = None,
    refer_audios: Optional[List[List[torch.Tensor]]] = None,
    metas: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]] = None,
    vocal_languages: Optional[Union[str, List[str]]] = None,
    infer_steps: int = 60,
    guidance_scale: float = 7.0,
    seed: Optional[Union[int, List[int]]] = None,
    return_intermediate: bool = False,
    repainting_start: Optional[Union[float, List[float]]] = None,
    repainting_end: Optional[Union[float, List[float]]] = None,
    instructions: Optional[Union[str, List[str]]] = None,
    audio_cover_strength: float = 1.0,
    cover_noise_strength: float = 0.0,
    use_adg: bool = False,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    shift: float = 1.0,
    audio_code_hints: Optional[Union[str, List[str]]] = None,
    infer_method: str = "ode",
    timesteps: Optional[List[float]] = None,
    noise_schedule: str = "linear",
) -> Dict[str, Any]:
    """Generate music latents and metadata from text/audio conditioning inputs.

    Args:
        captions: Caption string(s) describing the target generation.
        lyrics: Lyric string(s) aligned with each requested sample.
        keys: Optional sample identifiers.
        target_wavs: Optional target audio tensor for repaint/cover flows.
        refer_audios: Optional nested reference-audio tensors for style conditioning.
        metas: Optional metadata payload(s) for each sample.
        vocal_languages: Optional per-sample vocal language code(s).
        infer_steps: Number of diffusion steps to run.
        guidance_scale: CFG guidance strength.
        seed: Optional scalar or per-sample seed list.
        return_intermediate: Whether to include intermediate tensors in outputs.
        repainting_start: Optional repaint start time(s) in seconds.
        repainting_end: Optional repaint end time(s) in seconds.
        instructions: Optional per-sample instruction string(s).
        audio_cover_strength: Cover blend strength.
        cover_noise_strength: Cover noise blend strength.
        use_adg: Whether adaptive diffusion guidance is enabled.
        cfg_interval_start: CFG schedule start ratio.
        cfg_interval_end: CFG schedule end ratio.
        shift: Diffusion shift parameter.
        audio_code_hints: Optional serialized audio-code hints.
        infer_method: Diffusion inference method selector.
        timesteps: Optional explicit diffusion timestep sequence.

    Returns:
        Dict[str, Any]: Service output payload containing generated latents,
        timing fields, and optionally intermediate conditioning tensors.

    Raises:
        Exception: Propagates exceptions raised by downstream helper methods
            (e.g., normalization, diffusion execution, output assembly).
    """
    normalized = self._normalize_service_generate_inputs(
        captions=captions,
        lyrics=lyrics,
        keys=keys,
        metas=metas,
        vocal_languages=vocal_languages,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        instructions=instructions,
        audio_code_hints=audio_code_hints,
        infer_steps=infer_steps,
        seed=seed,
        return_intermediate=return_intermediate,
    )
    batch = self._prepare_batch(
        captions=normalized["captions"],
        lyrics=normalized["lyrics"],
        keys=normalized["keys"],
        target_wavs=target_wavs,
        refer_audios=refer_audios,
        metas=normalized["metas"],
        vocal_languages=normalized["vocal_languages"],
        repainting_start=normalized["repainting_start"],
        repainting_end=normalized["repainting_end"],
        instructions=normalized["instructions"],
        audio_code_hints=normalized["audio_code_hints"],
        audio_cover_strength=audio_cover_strength,
        cover_noise_strength=cover_noise_strength,
    )
    payload = self._unpack_service_processed_data(self.preprocess_batch(batch))
    seed_param = self._resolve_service_seed_param(normalized["seed_list"])
    self._ensure_silence_latent_on_device()
    generate_kwargs = self._build_service_generate_kwargs(
        payload=payload,
        seed_param=seed_param,
        infer_steps=normalized["infer_steps"],
        guidance_scale=guidance_scale,
        audio_cover_strength=audio_cover_strength,
        cover_noise_strength=cover_noise_strength,
        infer_method=infer_method,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
    )
    outputs, encoder_hidden_states, encoder_attention_mask, context_latents = (
        self._execute_service_generate_diffusion(
            payload=payload,
            generate_kwargs=generate_kwargs,
            seed_param=seed_param,
            infer_method=infer_method,
            shift=shift,
            audio_cover_strength=audio_cover_strength,
        )
    )
    return self._attach_service_generate_outputs(
        outputs=outputs,
        payload=payload,
        batch=batch,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        return_intermediate=normalized["return_intermediate"],
    )

# --- From service_generate_request.py ---

def _build_service_seed_list(
    self,
    seed: Optional[Union[int, List[int]]],
    batch_size: int,
) -> Optional[List[int]]:
    """Normalize ``seed`` into a per-item list or ``None`` for random sampling."""
    if seed is None:
        return None
    if isinstance(seed, list):
        seed_list = list(seed)
        if len(seed_list) < batch_size:
            while len(seed_list) < batch_size:
                seed_list.append(random.randint(0, 2**32 - 1))
        elif len(seed_list) > batch_size:
            seed_list = seed_list[:batch_size]
        return seed_list
    return [int(seed)] * batch_size

def _normalize_service_generate_inputs(
    self,
    captions: Union[str, List[str]],
    lyrics: Union[str, List[str]],
    keys: Optional[Union[str, List[str]]],
    metas: Optional[Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]],
    vocal_languages: Optional[Union[str, List[str]]],
    repainting_start: Optional[Union[float, List[float]]],
    repainting_end: Optional[Union[float, List[float]]],
    instructions: Optional[Union[str, List[str]]],
    audio_code_hints: Optional[Union[str, List[str]]],
    infer_steps: int,
    seed: Optional[Union[int, List[int]]],
    return_intermediate: bool = False,
) -> Dict[str, Any]:
    """Normalize scalar/list generation inputs and clamp turbo infer steps."""
    if self.config.is_turbo and infer_steps > 8:
        logger.warning(
            "[service_generate] dmd_gan version: infer_steps {} exceeds maximum 8, clamping to 8",
            infer_steps,
        )
        infer_steps = 8

    # Convert captions to list and determine batch size
    if isinstance(captions, str):
        captions = [captions]

    batch_size = len(captions)

    # Validate and clamp batch size to maximum supported
    if batch_size > MAX_BATCH_SIZE:
        logger.warning(
            "[service_generate] Batch size {} exceeds maximum supported batch size ({}). "
            "Clamping to {}.",
            batch_size,
            MAX_BATCH_SIZE,
            MAX_BATCH_SIZE,
        )
        batch_size = MAX_BATCH_SIZE
        captions = captions[:batch_size]

    # Now convert other inputs to lists, using the clamped batch_size
    if isinstance(lyrics, str):
        lyrics = [lyrics]
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(vocal_languages, str):
        vocal_languages = [vocal_languages]
    if isinstance(metas, (str, dict)):
        metas = [metas]
    if isinstance(repainting_start, (int, float)):
        repainting_start = [repainting_start]
    if isinstance(repainting_end, (int, float)):
        repainting_end = [repainting_end]

    # Truncate list inputs that exceed clamped batch size
    if keys is not None and len(keys) > batch_size:
        keys = keys[:batch_size]
    if metas is not None and len(metas) > batch_size:
        metas = metas[:batch_size]
    if vocal_languages is not None and len(vocal_languages) > batch_size:
        vocal_languages = vocal_languages[:batch_size]
    if repainting_start is not None and len(repainting_start) > batch_size:
        repainting_start = repainting_start[:batch_size]
    if repainting_end is not None and len(repainting_end) > batch_size:
        repainting_end = repainting_end[:batch_size]

    if len(lyrics) < batch_size:
        fill = lyrics[-1] if lyrics else ""
        lyrics = list(lyrics) + [fill] * (batch_size - len(lyrics))
    elif len(lyrics) > batch_size:
        lyrics = lyrics[:batch_size]

    if instructions is not None:
        instructions = self._normalize_instructions(
            instructions,
            batch_size,
            DEFAULT_DIT_INSTRUCTION,
        )
    if audio_code_hints is not None:
        audio_code_hints = self._normalize_audio_code_hints(audio_code_hints, batch_size)

    return {
        "captions": captions,
        "lyrics": lyrics,
        "keys": keys,
        "metas": metas,
        "vocal_languages": vocal_languages,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "instructions": instructions,
        "audio_code_hints": audio_code_hints,
        "infer_steps": infer_steps,
        "seed_list": self._build_service_seed_list(seed=seed, batch_size=batch_size),
        "return_intermediate": return_intermediate,
    }

# --- From service_generate_execute.py ---

def _unpack_service_processed_data(self, processed_data: Tuple[Any, ...]) -> Dict[str, Any]:
    """Convert batch preprocessing tuple into a keyed payload."""
    (
        keys,
        text_inputs,
        src_latents,
        target_latents,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        _audio_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
        chunk_mask,
        spans,
        is_covers,
        _audio_codes,
        lyric_token_idss,
        precomputed_lm_hints_25Hz,
        non_cover_text_hidden_states,
        non_cover_text_attention_masks,
    ) = processed_data
    return {
        "keys": keys,
        "text_inputs": text_inputs,
        "src_latents": src_latents,
        "target_latents": target_latents,
        "text_hidden_states": text_hidden_states,
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
        "refer_audio_order_mask": refer_audio_order_mask,
        "chunk_mask": chunk_mask,
        "spans": spans,
        "is_covers": is_covers,
        "lyric_token_idss": lyric_token_idss,
        "precomputed_lm_hints_25Hz": precomputed_lm_hints_25Hz,
        "non_cover_text_hidden_states": non_cover_text_hidden_states,
        "non_cover_text_attention_masks": non_cover_text_attention_masks,
    }

def _resolve_service_seed_param(self, seed_list: Optional[List[int]]) -> Any:
    """Return model seed parameter: per-item seed list or random single seed."""
    if seed_list is not None:
        return seed_list
    return random.randint(0, 2**32 - 1)

def _build_service_generate_kwargs(
    self,
    payload: Dict[str, Any],
    seed_param: Any,
    infer_steps: int,
    guidance_scale: float,
    audio_cover_strength: float,
    cover_noise_strength: float,
    infer_method: str,
    use_adg: bool,
    cfg_interval_start: float,
    cfg_interval_end: float,
    shift: float,
    timesteps: Optional[List[float]],
    noise_schedule: str = "linear",
) -> Dict[str, Any]:
    """Build kwargs passed to model generation backends."""
    kwargs = {
        "text_hidden_states": payload["text_hidden_states"],
        "text_attention_mask": payload["text_attention_mask"],
        "lyric_hidden_states": payload["lyric_hidden_states"],
        "lyric_attention_mask": payload["lyric_attention_mask"],
        "refer_audio_acoustic_hidden_states_packed": payload["refer_audio_acoustic_hidden_states_packed"],
        "refer_audio_order_mask": payload["refer_audio_order_mask"],
        "src_latents": payload["src_latents"],
        "chunk_masks": payload["chunk_mask"],
        "is_covers": payload["is_covers"],
        "silence_latent": self.silence_latent,
        "seed": seed_param,
        "non_cover_text_hidden_states": payload["non_cover_text_hidden_states"],
        "non_cover_text_attention_mask": payload["non_cover_text_attention_masks"],
        "precomputed_lm_hints_25Hz": payload["precomputed_lm_hints_25Hz"],
        "audio_cover_strength": audio_cover_strength,
        "cover_noise_strength": cover_noise_strength,
        "infer_method": infer_method,
        "infer_steps": infer_steps,
        "diffusion_guidance_sale": guidance_scale,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "noise_schedule": noise_schedule,
    }
    if timesteps is not None:
        kwargs["timesteps"] = torch.tensor(timesteps, dtype=torch.float32, device=self.device)
    return kwargs

def _execute_service_generate_diffusion(
    self,
    payload: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
    seed_param: Any,
    infer_method: str,
    shift: float,
    audio_cover_strength: float,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute condition preparation and diffusion using MLX or PyTorch backend."""
    dit_backend = (
        "MLX (native)" if (self.use_mlx_dit and self.mlx_decoder is not None) else f"PyTorch ({self.device})"
    )
    logger.info(f"[service_generate] Generating audio... (DiT backend: {dit_backend})")
    with torch.inference_mode():
        with self._load_model_context("model"):
            encoder_hidden_states, encoder_attention_mask, context_latents = self.model.prepare_condition(
                text_hidden_states=payload["text_hidden_states"],
                text_attention_mask=payload["text_attention_mask"],
                lyric_hidden_states=payload["lyric_hidden_states"],
                lyric_attention_mask=payload["lyric_attention_mask"],
                refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                refer_audio_order_mask=payload["refer_audio_order_mask"],
                hidden_states=payload["src_latents"],
                attention_mask=torch.ones(
                    payload["src_latents"].shape[0],
                    payload["src_latents"].shape[1],
                    device=payload["src_latents"].device,
                    dtype=payload["src_latents"].dtype,
                ),
                silence_latent=self.silence_latent,
                src_latents=payload["src_latents"],
                chunk_masks=payload["chunk_mask"],
                is_covers=payload["is_covers"],
                precomputed_lm_hints_25Hz=payload["precomputed_lm_hints_25Hz"],
            )

            if self.use_mlx_dit and self.mlx_decoder is not None:
                try:
                    enc_hs_nc, enc_am_nc, ctx_nc = None, None, None
                    if audio_cover_strength < 1.0 and payload["non_cover_text_hidden_states"] is not None:
                        non_is_covers = torch.zeros_like(payload["is_covers"])
                        sil_exp = self.silence_latent[:, :payload["src_latents"].shape[1], :].expand(
                            payload["src_latents"].shape[0], -1, -1
                        )
                        enc_hs_nc, enc_am_nc, ctx_nc = self.model.prepare_condition(
                            text_hidden_states=payload["non_cover_text_hidden_states"],
                            text_attention_mask=payload["non_cover_text_attention_masks"],
                            lyric_hidden_states=payload["lyric_hidden_states"],
                            lyric_attention_mask=payload["lyric_attention_mask"],
                            refer_audio_acoustic_hidden_states_packed=payload["refer_audio_acoustic_hidden_states_packed"],
                            refer_audio_order_mask=payload["refer_audio_order_mask"],
                            hidden_states=sil_exp,
                            attention_mask=torch.ones(
                                sil_exp.shape[0], sil_exp.shape[1], device=sil_exp.device, dtype=sil_exp.dtype
                            ),
                            silence_latent=self.silence_latent,
                            src_latents=sil_exp,
                            chunk_masks=payload["chunk_mask"],
                            is_covers=non_is_covers,
                        )

                    null_cond_emb = getattr(self.model, "null_condition_emb", None)

                    outputs = self._mlx_run_diffusion(
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        context_latents=context_latents,
                        src_latents=payload["src_latents"],
                        seed=seed_param,
                        infer_method=infer_method,
                        shift=shift,
                        timesteps=generate_kwargs.get("timesteps"),
                        infer_steps=generate_kwargs.get("infer_steps"),
                        guidance_scale=generate_kwargs.get("diffusion_guidance_sale", 1.0),
                        null_condition_emb=null_cond_emb,
                        cfg_interval_start=generate_kwargs.get("cfg_interval_start", 0.0),
                        cfg_interval_end=generate_kwargs.get("cfg_interval_end", 1.0),
                        audio_cover_strength=audio_cover_strength,
                        encoder_hidden_states_non_cover=enc_hs_nc,
                        encoder_attention_mask_non_cover=enc_am_nc,
                        context_latents_non_cover=ctx_nc,
                        noise_schedule=generate_kwargs.get("noise_schedule", "linear"),
                    )
                    _tc = outputs.get("time_costs", {})
                    logger.info(
                        "[service_generate] DiT diffusion complete via MLX (%.2fs total, %.3fs/step).",
                        _tc.get("diffusion_time_cost", 0),
                        _tc.get("diffusion_per_step_time_cost", 0),
                    )
                except Exception as exc:
                    logger.warning("[service_generate] MLX diffusion failed (%s); falling back to PyTorch.", exc)
                    outputs = self.model.generate_audio(**generate_kwargs)
            else:
                logger.info("[service_generate] DiT diffusion via PyTorch ({})...", self.device)
                outputs = self.model.generate_audio(**generate_kwargs)

    return outputs, encoder_hidden_states, encoder_attention_mask, context_latents

# --- From service_generate_outputs.py ---

def _attach_service_generate_outputs(
    self,
    outputs: Dict[str, Any],
    payload: Dict[str, Any],
    batch: Dict[str, Any],
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    context_latents: torch.Tensor,
    return_intermediate: bool = True,
) -> Dict[str, Any]:
    """Attach intermediate tensors required by downstream consumers."""
    outputs["spans"] = payload["spans"]
    if not return_intermediate:
        return outputs

    outputs["src_latents"] = payload["src_latents"]
    outputs["target_latents_input"] = payload["target_latents"]
    outputs["chunk_masks"] = payload["chunk_mask"]
    outputs["latent_masks"] = batch.get("latent_masks")
    outputs["encoder_hidden_states"] = encoder_hidden_states
    outputs["encoder_attention_mask"] = encoder_attention_mask
    outputs["context_latents"] = context_latents
    outputs["lyric_token_idss"] = payload["lyric_token_idss"]
    return outputs

# --- From diffusion.py ---

def _mlx_run_diffusion(
    self,
    encoder_hidden_states,
    encoder_attention_mask,
    context_latents,
    src_latents,
    seed,
    infer_method: str = "ode",
    shift: float = 3.0,
    timesteps=None,
    infer_steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    null_condition_emb: Optional[torch.Tensor] = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    audio_cover_strength: float = 1.0,
    encoder_hidden_states_non_cover=None,
    encoder_attention_mask_non_cover=None,
    context_latents_non_cover=None,
    disable_tqdm: bool = False,
    noise_schedule: str = "linear",
) -> Dict[str, Any]:
    """Run the MLX diffusion loop and return generated latents.

    Args:
        encoder_hidden_states: Prompt conditioning tensor.
        encoder_attention_mask: Unused; accepted for API compatibility.
        context_latents: Context/reference latent tensor.
        src_latents: Source latent tensor used for shape and initialization.
        seed: Random seed used by MLX diffusion.
        infer_method: Diffusion method (``"ode"``, ``"sde"``, or ``"auraflow"``).
        shift: Timestep shift value.
        timesteps: Optional iterable or tensor-like custom timesteps.
        infer_steps: Number of diffusion steps (overrides fixed 8-step table).
        guidance_scale: CFG guidance strength (>1.0 enables CFG).
        null_condition_emb: Null condition embedding tensor for CFG.
        cfg_interval_start: Timestep ratio below which CFG is disabled.
        cfg_interval_end: Timestep ratio above which CFG is disabled.
        audio_cover_strength: Blend factor for cover conditioning.
        encoder_hidden_states_non_cover: Optional non-cover conditioning tensor.
        encoder_attention_mask_non_cover: Unused; accepted for API compatibility.
        context_latents_non_cover: Optional non-cover context latent tensor.
        disable_tqdm: If True, suppress the diffusion progress bar.
        noise_schedule: Schedule type (``"linear"`` or ``"cosine"``).

    Returns:
        Dict[str, Any]: ``{"target_latents": torch.Tensor, "time_costs": dict}``.
    """
    import numpy as np

    _ = encoder_attention_mask, encoder_attention_mask_non_cover

    for required_attr in ("mlx_decoder", "device", "dtype"):
        if not hasattr(self, required_attr):
            raise AttributeError(f"DiffusionMixin host is missing required attribute '{required_attr}'")

    _valid = {"ode", "sde", "auraflow"}
    if infer_method not in _valid:
        raise ValueError(
            f"Unsupported infer_method '{infer_method}'. Expected one of {sorted(_valid)}."
        )

    if timesteps is not None and not (hasattr(timesteps, "__iter__") or hasattr(timesteps, "tolist")):
        raise TypeError("timesteps must be iterable, tensor-like, or None")

    if encoder_hidden_states.shape[0] != context_latents.shape[0]:
        raise ValueError(
            "Batch dimension mismatch: encoder_hidden_states and context_latents must share dim 0"
        )
    if encoder_hidden_states.shape[0] != src_latents.shape[0]:
        raise ValueError(
            "Batch dimension mismatch: encoder_hidden_states and src_latents must share dim 0"
        )
    if encoder_hidden_states_non_cover is not None and encoder_hidden_states_non_cover.shape[0] != encoder_hidden_states.shape[0]:
        raise ValueError(
            "Batch dimension mismatch: encoder_hidden_states_non_cover must share dim 0 with encoder_hidden_states"
        )
    if context_latents_non_cover is not None and context_latents_non_cover.shape[0] != context_latents.shape[0]:
        raise ValueError(
            "Batch dimension mismatch: context_latents_non_cover must share dim 0 with context_latents"
        )

    enc_np = encoder_hidden_states.detach().cpu().float().numpy()
    ctx_np = context_latents.detach().cpu().float().numpy()
    src_shape = (src_latents.shape[0], src_latents.shape[1], src_latents.shape[2])

    enc_nc_np = (
        encoder_hidden_states_non_cover.detach().cpu().float().numpy()
        if encoder_hidden_states_non_cover is not None else None
    )
    ctx_nc_np = (
        context_latents_non_cover.detach().cpu().float().numpy()
        if context_latents_non_cover is not None else None
    )

    null_cond_np = (
        null_condition_emb.detach().cpu().float().numpy()
        if null_condition_emb is not None else None
    )

    ts_list = None
    if timesteps is not None:
        if hasattr(timesteps, "tolist"):
            ts_list = timesteps.tolist()
        else:
            ts_list = list(timesteps)

    result = mlx_generate_diffusion(
        mlx_decoder=self.mlx_decoder,
        encoder_hidden_states_np=enc_np,
        context_latents_np=ctx_np,
        src_latents_shape=src_shape,
        seed=seed,
        infer_method=infer_method,
        shift=shift,
        timesteps=ts_list,
        infer_steps=infer_steps,
        guidance_scale=guidance_scale,
        null_condition_emb_np=null_cond_np,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        audio_cover_strength=audio_cover_strength,
        encoder_hidden_states_non_cover_np=enc_nc_np,
        context_latents_non_cover_np=ctx_nc_np,
        compile_model=getattr(self, "mlx_dit_compiled", False),
        disable_tqdm=disable_tqdm,
        noise_schedule=noise_schedule,
    )

    target_np = result["target_latents"]
    target_tensor = torch.from_numpy(target_np).to(device=self.device, dtype=self.dtype)

    return {
        "target_latents": target_tensor,
        "time_costs": result["time_costs"],
    }
