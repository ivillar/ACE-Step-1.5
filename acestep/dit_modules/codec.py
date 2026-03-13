"""Consolidated handler functions – codec module."""

import math
from typing import Optional
import torch
from loguru import logger
from acestep.gpu_config import get_gpu_memory_gb
from tqdm import tqdm
import os
from typing import Any
import time as _time
import numpy as np
import random


# --- From vae_encode.py ---

def tiled_encode(self, audio, chunk_size=None, overlap=None, offload_latent_to_cpu=True):
    """Encode audio to latents using overlap-discard tiling.

    Args:
        audio: Tensor shaped ``[batch, channels, samples]`` or ``[channels, samples]``.
        chunk_size: Audio chunk size in samples; auto-selected when ``None``.
        overlap: Overlap in samples; defaults to 2 seconds at 48kHz.
        offload_latent_to_cpu: Whether to offload chunk outputs to CPU.

    Returns:
        Latent tensor shaped ``[batch, latent_channels, latent_frames]``.
    """
    # ---- MLX fast path (macOS Apple Silicon) ----
    if self.use_mlx_vae and self.mlx_vae is not None:
        input_was_2d = audio.dim() == 2
        if input_was_2d:
            audio = audio.unsqueeze(0)
        try:
            result = self._mlx_vae_encode_sample(audio)
            if input_was_2d:
                result = result.squeeze(0)
            return result
        except Exception as exc:
            logger.warning(
                f"[tiled_encode] MLX VAE encode failed ({type(exc).__name__}: {exc}), "
                f"falling back to PyTorch VAE..."
            )
            if input_was_2d:
                audio = audio.squeeze(0)

    # ---- PyTorch path (CUDA / MPS / CPU) ----
    if chunk_size is None:
        gpu_memory = get_gpu_memory_gb()
        if gpu_memory <= 0 and self.device == "mps":
            mem_gb = self._get_effective_mps_memory_gb()
            if mem_gb is not None:
                gpu_memory = mem_gb
        chunk_size = 48000 * 15 if gpu_memory <= 8 else 48000 * 30
    if overlap is None:
        overlap = 48000 * 2

    input_was_2d = audio.dim() == 2
    if input_was_2d:
        audio = audio.unsqueeze(0)

    batch_size, _channels, samples = audio.shape

    if samples <= chunk_size:
        vae_input = audio.to(self.device).to(self.vae.dtype)
        with torch.inference_mode():
            latents = self.vae.encode(vae_input).latent_dist.sample()
        if input_was_2d:
            latents = latents.squeeze(0)
        return latents

    stride = chunk_size - 2 * overlap
    if stride <= 0:
        raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

    num_steps = math.ceil(samples / stride)
    if offload_latent_to_cpu:
        result = self._tiled_encode_offload_cpu(audio, batch_size, samples, stride, overlap, num_steps, chunk_size)
    else:
        result = self._tiled_encode_gpu(audio, batch_size, samples, stride, overlap, num_steps, chunk_size)

    if input_was_2d:
        result = result.squeeze(0)
    return result

# --- From vae_encode_chunks.py ---

def _tiled_encode_gpu(self, audio, batch_size, samples, stride, overlap, num_steps, chunk_size):
    """Standard tiled encode keeping all data on GPU."""
    _ = batch_size, chunk_size
    encoded_latent_list = []
    downsample_factor = None

    for i in tqdm(range(num_steps), desc="Encoding audio chunks", disable=self.disable_tqdm):
        core_start = i * stride
        core_end = min(core_start + stride, samples)
        win_start = max(0, core_start - overlap)
        win_end = min(samples, core_end + overlap)

        audio_chunk = audio[:, :, win_start:win_end].to(self.device).to(self.vae.dtype)
        with torch.inference_mode():
            latent_chunk = self.vae.encode(audio_chunk).latent_dist.sample()

        if downsample_factor is None:
            downsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

        added_start = core_start - win_start
        trim_start = int(round(added_start / downsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end / downsample_factor))

        latent_len = latent_chunk.shape[-1]
        end_idx = latent_len - trim_end if trim_end > 0 else latent_len
        latent_core = latent_chunk[:, :, trim_start:end_idx]
        encoded_latent_list.append(latent_core)
        del audio_chunk

    return torch.cat(encoded_latent_list, dim=-1)

def _tiled_encode_offload_cpu(self, audio, batch_size, samples, stride, overlap, num_steps, chunk_size):
    """Tiled encode that offloads latent chunks to CPU immediately."""
    _ = chunk_size
    first_core_end = min(stride, samples)
    first_win_end = min(samples, first_core_end + overlap)

    first_audio_chunk = audio[:, :, 0:first_win_end].to(self.device).to(self.vae.dtype)
    with torch.inference_mode():
        first_latent_chunk = self.vae.encode(first_audio_chunk).latent_dist.sample()

    downsample_factor = first_audio_chunk.shape[-1] / first_latent_chunk.shape[-1]
    latent_channels = first_latent_chunk.shape[1]

    total_latent_length = int(round(samples / downsample_factor))
    final_latents = torch.zeros(
        batch_size,
        latent_channels,
        total_latent_length,
        dtype=first_latent_chunk.dtype,
        device="cpu",
    )

    first_added_end = first_win_end - first_core_end
    first_trim_end = int(round(first_added_end / downsample_factor))
    first_latent_len = first_latent_chunk.shape[-1]
    first_end_idx = first_latent_len - first_trim_end if first_trim_end > 0 else first_latent_len

    first_latent_core = first_latent_chunk[:, :, :first_end_idx]
    latent_write_pos = first_latent_core.shape[-1]
    final_latents[:, :, :latent_write_pos] = first_latent_core.cpu()
    del first_audio_chunk, first_latent_chunk, first_latent_core

    for i in tqdm(range(1, num_steps), desc="Encoding audio chunks", disable=self.disable_tqdm):
        core_start = i * stride
        core_end = min(core_start + stride, samples)
        win_start = max(0, core_start - overlap)
        win_end = min(samples, core_end + overlap)

        audio_chunk = audio[:, :, win_start:win_end].to(self.device).to(self.vae.dtype)
        with torch.inference_mode():
            latent_chunk = self.vae.encode(audio_chunk).latent_dist.sample()

        added_start = core_start - win_start
        trim_start = int(round(added_start / downsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end / downsample_factor))

        latent_len = latent_chunk.shape[-1]
        end_idx = latent_len - trim_end if trim_end > 0 else latent_len
        latent_core = latent_chunk[:, :, trim_start:end_idx]

        core_len = latent_core.shape[-1]
        final_latents[:, :, latent_write_pos : latent_write_pos + core_len] = latent_core.cpu()
        latent_write_pos += core_len
        del audio_chunk, latent_chunk, latent_core

    return final_latents[:, :, :latent_write_pos]

# --- From vae_decode.py ---

_MPS_DECODE_CHUNK_SIZE = 32
_MPS_DECODE_OVERLAP = 8

def tiled_decode(
    self,
    latents,
    chunk_size: Optional[int] = None,
    overlap: int = 64,
    offload_wav_to_cpu: Optional[bool] = None,
):
    """Decode latents using tiling to reduce VRAM usage.

    Uses overlap-discard chunking to avoid boundary artifacts while
    constraining peak decode memory.

    Args:
        latents: Tensor shaped ``[batch, channels, latent_frames]``.
        chunk_size: Chunk size in latent frames. When ``None``, an
            auto-tuned value is selected by runtime policy.
        overlap: Overlap in latent frames between adjacent windows.
        offload_wav_to_cpu: Whether decoded waveform chunks should be
            offloaded to CPU immediately to reduce VRAM pressure.

    Returns:
        Decoded waveform tensor shaped ``[batch, audio_channels, samples]``.
    """
    # ---- MLX fast path (macOS Apple Silicon) ----
    if self.use_mlx_vae and self.mlx_vae is not None:
        try:
            result = self._mlx_vae_decode(latents)
            return result
        except Exception as exc:
            logger.warning(
                f"[tiled_decode] MLX VAE decode failed ({type(exc).__name__}: {exc}), "
                f"falling back to PyTorch VAE..."
            )

    # ---- PyTorch path (CUDA / MPS / CPU) ----
    if chunk_size is None:
        chunk_size = self._get_auto_decode_chunk_size()
    if offload_wav_to_cpu is None:
        offload_wav_to_cpu = self._should_offload_wav_to_cpu()

    logger.info(
        f"[tiled_decode] chunk_size={chunk_size}, offload_wav_to_cpu={offload_wav_to_cpu}, "
        f"latents_shape={latents.shape}"
    )

    # MPS Conv1d has a hard output-size limit during temporal upsampling.
    _is_mps = self.device == "mps"
    if _is_mps:
        _mps_chunk = self._MPS_DECODE_CHUNK_SIZE
        _mps_overlap = self._MPS_DECODE_OVERLAP
        _needs_reduction = (chunk_size > _mps_chunk) or (overlap > _mps_overlap)
        if _needs_reduction:
            logger.info(
                f"[tiled_decode] VAE decode via PyTorch MPS; reducing chunk_size from {chunk_size} "
                f"to {min(chunk_size, _mps_chunk)} and overlap from {overlap} "
                f"to {min(overlap, _mps_overlap)} to avoid MPS conv output limit."
            )
            chunk_size = min(chunk_size, _mps_chunk)
            overlap = min(overlap, _mps_overlap)

    try:
        return self._tiled_decode_inner(latents, chunk_size, overlap, offload_wav_to_cpu)
    except (NotImplementedError, RuntimeError) as exc:
        if not _is_mps:
            raise
        logger.warning(
            f"[tiled_decode] MPS decode failed ({type(exc).__name__}: {exc}), "
            f"falling back to CPU VAE decode..."
        )
        return self._tiled_decode_cpu_fallback(latents)

def _tiled_decode_cpu_fallback(self, latents):
    """Last-resort CPU VAE decode when MPS fails unexpectedly."""
    _first_param = next(self.vae.parameters())
    vae_device = _first_param.device
    vae_dtype = _first_param.dtype
    try:
        self.vae = self.vae.cpu().float()
        latents_cpu = latents.to(device="cpu", dtype=torch.float32)
        decoder_output = self.vae.decode(latents_cpu)
        result = decoder_output.sample
        del decoder_output
        return result
    finally:
        # Always restore VAE to original device/dtype
        self.vae = self.vae.to(vae_dtype).to(vae_device)

def _decode_on_cpu(self, latents):
    """Move VAE to CPU, decode there, then restore original device."""
    logger.warning("[_decode_on_cpu] Moving VAE to CPU for decode (VRAM too tight for GPU decode)")

    try:
        original_device = next(self.vae.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")

    vae_cpu_dtype = self._get_vae_dtype("cpu")
    self._recursive_to_device(self.vae, "cpu", vae_cpu_dtype)
    self._empty_cache()

    latents_cpu = latents.cpu().to(vae_cpu_dtype)
    try:
        with torch.inference_mode():
            decoder_output = self.vae.decode(latents_cpu)
            result = decoder_output.sample
            del decoder_output
    finally:
        if original_device.type != "cpu":
            vae_gpu_dtype = self._get_vae_dtype(str(original_device))
            self._recursive_to_device(self.vae, original_device, vae_gpu_dtype)

    logger.info(f"[_decode_on_cpu] CPU decode complete, result shape={result.shape}")
    return result

# --- From vae_decode_chunks.py ---

def _tiled_decode_inner(self, latents, chunk_size, overlap, offload_wav_to_cpu):
    """Run tiled decode with adaptive overlap and OOM fallbacks."""
    bsz, _channels, latent_frames = latents.shape

    # Batch-sequential decode keeps peak VRAM stable across batch sizes.
    if bsz > 1:
        logger.info(f"[tiled_decode] Batch size {bsz} > 1; decoding samples sequentially to save VRAM")
        per_sample_results = []
        for b_idx in range(bsz):
            single = latents[b_idx : b_idx + 1]
            decoded = self._tiled_decode_inner(single, chunk_size, overlap, offload_wav_to_cpu)
            per_sample_results.append(decoded.cpu() if decoded.device.type != "cpu" else decoded)
            self._empty_cache()
        result = torch.cat(per_sample_results, dim=0)
        if latents.device.type != "cpu" and not offload_wav_to_cpu:
            result = result.to(latents.device)
        return result

    effective_overlap = overlap
    while chunk_size - 2 * effective_overlap <= 0 and effective_overlap > 0:
        effective_overlap = effective_overlap // 2
    if effective_overlap != overlap:
        logger.warning(
            f"[tiled_decode] Reduced overlap from {overlap} to {effective_overlap} for chunk_size={chunk_size}"
        )
    overlap = effective_overlap

    if latent_frames <= chunk_size:
        try:
            decoder_output = self.vae.decode(latents)
            result = decoder_output.sample
            del decoder_output
            return result
        except torch.cuda.OutOfMemoryError:
            logger.warning("[tiled_decode] OOM on direct decode, falling back to CPU VAE decode")
            self._empty_cache()
            return self._decode_on_cpu(latents)

    stride = chunk_size - 2 * overlap
    if stride <= 0:
        raise ValueError(f"chunk_size {chunk_size} must be > 2 * overlap {overlap}")

    num_steps = math.ceil(latent_frames / stride)

    if offload_wav_to_cpu:
        try:
            return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"[tiled_decode] OOM during offload_cpu decode with chunk_size={chunk_size}, "
                "falling back to CPU VAE decode"
            )
            self._empty_cache()
            return self._decode_on_cpu(latents)

    try:
        return self._tiled_decode_gpu(latents, stride, overlap, num_steps)
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            f"[tiled_decode] OOM during GPU decode with chunk_size={chunk_size}, "
            "falling back to CPU offload path"
        )
        self._empty_cache()
        try:
            return self._tiled_decode_offload_cpu(latents, bsz, latent_frames, stride, overlap, num_steps)
        except torch.cuda.OutOfMemoryError:
            logger.warning("[tiled_decode] OOM even with offload path, falling back to full CPU VAE decode")
            self._empty_cache()
            return self._decode_on_cpu(latents)

def _tiled_decode_gpu(self, latents, stride, overlap, num_steps):
    """Decode chunks and keep decoded audio tensors on GPU."""
    decoded_audio_list = []
    upsample_factor = None

    for i in tqdm(range(num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
        core_start = i * stride
        core_end = min(core_start + stride, latents.shape[-1])
        win_start = max(0, core_start - overlap)
        win_end = min(latents.shape[-1], core_end + overlap)

        latent_chunk = latents[:, :, win_start:win_end]
        decoder_output = self.vae.decode(latent_chunk)
        audio_chunk = decoder_output.sample
        del decoder_output

        if upsample_factor is None:
            upsample_factor = audio_chunk.shape[-1] / latent_chunk.shape[-1]

        added_start = core_start - win_start
        trim_start = int(round(added_start * upsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end * upsample_factor))

        audio_len = audio_chunk.shape[-1]
        end_idx = audio_len - trim_end if trim_end > 0 else audio_len
        audio_core = audio_chunk[:, :, trim_start:end_idx]
        decoded_audio_list.append(audio_core)

    return torch.cat(decoded_audio_list, dim=-1)

def _tiled_decode_offload_cpu(self, latents, bsz, latent_frames, stride, overlap, num_steps):
    """Decode chunks on GPU and copy trimmed audio cores to a CPU buffer."""
    first_core_end = min(stride, latent_frames)
    first_win_end = min(latent_frames, first_core_end + overlap)
    first_latent_chunk = latents[:, :, 0:first_win_end]
    first_decoder_output = self.vae.decode(first_latent_chunk)
    first_audio_chunk = first_decoder_output.sample
    del first_decoder_output

    upsample_factor = first_audio_chunk.shape[-1] / first_latent_chunk.shape[-1]
    audio_channels = first_audio_chunk.shape[1]

    total_audio_length = int(round(latent_frames * upsample_factor))
    final_audio = torch.zeros(bsz, audio_channels, total_audio_length, dtype=first_audio_chunk.dtype, device="cpu")

    first_added_end = first_win_end - first_core_end
    first_trim_end = int(round(first_added_end * upsample_factor))
    first_audio_len = first_audio_chunk.shape[-1]
    first_end_idx = first_audio_len - first_trim_end if first_trim_end > 0 else first_audio_len

    first_audio_core = first_audio_chunk[:, :, :first_end_idx]
    audio_write_pos = first_audio_core.shape[-1]
    final_audio[:, :, :audio_write_pos] = first_audio_core.cpu()

    del first_audio_chunk, first_audio_core, first_latent_chunk

    for i in tqdm(range(1, num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
        core_start = i * stride
        core_end = min(core_start + stride, latent_frames)
        win_start = max(0, core_start - overlap)
        win_end = min(latent_frames, core_end + overlap)

        latent_chunk = latents[:, :, win_start:win_end]
        decoder_output = self.vae.decode(latent_chunk)
        audio_chunk = decoder_output.sample
        del decoder_output

        added_start = core_start - win_start
        trim_start = int(round(added_start * upsample_factor))
        added_end = win_end - core_end
        trim_end = int(round(added_end * upsample_factor))

        audio_len = audio_chunk.shape[-1]
        end_idx = audio_len - trim_end if trim_end > 0 else audio_len
        audio_core = audio_chunk[:, :, trim_start:end_idx]

        core_len = audio_core.shape[-1]
        final_audio[:, :, audio_write_pos : audio_write_pos + core_len] = audio_core.cpu()
        audio_write_pos += core_len

        del audio_chunk, audio_core, latent_chunk

    return final_audio[:, :, :audio_write_pos]

# --- From mlx_dit_init.py ---

def _init_mlx_dit(self, compile_model: bool = False) -> bool:
    """Initialize the MLX DiT decoder when platform support is available.

    Args:
        compile_model: Whether MLX diffusion should use ``mx.compile``.

    Returns:
        bool: ``True`` when MLX DiT is initialized successfully, else ``False``.
    """
    try:
        from acestep.models.mlx import mlx_available

        if not mlx_available():
            logger.info("[MLX-DiT] MLX not available on this platform; skipping.")
            return False

        from acestep.models.mlx.dit_model import MLXDiTDecoder
        from acestep.models.mlx.dit_convert import convert_and_load

        mlx_decoder = MLXDiTDecoder.from_config(self.config)
        convert_and_load(self.model, mlx_decoder)
        self.mlx_decoder = mlx_decoder
        self.use_mlx_dit = True
        self.mlx_dit_compiled = compile_model
        logger.info(
            "[MLX-DiT] Native MLX DiT decoder initialized successfully "
            f"(mx.compile={compile_model})."
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[MLX-DiT] Failed to initialize MLX decoder (non-fatal): {exc}")
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False
        return False

# --- From mlx_vae_init.py ---

def _init_mlx_vae(self) -> bool:
    """Initialize native MLX VAE runtime state from ``self.vae``.

    The ``_init_mlx_vae`` path converts the loaded PyTorch VAE in
    ``self.vae`` to an MLX implementation, optionally applies float16
    conversion based on ``ACESTEP_MLX_VAE_FP16``, and prepares decode/encode
    callables.

    Side effects:
        Mutates ``self.mlx_vae`` and ``self.use_mlx_vae`` and updates
        ``self._mlx_compiled_decode``, ``self._mlx_compiled_encode_sample``,
        and ``self._mlx_vae_dtype``.

    Returns:
        bool: ``True`` when MLX VAE is initialized successfully, else ``False``.

    Error behavior:
        Returns ``False`` when MLX is unavailable or when any conversion/
        initialization step raises an exception. Failures are logged as
        non-fatal.
    """
    try:
        from acestep.models.mlx import mlx_available

        if not mlx_available():
            logger.info("[MLX-VAE] MLX not available on this platform; skipping.")
            return False

        import mlx.core as mx
        from mlx.utils import tree_map
        from acestep.models.mlx.vae_model import MLXAutoEncoderOobleck
        from acestep.models.mlx.vae_convert import convert_and_load

        mlx_vae = MLXAutoEncoderOobleck.from_pytorch_config(self.vae)
        convert_and_load(self.vae, mlx_vae)

        use_fp16 = os.environ.get("ACESTEP_MLX_VAE_FP16", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        vae_dtype = mx.float16 if use_fp16 else mx.float32

        if use_fp16:
            try:

                def _to_fp16(value: Any):
                    """Cast floating MLX arrays to float16 while preserving other values."""
                    if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
                        return value.astype(mx.float16)
                    return value

                mlx_vae.update(tree_map(_to_fp16, mlx_vae.parameters()))
                mx.eval(mlx_vae.parameters())
                logger.info("[MLX-VAE] Model weights converted to float16.")
            except Exception as exc:
                logger.warning(f"[MLX-VAE] Float16 conversion failed ({exc}); using float32.")
                vae_dtype = mx.float32

        compiled = True
        try:
            self._mlx_compiled_decode = mx.compile(mlx_vae.decode)
            self._mlx_compiled_encode_sample = mx.compile(mlx_vae.encode_and_sample)
            logger.info("[MLX-VAE] Decode/encode compiled with mx.compile().")
        except Exception as exc:
            compiled = False
            logger.warning(f"[MLX-VAE] mx.compile() failed ({exc}); using uncompiled path.")
            self._mlx_compiled_decode = mlx_vae.decode
            self._mlx_compiled_encode_sample = mlx_vae.encode_and_sample

        self.mlx_vae = mlx_vae
        self.use_mlx_vae = True
        self._mlx_vae_dtype = vae_dtype
        logger.info(
            f"[MLX-VAE] Native MLX VAE initialized (dtype={vae_dtype}, compiled={compiled})."
        )
        return True
    except Exception as exc:
        logger.warning(f"[MLX-VAE] Failed to initialize MLX VAE (non-fatal): {exc}")
        self.mlx_vae = None
        self.use_mlx_vae = False
        self._mlx_compiled_decode = None
        self._mlx_compiled_encode_sample = None
        self._mlx_vae_dtype = None
        return False

# --- From mlx_vae_encode_native.py ---

def _resolve_mlx_encode_fn(self):
    """Resolve the active MLX encode callable from compiled or model state.

    Returns:
        Any: Callable that encodes ``[1, S, C]`` MLX audio samples.

    Raises:
        RuntimeError: If no compiled callable exists and ``self.mlx_vae`` is missing.
    """
    encode_fn = getattr(self, "_mlx_compiled_encode_sample", None)
    if encode_fn is not None:
        return encode_fn
    if self.mlx_vae is None:
        raise RuntimeError("MLX VAE encode requested but mlx_vae is not initialized.")
    return self.mlx_vae.encode_and_sample

def _mlx_vae_encode_sample(self, audio_torch):
    """Encode batched PyTorch audio to MLX latents.

    Args:
        audio_torch: Audio tensor shaped ``[batch, channels, samples]``.

    Returns:
        torch.Tensor: Latent tensor shaped ``[batch, channels, frames]``.
    """
    import mlx.core as mx

    audio_np = audio_torch.detach().cpu().float().numpy()
    audio_nlc = np.transpose(audio_np, (0, 2, 1))
    batch_size = audio_nlc.shape[0]
    sample_frames = audio_nlc.shape[1]

    mlx_encode_chunk = 48000 * 30
    mlx_encode_overlap = 48000 * 2
    if sample_frames <= mlx_encode_chunk:
        chunks_per_sample = 1
    else:
        stride = mlx_encode_chunk - 2 * mlx_encode_overlap
        chunks_per_sample = math.ceil(sample_frames / stride)
    total_work = batch_size * chunks_per_sample

    t_start = _time.time()
    vae_dtype = getattr(self, "_mlx_vae_dtype", mx.float32)
    encode_fn = self._resolve_mlx_encode_fn()

    latent_parts = []
    pbar = tqdm(
        total=total_work,
        desc=f"MLX VAE Encode (native, n={batch_size})",
        disable=self.disable_tqdm,
        unit="chunk",
    )
    for idx in range(batch_size):
        single = mx.array(audio_nlc[idx : idx + 1])
        if single.dtype != vae_dtype:
            single = single.astype(vae_dtype)
        latent = self._mlx_encode_single(single, pbar=pbar, encode_fn=encode_fn)
        if latent.dtype != mx.float32:
            latent = latent.astype(mx.float32)
        mx.eval(latent)
        latent_parts.append(np.array(latent))
        mx.clear_cache()
    pbar.close()

    elapsed = _time.time() - t_start
    logger.info(
        f"[MLX-VAE] Encoded {batch_size} sample(s), {sample_frames} audio frames -> "
        f"latent in {elapsed:.2f}s (dtype={vae_dtype})"
    )

    latent_nlc = np.concatenate(latent_parts, axis=0)
    latent_ncl = np.transpose(latent_nlc, (0, 2, 1))
    return torch.from_numpy(latent_ncl)

def _mlx_encode_single(self, audio_nlc, pbar=None, encode_fn=None):
    """Encode one MLX audio sample with optional overlap-discard tiling.

    Args:
        audio_nlc: MLX array in ``[1, samples, channels]`` layout.
        pbar: Optional progress-bar object with ``update``.
        encode_fn: Optional encode callable; falls back to compiled encode.

    Returns:
        Any: MLX array in ``[1, frames, channels]`` layout.
    """
    import mlx.core as mx

    if encode_fn is None:
        encode_fn = self._resolve_mlx_encode_fn()

    sample_frames = audio_nlc.shape[1]
    mlx_encode_chunk = 48000 * 30
    mlx_encode_overlap = 48000 * 2

    if sample_frames <= mlx_encode_chunk:
        result = encode_fn(audio_nlc)
        mx.eval(result)
        if pbar is not None:
            pbar.update(1)
        return result

    stride = mlx_encode_chunk - 2 * mlx_encode_overlap
    num_steps = math.ceil(sample_frames / stride)
    encoded_parts = []
    downsample_factor = None

    for idx in range(num_steps):
        core_start = idx * stride
        core_end = min(core_start + stride, sample_frames)
        win_start = max(0, core_start - mlx_encode_overlap)
        win_end = min(sample_frames, core_end + mlx_encode_overlap)

        chunk = audio_nlc[:, win_start:win_end, :]
        latent_chunk = encode_fn(chunk)
        mx.eval(latent_chunk)
        if downsample_factor is None:
            downsample_factor = chunk.shape[1] / latent_chunk.shape[1]

        trim_start = int(round((core_start - win_start) / downsample_factor))
        trim_end = int(round((win_end - core_end) / downsample_factor))
        latent_len = latent_chunk.shape[1]
        end_idx = latent_len - trim_end if trim_end > 0 else latent_len
        encoded_parts.append(latent_chunk[:, trim_start:end_idx, :])
        if pbar is not None:
            pbar.update(1)

    return mx.concatenate(encoded_parts, axis=1)

# --- From mlx_vae_decode_native.py ---

def _resolve_mlx_decode_fn(self):
    """Resolve the active MLX decode callable from compiled or model state.

    Returns:
        Any: Callable that decodes ``[1, T, C]`` MLX latents.

    Raises:
        RuntimeError: If no compiled callable exists and ``self.mlx_vae`` is missing.
    """
    decode_fn = getattr(self, "_mlx_compiled_decode", None)
    if decode_fn is not None:
        return decode_fn
    if self.mlx_vae is None:
        raise RuntimeError("MLX VAE decode requested but mlx_vae is not initialized.")
    return self.mlx_vae.decode

def _mlx_vae_decode(self, latents_torch):
    """Decode batched PyTorch latents using native MLX VAE decode.

    Args:
        latents_torch: Latent tensor shaped ``[batch, channels, frames]``.

    Returns:
        torch.Tensor: Decoded audio shaped ``[batch, channels, samples]``.
    """
    import mlx.core as mx

    t_start = _time.time()
    latents_np = latents_torch.detach().cpu().float().numpy()
    latents_nlc = np.transpose(latents_np, (0, 2, 1))
    batch_size = latents_nlc.shape[0]
    latent_frames = latents_nlc.shape[1]

    vae_dtype = getattr(self, "_mlx_vae_dtype", mx.float32)
    latents_mx = mx.array(latents_nlc).astype(vae_dtype)
    t_convert = _time.time()

    decode_fn = self._resolve_mlx_decode_fn()
    audio_parts = []
    for idx in range(batch_size):
        decoded = self._mlx_decode_single(latents_mx[idx : idx + 1], decode_fn=decode_fn)
        if decoded.dtype != mx.float32:
            decoded = decoded.astype(mx.float32)
        mx.eval(decoded)
        audio_parts.append(np.array(decoded))
        mx.clear_cache()

    t_decode = _time.time()
    audio_nlc = np.concatenate(audio_parts, axis=0)
    audio_ncl = np.transpose(audio_nlc, (0, 2, 1))
    elapsed = _time.time() - t_start
    logger.info(
        f"[MLX-VAE] Decoded {batch_size} sample(s), {latent_frames} latent frames -> "
        f"audio in {elapsed:.2f}s "
        f"(convert={t_convert - t_start:.3f}s, decode={t_decode - t_convert:.2f}s, "
        f"dtype={vae_dtype})"
    )
    return torch.from_numpy(audio_ncl)

def _mlx_decode_single(self, z_nlc, decode_fn=None):
    """Decode a single MLX latent sample with optional tiling.

    Args:
        z_nlc: MLX array in ``[1, frames, channels]`` layout.
        decode_fn: Optional decode callable; falls back to compiled decode.

    Returns:
        Any: MLX array in ``[1, samples, channels]`` layout.
    """
    import mlx.core as mx

    if decode_fn is None:
        decode_fn = self._resolve_mlx_decode_fn()

    latent_frames = z_nlc.shape[1]
    mlx_chunk = 2048
    mlx_overlap = 64

    if latent_frames <= mlx_chunk:
        return decode_fn(z_nlc)

    stride = mlx_chunk - 2 * mlx_overlap
    num_steps = math.ceil(latent_frames / stride)
    decoded_parts = []
    upsample_factor = None

    for idx in tqdm(range(num_steps), desc="Decoding audio chunks", disable=self.disable_tqdm):
        core_start = idx * stride
        core_end = min(core_start + stride, latent_frames)
        win_start = max(0, core_start - mlx_overlap)
        win_end = min(latent_frames, core_end + mlx_overlap)

        chunk = z_nlc[:, win_start:win_end, :]
        audio_chunk = decode_fn(chunk)
        mx.eval(audio_chunk)
        if upsample_factor is None:
            upsample_factor = audio_chunk.shape[1] / chunk.shape[1]

        trim_start = int(round((core_start - win_start) * upsample_factor))
        trim_end = int(round((win_end - core_end) * upsample_factor))
        audio_len = audio_chunk.shape[1]
        end_idx = audio_len - trim_end if trim_end > 0 else audio_len
        decoded_parts.append(audio_chunk[:, trim_start:end_idx, :])

    return mx.concatenate(decoded_parts, axis=1)

# --- From io_audio.py ---

def _normalize_audio_to_stereo_48k(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
    """Normalize audio tensor to stereo at 48kHz.

    Args:
        audio: Tensor in [channels, samples] or [samples] format.
        sr: Source sample rate.

    Returns:
        Tensor in [2, samples] at 48kHz, clamped to [-1.0, 1.0].
    """
    if audio.shape[0] == 1:
        audio = torch.cat([audio, audio], dim=0)

    audio = audio[:2]

    if sr != 48000:
        import torchaudio
        audio = torchaudio.transforms.Resample(sr, 48000)(audio)

    return torch.clamp(audio, -1.0, 1.0)

def process_target_audio(self, audio_file: Optional[str]) -> Optional[torch.Tensor]:
    """Load and normalize target audio file.

    Args:
        audio_file: Path to target audio file.

    Returns:
        Normalized stereo 48kHz tensor, or ``None`` on error/empty input.
    """
    if audio_file is None:
        return None

    try:
        import soundfile as sf
        audio_np, sr = sf.read(audio_file, dtype="float32")
        if audio_np.ndim == 1:
            audio = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            audio = torch.from_numpy(audio_np.T)
        return self._normalize_audio_to_stereo_48k(audio, sr)
    except (OSError, RuntimeError, ValueError):
        logger.exception("[process_target_audio] Error processing target audio")
        return None

def process_reference_audio(self, audio_file: Optional[str]) -> Optional[torch.Tensor]:
    """Load and normalize reference audio, then sample 3x10s segments.

    Args:
        audio_file: Path to reference audio file.

    Returns:
        30-second stereo tensor from sampled front/middle/back segments,
        or ``None`` for empty/silent/error cases.
    """
    if audio_file is None:
        return None

    try:
        import torchaudio
        audio, sr = torchaudio.load(audio_file)
        logger.debug(f"[process_reference_audio] Reference audio shape: {audio.shape}")
        logger.debug(f"[process_reference_audio] Reference audio sample rate: {sr}")
        logger.debug(
            f"[process_reference_audio] Reference audio duration: {audio.shape[-1] / sr:.6f} seconds"
        )

        audio = self._normalize_audio_to_stereo_48k(audio, sr)
        if self.is_silence(audio):
            return None

        target_frames = 30 * 48000
        segment_frames = 10 * 48000

        if audio.shape[-1] < target_frames:
            repeat_times = math.ceil(target_frames / audio.shape[-1])
            audio = audio.repeat(1, repeat_times)

        total_frames = audio.shape[-1]
        segment_size = total_frames // 3

        front_start = random.randint(0, max(0, segment_size - segment_frames))
        front_audio = audio[:, front_start : front_start + segment_frames]

        middle_start = segment_size + random.randint(0, max(0, segment_size - segment_frames))
        middle_audio = audio[:, middle_start : middle_start + segment_frames]

        back_start = 2 * segment_size + random.randint(
            0, max(0, (total_frames - 2 * segment_size) - segment_frames)
        )
        back_audio = audio[:, back_start : back_start + segment_frames]

        return torch.cat([front_audio, middle_audio, back_audio], dim=-1)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"[process_reference_audio] Invalid or unsupported reference audio: {exc}")
        return None

def process_src_audio(self, audio_file: Optional[str]) -> Optional[torch.Tensor]:
    """Load and normalize source audio for remix/extract flows.

    Args:
        audio_file: Path to source audio file.

    Returns:
        Normalized stereo 48kHz tensor, or ``None`` on error/empty input.
    """
    if audio_file is None:
        return None

    try:
        import torchaudio
        audio, sr = torchaudio.load(audio_file)
        return self._normalize_audio_to_stereo_48k(audio, sr)
    except (OSError, RuntimeError, ValueError):
        logger.exception("[process_src_audio] Error processing source audio")
        return None
