"""Core LM methods (reference file for building lm_wrapper.py)."""

import os
import random
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any

import torch
import yaml
from loguru import logger
from transformers import AutoTokenizer

from acestep.constants import (
    DEFAULT_LM_INSTRUCTION,
    DURATION_MAX,
    DURATION_MIN,
)
from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor
from acestep.env_utils import env_is_truthy
from acestep.gpu_config import get_global_gpu_config, get_gpu_memory_gb, get_lm_gpu_memory_ratio

VRAM_SAFE_FREE_GB = 2.0
def _warn_if_prerelease_python():
    v = sys.version_info
    if getattr(v, "releaselevel", "final") != "final" and sys.platform.startswith("linux"):
        warnings.warn(
            f"Detected pre-release Python {sys.version.split()[0]} ({getattr(v, 'releaselevel', '')}). "
            "This is known to cause segmentation faults with vLLM/nano-vllm on Linux. "
            "Please install a stable Python release (e.g. 3.11.12+), or use --backend pt as a workaround.",
            RuntimeWarning,
            stacklevel=2,
        )

STOP_REASONING_TAG = "</think>"
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None


def __init__(self, persistent_storage_path: str | None = None):
    """Initialize AceStepLMWrapper with default values."""
    self.llm = None
    self.llm_tokenizer = None
    self.llm_initialized = False
    self.llm_backend = None
    self.max_model_len = 4096
    self.device = "cpu"
    self.dtype = torch.float32
    self.offload_to_cpu = False
    self.disable_tqdm = env_is_truthy("ACESTEP_DISABLE_TQDM") or not (hasattr(sys.stderr, 'isatty') and sys.stderr.isatty())

    # HuggingFace Space persistent storage support
    if persistent_storage_path is None and self.IS_HUGGINGFACE_SPACE:
        persistent_storage_path = "/data"
    self.persistent_storage_path = persistent_storage_path

    # Shared constrained decoding processor
    self.constrained_processor: MetadataConstrainedLogitsProcessor | None = None

    # Shared HuggingFace model for perplexity calculation
    self._hf_model_for_scoring = None

    # MLX model reference (used when llm_backend == "mlx")
    self._mlx_model = None
    self._mlx_model_path = None


def unload(self) -> None:
    """Release LM weights/tokenizer and clear caches to free memory."""
    try:
        if self.llm_backend == "vllm":
            try:
                if hasattr(self.llm, "reset"):
                    self.llm.reset()
            except Exception:
                pass
            self._cleanup_torch_distributed_state()
        self.llm = None
        self.llm_tokenizer = None
        self.constrained_processor = None
        self.llm_initialized = False
        self.llm_backend = None
        self._mlx_model = None
        self._mlx_model_path = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    except Exception:
        pass


def _cleanup_torch_distributed_state(self) -> None:
    """Destroy default torch distributed process group when already initialized."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            logger.warning("[LLM vLLM] Destroying stale default process group before/after vLLM lifecycle")
            dist.destroy_process_group()
    except Exception as exc:
        logger.warning(f"[LLM vLLM] Failed to clean torch distributed state: {exc}")


def _get_checkpoint_dir(self) -> str:
    """Get checkpoint directory, prioritizing persistent storage"""
    if self.persistent_storage_path:
        return os.path.join(self.persistent_storage_path, "checkpoints")
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return os.path.join(project_root, "checkpoints")


def get_available_5hz_lm_models(self, checkpoint_dir = None) -> list[str]:
    """Scan and return all model directory names starting with 'acestep-5Hz-lm-'"""
    if not checkpoint_dir:
        checkpoint_dir = self._get_checkpoint_dir()

    models = []
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("acestep-5Hz-lm-"):
                models.append(item)

    models.sort()
    return models


def get_gpu_memory_utilization(self, model_path: str = None, minimal_gpu: float = 8, min_ratio: float = 0.2, max_ratio: float = 0.9) -> tuple[float, bool]:
    """
    Get GPU memory utilization ratio based on LM model size and available GPU memory.

    Args:
        model_path: LM model path (e.g., "acestep-5Hz-lm-0.6B"). Used to determine target memory.
        minimal_gpu: Minimum GPU memory requirement in GB (fallback)
        min_ratio: Minimum memory utilization ratio
        max_ratio: Maximum memory utilization ratio

    Returns:
        Tuple of (gpu_memory_utilization_ratio, low_gpu_memory_mode)
    """
    try:
        device = torch.device("cuda:0")
        total_gpu_mem_bytes = torch.cuda.get_device_properties(device).total_memory
        total_gpu = total_gpu_mem_bytes / 1024**3

        low_gpu_memory_mode = False

        # Use adaptive GPU memory ratio based on model size
        if model_path:
            ratio, target_memory_gb = get_lm_gpu_memory_ratio(model_path, total_gpu)
            logger.info(f"Adaptive LM memory allocation: model={model_path}, target={target_memory_gb}GB, ratio={ratio:.3f}, total_gpu={total_gpu:.1f}GB")

            # Enable low memory mode for small GPUs
            if total_gpu < 8:
                low_gpu_memory_mode = True

            return ratio, low_gpu_memory_mode

        # Fallback to original logic if no model_path provided
        reserved_mem_bytes = torch.cuda.memory_reserved(device)
        reserved_gpu = reserved_mem_bytes / 1024**3
        available_gpu = total_gpu - reserved_gpu

        if total_gpu < minimal_gpu:
            minimal_gpu = 0.5 * total_gpu
            low_gpu_memory_mode = True

        if available_gpu >= minimal_gpu:
            ratio = min(max_ratio, max(min_ratio, minimal_gpu / total_gpu))
        else:
            ratio = min(max_ratio, max(min_ratio, (available_gpu * 0.8) / total_gpu))

        return ratio, low_gpu_memory_mode
    except Exception as e:
        logger.warning(f"Failed to calculate GPU memory utilization: {e}")
        return 0.9, False


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


def initialize(
    self,
    checkpoint_dir: str,
    lm_model_path: str,
    backend: str = "vllm",
    device: str = "auto",
    offload_to_cpu: bool = False,
    dtype: torch.dtype | None = None,
) -> tuple[str, bool]:
    """
    Initialize 5Hz LM model

    Args:
        checkpoint_dir: Checkpoint directory path
        lm_model_path: LM model path (relative to checkpoint_dir)
        backend: Backend type ("vllm" or "pt")
        device: Device type ("auto", "cuda", "mps", "xpu", or "cpu")
        offload_to_cpu: Whether to offload to CPU
        dtype: Data type (if None, auto-detect based on device)

    Returns:
        (status_message, success)
    """
    try:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.warning("[initialize] CUDA requested but unavailable. Falling back to MPS.")
                device = "mps"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                logger.warning("[initialize] CUDA requested but unavailable. Falling back to XPU.")
                device = "xpu"
            else:
                logger.warning("[initialize] CUDA requested but unavailable. Falling back to CPU.")
                device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            if torch.cuda.is_available():
                logger.warning("[initialize] MPS requested but unavailable. Falling back to CUDA.")
                device = "cuda"
            elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                logger.warning("[initialize] MPS requested but unavailable. Falling back to XPU.")
                device = "xpu"
            else:
                logger.warning("[initialize] MPS requested but unavailable. Falling back to CPU.")
                device = "cpu"
        elif device == "xpu" and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            if torch.cuda.is_available():
                logger.warning("[initialize] XPU requested but unavailable. Falling back to CUDA.")
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.warning("[initialize] XPU requested but unavailable. Falling back to MPS.")
                device = "mps"
            else:
                logger.warning("[initialize] XPU requested but unavailable. Falling back to CPU.")
                device = "cpu"

        self.device = device
        self.offload_to_cpu = offload_to_cpu

        # Set dtype based on device: bfloat16 for cuda/xpu, float32 for mps/cpu
        # Note: LLM stays in float32 on MPS because autoregressive generation is
        # latency-bound (not compute-bound), and many LLM weights trained in bfloat16
        # produce NaN/inf when naively converted to float16 (different exponent range).
        # The DiT and VAE use float16 on MPS where it actually helps throughput.
        if dtype is None:
            if device in ["cuda", "xpu"]:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype
            # Keep LM in float32 on MPS for stability.
            if device == "mps" and self.dtype != torch.float32:
                logger.warning(
                    f"[initialize] Overriding requested dtype {self.dtype} to float32 for LM on MPS."
                )
                self.dtype = torch.float32

        # If lm_model_path is None, use default
        if lm_model_path is None:
            lm_model_path = "acestep-5Hz-lm-1.7B"
            logger.info(f"[initialize] lm_model_path is None, using default: {lm_model_path}")

        full_lm_model_path = os.path.join(checkpoint_dir, lm_model_path)
        if not os.path.exists(full_lm_model_path):
            return f"❌ 5Hz LM model not found at {full_lm_model_path}", False

        # Proactive CUDA cleanup before LM load to reduce fragmentation on mode/model switch
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("loading 5Hz LM tokenizer... it may take 80~90s")
        start_time = time.time()
        # TODO: load tokenizer too slow, not found solution yet
        llm_tokenizer = AutoTokenizer.from_pretrained(full_lm_model_path, use_fast=True)
        logger.info(f"5Hz LM tokenizer loaded successfully in {time.time() - start_time:.2f} seconds")
        self.llm_tokenizer = llm_tokenizer

        # Initialize shared constrained decoding processor (one-time initialization)
        # Use GPU-based max_duration to limit duration values in constrained decoding
        logger.info("Initializing constrained decoding processor...")
        processor_start = time.time()

        gpu_config = get_global_gpu_config()
        # Use max_duration_with_lm since LM is being initialized
        max_duration_for_constraint = gpu_config.max_duration_with_lm
        logger.info(f"Setting constrained decoding max_duration to {max_duration_for_constraint}s based on GPU config (tier: {gpu_config.tier})")

        self.constrained_processor = MetadataConstrainedLogitsProcessor(
            tokenizer=self.llm_tokenizer,
            enabled=True,
            debug=False,
            max_duration=max_duration_for_constraint,
        )
        logger.info(f"Constrained processor initialized in {time.time() - processor_start:.2f} seconds")

        # Disable CUDA/HIP graph capture on ROCm (unverified on RDNA3 Windows)
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        enforce_eager_for_vllm = bool(is_rocm)

        # Auto-detect best backend on Apple Silicon
        if backend == "mlx" or (backend == "vllm" and device == "mps"):
            # On Apple Silicon, prefer MLX (native acceleration) over PyTorch MPS
            if self._is_mlx_available():
                logger.info("Attempting MLX backend for Apple Silicon acceleration...")
                mlx_success, mlx_status = self._load_mlx_model(full_lm_model_path)
                if mlx_success:
                    return mlx_status, True
                else:
                    logger.warning(f"MLX backend failed: {mlx_status}")
                    if backend == "mlx":
                        # User explicitly requested MLX, fall back to PyTorch
                        logger.warning("MLX explicitly requested but failed, falling back to PyTorch backend")
                        success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                        if not success:
                            return status_msg, False
                        status_msg = f"✅ 5Hz LM initialized (PyTorch fallback from MLX)\nModel: {full_lm_model_path}\nBackend: PyTorch"
                        return status_msg, True
                    # else: backend was "vllm" on MPS, continue to vllm attempt below
            elif backend == "mlx":
                logger.warning("MLX not available (requires Apple Silicon + mlx-lm package)")
                # Fall back to PyTorch
                success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                if not success:
                    return status_msg, False
                status_msg = f"✅ 5Hz LM initialized (PyTorch fallback, MLX not available)\nModel: {full_lm_model_path}\nBackend: PyTorch"
                return status_msg, True

        if backend == "vllm" and device != "cuda":
            logger.info(
                f"[initialize] vllm backend requires CUDA, using PyTorch backend for device={device}."
            )
            backend = "pt"

        # Initialize based on user-selected backend
        if backend == "vllm":
            _warn_if_prerelease_python()
            total_gb = get_gpu_memory_gb() if device == "cuda" else 0.0
            free_gb = 0.0
            if device == "cuda" and torch.cuda.is_available():
                try:
                    if hasattr(torch.cuda, "mem_get_info"):
                        free_bytes, _ = torch.cuda.mem_get_info()
                        free_gb = free_bytes / (1024**3)
                    else:
                        total_bytes = torch.cuda.get_device_properties(0).total_memory
                        free_gb = (total_bytes - torch.cuda.memory_reserved(0)) / (1024**3)
                except Exception:
                    free_gb = 0.0
            if device == "cuda" and free_gb < VRAM_SAFE_FREE_GB:
                logger.warning(
                    f"vLLM disabled due to insufficient free VRAM (total={total_gb:.2f}GB, free={free_gb:.2f}GB, need>={VRAM_SAFE_FREE_GB}GB free) — falling back to PyTorch backend"
                )
                success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                if not success:
                    return status_msg, False
                status_msg = f"✅ 5Hz LM initialized successfully (PyTorch fallback)\nModel: {full_lm_model_path}\nBackend: PyTorch"
            else:
                status_msg = self._initialize_5hz_lm_vllm(
                    full_lm_model_path,
                    enforce_eager=enforce_eager_for_vllm,
                )
                logger.info(f"5Hz LM status message: {status_msg}")
                if status_msg.startswith("❌"):
                    if not self.llm_initialized:
                        if device == "mps" and self._is_mlx_available():
                            logger.warning("vllm failed on MPS, trying MLX backend...")
                            mlx_success, mlx_status = self._load_mlx_model(full_lm_model_path)
                            if mlx_success:
                                return mlx_status, True
                            logger.warning(f"MLX also failed: {mlx_status}, falling back to PyTorch")
                        logger.warning("Falling back to PyTorch backend")
                        success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                        if not success:
                            return status_msg, False
                        status_msg = f"✅ 5Hz LM initialized successfully (PyTorch fallback)\nModel: {full_lm_model_path}\nBackend: PyTorch"
        elif backend != "mlx":
            success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
            if not success:
                return status_msg, False

        return status_msg, True

    except Exception as e:
        return f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", False


def has_all_metas(self, user_metadata: dict[str, str | None] | None) -> bool:
    """Check if all required metadata are present."""
    if user_metadata is None:
        return False
    if 'bpm' in user_metadata and 'keyscale' in user_metadata and 'timesignature' in user_metadata and 'duration' in user_metadata:
        return True
    return False


def _format_metadata_as_cot(self, metadata: dict[str, Any]) -> str:
    """
    Format parsed metadata as CoT text using YAML format (matching training format).

    Args:
        metadata: Dictionary with keys: bpm, caption, duration, keyscale, language, timesignature

    Returns:
        Formatted CoT text: "<think>\n{yaml_content}\n</think>"
    """
    # Build cot_items dict with only non-None values
    cot_items = {}
    for key in ['bpm', 'caption', 'duration', 'keyscale', 'language', 'timesignature']:
        if key in metadata and metadata[key] is not None:
            value = metadata[key]
            if key == "timesignature" and value.endswith("/4"):
                value = value.split("/")[0]
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            cot_items[key] = value

    # Format as YAML (sorted keys, unicode support)
    if len(cot_items) > 0:
        cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()
    else:
        cot_yaml = ""

    return f"<think>\n{cot_yaml}\n</think>"


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
                import re
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


def build_formatted_prompt(self, caption: str, lyrics: str = "", is_negative_prompt: bool = False, generation_phase: str = "cot", negative_prompt: str = "NO USER INPUT") -> str:
    """
    Build the chat-formatted prompt for 5Hz LM from caption/lyrics.
    Raises a ValueError if the tokenizer is not initialized.

    Args:
        caption: Caption text
        lyrics: Lyrics text
        is_negative_prompt: If True, builds unconditional prompt for CFG
        generation_phase: "cot" or "codes" - affects unconditional prompt format
        negative_prompt: Negative prompt for CFG (used when is_negative_prompt=True)

    Example:
        prompt = handler.build_formatted_prompt("calm piano", "hello world")
    """
    if self.llm_tokenizer is None:
        raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")

    if is_negative_prompt:
        # Unconditional prompt for CFG
        # Check if user provided a meaningful negative prompt (not the default)
        has_negative_prompt = self._has_meaningful_negative_prompt(negative_prompt)

        if generation_phase == "cot":
            # CoT phase unconditional prompt
            if has_negative_prompt:
                # If negative prompt provided, use it as caption
                prompt = f"# Caption\n{negative_prompt}\n\n# Lyric\n{lyrics}\n"
            else:
                # No negative prompt: remove caption, keep only lyrics
                prompt = f"# Lyric\n{lyrics}\n"
        else:
            # Codes phase: will be handled by build_formatted_prompt_with_cot
            # For backward compatibility, use simple caption as before
            prompt = caption
    else:
        # Conditional prompt: include both caption and lyrics
        prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"

    return self.llm_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_formatted_prompt_with_cot(self, caption: str, lyrics: str, cot_text: str, is_negative_prompt: bool = False, negative_prompt: str = "NO USER INPUT") -> str:
    """
    Build the chat-formatted prompt for codes generation phase with pre-generated CoT.

    Args:
        caption: Caption text
        lyrics: Lyrics text
        cot_text: Pre-generated CoT text (e.g., "<think>\\nbpm: 120\\n...\\n</think>")
        is_negative_prompt: If True, uses empty CoT for CFG unconditional prompt
        negative_prompt: Negative prompt for CFG (used when is_negative_prompt=True)

    Returns:
        Formatted prompt string

    Example:
        cot = "<think>\\nbpm: 120\\ncaption: calm piano\\n...\\n</think>"
        prompt = handler.build_formatted_prompt_with_cot("calm piano", "hello", cot)
    """
    if self.llm_tokenizer is None:
        raise ValueError("LLM tokenizer is not initialized. Call initialize() first.")

    if is_negative_prompt:
        # Unconditional prompt for codes phase
        # Check if user provided a meaningful negative prompt
        has_negative_prompt = self._has_meaningful_negative_prompt(negative_prompt)

        # Use empty CoT for unconditional
        cot_for_prompt = "<think>\n</think>"

        if has_negative_prompt:
            # If negative prompt provided, use it as caption
            caption_for_prompt = negative_prompt
        else:
            # No negative prompt: use original caption
            caption_for_prompt = caption
    else:
        # Conditional prompt: use the full CoT and original caption
        cot_for_prompt = cot_text
        caption_for_prompt = caption

    # Build user prompt with caption and lyrics ONLY (no COT)
    # COT should be in the assistant's message, not user's
    user_prompt = f"# Caption\n{caption_for_prompt}\n\n# Lyric\n{lyrics}\n"

    # Build the chat with assistant message containing the COT
    # The model will continue generation after the COT
    formatted = self.llm_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": cot_for_prompt},
        ],
        tokenize=False,
        add_generation_prompt=False,  # Don't add generation prompt, COT is already in assistant
    )

    # Add a newline after </think> so model generates audio codes on next line
    if not formatted.endswith('\n'):
        formatted += '\n'

    return formatted


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
        import traceback
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


def parse_lm_output(self, output_text: str) -> tuple[dict[str, Any], str]:
    """
    Parse LM output to extract metadata and audio codes.

    Expected format:
    <think>
    bpm: 73
    caption: A calm piano melody
    duration: 273
    genres: Chinese folk
    keyscale: G major
    language: en
    timesignature: 4
    </think>

    <|audio_code_56535|><|audio_code_62918|>...

    Returns:
        Tuple of (metadata_dict, audio_codes_string)
    """
    debug_output_text = output_text.split("</think>")[0]
    logger.debug(f"Debug output text: {debug_output_text}")
    metadata = {}
    audio_codes = ""

    import re

    # Extract audio codes - find all <|audio_code_XXX|> patterns
    code_pattern = r'<\|audio_code_\d+\|>'
    code_matches = re.findall(code_pattern, output_text)
    if code_matches:
        audio_codes = "".join(code_matches)

    # Extract metadata from reasoning section
    # Try different reasoning tag patterns
    reasoning_patterns = [
        r'<think>(.*?)</think>',
        r'<think>(.*?)</think>',
        r'<reasoning>(.*?)</reasoning>',
    ]

    reasoning_text = None
    for pattern in reasoning_patterns:
        match = re.search(pattern, output_text, re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()
            break

    # If no reasoning tags found, try to parse metadata from the beginning of output
    if not reasoning_text:
        # Look for metadata lines before audio codes
        lines_before_codes = output_text.split('<|audio_code_')[0] if '<|audio_code_' in output_text else output_text
        reasoning_text = lines_before_codes.strip()

    # Parse metadata fields with YAML multi-line value support
    if reasoning_text:
        lines = reasoning_text.split('\n')
        current_key = None
        current_value_lines = []

        def save_current_field():
            """Save the accumulated field value"""
            nonlocal current_key, current_value_lines
            if current_key and current_value_lines:
                # Join multi-line value
                value = '\n'.join(current_value_lines)

                if current_key == 'bpm':
                    try:
                        metadata['bpm'] = int(value.strip())
                    except (ValueError, TypeError):
                        metadata['bpm'] = value.strip()
                elif current_key == 'caption':
                    # Post-process caption to remove YAML multi-line formatting
                    metadata['caption'] = MetadataConstrainedLogitsProcessor.postprocess_caption(value)
                elif current_key == 'duration':
                    try:
                        metadata['duration'] = int(value.strip())
                    except (ValueError, TypeError):
                        metadata['duration'] = value.strip()
                elif current_key == 'genres':
                    metadata['genres'] = value.strip()
                elif current_key == 'keyscale':
                    metadata['keyscale'] = value.strip()
                elif current_key == 'language':
                    metadata['language'] = value.strip()
                    metadata['vocal_language'] = value.strip()
                elif current_key == 'timesignature':
                    metadata['timesignature'] = value.strip()
                elif current_key == 'lyrics':
                    metadata['lyrics'] = value.strip()

            current_key = None
            current_value_lines = []

        for line in lines:
            # Skip lines starting with '<' (tags)
            if line.strip().startswith('<'):
                continue

            # Check if this is a new field (no leading spaces and contains ':')
            if line and not line[0].isspace() and ':' in line:
                # Save previous field if any
                save_current_field()

                # Parse new field
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_key = parts[0].strip().lower()
                    # First line of value (after colon)
                    first_value = parts[1]
                    if first_value.strip():
                        current_value_lines.append(first_value)
            elif line.startswith(' ') or line.startswith('\t'):
                # Continuation line (YAML multi-line value)
                if current_key:
                    current_value_lines.append(line)

        # Don't forget to save the last field
        save_current_field()

    return metadata, audio_codes


@contextmanager
def _load_model_context(self):
    """
    Context manager to load a model to GPU and offload it back to CPU after use.
    Only used for PyTorch backend when offload_to_cpu is True.
    """
    if not self.offload_to_cpu:
        yield
        return

    # If using nanovllm or MLX, do not offload (managed differently)
    if self.llm_backend in ("vllm", "mlx"):
        yield
        return

    model = self.llm
    if model is None:
        yield
        return

    # Reentrancy guard: if an outer context already loaded the model
    # to the target device, skip the inner load/offload to avoid
    # redundant CPU↔GPU transfers during batch processing.
    try:
        current_device = next(model.parameters()).device.type
    except StopIteration:
        current_device = None
    target_device = str(self.device).split(":")[0]
    if current_device == target_device:
        yield
        return

    # Load to GPU
    logger.info(f"Loading LLM to {self.device}")
    start_time = time.time()
    if hasattr(model, "to"):
        model.to(self.device).to(self.dtype)
    load_time = time.time() - start_time
    logger.info(f"Loaded LLM to {self.device} in {load_time:.4f}s")

    try:
        yield
    finally:
        # Offload to CPU
        logger.info("Offloading LLM to CPU")
        start_time = time.time()
        if hasattr(model, "to"):
            model.to("cpu")
        # Clear accelerator cache after offloading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        offload_time = time.time() - start_time
        logger.info(f"Offloaded LLM to CPU in {offload_time:.4f}s")
