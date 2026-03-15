"""AceStepLMWrapper — thin model wrapper for 5Hz LM generation.

Init methods are inlined; remaining functionality bound
programmatically from handler modules.
"""

import os
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any

import torch
from loguru import logger
from transformers import AutoTokenizer

from acestep.download_utils import get_checkpoints_dir, check_model_exists, ensure_download, ensure_lm_model, SUBMODEL_REGISTRY
from acestep.env_utils import env_is_truthy
from acestep.gpu_utils import get_global_gpu_config, get_gpu_memory_gb, get_lm_gpu_memory_ratio
from acestep.models.lm import generation as lm_generation
from acestep.models.lm import utils as lm_utils
from acestep.models.lm.backends import mlx as mlx_backend
from acestep.models.lm.backends import pt as pt_backend
from acestep.models.lm.backends import vllm as vllm_backend
from acestep.models.lm.constrained_logits_processor import MetadataConstrainedLogitsProcessor

# Module-level constants
VRAM_SAFE_FREE_GB = 2.0
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None
STOP_REASONING_TAG = "</think>"


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


class AceStepLMWrapper:
    """5Hz LM wrapper for audio code generation."""

    STOP_REASONING_TAG = STOP_REASONING_TAG
    IS_HUGGINGFACE_SPACE = IS_HUGGINGFACE_SPACE

    def __init__(
        self,
        checkpoint_dir: str | None = None,
        lm_model_path: str | None = None,
        backend: str = "vllm",
        device: str = "auto",
        offload_to_cpu: bool = False,
        dtype: torch.dtype | None = None,
        persistent_storage_path: str | None = None,
    ):
        """Initialize LM wrapper with default values, optionally loading a model."""
        self.llm = None
        self.llm_tokenizer = None
        self.llm_initialized = False
        self.llm_backend = None
        self.max_model_len = 4096
        self.device = "cpu"
        self.dtype = torch.float32
        self.offload_to_cpu = False
        self.disable_tqdm = (
            env_is_truthy("ACESTEP_DISABLE_TQDM")
            or not (hasattr(sys.stderr, 'isatty') and sys.stderr.isatty())
        )

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

        if checkpoint_dir is not None:
            self.load_model(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=offload_to_cpu,
                dtype=dtype,
            )

    # =====================================================================
    # Init / lifecycle methods (inlined from lm/init.py)
    # =====================================================================

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

    def get_available_5hz_lm_models(self, checkpoint_dir=None) -> list[str]:
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
        """Get GPU memory utilization ratio based on LM model size and available GPU memory."""
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

    def load_model(
        self,
        checkpoint_dir: str,
        lm_model_path: str | None = None,
        backend: str = "vllm",
        device: str = "auto",
        offload_to_cpu: bool = False,
        dtype: torch.dtype | None = None,
    ) -> tuple[str, bool]:
        """Resolve model path (downloading if needed) and initialize the 5Hz LM."""
        try:
            # --- Model path resolution ---
            checkpoints_dir = get_checkpoints_dir(checkpoint_dir)

            if lm_model_path is None:
                available = self.get_available_5hz_lm_models(checkpoints_dir)
                if not available:
                    ensure_download(ensure_lm_model, checkpoints_dir)
                    available = self.get_available_5hz_lm_models(checkpoints_dir)
                if not available:
                    raise RuntimeError(
                        "No LM models available. Please specify lm_model_path "
                        "or disable params.thinking."
                    )
                lm_model_path = available[0]
                logger.info("Using default LM model: {}", lm_model_path)
            else:
                lm_model_path = str(lm_model_path)
                if not (os.path.isabs(lm_model_path) and os.path.exists(lm_model_path)):
                    if not check_model_exists(lm_model_path, checkpoints_dir):
                        if lm_model_path in SUBMODEL_REGISTRY:
                            ensure_download(ensure_lm_model, lm_model_path, checkpoints_dir)
                        else:
                            raise RuntimeError(
                                f"LM model '{lm_model_path}' not found locally and not in registry. "
                                "Please provide a valid lm_model_path."
                            )

            logger.info("Initializing LM wrapper with model: {}", lm_model_path)
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

            full_lm_model_path = os.path.join(checkpoint_dir, lm_model_path)
            if not os.path.exists(full_lm_model_path):
                return f"\u274c 5Hz LM model not found at {full_lm_model_path}", False

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
                            status_msg = f"\u2705 5Hz LM initialized (PyTorch fallback from MLX)\nModel: {full_lm_model_path}\nBackend: PyTorch"
                            return status_msg, True
                        # else: backend was "vllm" on MPS, continue to vllm attempt below
                elif backend == "mlx":
                    logger.warning("MLX not available (requires Apple Silicon + mlx-lm package)")
                    # Fall back to PyTorch
                    success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                    if not success:
                        return status_msg, False
                    status_msg = f"\u2705 5Hz LM initialized (PyTorch fallback, MLX not available)\nModel: {full_lm_model_path}\nBackend: PyTorch"
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
                        f"vLLM disabled due to insufficient free VRAM (total={total_gb:.2f}GB, free={free_gb:.2f}GB, need>={VRAM_SAFE_FREE_GB}GB free) \u2014 falling back to PyTorch backend"
                    )
                    success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                    if not success:
                        return status_msg, False
                    status_msg = f"\u2705 5Hz LM initialized successfully (PyTorch fallback)\nModel: {full_lm_model_path}\nBackend: PyTorch"
                else:
                    status_msg = self._initialize_5hz_lm_vllm(
                        full_lm_model_path,
                        enforce_eager=enforce_eager_for_vllm,
                    )
                    logger.info(f"5Hz LM status message: {status_msg}")
                    if status_msg.startswith("\u274c"):
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
                            status_msg = f"\u2705 5Hz LM initialized successfully (PyTorch fallback)\nModel: {full_lm_model_path}\nBackend: PyTorch"
            elif backend != "mlx":
                success, status_msg = self._load_pytorch_model(full_lm_model_path, device)
                if not success:
                    return status_msg, False

            return status_msg, True

        except Exception as e:
            return f"\u274c Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", False

    @contextmanager
    def _load_model_context(self):
        """Context manager to load a model to GPU and offload it back to CPU after use."""
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
        # redundant CPU<->GPU transfers during batch processing.
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


# =====================================================================
# Programmatic bindings for remaining handler modules
# =====================================================================

def _collect_lm_wrapper_bindings():
    bindings = {}
    for mod in [lm_generation, lm_utils, vllm_backend, pt_backend, mlx_backend]:
        for name in mod.__all__:
            bindings[name] = getattr(mod, name)
    return bindings


for _name, _obj in _collect_lm_wrapper_bindings().items():
    setattr(AceStepLMWrapper, _name, _obj)
