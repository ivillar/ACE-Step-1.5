"""Consolidated handler functions – init module."""

import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from acestep import gpu_config
from acestep.model_downloader import (
    check_main_model_exists,
    check_model_exists,
    ensure_dit_model,
    ensure_main_model,
)

# --- From init_service_setup.py ---

def _resolve_initialize_device(self, requested_device: str) -> str:
    """Resolve a concrete runtime device, applying backend fallback rules."""
    device = requested_device
    if device == "auto":
        if gpu_config.is_cuda_available():
            return "cuda"
        if gpu_config.is_mps_available():
            return "mps"
        if gpu_config.is_xpu_available():
            return "xpu"
        return "cpu"

    if device == "cuda" and not gpu_config.is_cuda_available():
        if gpu_config.is_mps_available():
            logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to MPS.")
            return "mps"
        if gpu_config.is_xpu_available():
            logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to XPU.")
            return "xpu"
        logger.warning("[initialize_service] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if device == "mps" and not gpu_config.is_mps_available():
        if gpu_config.is_cuda_available():
            logger.warning("[initialize_service] MPS requested but unavailable. Falling back to CUDA.")
            return "cuda"
        if gpu_config.is_xpu_available():
            logger.warning("[initialize_service] MPS requested but unavailable. Falling back to XPU.")
            return "xpu"
        logger.warning("[initialize_service] MPS requested but unavailable. Falling back to CPU.")
        return "cpu"

    if device == "xpu" and not gpu_config.is_xpu_available():
        if gpu_config.is_cuda_available():
            logger.warning("[initialize_service] XPU requested but unavailable. Falling back to CUDA.")
            return "cuda"
        if gpu_config.is_mps_available():
            logger.warning("[initialize_service] XPU requested but unavailable. Falling back to MPS.")
            return "mps"
        logger.warning("[initialize_service] XPU requested but unavailable. Falling back to CPU.")
        return "cpu"

    return device

def _configure_initialize_runtime(
    self,
    *,
    device: str,
    compile_model: bool,
    quantization: str | None,
) -> tuple[bool, str | None, bool]:
    """Apply backend constraints and return normalized compile/quantization settings."""
    mlx_compile_requested = False
    normalized_compile = compile_model
    normalized_quantization = quantization

    if device == "mps":
        if normalized_compile:
            logger.info(
                "[initialize_service] MPS detected: torch.compile is not "
                "supported - redirecting to mx.compile for MLX components."
            )
            mlx_compile_requested = True
            normalized_compile = False
        if normalized_quantization is not None:
            logger.warning("[initialize_service] Quantization (torchao) is not supported on MPS; disabling.")
            normalized_quantization = None

    return normalized_compile, normalized_quantization, mlx_compile_requested

@staticmethod
def _ensure_len_for_compile(model: Any, method_name: str) -> None:
    """Inject a fallback ``__len__`` implementation for torch.compile introspection.

    Args:
        model: Model instance whose class may need a ``__len__`` shim.
        method_name: Label used in debug logs to identify the target model.
    """
    if hasattr(model.__class__, "__len__"):
        return

    def _len_impl(_model_self):
        """Return a neutral length for torch.compile compatibility."""
        return 0

    model.__class__.__len__ = _len_impl
    logger.debug(f"[initialize_service] Injected __len__ into {method_name} class for torch.compile")

def _validate_quantization_setup(self, *, quantization: str | None, compile_model: bool) -> None:
    """Validate quantization prerequisites before model loading."""
    if quantization is None:
        return
    if not compile_model:
        raise ValueError("Quantization requires compile_model to be True")
    try:
        import torchao  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchao is required for quantization but is not installed. "
            "Please install torchao to use quantization features."
        ) from exc

def _initialize_mlx_backends(
    self,
    *,
    device: str,
    use_mlx_dit: bool,
    mlx_compile_requested: bool,
) -> tuple[str, str]:
    """Initialize MLX DiT/VAE integrations and return status labels."""
    mlx_dit_status = "Disabled"
    if use_mlx_dit and device in ("mps", "cpu"):
        mlx_ok = self._init_mlx_dit(compile_model=mlx_compile_requested)
        if mlx_ok:
            mlx_dit_status = (
                "Active (native MLX, mx.compile)"
                if mlx_compile_requested
                else "Active (native MLX)"
            )
        else:
            mlx_dit_status = "Unavailable (PyTorch fallback)"
    elif not use_mlx_dit:
        mlx_dit_status = "Disabled by user"
        self.mlx_decoder = None
        self.use_mlx_dit = False

    mlx_vae_status = "Disabled"
    if device in ("mps", "cpu"):
        mlx_vae_ok = self._init_mlx_vae()
        mlx_vae_status = "Active (native MLX)" if mlx_vae_ok else "Unavailable (PyTorch fallback)"
    else:
        self.mlx_vae = None
        self.use_mlx_vae = False

    return mlx_dit_status, mlx_vae_status

@staticmethod
def _build_initialize_status_message(
    *,
    device: str,
    model_path: str,
    vae_path: str,
    text_encoder_path: str,
    dtype: torch.dtype,
    attention: str,
    compile_model: bool,
    mlx_compile_requested: bool,
    offload_to_cpu: bool,
    offload_dit_to_cpu: bool,
    mlx_dit_status: str,
    mlx_vae_status: str,
) -> str:
    """Format initialize_service status output for UI/API consumers."""
    status_msg = f"[OK] Model initialized successfully on {device}\n"
    status_msg += f"Main model: {model_path}\n"
    status_msg += f"VAE: {vae_path}\n"
    status_msg += f"Text encoder: {text_encoder_path}\n"
    status_msg += f"Dtype: {dtype}\n"
    status_msg += f"Attention: {attention}\n"
    compiled_label = "mx.compile (MLX)" if mlx_compile_requested else str(compile_model)
    status_msg += f"Compiled: {compiled_label}\n"
    status_msg += f"Offload to CPU: {offload_to_cpu}\n"
    status_msg += f"Offload DiT to CPU: {offload_dit_to_cpu}\n"
    status_msg += f"MLX DiT: {mlx_dit_status}\n"
    status_msg += f"MLX VAE: {mlx_vae_status}"
    return status_msg

# --- From init_service_catalog.py ---

def _device_type(self) -> str:
    """Normalize the host device value to a backend type string."""
    if isinstance(self.device, str):
        return self.device.split(":", 1)[0]
    return self.device.type

def get_available_checkpoints(self) -> list[str]:
    """Return available checkpoint directory paths under the project root."""
    project_root = self._get_project_root()
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    if os.path.exists(checkpoint_dir):
        return [checkpoint_dir]
    return []

def get_available_acestep_v15_models(self, checkpoints_dir=None) -> list[str]:
    """Scan and return all model directory names starting with ``acestep-v15-``."""
    project_root = self._get_project_root()
    if not checkpoints_dir:
        checkpoints_dir = os.path.join(project_root, "checkpoints")

    models = []
    if os.path.exists(checkpoints_dir):
        for item in os.listdir(checkpoints_dir):
            item_path = os.path.join(checkpoints_dir, item)
            if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                models.append(item)

    models.sort()
    return models

def is_flash_attention_available(self, device: str | None = None) -> bool:
    """Check whether flash attention can be used on the target device."""
    target_device = str(device or self.device or "auto").split(":", 1)[0]
    if target_device == "auto":
        if not torch.cuda.is_available():
            return False
    else:
        if target_device != "cuda" or not torch.cuda.is_available():
            return False

    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.info(
                f"[is_flash_attention_available] GPU compute capability {major}.x < 8.0 "
                f"(pre-Ampere) — FlashAttention not supported, will use SDPA instead."
            )
            return False
    except Exception:
        return False

    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False

def is_turbo_model(self) -> bool:
    """Check whether the currently loaded model is a turbo variant."""
    if self.config is None:
        return False
    return getattr(self.config, "is_turbo", False)

# --- From init_service_downloads.py ---

def _ensure_models_present(
    self,
    *,
    checkpoint_path: Path,
    config_path: str,
    prefer_source: str | None,
) -> tuple[str, bool] | None:
    """Ensure required checkpoint assets exist locally, downloading when missing."""
    if not check_main_model_exists(checkpoint_path):
        logger.info("[initialize_service] Main model not found, starting auto-download...")
        success, msg = ensure_main_model(checkpoint_path, prefer_source=prefer_source)
        if not success:
            return f"ERROR: Failed to download main model: {msg}", False
        logger.info(f"[initialize_service] {msg}")

    if config_path == "":
        logger.warning(
            "[initialize_service] Empty config_path; pass None to use the default model."
        )

    if not check_model_exists(config_path, checkpoint_path):
        logger.info(f"[initialize_service] DiT model '{config_path}' not found, starting auto-download...")
        success, msg = ensure_dit_model(config_path, checkpoint_path, prefer_source=prefer_source)
        if not success:
            return f"ERROR: Failed to download DiT model '{config_path}': {msg}", False
        logger.info(f"[initialize_service] {msg}")

    return None

@staticmethod
def _sync_model_code_if_needed(config_path: str, checkpoint_path: Path) -> None:
    """Sync model-side python files when checkpoint code metadata diverges."""
    from acestep.model_downloader import _check_code_mismatch, _sync_model_code_files

    mismatched = _check_code_mismatch(config_path, checkpoint_path)
    if mismatched:
        logger.warning(
            f"[initialize_service] Model code mismatch detected for '{config_path}': "
            f"{mismatched}. Auto-syncing from acestep/models/..."
        )
        _sync_model_code_files(config_path, checkpoint_path)
        logger.info("[initialize_service] Model code files synced successfully.")

# --- From init_service_loader.py ---

def _load_main_model_from_checkpoint(
    self,
    *,
    model_checkpoint_path: str,
    device: str,
    use_flash_attention: bool,
    compile_model: bool,
    quantization: str | None,
) -> str:
    """Load DiT, apply compile/quantization options, and return selected attention backend."""
    from transformers import AutoModel

    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {model_checkpoint_path}")

    if torch.cuda.is_available():
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if use_flash_attention and self.is_flash_attention_available(device):
        attn_implementation = "flash_attention_2"
    else:
        if use_flash_attention:
            logger.warning(
                f"[initialize_service] Flash attention requested but unavailable for device={device}. "
                "Falling back to SDPA."
            )
        attn_implementation = "sdpa"

    attn_candidates = [attn_implementation]
    if "sdpa" not in attn_candidates:
        attn_candidates.append("sdpa")
    if "eager" not in attn_candidates:
        attn_candidates.append("eager")

    last_attn_error = None
    self.model = None
    for candidate in attn_candidates:
        try:
            logger.info(f"[initialize_service] Attempting to load model with attention implementation: {candidate}")
            self.model = AutoModel.from_pretrained(
                model_checkpoint_path,
                trust_remote_code=True,
                attn_implementation=candidate,
                torch_dtype=self.dtype,
            )
            attn_implementation = candidate
            break
        except Exception as exc:
            last_attn_error = exc
            logger.warning(f"[initialize_service] Failed to load model with {candidate}: {exc}")

    if self.model is None:
        raise RuntimeError(
            f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
        ) from last_attn_error

    self.model.config._attn_implementation = attn_implementation
    self.config = self.model.config

    if not self.offload_to_cpu:
        self.model = self.model.to(device).to(self.dtype)
    elif not self.offload_dit_to_cpu:
        logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
        self.model = self.model.to(device).to(self.dtype)
    else:
        self.model = self.model.to("cpu").to(self.dtype)
    self.model.eval()

    if compile_model:
        self._ensure_len_for_compile(self.model, "model")
        self.model = torch.compile(self.model)

        if quantization is not None:
            from torchao.quantization import quantize_
            from torchao.quantization.quant_api import _is_linear
            if quantization == "int8_weight_only":
                from torchao.quantization import Int8WeightOnlyConfig
                quant_config = Int8WeightOnlyConfig()
            elif quantization == "fp8_weight_only":
                from torchao.quantization import Float8WeightOnlyConfig
                quant_config = Float8WeightOnlyConfig()
            elif quantization == "w8a8_dynamic":
                from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
                quant_config = Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
            else:
                raise ValueError(f"Unsupported quantization type: {quantization}")

            def _dit_filter_fn(module, fqn):
                """Keep only DiT linear layers and exclude tokenizer/detokenizer paths."""
                if not _is_linear(module, fqn):
                    return False
                for part in fqn.split("."):
                    if part in ("tokenizer", "detokenizer"):
                        return False
                return True

            quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
            logger.info(f"[initialize_service] DiT quantized with: {quantization}")

    silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
    if not os.path.exists(silence_latent_path):
        raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
    self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
    self.silence_latent = self.silence_latent.to(device).to(self.dtype)
    return attn_implementation

def _load_vae_model(self, *, checkpoint_dir: str, device: str, compile_model: bool) -> str:
    """Load and optionally compile the VAE module."""
    from diffusers.models import AutoencoderOobleck

    vae_checkpoint_path = os.path.join(checkpoint_dir, "vae")
    if not os.path.exists(vae_checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")

    self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
    if not self.offload_to_cpu:
        vae_dtype = self._get_vae_dtype(device)
        self.vae = self.vae.to(device).to(vae_dtype)
    else:
        vae_dtype = self._get_vae_dtype("cpu")
        self.vae = self.vae.to("cpu").to(vae_dtype)
    self.vae.eval()

    if compile_model:
        self._ensure_len_for_compile(self.vae, "vae")
        self.vae = torch.compile(self.vae)

    return vae_checkpoint_path

def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir: str, device: str) -> str:
    """Load text tokenizer and embedding model."""
    from transformers import AutoModel, AutoTokenizer

    text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
    if not os.path.exists(text_encoder_path):
        raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")

    self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
    if not self.offload_to_cpu:
        self.text_encoder = self.text_encoder.to(device).to(self.dtype)
    else:
        self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)
    self.text_encoder.eval()
    return text_encoder_path

# --- From init_service_orchestrator.py ---

def initialize_service(
    self,
    project_root: str,
    config_path: str,
    device: str = "auto",
    use_flash_attention: bool = False,
    compile_model: bool = False,
    offload_to_cpu: bool = False,
    offload_dit_to_cpu: bool = False,
    quantization: str | None = None,
    prefer_source: str | None = None,
    checkpoint_dir: str | None = None,
    use_mlx_dit: bool = True,
) -> tuple[str, bool]:
    """Initialize model artifacts and runtime backends for generation.

    This method intentionally supports repeated calls to reinitialize models
    with new settings; it does not short-circuit when components are already loaded.
    """
    try:
        if config_path is None:
            config_path = "acestep-v15-turbo"
            logger.warning(
                "[initialize_service] config_path not set; defaulting to 'acestep-v15-turbo'."
            )

        resolved_device = self._resolve_initialize_device(device)
        self.device = resolved_device
        self.offload_to_cpu = offload_to_cpu
        self.offload_dit_to_cpu = offload_dit_to_cpu

        normalized_compile, normalized_quantization, mlx_compile_requested = self._configure_initialize_runtime(
            device=resolved_device,
            compile_model=compile_model,
            quantization=quantization,
        )
        self.compiled = normalized_compile
        self.dtype = torch.bfloat16 if resolved_device in ["cuda", "xpu"] else torch.float32
        self.quantization = normalized_quantization
        self._validate_quantization_setup(
            quantization=self.quantization,
            compile_model=normalized_compile,
        )

        base_root = project_root or self._get_project_root()
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(base_root, "checkpoints")
        checkpoint_path = Path(checkpoint_dir)

        precheck_failure = self._ensure_models_present(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            prefer_source=prefer_source,
        )
        if precheck_failure is not None:
            self.model = None
            self.vae = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.config = None
            self.silence_latent = None
            return precheck_failure

        self._sync_model_code_if_needed(config_path, checkpoint_path)

        model_path = os.path.join(checkpoint_dir, config_path)
        self._load_main_model_from_checkpoint(
            model_checkpoint_path=model_path,
            device=resolved_device,
            use_flash_attention=use_flash_attention,
            compile_model=normalized_compile,
            quantization=self.quantization,
        )
        vae_path = self._load_vae_model(
            checkpoint_dir=checkpoint_dir,
            device=resolved_device,
            compile_model=normalized_compile,
        )
        text_encoder_path = self._load_text_encoder_and_tokenizer(
            checkpoint_dir=checkpoint_dir,
            device=resolved_device,
        )

        mlx_dit_status, mlx_vae_status = self._initialize_mlx_backends(
            device=resolved_device,
            use_mlx_dit=use_mlx_dit,
            mlx_compile_requested=mlx_compile_requested,
        )

        status_msg = self._build_initialize_status_message(
            device=resolved_device,
            model_path=model_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            dtype=self.dtype,
            attention=getattr(self.config, "_attn_implementation", "eager"),
            compile_model=normalized_compile,
            mlx_compile_requested=mlx_compile_requested,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            mlx_dit_status=mlx_dit_status,
            mlx_vae_status=mlx_vae_status,
        )

        self.last_init_params = {
            "project_root": project_root,
            "config_path": config_path,
            "device": resolved_device,
            "use_flash_attention": use_flash_attention,
            "compile_model": normalized_compile,
            "offload_to_cpu": offload_to_cpu,
            "offload_dit_to_cpu": offload_dit_to_cpu,
            "quantization": self.quantization,
            "use_mlx_dit": use_mlx_dit,
            "prefer_source": prefer_source,
        }

        return status_msg, True
    except Exception as exc:
        self.model = None
        self.vae = None
        self.text_encoder = None
        self.text_tokenizer = None
        self.config = None
        self.silence_latent = None
        error_msg = f"Error initializing model: {str(exc)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.exception(error_msg)
        return error_msg, False

# --- From init_service_memory_basic.py ---

def _empty_cache(self):
    """Clear accelerator memory cache (CUDA, XPU, or MPS)."""
    device_type = self._device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _synchronize(self):
    """Synchronize accelerator operations (CUDA, XPU, or MPS)."""
    device_type = self._device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif device_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

def _memory_allocated(self):
    """Get current accelerator memory usage in bytes, or 0 for unsupported backends."""
    device_type = self._device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0

def _max_memory_allocated(self):
    """Get peak accelerator memory usage in bytes, or 0 for unsupported backends."""
    device_type = self._device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0

def _is_on_target_device(self, tensor, target_device):
    """Check if tensor is on the target device (handles cuda vs cuda:0 comparison)."""
    if tensor is None:
        return True
    try:
        if isinstance(target_device, torch.device):
            target_type = target_device.type
        else:
            target_type = torch.device(str(target_device)).type
    except Exception:
        target_type = str(target_device).strip().lower().split(":", 1)[0]
        if not target_type:
            logger.warning(
                "[_is_on_target_device] Malformed target device value: {!r}",
                target_device,
            )
            return False
    return tensor.device.type == target_type

@staticmethod
def _get_affine_quantized_tensor_class():
    """Return the AffineQuantizedTensor class from torchao, or None if unavailable."""
    try:
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
        return AffineQuantizedTensor
    except ImportError:
        pass
    try:
        from torchao.quantization.affine_quantized import AffineQuantizedTensor
        return AffineQuantizedTensor
    except ImportError:
        pass
    return None

def _is_quantized_tensor(self, t):
    """True if ``t`` is a torchao AffineQuantizedTensor."""
    if t is None:
        return False
    cls = self._get_affine_quantized_tensor_class()
    if cls is None:
        return False
    return isinstance(t, cls)

def _has_quantized_params(self, module):
    """True if module (or any submodule) has an AffineQuantizedTensor parameter."""
    cls = self._get_affine_quantized_tensor_class()
    if cls is None:
        return False
    for _, param in module.named_parameters():
        if param is not None and isinstance(param, cls):
            return True
    return False

def _ensure_silence_latent_on_device(self):
    """Ensure ``silence_latent`` is on ``self.device``."""
    if hasattr(self, "silence_latent") and self.silence_latent is not None:
        if not self._is_on_target_device(self.silence_latent, self.device):
            self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

# --- From init_service_memory_transfer.py ---

def _move_module_recursive(self, module, target_device, dtype=None, visited=None):
    """Recursively move a module and all submodules to the target device."""
    if visited is None:
        visited = set()

    module_id = id(module)
    if module_id in visited:
        return
    visited.add(module_id)

    module.to(target_device)
    if dtype is not None:
        module.to(dtype)

    for param_name, param in module._parameters.items():
        if param is not None and not self._is_on_target_device(param, target_device):
            if self._is_quantized_tensor(param):
                moved_param = self._move_quantized_param(param, target_device)
            else:
                moved_param = torch.nn.Parameter(
                    param.data.to(target_device), requires_grad=param.requires_grad
                )
            if dtype is not None and moved_param.is_floating_point():
                moved_param = torch.nn.Parameter(
                    moved_param.data.to(dtype), requires_grad=param.requires_grad
                )
            module._parameters[param_name] = moved_param

    for buf_name, buf in module._buffers.items():
        if buf is not None and not self._is_on_target_device(buf, target_device):
            module._buffers[buf_name] = buf.to(target_device)

    for _, child in module._modules.items():
        if child is not None:
            self._move_module_recursive(child, target_device, dtype, visited)

    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(module, attr_name, None)
            if isinstance(attr, torch.nn.Module) and id(attr) not in visited:
                self._move_module_recursive(attr, target_device, dtype, visited)
        except (AttributeError, TypeError) as exc:
            log = getattr(self, "logger", logger)
            log.warning(
                f"[_move_module_recursive] Skipping attr '{attr_name}' during recursive move: {exc}"
            )

def _move_quantized_param(self, param, target_device):
    """Move an AffineQuantizedTensor to target device using ``_apply_fn_to_data`` when available."""
    if hasattr(param, "_apply_fn_to_data"):
        return torch.nn.Parameter(
            param._apply_fn_to_data(lambda x: x.to(target_device)),
            requires_grad=param.requires_grad,
        )
    moved = param.to(target_device)
    return torch.nn.Parameter(moved, requires_grad=param.requires_grad)

def _recursive_to_device(self, model, device, dtype=None):
    """Recursively move parameters and buffers to the specified device."""
    target_device = torch.device(device) if isinstance(device, str) else device

    try:
        model.to(target_device)
        if dtype is not None:
            model.to(dtype)
    except NotImplementedError:
        logger.info(
            "[_recursive_to_device] model.to() raised NotImplementedError "
            "(AffineQuantizedTensor on older torch). Moving parameters individually."
        )
        for module in model.modules():
            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                if self._is_on_target_device(param, target_device):
                    continue
                if self._is_quantized_tensor(param):
                    module._parameters[param_name] = self._move_quantized_param(param, target_device)
                else:
                    module._parameters[param_name] = torch.nn.Parameter(
                        param.data.to(target_device), requires_grad=param.requires_grad
                    )
                    if dtype is not None:
                        module._parameters[param_name] = torch.nn.Parameter(
                            module._parameters[param_name].data.to(dtype),
                            requires_grad=param.requires_grad,
                        )
            for buf_name, buf in module._buffers.items():
                if buf is not None and not self._is_on_target_device(buf, target_device):
                    module._buffers[buf_name] = buf.to(target_device)

    try:
        self._move_module_recursive(model, target_device, dtype)
    except NotImplementedError:
        pass

    wrong_device_params = []
    for name, param in model.named_parameters():
        if not self._is_on_target_device(param, device):
            wrong_device_params.append(name)

    if wrong_device_params and device != "cpu":
        logger.warning(
            f"[_recursive_to_device] {len(wrong_device_params)} parameters on wrong device after initial move, retrying individually"
        )
        for module in model.modules():
            for param_name, param in module._parameters.items():
                if param is None or self._is_on_target_device(param, target_device):
                    continue
                if self._is_quantized_tensor(param):
                    module._parameters[param_name] = self._move_quantized_param(param, target_device)
                else:
                    module._parameters[param_name] = torch.nn.Parameter(
                        param.data.to(target_device), requires_grad=param.requires_grad
                    )
                    if dtype is not None and module._parameters[param_name].is_floating_point():
                        module._parameters[param_name] = torch.nn.Parameter(
                            module._parameters[param_name].data.to(dtype),
                            requires_grad=param.requires_grad,
                        )

    if device != "cpu":
        self._synchronize()

    if device != "cpu":
        still_wrong = []
        for name, param in model.named_parameters():
            if not self._is_on_target_device(param, device):
                still_wrong.append(f"{name} on {param.device}")
        if still_wrong:
            logger.error(
                f"[_recursive_to_device] CRITICAL: {len(still_wrong)} parameters still on wrong device: {still_wrong[:10]}"
            )

# --- From init_service_offload_context.py ---

@contextmanager
def _load_model_context(self, model_name: str):
    """Load a model to device for the context and offload back to CPU on exit."""
    if not self.offload_to_cpu:
        yield
        return

    if model_name == "model" and not self.offload_dit_to_cpu:
        model = getattr(self, model_name, None)
        if model is not None:
            try:
                param = next(model.parameters())
                if param.device.type == "cpu":
                    logger.info(f"[_load_model_context] Moving {model_name} to {self.device} (persistent)")
                    self._recursive_to_device(model, self.device, self.dtype)
                    if hasattr(self, "silence_latent"):
                        self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)
            except StopIteration:
                pass
        yield
        return

    model = getattr(self, model_name, None)
    if model is None:
        yield
        return

    logger.info(f"[_load_model_context] Loading {model_name} to {self.device}")
    start_time = time.time()
    if model_name == "vae":
        vae_dtype = self._get_vae_dtype()
        self._recursive_to_device(model, self.device, vae_dtype)
    else:
        self._recursive_to_device(model, self.device, self.dtype)

    if model_name == "model" and hasattr(self, "silence_latent"):
        self.silence_latent = self.silence_latent.to(self.device).to(self.dtype)

    load_time = time.time() - start_time
    self.current_offload_cost += load_time
    logger.info(f"[_load_model_context] Loaded {model_name} to {self.device} in {load_time:.4f}s")

    try:
        yield
    finally:
        logger.info(f"[_load_model_context] Offloading {model_name} to CPU")
        start_time = time.time()
        if model_name == "vae":
            self._recursive_to_device(model, "cpu", self._get_vae_dtype("cpu"))
        else:
            self._recursive_to_device(model, "cpu")

        self._empty_cache()
        offload_time = time.time() - start_time
        self.current_offload_cost += offload_time
        logger.info(f"[_load_model_context] Offloaded {model_name} to CPU in {offload_time:.4f}s")
