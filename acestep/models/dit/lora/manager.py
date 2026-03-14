"""Handler-bound LoRA management: registry, controls, and lifecycle.

All functions in this module take ``self`` as their first argument and are
assigned onto ``AceStepDiTWrapper`` at class-definition time via utils.py.
They manage LoRA state directly on ``self.*`` attributes — no intermediate
facade class.
"""

import json
import math
import os
from typing import Any

from loguru import logger

from acestep.constants import DEBUG_MODEL_LOADING
from acestep.debug_utils import debug_log
from acestep.training.configs import LoKRConfig

from .ops import (
    apply_scale_to_adapter as _pure_apply_scale,
)
from .ops import (
    build_lora_registry as _pure_build_registry,
)
from .ops import (
    collect_adapter_names as _pure_collect_names,
)
from .ops import (
    keep_adapter_agnostic_targets,
)

LOKR_WEIGHTS_FILENAME = "lokr_weights.safetensors"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _decoder_from_host(self):
    model = getattr(self, "model", None)
    return getattr(model, "decoder", None) if model is not None else None


def _copy_registry(registry: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for adapter_name, meta in registry.items():
        targets: list[dict[str, Any]] = []
        for target in meta.get("targets", []):
            target_copy = dict(target)
            module = target_copy.pop("module", None)
            if "module_class" not in target_copy:
                target_copy["module_class"] = module.__class__.__name__ if module is not None else None
            targets.append(target_copy)
        copied[adapter_name] = {
            "path": meta.get("path"),
            "targets": targets,
        }
    return copied


# ---------------------------------------------------------------------------
# Registry state management (replaces registry_state.py + LoraService)
# ---------------------------------------------------------------------------

def sync_lora_state(self) -> None:
    """Sync handler-visible snapshots from authoritative state."""
    self._lora_adapter_registry = _copy_registry(self._lora_registry)
    self._lora_scale_state = dict(self._lora_scale_state_internal)
    self._lora_active_adapter = self._lora_active_adapter_internal
    self._lora_last_scale_report = dict(self._lora_last_scale_report_internal)


def ensure_lora_registry(self) -> None:
    decoder = _decoder_from_host(self)

    if not hasattr(self, "_lora_registry"):
        self._lora_registry = {}
    if not hasattr(self, "_lora_scale_state_internal"):
        self._lora_scale_state_internal = {}
    if not hasattr(self, "_lora_active_adapter_internal"):
        self._lora_active_adapter_internal = None
    if not hasattr(self, "_lora_last_scale_report_internal"):
        self._lora_last_scale_report_internal = {}
    if not hasattr(self, "_lora_synthetic_default_mode"):
        self._lora_synthetic_default_mode = False
    if not hasattr(self, "_lora_decoder"):
        self._lora_decoder = None

    self._lora_decoder = decoder

    if not hasattr(self, "_lora_adapter_registry"):
        self._lora_adapter_registry = {}
    if not hasattr(self, "_lora_active_adapter"):
        self._lora_active_adapter = None
    if not hasattr(self, "_lora_scale_state"):
        self._lora_scale_state = {}
    if not hasattr(self, "_active_loras"):
        self._active_loras = {}
    if not hasattr(self, "_lora_last_scale_report"):
        self._lora_last_scale_report = {}

    sync_lora_state(self)


def _discover_adapter_names(self) -> list[str]:
    decoder = self._lora_decoder
    if decoder is None:
        return []
    return [name for name in _pure_collect_names(decoder) if isinstance(name, str) and name]


def rebuild_lora_registry(self, lora_path: str | None = None) -> tuple[int, list[str]]:
    """Build explicit adapter->target mapping used for deterministic scaling."""
    self._ensure_lora_registry()
    decoder = self._lora_decoder

    if decoder is None:
        self._lora_registry = {}
        self._lora_scale_state_internal = {}
        self._lora_active_adapter_internal = None
        self._lora_synthetic_default_mode = False
        sync_lora_state(self)
        return 0, []

    adapter_names = _discover_adapter_names(self)
    synthetic_default = not adapter_names
    self._lora_synthetic_default_mode = synthetic_default
    effective_names = adapter_names or ["default"]

    rebuilt_registry, _ = _pure_build_registry(
        decoder=decoder,
        adapter_names=effective_names,
        lora_path=lora_path,
    )
    if synthetic_default:
        rebuilt_registry = keep_adapter_agnostic_targets(rebuilt_registry)

    self._lora_registry = rebuilt_registry
    self._lora_scale_state_internal = {}
    total_targets = sum(len(meta.get("targets", [])) for meta in self._lora_registry.values())
    if self._lora_active_adapter_internal not in self._lora_registry:
        self._lora_active_adapter_internal = next(iter(self._lora_registry.keys()), None)

    adapters = list(self._lora_registry.keys())
    sync_lora_state(self)

    if not adapters:
        logger.warning("No adapter names discovered from decoder; LoRA registry will be empty.")
        debug_log(
            "No adapter names discovered; skipping adapter target registration.",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )

    return total_targets, adapters


def debug_lora_registry_snapshot(self, max_targets_per_adapter: int = 20) -> dict[str, Any]:
    """Return debugger-friendly snapshot of LoRA adapter registry."""
    self._ensure_lora_registry()
    adapters: dict[str, Any] = {}
    for adapter_name, meta in self._lora_registry.items():
        targets = meta.get("targets", [])
        entries = []
        for target in targets[:max_targets_per_adapter]:
            module = target.get("module")
            entries.append(
                {
                    "kind": target.get("kind"),
                    "module_name": target.get("module_name"),
                    "adapter": target.get("adapter"),
                    "module_class": module.__class__.__name__ if module is not None else None,
                }
            )
        adapters[adapter_name] = {
            "path": meta.get("path"),
            "target_count": len(targets),
            "targets": entries,
            "truncated": len(targets) > max_targets_per_adapter,
        }
    return {
        "active_adapter": self._lora_active_adapter_internal,
        "adapter_names": list(self._lora_registry.keys()),
        "synthetic_default_mode": self._lora_synthetic_default_mode,
        "adapters": adapters,
    }


# ---------------------------------------------------------------------------
# Adapter discovery wrapper
# ---------------------------------------------------------------------------

def collect_adapter_names(self) -> list[str]:
    """Best-effort adapter name discovery across PEFT runtime variants."""
    self._ensure_lora_registry()
    return _discover_adapter_names(self)


# ---------------------------------------------------------------------------
# Scale application wrapper
# ---------------------------------------------------------------------------

def apply_scale_to_adapter(self, adapter_name: str, scale: float) -> int:
    """Apply scale to registered targets for one adapter."""
    self._ensure_lora_registry()
    modified, report = _pure_apply_scale(
        registry=self._lora_registry,
        scale_state=self._lora_scale_state_internal,
        adapter_name=adapter_name,
        scale=scale,
        warn_hook=logger.warning,
        debug_hook=lambda message: debug_log(message, mode=DEBUG_MODEL_LOADING, prefix="lora"),
    )
    self._lora_last_scale_report_internal = report
    sync_lora_state(self)
    return modified


# ---------------------------------------------------------------------------
# Controls (replaces controls.py)
# ---------------------------------------------------------------------------

def _toggle_lokr(decoder, enable: bool, scale: float = 1.0) -> bool:
    """Toggle a LyCORIS LoKr adapter via its multiplier."""
    lycoris_net = getattr(decoder, "_lycoris_net", None)
    if lycoris_net is None:
        return False
    set_mul = getattr(lycoris_net, "set_multiplier", None)
    if not callable(set_mul):
        return False
    target = float(scale) if enable else 0.0
    set_mul(target)
    logger.info(f"LoKr multiplier set to {target}")
    return True


def set_use_lora(self, use_lora: bool) -> str:
    """Toggle LoRA/LoKr usage for inference."""
    if use_lora and not self.lora_loaded:
        return "❌ No LoRA adapter loaded. Please load a LoRA first."

    self.use_lora = use_lora
    model = getattr(self, "model", None)
    decoder = getattr(model, "decoder", None) if model is not None else None
    if self.lora_loaded and decoder is None:
        logger.warning("LoRA is marked as loaded, but model/decoder is unavailable during toggle.")

    if self.lora_loaded and decoder is not None:
        adapter_type = getattr(self, "_adapter_type", None)

        if adapter_type == "lokr":
            active = getattr(self, "_lora_active_adapter", None)
            scale = getattr(self, "_active_loras", {}).get(active, 1.0) if active else self.lora_scale
            toggled = _toggle_lokr(decoder, use_lora, scale=scale)
            if not toggled:
                logger.warning("LoKr adapter type set but no _lycoris_net found on decoder")

        elif hasattr(decoder, "disable_adapter_layers"):
            try:
                if use_lora:
                    active = getattr(self, "_lora_active_adapter", None)
                    if active and hasattr(decoder, "set_adapter"):
                        try:
                            decoder.set_adapter(active)
                        except Exception:
                            pass
                    decoder.enable_adapter_layers()
                    logger.info("LoRA adapter enabled")
                    scale = getattr(self, "_active_loras", {}).get(active, 1.0)
                    if active and scale != 1.0:
                        self.set_lora_scale(active, scale)
                else:
                    decoder.disable_adapter_layers()
                    logger.info("LoRA adapter disabled")
            except Exception as e:
                logger.warning(f"Could not toggle adapter layers: {e}")

    adapter_label = "LoKr" if getattr(self, "_adapter_type", None) == "lokr" else "LoRA"
    status = "enabled" if use_lora else "disabled"
    return f"✅ {adapter_label} {status}"


def set_lora_scale(self, adapter_name_or_scale: str | float, scale: float | None = None) -> str:
    """Set LoRA scale (0–1). Call as set_lora_scale(scale) or set_lora_scale(adapter_name, scale)."""
    if not self.lora_loaded:
        return "⚠️ No LoRA loaded"

    if scale is None:
        scale_value = adapter_name_or_scale
        effective_name = None
    else:
        effective_name = adapter_name_or_scale if isinstance(adapter_name_or_scale, str) else None
        scale_value = scale

    try:
        scale_value = float(scale_value)
    except (TypeError, ValueError):
        return "❌ Invalid LoRA scale: please provide a numeric value between 0 and 1."
    if not math.isfinite(scale_value):
        return "❌ Invalid LoRA scale: please provide a finite numeric value between 0 and 1."

    scale_value = max(0.0, min(1.0, scale_value))
    _active_loras = getattr(self, "_active_loras", None) or {}
    if not effective_name:
        effective_name = getattr(self, "_lora_active_adapter", None) or (
            next(iter(_active_loras), None) if _active_loras else None
        )
    if not effective_name:
        return "❌ No adapter specified and no active adapter. Load a LoRA or pass adapter_name."

    self._active_loras[effective_name] = scale_value
    self.lora_scale = scale_value

    adapter_label = "LoKr" if getattr(self, "_adapter_type", None) == "lokr" else "LoRA"

    if not self.use_lora:
        logger.info(f"{adapter_label} scale for '{effective_name}' set to {scale_value:.2f} (will apply when enabled)")
        return f"✅ {adapter_label} scale ({effective_name}): {scale_value:.2f} ({adapter_label} disabled)"

    if getattr(self, "_adapter_type", None) == "lokr":
        decoder = getattr(getattr(self, "model", None), "decoder", None)
        if decoder is not None:
            toggled = _toggle_lokr(decoder, True, scale=scale_value)
            if toggled:
                return f"✅ {adapter_label} scale ({effective_name}): {scale_value:.2f}"
            logger.warning("LoKr adapter type set but no _lycoris_net found for scale")
        return f"⚠️ {adapter_label} scale set to {scale_value:.2f} (no LyCORIS net found)"

    try:
        rebuilt_adapters: list[str] | None = None
        if not getattr(self, "_lora_adapter_registry", None):
            _, rebuilt_adapters = self._rebuild_lora_registry()

        if rebuilt_adapters is not None:
            if effective_name not in (rebuilt_adapters or []):
                return f"❌ Adapter '{effective_name}' not in loaded adapters: {rebuilt_adapters}"
            active_adapter = self._lora_active_adapter_internal or effective_name
            if active_adapter != effective_name:
                if effective_name in self._lora_registry:
                    self._lora_active_adapter_internal = effective_name
                    self._lora_active_adapter = effective_name
                if getattr(self.model, "decoder", None) and hasattr(self.model.decoder, "set_adapter"):
                    try:
                        self.model.decoder.set_adapter(effective_name)
                    except Exception:
                        pass
        else:
            if self._lora_active_adapter_internal is None and self._lora_registry:
                self._lora_active_adapter_internal = next(iter(self._lora_registry.keys()))
            self._lora_active_adapter = self._lora_active_adapter_internal
        sync_lora_state(self)
        adapter_names = list(self._lora_registry.keys())

        debug_log(
            lambda: (
                f"LoRA scale request: adapter={effective_name} scale={scale_value:.3f} "
                f"adapters={adapter_names}"
            ),
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )

        modified = self._apply_scale_to_adapter(effective_name, scale_value)
        report = getattr(self, "_lora_last_scale_report", {})
        skipped_total = sum(report.get("skipped_by_kind", {}).values())

        if modified > 0:
            logger.info(
                f"LoRA scale for '{effective_name}' set to {scale_value:.2f} "
                f"(modified={modified}, by_kind={report.get('modified_by_kind', {})}, skipped={report.get('skipped_by_kind', {})})"
            )
            return (
                f"✅ LoRA scale ({effective_name}): {scale_value:.2f}"
                if skipped_total == 0
                else f"✅ LoRA scale ({effective_name}): {scale_value:.2f} (skipped {skipped_total} targets)"
            )

        if skipped_total > 0:
            logger.warning(
                f"No LoRA targets were modified for adapter '{effective_name}' "
                f"(skipped={report.get('skipped_by_kind', {})})"
            )
            return f"⚠️ LoRA scale unchanged: {scale_value:.2f} (skipped {skipped_total} targets)"

        logger.warning(f"No registered LoRA scaling targets found for adapter '{effective_name}'")
        return f"⚠️ Scale set to {scale_value:.2f} (no modules found)"
    except Exception as e:
        logger.warning(f"Could not set LoRA scale: {e}")
        return f"⚠️ Scale set to {scale_value:.2f} (partial)"


def set_active_lora_adapter(self, adapter_name: str) -> str:
    """Set the active LoRA adapter for scaling/inference."""
    self._ensure_lora_registry()
    if adapter_name not in self._lora_registry:
        return f"❌ Unknown adapter: {adapter_name}"
    self._lora_active_adapter_internal = adapter_name
    self._lora_active_adapter = adapter_name
    debug_log(f"Selected active LoRA adapter: {adapter_name}", mode=DEBUG_MODEL_LOADING, prefix="lora")
    if self.model is not None and hasattr(self.model, "decoder") and hasattr(self.model.decoder, "set_adapter"):
        try:
            self.model.decoder.set_adapter(adapter_name)
        except Exception:
            pass
    return f"✅ Active LoRA adapter: {adapter_name}"


def get_lora_status(self) -> dict[str, Any]:
    """Get current LoRA status."""
    self._ensure_lora_registry()
    _active_loras = getattr(self, "_active_loras", None) or {}
    return {
        "loaded": self.lora_loaded,
        "active": self.use_lora,
        "scale": self.lora_scale,
        "scales": dict(_active_loras),
        "active_adapter": self._lora_active_adapter,
        "adapters": list(self._lora_registry.keys()),
        "synthetic_default_mode": self._lora_synthetic_default_mode,
    }


# ---------------------------------------------------------------------------
# Lifecycle (replaces lifecycle.py)
# ---------------------------------------------------------------------------

def _is_lokr_safetensors(weights_path: str) -> bool:
    """Return whether ``weights_path`` looks like a LoKr/LyCORIS safetensors file."""
    if not os.path.isfile(weights_path) or not weights_path.lower().endswith(".safetensors"):
        return False
    if os.path.basename(weights_path) == LOKR_WEIGHTS_FILENAME:
        return True

    try:
        from safetensors import safe_open
    except ImportError:
        return False

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception:
        return False

    raw_config = metadata.get("lokr_config")
    return isinstance(raw_config, str) and bool(raw_config.strip())


def _resolve_lokr_weights_path(adapter_path: str) -> str | None:
    """Return LoKr safetensors path when ``adapter_path`` points to LoKr artifacts."""
    if os.path.isfile(adapter_path):
        return adapter_path if _is_lokr_safetensors(adapter_path) else None
    if os.path.isdir(adapter_path):
        weights_path = os.path.join(adapter_path, LOKR_WEIGHTS_FILENAME)
        if os.path.exists(weights_path):
            return weights_path
        try:
            entries = os.listdir(adapter_path)
        except OSError:
            return None
        for name in entries:
            candidate = os.path.join(adapter_path, name)
            if _is_lokr_safetensors(candidate):
                return candidate
    return None


def _load_lokr_config(weights_path: str) -> LoKRConfig:
    """Build ``LoKRConfig`` from safetensors metadata, with defaults on parse failure."""
    config = LoKRConfig()
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors metadata reader unavailable; using default LoKr config.")
        return config

    try:
        with safe_open(weights_path, framework="pt", device="cpu") as sf:
            metadata: dict[str, Any] = sf.metadata() or {}
    except Exception as exc:
        logger.warning(f"Unable to read LoKr metadata from {weights_path}: {exc}")
        return config

    raw_config = metadata.get("lokr_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return config

    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        logger.warning(f"Invalid LoKr metadata config JSON in {weights_path}: {exc}")
        return config

    if not isinstance(parsed, dict):
        return config

    allowed_keys = set(LoKRConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in parsed.items() if k in allowed_keys}
    if not filtered:
        return config

    try:
        return LoKRConfig(**filtered)
    except Exception as exc:
        logger.warning(f"Failed to apply LoKr metadata config from {weights_path}: {exc}")
        return config


def _load_lokr_adapter(decoder: Any, weights_path: str) -> Any:
    """Inject and load a LoKr LyCORIS adapter into ``decoder``."""
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:
        raise ImportError("LyCORIS library not installed. Please install with: pip install lycoris-lora") from exc

    lokr_config = _load_lokr_config(weights_path)
    LycorisNetwork.apply_preset(
        {
            "unet_target_name": lokr_config.target_modules,
            "target_name": lokr_config.target_modules,
        }
    )
    lycoris_net = create_lycoris(
        decoder,
        1.0,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        try:
            lycoris_net = create_lycoris(
                decoder,
                1.0,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
        except Exception as exc:
            logger.warning(f"DoRA mode not supported in current LyCORIS build: {exc}")

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net
    lycoris_net.load_weights(weights_path)
    return lycoris_net


def _default_adapter_name_from_path(lora_path: str) -> str:
    """Derive a default adapter name from path (e.g. 'final' from './lora/final')."""
    name = os.path.basename(lora_path.rstrip(os.sep))
    return name if name else "default"


def _reset_lora_state(self) -> None:
    """Clear all LoRA bookkeeping to a clean slate."""
    self._lora_registry = {}
    self._lora_scale_state_internal = {}
    self._lora_active_adapter_internal = None
    self._lora_last_scale_report_internal = {}
    self._lora_adapter_registry = {}
    self._lora_active_adapter = None
    self._lora_scale_state = {}


def add_lora(self, lora_path: str, adapter_name: str | None = None) -> str:
    """Load a LoRA adapter into the decoder under the given name."""
    if self.model is None:
        return "❌ Model not initialized. Please initialize service first."

    if self.quantization is not None:
        return (
            "❌ LoRA loading is not supported on quantized models. "
            f"Current quantization: {self.quantization}. "
            "Please re-initialize the service with quantization disabled, then try loading the LoRA adapter again."
        )

    if not lora_path or not lora_path.strip():
        return "❌ Please provide a LoRA path."

    lora_path = lora_path.strip()
    if not os.path.exists(lora_path):
        return f"❌ LoRA path not found: {lora_path}"

    lokr_weights_path = _resolve_lokr_weights_path(lora_path)
    if lokr_weights_path is None:
        config_file = os.path.join(lora_path, "adapter_config.json")
        if not os.path.exists(config_file):
            return (
                "❌ Invalid adapter: expected PEFT LoRA directory containing adapter_config.json "
                f"or LoKr artifact {LOKR_WEIGHTS_FILENAME} in {lora_path}"
            )

    try:
        from peft import PeftModel
    except ImportError:
        if lokr_weights_path is None:
            return "❌ PEFT library not installed. Please install with: pip install peft"
        PeftModel = None  # type: ignore[assignment]

    effective_name = adapter_name.strip() if isinstance(adapter_name, str) and adapter_name.strip() else _default_adapter_name_from_path(lora_path)
    _active_loras = getattr(self, "_active_loras", None)
    if _active_loras is None:
        self._active_loras = {}
        _active_loras = self._active_loras
    if effective_name in _active_loras:
        return f"❌ Adapter name already in use: {effective_name}. Use a different name or remove it first."

    try:
        decoder = self.model.decoder
        is_peft = PeftModel is not None and isinstance(decoder, PeftModel)

        if not is_peft:
            if self._base_decoder is None:
                if hasattr(self, "_memory_allocated"):
                    mem_before = self._memory_allocated() / (1024**3)
                    logger.info(f"VRAM before LoRA load: {mem_before:.2f}GB")
                try:
                    state_dict = decoder.state_dict()
                    if not state_dict:
                        raise ValueError("state_dict is empty - cannot backup decoder")
                    self._base_decoder = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
                except Exception as e:
                    logger.error(f"Failed to create state_dict backup: {e}")
                    raise
                backup_size_mb = sum(v.numel() * v.element_size() for v in self._base_decoder.values()) / (1024**2)
                logger.info(f"Base decoder state_dict backed up to CPU ({backup_size_mb:.1f}MB)")

            if lokr_weights_path is not None:
                logger.info(f"Loading LoKr adapter from {lokr_weights_path} as '{effective_name}'")
                _load_lokr_adapter(decoder, lokr_weights_path)
                self.model.decoder = decoder
                self._adapter_type = "lokr"
            else:
                logger.info(f"Loading LoRA adapter from {lora_path} as '{effective_name}'")
                self.model.decoder = PeftModel.from_pretrained(
                    decoder, lora_path, adapter_name=effective_name, is_trainable=False
                )
                self._adapter_type = "lora"
        else:
            if lokr_weights_path is not None:
                return "❌ LoKr cannot be added as a second adapter when PEFT is already loaded."
            logger.info(f"Loading additional LoRA from {lora_path} as '{effective_name}'")
            self.model.decoder.load_adapter(lora_path, adapter_name=effective_name)
            self._adapter_type = "lora"

        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

        if hasattr(self, "_memory_allocated"):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA load: {mem_after:.2f}GB")

        self.lora_loaded = True
        self.use_lora = True
        self._active_loras[effective_name] = 1.0
        self._ensure_lora_registry()
        self._lora_active_adapter = None
        target_count, adapters = self._rebuild_lora_registry(lora_path=lora_path)
        if effective_name in self._lora_registry:
            self._lora_active_adapter_internal = effective_name
            self._lora_active_adapter = effective_name
        if hasattr(self.model.decoder, "set_adapter"):
            try:
                self.model.decoder.set_adapter(effective_name)
            except Exception:
                pass

        logger.info(
            f"LoRA adapter '{effective_name}' loaded from {lora_path} "
            f"(adapters={adapters}, targets={target_count})"
        )
        debug_log(
            lambda: f"LoRA registry snapshot: {self._debug_lora_registry_snapshot()}",
            mode=DEBUG_MODEL_LOADING,
            prefix="lora",
        )
        return f"✅ LoRA '{effective_name}' loaded from {lora_path}"
    except Exception as e:
        logger.exception("Failed to load LoRA adapter")
        return f"❌ Failed to load LoRA: {str(e)}"


def load_lora(self, lora_path: str) -> str:
    """Load a single adapter (backward-compat), including LyCORIS LoKr paths."""
    lokr_weights_path = _resolve_lokr_weights_path(lora_path.strip()) if isinstance(lora_path, str) else None
    message = self.add_lora(lora_path, adapter_name=None)
    if lokr_weights_path is not None and message.startswith("✅"):
        return f"✅ LoKr loaded from {lokr_weights_path}"
    return message


def add_voice_lora(self, lora_path: str, scale: float = 1.0) -> str:
    """Load a LoRA as the 'voice' adapter and set its scale."""
    msg = self.add_lora(lora_path, adapter_name="voice")
    if not msg.startswith("✅"):
        return msg
    return self.set_lora_scale("voice", scale)


def remove_lora(self, adapter_name: str) -> str:
    """Remove one LoRA adapter by name. If no adapters remain, restores base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    _active_loras = getattr(self, "_active_loras", None) or {}
    if adapter_name not in _active_loras:
        return f"❌ Unknown adapter: {adapter_name}. Loaded: {list(_active_loras.keys())}"

    try:
        from peft import PeftModel
    except ImportError:
        return "❌ PEFT library not installed."

    decoder = getattr(self.model, "decoder", None)
    if decoder is None or not isinstance(decoder, PeftModel):
        _active_loras.pop(adapter_name, None)
        if not _active_loras:
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
        return "⚠️ Adapter removed from registry (decoder was not PEFT)."

    if adapter_name not in (getattr(decoder, "peft_config", None) or {}):
        _active_loras.pop(adapter_name, None)
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        return f"✅ Adapter '{adapter_name}' removed (was not in PEFT)."

    try:
        decoder.delete_adapter(adapter_name)
        _active_loras.pop(adapter_name, None)
        remaining = list(_active_loras.keys())

        if not remaining:
            if self._base_decoder is None:
                self.lora_loaded = False
                self.use_lora = False
                self._adapter_type = None
                self._active_loras.clear()
                self._ensure_lora_registry()
                _reset_lora_state(self)
                return "✅ Last adapter removed; base decoder still wrapped (no backup). Restart or load a new LoRA."
            mem_before = None
            if hasattr(self, "_memory_allocated"):
                mem_before = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")
            self.model.decoder = decoder.get_base_model()
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
            self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
            self.model.decoder.eval()
            self.lora_loaded = False
            self.use_lora = False
            self._adapter_type = None
            self._active_loras = {}
            self._ensure_lora_registry()
            _reset_lora_state(self)
            if mem_before is not None and hasattr(self, "_memory_allocated"):
                mem_after = self._memory_allocated() / (1024**3)
                logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")
            logger.info("LoRA unloaded, base decoder restored")
            return "✅ LoRA unloaded, using base model"
        next_active = remaining[0]
        if hasattr(decoder, "set_adapter"):
            try:
                decoder.set_adapter(next_active)
            except Exception:
                pass
        self._lora_active_adapter = next_active
        self._ensure_lora_registry()
        self._rebuild_lora_registry()
        self._lora_active_adapter_internal = next_active
        scale = self._active_loras.get(next_active, 1.0)
        self._apply_scale_to_adapter(next_active, scale)
        logger.info(f"Adapter '{adapter_name}' removed. Active: {next_active}")
        return f"✅ Adapter '{adapter_name}' removed. Active: {next_active}"
    except Exception as e:
        logger.exception("Failed to remove LoRA adapter")
        return f"❌ Failed to remove LoRA: {str(e)}"


def unload_lora(self) -> str:
    """Unload all LoRA adapters and restore base decoder."""
    if not self.lora_loaded:
        return "⚠️ No LoRA adapter loaded."

    if self._base_decoder is None:
        return "❌ Base decoder backup not found. Cannot restore."

    try:
        mem_before = None
        if hasattr(self, "_memory_allocated"):
            mem_before = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM before LoRA unload: {mem_before:.2f}GB")

        lycoris_net = getattr(self.model.decoder, "_lycoris_net", None)
        if lycoris_net is not None:
            restore_fn = getattr(lycoris_net, "restore", None)
            if callable(restore_fn):
                logger.info("Restoring decoder structure from LyCORIS adapter")
                restore_fn()
            else:
                logger.warning("Decoder has _lycoris_net but no restore() method; continuing with state_dict restore")
            self.model.decoder._lycoris_net = None

        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None  # type: ignore[assignment]

        if PeftModel is not None and isinstance(self.model.decoder, PeftModel):
            logger.info("Extracting base model from PEFT wrapper")
            self.model.decoder = self.model.decoder.get_base_model()
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")
        else:
            logger.info("Restoring base decoder from state_dict backup")
            load_result = self.model.decoder.load_state_dict(self._base_decoder, strict=False)
            if load_result.missing_keys:
                logger.warning(f"Missing keys when restoring decoder: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected keys when restoring decoder: {load_result.unexpected_keys[:5]}")

        self.model.decoder = self.model.decoder.to(self.device).to(self.dtype)
        self.model.decoder.eval()

        self.lora_loaded = False
        self.use_lora = False
        self._adapter_type = None
        self.lora_scale = 1.0
        _active_loras = getattr(self, "_active_loras", None)
        if _active_loras is not None:
            _active_loras.clear()
        self._ensure_lora_registry()
        _reset_lora_state(self)

        if mem_before is not None and hasattr(self, "_memory_allocated"):
            mem_after = self._memory_allocated() / (1024**3)
            logger.info(f"VRAM after LoRA unload: {mem_after:.2f}GB (freed: {mem_before - mem_after:.2f}GB)")

        logger.info("LoRA unloaded, base decoder restored")
        return "✅ LoRA unloaded, using base model"
    except Exception as e:
        logger.exception("Failed to unload LoRA")
        return f"❌ Failed to unload LoRA: {str(e)}"
