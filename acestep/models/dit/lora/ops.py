"""Pure LoRA operations: introspection, registry construction, and scale application."""

import math
import traceback
from collections.abc import Callable
from typing import Any

MIN_PREV_SCALE = 1e-12


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def collect_adapter_names(decoder: Any) -> list[str]:
    """Best-effort adapter name discovery across PEFT runtime variants."""

    def _extract_names(value: Any) -> list[str]:
        names: list[str] = []

        def _append_name(name: Any) -> None:
            if isinstance(name, str) and name and name not in names:
                names.append(name)

        def _walk(obj: Any) -> None:
            if obj is None:
                return
            if isinstance(obj, str):
                _append_name(obj)
                return
            if isinstance(obj, dict):
                for key in obj.keys():
                    _append_name(key)
                return
            if isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _walk(item)
                return
            if hasattr(obj, "keys") and callable(obj.keys):
                try:
                    for key in obj.keys():
                        _append_name(key)
                except Exception:
                    pass
            if hasattr(obj, "adapters"):
                _walk(obj.adapters)
            if hasattr(obj, "adapter_names"):
                _walk(obj.adapter_names)
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                try:
                    _walk(obj.to_dict())
                except Exception:
                    pass

        _walk(value)
        return list(dict.fromkeys(names))

    ordered: list[str] = []
    source_groups: list[list[str]] = []

    if hasattr(decoder, "get_adapter_names") and callable(decoder.get_adapter_names):
        try:
            names_value = decoder.get_adapter_names()
            source_groups.append(_extract_names(names_value() if callable(names_value) else names_value))
        except Exception:
            pass

    for attr in ("active_adapter", "active_adapters", "peft_config"):
        if not hasattr(decoder, attr):
            continue
        try:
            value = getattr(decoder, attr)
            source_groups.append(_extract_names(value() if callable(value) else value))
        except Exception:
            pass

    for group in source_groups:
        for name in group:
            if name not in ordered:
                ordered.append(name)
    return ordered


def is_lora_like_module(name: str, module: Any) -> bool:
    """Conservative LoRA module detection for mixed PEFT implementations."""
    name_l = name.lower()
    cls_l = module.__class__.__name__.lower()
    mod_l = module.__class__.__module__.lower()
    has_lora_signals = (
        "lora" in name_l
        or "lora" in cls_l
        or ("peft" in mod_l and "lora" in mod_l)
        or hasattr(module, "lora_A")
        or hasattr(module, "lora_B")
    )
    has_scaling_api = hasattr(module, "scaling") or hasattr(module, "set_scale") or hasattr(module, "scale_layer")
    return has_lora_signals and has_scaling_api


def read_adapter_value(value: Any, adapter: str) -> Any:
    """Read adapter-specific value from mapping-like or scalar containers."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(adapter)
    if hasattr(value, "keys") and callable(value.keys):
        try:
            return value.get(adapter)
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return value
    return None


def is_peft_factor_set_scale_module(module: Any) -> bool:
    """Detect modules where set_scale(adapter, factor) semantics are expected."""
    return hasattr(module, "set_scale") and hasattr(module, "lora_alpha") and hasattr(module, "r")


def get_peft_initial_scale(
    module: Any,
    adapter: str,
    debug_hook: Callable[[str], None] | None = None,
) -> float | None:
    """Return PEFT LoRA baseline scale (alpha/r or alpha/sqrt(r)) for adapter."""
    try:
        alpha = read_adapter_value(getattr(module, "lora_alpha", None), adapter)
        r_val = read_adapter_value(getattr(module, "r", None), adapter)
        if not isinstance(alpha, (int, float)) or not isinstance(r_val, (int, float)) or not r_val:
            return None
        use_rslora_raw = getattr(module, "use_rslora", False)
        use_rslora = bool(use_rslora_raw.get(adapter, False)) if isinstance(use_rslora_raw, dict) else bool(use_rslora_raw)
        return (alpha / math.sqrt(r_val)) if use_rslora else (alpha / r_val)
    except Exception as exc:
        if debug_hook is not None:
            debug_hook(f"Failed to compute initial scale (adapter={adapter}, err={exc})")
        return None


# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------

def build_lora_registry(
    decoder: Any,
    adapter_names: list[str],
    lora_path: str | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Build explicit adapter->target mapping used for deterministic scaling."""
    registry: dict[str, dict[str, Any]] = {name: {"path": lora_path, "targets": []} for name in adapter_names}

    for module_name, module in decoder.named_modules():
        if not is_lora_like_module(module_name, module):
            continue

        if is_peft_factor_set_scale_module(module):
            for adapter in adapter_names:
                scaling = getattr(module, "scaling", None)
                current_scale = read_adapter_value(scaling, adapter)
                initial_scale = get_peft_initial_scale(module, adapter)
                base_factor = (
                    float(current_scale) / float(initial_scale)
                    if isinstance(current_scale, (int, float))
                    and isinstance(initial_scale, (int, float))
                    and initial_scale != 0
                    else None
                )
                registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "set_scale_factor",
                        "adapter": adapter,
                        "module_name": module_name,
                        "base_factor": base_factor,
                    }
                )
            continue

        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            for adapter in adapter_names:
                if adapter in module.scaling:
                    registry[adapter]["targets"].append(
                        {
                            "module": module,
                            "kind": "scaling_dict",
                            "adapter": adapter,
                            "module_name": module_name,
                            "base_scale": module.scaling[adapter],
                        }
                    )
            continue

        if hasattr(module, "set_scale"):
            for adapter in adapter_names:
                registry[adapter]["targets"].append(
                    {
                        "module": module,
                        "kind": "set_scale_unknown",
                        "adapter": adapter,
                        "module_name": module_name,
                        "base_scale": read_adapter_value(getattr(module, "scaling", None), adapter),
                    }
                )
            continue

        if hasattr(module, "scale_layer") and len(adapter_names) == 1:
            adapter = adapter_names[0]
            base_scale = read_adapter_value(getattr(module, "scaling", None), adapter)
            registry[adapter]["targets"].append(
                {
                    "module": module,
                    "kind": "scale_layer",
                    "module_name": module_name,
                    "base_scale": float(base_scale) if isinstance(base_scale, (int, float)) else None,
                }
            )
            continue

        if hasattr(module, "scaling") and isinstance(module.scaling, (int, float)) and len(adapter_names) == 1:
            adapter = adapter_names[0]
            registry[adapter]["targets"].append(
                {
                    "module": module,
                    "kind": "scaling_scalar",
                    "module_name": module_name,
                    "base_scale": float(module.scaling),
                }
            )

    total_targets = sum(len(meta["targets"]) for meta in registry.values())
    return registry, total_targets


def keep_adapter_agnostic_targets(registry: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep only adapter-agnostic targets when no adapter names are discoverable."""
    allowed_kinds = {"scale_layer", "scaling_scalar"}
    filtered: dict[str, dict[str, Any]] = {}
    for adapter_name, meta in registry.items():
        targets = [target for target in meta.get("targets", []) if target.get("kind") in allowed_kinds]
        if targets:
            filtered[adapter_name] = {
                "path": meta.get("path"),
                "targets": targets,
            }
    return filtered


# ---------------------------------------------------------------------------
# Scale application
# ---------------------------------------------------------------------------

def _inc(store: dict[str, int], key: str) -> None:
    store[key] = store.get(key, 0) + 1


def apply_scale_to_adapter(
    registry: dict[str, dict[str, Any]],
    scale_state: dict[tuple[int, str, str], float],
    adapter_name: str,
    scale: float,
    warn_hook: Callable[[str], None] | None = None,
    debug_hook: Callable[[str], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Apply scale to one adapter and return `(modified_count, report)`."""
    meta = registry.get(adapter_name)
    if not meta:
        report = {
            "adapter": adapter_name,
            "modified_total": 0,
            "modified_by_kind": {},
            "skipped_by_kind": {"no_registry": 1},
        }
        return 0, report

    modified = 0
    modified_by_kind: dict[str, int] = {}
    skipped_by_kind: dict[str, int] = {}

    for target in meta.get("targets", []):
        module = target.get("module")
        kind = target.get("kind")
        kind_key = kind if isinstance(kind, str) and kind else "unknown_kind"
        module_name = target.get("module_name")
        if module is None:
            _inc(skipped_by_kind, kind_key)
            continue

        try:
            if kind == "scaling_dict":
                adapter = target.get("adapter")
                if adapter not in module.scaling:
                    _inc(skipped_by_kind, kind_key)
                    continue
                module.scaling[adapter] = target.get("base_scale", module.scaling[adapter]) * scale
                modified += 1
                _inc(modified_by_kind, kind_key)
            elif kind == "set_scale_factor":
                base_factor = target.get("base_factor", None)
                if isinstance(base_factor, (int, float)):
                    module.set_scale(adapter_name, base_factor * scale)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
                else:
                    _inc(skipped_by_kind, "set_scale_factor_unanchored")
                    if warn_hook:
                        warn_hook(
                            f"Skipping set_scale_factor target without anchor "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
            elif kind == "set_scale_unknown":
                base_scale = target.get("base_scale", None)
                if isinstance(base_scale, (int, float)):
                    module.set_scale(adapter_name, base_scale * scale)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
                else:
                    _inc(skipped_by_kind, kind_key)
                    if warn_hook:
                        warn_hook(
                            f"Skipping set_scale target with unknown semantics and no base "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                    if debug_hook:
                        debug_hook(
                            f"Skipped unanchored set_scale target "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
            elif kind == "scale_layer":
                base_scale = target.get("base_scale", None)
                desired = (base_scale * scale) if isinstance(base_scale, (int, float)) else scale
                state_key = (id(module), kind_key, adapter_name)
                if hasattr(module, "unscale_layer"):
                    module.unscale_layer()
                    module.scale_layer(desired)
                    scale_state[state_key] = float(desired)
                    modified += 1
                    _inc(modified_by_kind, kind_key if base_scale is not None else "scale_layer_fallback")
                elif base_scale is None:
                    _inc(skipped_by_kind, "scale_layer_unanchored")
                    if warn_hook:
                        warn_hook(
                            f"Skipping unanchored scale_layer target without unscale_layer "
                            f"(adapter={adapter_name}, module={target.get('module_name')})"
                        )
                else:
                    prev = scale_state.get(state_key)
                    module.scale_layer(
                        desired / prev if isinstance(prev, (int, float)) and prev > MIN_PREV_SCALE else desired
                    )
                    scale_state[state_key] = float(desired)
                    modified += 1
                    _inc(modified_by_kind, kind_key)
            elif kind == "scaling_scalar":
                base_scale = target.get("base_scale", None)
                if not isinstance(base_scale, (int, float)):
                    current_scaling = getattr(module, "scaling", None)
                    if not isinstance(current_scaling, (int, float)):
                        _inc(skipped_by_kind, kind_key)
                        if warn_hook:
                            warn_hook(
                                f"Skipping scaling_scalar target with non-numeric scaling "
                                f"(adapter={adapter_name}, module={module_name})"
                            )
                        continue
                    base_scale = float(current_scaling)
                module.scaling = base_scale * scale
                modified += 1
                _inc(modified_by_kind, kind_key)
        except Exception as exc:
            _inc(skipped_by_kind, kind_key)
            if warn_hook:
                warn_hook(
                    f"Failed to apply LoRA scale target "
                    f"(adapter={adapter_name}, module={module_name}, kind={kind_key}, err={exc})"
                )
            if debug_hook:
                err_tb = traceback.format_exc()
                debug_hook(
                    f"Scale application exception for target "
                    f"(adapter={adapter_name}, module={module_name}, kind={kind_key}, err={exc}, tb={err_tb})"
                )

    report = {
        "adapter": adapter_name,
        "modified_total": modified,
        "modified_by_kind": modified_by_kind,
        "skipped_by_kind": skipped_by_kind,
    }
    return modified, report
