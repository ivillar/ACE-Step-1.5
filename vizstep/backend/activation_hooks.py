"""PyTorch forward hook manager for extracting activation statistics."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.hooks import RemovableHook


@dataclass
class ActivationStats:
    """Summary statistics for a single layer's activation."""

    layer_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    norm: float
    heatmap: np.ndarray | None = None


@dataclass
class HookManager:
    """Attaches/detaches forward hooks on named modules.

    Hooks compute summary stats on-GPU (mean/std/max), copying only scalars to CPU.
    For detailed view: quantize to uint8, downsample, zlib compress.
    """

    _hooks: list[RemovableHook] = field(default_factory=list)
    _stats_buffer: list[ActivationStats] = field(default_factory=list)
    _detailed: bool = False
    _max_heatmap_size: int = 128

    def attach(
        self,
        model: nn.Module,
        layer_patterns: list[str],
        detailed: bool = False,
    ) -> int:
        """Attach forward hooks to modules matching the given name patterns.

        Args:
            model: The PyTorch model to hook into.
            layer_patterns: List of glob/regex patterns to match against module names.
            detailed: If True, also capture downsampled heatmaps.

        Returns:
            Number of hooks attached.
        """
        self._detailed = detailed
        count = 0
        for name, module in model.named_modules():
            if _matches_any(name, layer_patterns):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
                count += 1
        return count

    def detach_all(self) -> None:
        """Remove all attached hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def flush_stats(self) -> list[ActivationStats]:
        """Return and clear all buffered activation stats."""
        stats = list(self._stats_buffer)
        self._stats_buffer.clear()
        return stats

    def _make_hook(self, layer_name: str) -> Callable:
        """Create a forward hook closure for the given layer."""

        def hook_fn(module: nn.Module, input: tuple, output: torch.Tensor | tuple) -> None:
            tensor = output
            if isinstance(output, tuple):
                tensor = output[0]
            if not isinstance(tensor, torch.Tensor):
                return

            with torch.no_grad():
                flat = tensor.float().flatten()
                stats = ActivationStats(
                    layer_name=layer_name,
                    mean=flat.mean().item(),
                    std=flat.std().item(),
                    min_val=flat.min().item(),
                    max_val=flat.max().item(),
                    norm=flat.norm().item(),
                )

                if self._detailed:
                    stats.heatmap = self._make_heatmap(tensor)

            self._stats_buffer.append(stats)

        return hook_fn

    def _make_heatmap(self, tensor: torch.Tensor) -> np.ndarray:
        """Create a downsampled uint8 heatmap from a tensor.

        Reduces the tensor to 2D by taking mean over batch and channel dims,
        then downsamples to at most _max_heatmap_size on each side.
        """
        with torch.no_grad():
            # Reduce to 2D: take mean over all dims except last two
            t = tensor.float()
            while t.dim() > 2:
                t = t.mean(dim=0)

            # Downsample if needed
            h, w = t.shape
            max_s = self._max_heatmap_size
            if h > max_s or w > max_s:
                scale = min(max_s / h, max_s / w)
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                t = t.unsqueeze(0).unsqueeze(0)
                t = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
                t = t.squeeze(0).squeeze(0)

            # Normalize to 0-255
            t_min, t_max = t.min(), t.max()
            if t_max - t_min > 1e-8:
                t = (t - t_min) / (t_max - t_min) * 255
            else:
                t = torch.zeros_like(t)

            return t.byte().cpu().numpy()


def _matches_any(name: str, patterns: list[str]) -> bool:
    """Check if a module name matches any of the given patterns (glob or regex)."""
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        try:
            if re.fullmatch(pattern, name):
                return True
        except re.error:
            pass
    return False
