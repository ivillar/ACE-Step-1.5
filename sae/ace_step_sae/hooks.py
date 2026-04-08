"""Activation collection hooks for ACE-Step's Diffusion Transformer.

Registers PyTorch forward hooks on DiT layers to capture residual-stream
activations during inference.  Collected activations are used to train
Sparse Autoencoders.

The collector supports two modes:
  1. **In-memory**: activations accumulated in RAM (fast, limited by memory).
  2. **Streaming**: activations flushed to disk shards in memory-mapped
     format (slower, but supports arbitrarily large datasets).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class ActivationCollector:
    """Collect residual-stream activations from ACE-Step DiT layers.

    Usage::

        collector = ActivationCollector(model.decoder, layers=[20, 24, 28, 31])
        collector.install()
        # ... run generation(s) ...
        activations = collector.harvest()   # dict[layer_idx -> Tensor]
        collector.remove()

    For diffusion models, activations carry an implicit timestep dimension
    because the same layers are called at every denoising step.  The
    ``timestep_filter`` lets you restrict collection to specific timesteps.
    """

    def __init__(
        self,
        dit_model: nn.Module,
        layers: list[int] | None = None,
        max_samples: int | None = None,
        timestep_filter: tuple[float, float] | None = None,
        stream_dir: str | None = None,
        stream_shard_size: int = 100_000,
    ):
        """
        Args:
            dit_model: The ``AceStepDiTModel`` instance whose ``.layers``
                attribute contains the ``AceStepDiTLayer`` modules.
            layers: Layer indices to hook.  ``None`` = all layers.
            max_samples: Stop collecting after this many vectors (per
                layer).  ``None`` = unlimited.
            timestep_filter: ``(t_min, t_max)`` — only collect when the
                current diffusion timestep falls in this range.  Useful
                for focusing on early or late denoising steps.
            stream_dir: If set, flush activations to disk shards in this
                directory instead of holding everything in RAM.
            stream_shard_size: Number of vectors per shard file.
        """
        self.dit_model = dit_model
        self.num_layers = len(dit_model.layers)
        self.layers = layers if layers is not None else list(range(self.num_layers))
        self.max_samples = max_samples
        self.timestep_filter = timestep_filter

        # Streaming config
        self.stream_dir = Path(stream_dir) if stream_dir else None
        self.stream_shard_size = stream_shard_size

        # Internal state
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._buffers: dict[int, list[torch.Tensor]] = {
            i: [] for i in self.layers
        }
        self._counts: dict[int, int] = {i: 0 for i in self.layers}
        self._shard_idx: dict[int, int] = {i: 0 for i in self.layers}
        self._current_timestep: float | None = None
        self._collecting = True

        if self.stream_dir:
            self.stream_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Timestep tracking
    # ------------------------------------------------------------------

    def set_timestep(self, t: float):
        """Call this before each diffusion step to update the timestep."""
        self._current_timestep = t

    def _should_collect(self) -> bool:
        if not self._collecting:
            return False
        if self.timestep_filter is not None and self._current_timestep is not None:
            t_min, t_max = self.timestep_filter
            if not (t_min <= self._current_timestep <= t_max):
                return False
        return True

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install(self):
        """Register forward hooks on the target layers."""
        self.remove()
        for layer_idx in self.layers:
            layer = self.dit_model.layers[layer_idx]
            handle = layer.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)

    def remove(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            if not self._should_collect():
                return
            if (
                self.max_samples is not None
                and self._counts[layer_idx] >= self.max_samples
            ):
                return
            # output is a tuple; first element is hidden_states
            hidden_states = output[0] if isinstance(output, tuple) else output
            # hidden_states: [batch, seq_len, d_model]
            # Flatten batch and sequence dims -> [batch * seq_len, d_model]
            flat = hidden_states.detach().float().cpu().reshape(-1, hidden_states.shape[-1])
            self._counts[layer_idx] += flat.shape[0]

            if self.stream_dir:
                self._buffers[layer_idx].append(flat)
                total_buffered = sum(b.shape[0] for b in self._buffers[layer_idx])
                if total_buffered >= self.stream_shard_size:
                    self._flush_shard(layer_idx)
            else:
                self._buffers[layer_idx].append(flat)

            # Check max
            if (
                self.max_samples is not None
                and all(
                    self._counts[i] >= self.max_samples for i in self.layers
                )
            ):
                self._collecting = False

        return hook_fn

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def harvest(self) -> dict[int, torch.Tensor]:
        """Concatenate collected activations per layer.

        Returns:
            ``{layer_idx: Tensor[N, d_model]}``.
        """
        if self.stream_dir:
            # Flush remaining
            for layer_idx in self.layers:
                if self._buffers[layer_idx]:
                    self._flush_shard(layer_idx)
            return self._load_shards()

        result = {}
        for layer_idx in self.layers:
            if self._buffers[layer_idx]:
                result[layer_idx] = torch.cat(self._buffers[layer_idx], dim=0)
            else:
                result[layer_idx] = torch.empty(0)
        return result

    def clear(self):
        """Clear all buffered activations."""
        for layer_idx in self.layers:
            self._buffers[layer_idx].clear()
            self._counts[layer_idx] = 0

    # ------------------------------------------------------------------
    # Disk streaming
    # ------------------------------------------------------------------

    def _flush_shard(self, layer_idx: int):
        if not self._buffers[layer_idx]:
            return
        data = torch.cat(self._buffers[layer_idx], dim=0)
        shard_path = (
            self.stream_dir / f"layer_{layer_idx:03d}_shard_{self._shard_idx[layer_idx]:06d}.pt"
        )
        torch.save(data, shard_path)
        self._shard_idx[layer_idx] += 1
        self._buffers[layer_idx].clear()

    def _load_shards(self) -> dict[int, torch.Tensor]:
        result = {}
        for layer_idx in self.layers:
            shards = sorted(
                self.stream_dir.glob(f"layer_{layer_idx:03d}_shard_*.pt")
            )
            if shards:
                result[layer_idx] = torch.cat(
                    [torch.load(s, weights_only=True) for s in shards], dim=0
                )
            else:
                result[layer_idx] = torch.empty(0)
        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *args):
        self.remove()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[int, int]:
        """Return ``{layer_idx: num_vectors_collected}``."""
        return dict(self._counts)


class ActivationDataset(torch.utils.data.Dataset):
    """Simple dataset wrapping a tensor of activation vectors.

    Supports loading from a single file or a directory of shards
    produced by ``ActivationCollector`` in streaming mode.
    """

    def __init__(self, data: torch.Tensor | str | Path):
        if isinstance(data, (str, Path)):
            p = Path(data)
            if p.is_dir():
                shards = sorted(p.glob("*.pt"))
                self.data = torch.cat(
                    [torch.load(s, weights_only=True) for s in shards], dim=0
                )
            else:
                self.data = torch.load(p, weights_only=True)
        else:
            self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
