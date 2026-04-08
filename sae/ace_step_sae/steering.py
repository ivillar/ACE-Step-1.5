"""Activation steering for ACE-Step DiT using trained SAE features.

Steers music generation by injecting scaled SAE feature directions into
the residual stream at specified hook points.  Adapts the approach from:
  - Singh, Cherep & Maes (2025) for concept steering in music models
  - "Steering Diffusion Transformers with Sparse Autoencoders" for
    multi-layer, timestep-aware steering in DiTs

Steering equation:
    x_steered = x + sum_i(alpha_i * W_dec[feature_i])

where ``alpha_i`` is the per-feature steering coefficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ace_step_sae.model import TopKSparseAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class SteeringSpec:
    """Specification for a single steering intervention.

    Attributes:
        feature_idx: Index of the SAE feature to steer.
        coefficient: Scaling factor.  Positive amplifies the concept,
            negative suppresses it.  Typical range: [-20, 20].
        layer_idx: Which DiT layer(s) to apply steering at.
            ``None`` = use the layer the SAE was trained on.
        timestep_range: ``(t_min, t_max)`` — only steer when the
            diffusion timestep falls in this range.
            ``None`` = steer at all timesteps.
    """

    feature_idx: int
    coefficient: float = 5.0
    layer_idx: int | None = None
    timestep_range: tuple[float, float] | None = None


class SteeringHook:
    """Install steering hooks on ACE-Step DiT layers.

    Usage::

        sae = TopKSparseAutoencoder.load("sae_layer_24.pt")
        hook = SteeringHook(dit_model, sae, default_layer=24)
        hook.add(SteeringSpec(feature_idx=42, coefficient=10.0))
        hook.add(SteeringSpec(feature_idx=100, coefficient=-5.0))
        hook.install()

        # ... run generation ...

        hook.remove()

    For multi-layer steering, load SAEs for each layer and create
    separate ``SteeringHook`` instances.
    """

    def __init__(
        self,
        dit_model: nn.Module,
        sae: TopKSparseAutoencoder,
        default_layer: int = -1,
    ):
        """
        Args:
            dit_model: ``AceStepDiTModel`` with ``.layers``.
            sae: Trained SAE whose decoder columns define the feature
                directions.
            default_layer: Layer to steer if ``SteeringSpec.layer_idx``
                is ``None``.  ``-1`` means last layer.
        """
        self.dit_model = dit_model
        self.sae = sae
        self.default_layer = (
            default_layer if default_layer >= 0 else len(dit_model.layers) - 1
        )
        self.specs: list[SteeringSpec] = []
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._current_timestep: float | None = None

        # Pre-cache feature directions on the model's device
        self._device = next(dit_model.parameters()).device
        self._dtype = next(dit_model.parameters()).dtype

    # ------------------------------------------------------------------
    # Spec management
    # ------------------------------------------------------------------

    def add(self, spec: SteeringSpec):
        self.specs.append(spec)

    def clear_specs(self):
        self.specs.clear()

    def set_timestep(self, t: float):
        self._current_timestep = t

    # ------------------------------------------------------------------
    # Hook logic
    # ------------------------------------------------------------------

    def install(self):
        """Register forward hooks on the target layers."""
        self.remove()
        # Group specs by layer
        layer_specs: dict[int, list[SteeringSpec]] = {}
        for spec in self.specs:
            layer = spec.layer_idx if spec.layer_idx is not None else self.default_layer
            layer_specs.setdefault(layer, []).append(spec)

        for layer_idx, specs in layer_specs.items():
            # Pre-compute the combined steering vector per timestep-range group
            layer_module = self.dit_model.layers[layer_idx]
            handle = layer_module.register_forward_hook(
                self._make_hook(specs)
            )
            self._handles.append(handle)

        logger.info(
            f"Installed steering hooks: {len(self.specs)} features across "
            f"{len(layer_specs)} layers"
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, specs: list[SteeringSpec]):
        """Create a hook that adds the steering directions to hidden_states."""
        # Pre-fetch decoder columns
        feature_indices = [s.feature_idx for s in specs]
        directions = self.sae.get_feature_directions(feature_indices)
        directions = directions.to(device=self._device, dtype=self._dtype)

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            t = self._current_timestep

            # Compute combined steering vector
            steering_vec = torch.zeros(
                hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            for i, spec in enumerate(specs):
                if spec.timestep_range is not None and t is not None:
                    t_min, t_max = spec.timestep_range
                    if not (t_min <= t <= t_max):
                        continue
                steering_vec += spec.coefficient * directions[i].to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

            # Add to all positions in the sequence
            hidden_states = hidden_states + steering_vec.unsqueeze(0).unsqueeze(0)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook_fn

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *args):
        self.remove()


class MultiLayerSteeringHook:
    """Convenience wrapper for steering across multiple layers.

    Implements the multi-layer steering approach where the same concept
    is reinforced at multiple DiT layers with a similarity-based layer
    selection criterion.

    Usage::

        hook = MultiLayerSteeringHook(dit_model)
        hook.add_sae(layer_idx=20, sae=sae_20)
        hook.add_sae(layer_idx=24, sae=sae_24)
        hook.add_sae(layer_idx=28, sae=sae_28)

        # Steer feature 42 across all loaded layers
        hook.steer_feature(feature_idx=42, coefficient=10.0)
        hook.install()

        # ... run generation ...

        hook.remove()
    """

    def __init__(self, dit_model: nn.Module):
        self.dit_model = dit_model
        self.hooks: list[SteeringHook] = []
        self._layer_saes: dict[int, TopKSparseAutoencoder] = {}

    def add_sae(self, layer_idx: int, sae: TopKSparseAutoencoder):
        self._layer_saes[layer_idx] = sae

    def steer_feature(
        self,
        feature_idx: int,
        coefficient: float = 5.0,
        timestep_range: tuple[float, float] | None = None,
        layers: list[int] | None = None,
    ):
        """Add a steering spec for the given feature across all (or specific) layers."""
        target_layers = layers or list(self._layer_saes.keys())
        for layer_idx in target_layers:
            if layer_idx not in self._layer_saes:
                logger.warning(
                    f"No SAE loaded for layer {layer_idx} — skipping"
                )
                continue
            sae = self._layer_saes[layer_idx]
            hook = SteeringHook(self.dit_model, sae, default_layer=layer_idx)
            hook.add(
                SteeringSpec(
                    feature_idx=feature_idx,
                    coefficient=coefficient,
                    layer_idx=layer_idx,
                    timestep_range=timestep_range,
                )
            )
            self.hooks.append(hook)

    def install(self):
        for h in self.hooks:
            h.install()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def set_timestep(self, t: float):
        for h in self.hooks:
            h.set_timestep(t)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *args):
        self.remove()
