"""Configuration for Sparse Autoencoders on ACE-Step DiT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SAEConfig:
    """Configuration for a TopK Sparse Autoencoder.

    Attributes:
        d_model: Dimension of the model's residual stream (hidden_size).
            ACE-Step turbo: 2048, base: 4096.
        expansion_factor: Multiplier for the SAE latent dimension.
            Latent dim = d_model * expansion_factor.
            Typical values: 4, 8, 16, 32.
        k: Number of active (non-zero) latents after TopK projection.
            Typical values: 32, 64, 128.
        hook_layers: Which DiT layers to train SAEs on.  ``None`` means
            all layers.  Later layers tend to produce more interpretable
            features.
        hook_point: Where in each layer to collect activations.
            ``"resid_post"`` = residual stream after the full layer.
        aux_loss_coeff: Coefficient for the auxiliary dead-neuron loss.
        normalize_decoder: Whether to normalize decoder columns to unit
            norm after each optimizer step.
        dead_feature_window: Steps over which a feature must fire at
            least ``dead_feature_threshold`` times to stay alive.
        dead_feature_threshold: Minimum activation count within window.
        lr: Peak learning rate.
        batch_size: Training batch size (number of activation vectors).
        num_steps: Total training steps.
        warmup_steps: Linear warmup steps.
        seed: Random seed.
        dtype: Torch dtype (``"float32"`` or ``"bfloat16"``).
        device: Torch device string.
        log_every: Log metrics every N steps.
        save_every: Checkpoint every N steps.
        output_dir: Where to save checkpoints and logs.
    """

    # Architecture
    d_model: int = 2048
    expansion_factor: int = 8
    k: int = 32

    # Hook config
    hook_layers: Optional[List[int]] = None
    hook_point: str = "resid_post"

    # Training regularization
    aux_loss_coeff: float = 1 / 32
    normalize_decoder: bool = True

    # Dead feature handling
    dead_feature_window: int = 5000
    dead_feature_threshold: int = 1

    # Optimizer
    lr: float = 3e-4
    batch_size: int = 4096
    num_steps: int = 100_000
    warmup_steps: int = 1000

    # Misc
    seed: int = 42
    dtype: str = "float32"
    device: str = "cuda"
    log_every: int = 100
    save_every: int = 10_000
    output_dir: str = "./sae_output"

    @property
    def d_sae(self) -> int:
        """Latent dimension of the SAE."""
        return self.d_model * self.expansion_factor

    def resolve_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
