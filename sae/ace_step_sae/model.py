"""TopK Sparse Autoencoder for ACE-Step DiT activations.

Implements the k-sparse autoencoder architecture following:
  - Gao et al. (2024) "Scaling and evaluating sparse autoencoders"
  - Singh, Cherep & Maes (2025) "Discovering and Steering Interpretable
    Concepts in Large Generative Music Models"

Architecture:
    Encoder:  h = W_enc @ (x - b_dec) + b_enc
    TopK:     z = TopK(h, k)
    Decoder:  x_hat = W_dec @ z + b_dec

Loss:
    L = MSE(x, x_hat) + aux_coeff * AuxLoss(dead features)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ace_step_sae.config import SAEConfig


class TopKSparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder.

    Encodes residual-stream activations into a high-dimensional sparse
    representation, then decodes back to reconstruct the original.
    Sparsity is enforced via a hard TopK activation: only the ``k``
    largest latent activations survive; the rest are zeroed.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        d_sae = cfg.d_sae
        self.k = cfg.k

        # Encoder: (x - b_dec) @ W_enc + b_enc
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: z @ W_dec + b_dec
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Dead feature tracking buffers
        self.register_buffer(
            "feature_activation_count",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer(
            "steps_since_reset", torch.tensor(0, dtype=torch.long)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize encoder as transpose of decoder (Gao et al.)."""
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
            self.W_enc.data = self.W_dec.data.T.clone()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to pre-TopK latent activations ``[..., d_sae]``."""
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    @staticmethod
    def _topk(h: torch.Tensor, k: int):
        """Return (sparse_z, topk_values, topk_indices)."""
        topk_values, topk_indices = torch.topk(h, k, dim=-1)
        z = torch.zeros_like(h)
        z.scatter_(-1, topk_indices, topk_values)
        return z, topk_values, topk_indices

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse latents to model space ``[..., d_model]``."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward: encode -> topk -> decode.

        Args:
            x: ``[batch, d_model]`` activation vectors.

        Returns:
            Dict with ``x_hat, z, h, topk_indices, loss, mse_loss,
            aux_loss``.
        """
        h = self.encode(x)
        z, topk_values, topk_indices = self._topk(h, self.k)
        x_hat = self.decode(z)

        mse_loss = F.mse_loss(x_hat, x)
        aux_loss = self._aux_dead_feature_loss(x, h, topk_indices)
        loss = mse_loss + self.cfg.aux_loss_coeff * aux_loss

        return {
            "x_hat": x_hat,
            "z": z,
            "h": h,
            "topk_indices": topk_indices,
            "loss": loss,
            "mse_loss": mse_loss,
            "aux_loss": aux_loss,
        }

    # ------------------------------------------------------------------
    # Auxiliary dead-feature loss  (Gao et al. 2024)
    # ------------------------------------------------------------------

    def _aux_dead_feature_loss(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the residual using top-k *dead* features."""
        if self.cfg.aux_loss_coeff == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Track activations
        if self.training:
            fired = torch.zeros(
                self.cfg.d_sae, device=x.device, dtype=torch.long
            )
            fired.scatter_add_(
                0,
                topk_indices.reshape(-1),
                torch.ones(
                    topk_indices.numel(), device=x.device, dtype=torch.long
                ),
            )
            self.feature_activation_count.add_(
                fired.to(self.feature_activation_count.device)
            )
            self.steps_since_reset += 1

        dead_mask = (
            self.feature_activation_count < self.cfg.dead_feature_threshold
        )
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Residual from alive reconstruction
        with torch.no_grad():
            z_alive = torch.zeros_like(h)
            alive_vals = torch.gather(h, -1, topk_indices)
            z_alive.scatter_(-1, topk_indices, alive_vals)
            residual = x - (z_alive @ self.W_dec + self.b_dec)

        # Use dead features to reconstruct residual
        h_dead = h.clone()
        h_dead[:, ~dead_mask] = -float("inf")
        k_dead = min(self.k, num_dead)
        topk_dead_vals, topk_dead_idx = torch.topk(h_dead, k_dead, dim=-1)
        z_dead = torch.zeros_like(h)
        z_dead.scatter_(-1, topk_dead_idx, topk_dead_vals)
        x_hat_dead = z_dead @ self.W_dec  # no bias for residual

        return F.mse_loss(x_hat_dead, residual)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset_dead_feature_tracking(self):
        self.feature_activation_count.zero_()
        self.steps_since_reset.zero_()

    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    # ------------------------------------------------------------------
    # Feature access helpers (used by steering & discovery)
    # ------------------------------------------------------------------

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Decoder column for a single feature ``[d_model]``."""
        return self.W_dec[feature_idx]

    def get_feature_directions(self, feature_indices: list[int]) -> torch.Tensor:
        """Decoder columns for multiple features ``[n, d_model]``."""
        return self.W_dec[feature_indices]

    def get_alive_features(self) -> torch.Tensor:
        """Boolean mask of alive features ``[d_sae]``."""
        return self.feature_activation_count >= self.cfg.dead_feature_threshold

    @property
    def num_alive(self) -> int:
        return self.get_alive_features().sum().item()

    @property
    def num_dead(self) -> int:
        return self.cfg.d_sae - self.num_alive

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {"cfg": self.cfg, "state_dict": self.state_dict()}, path
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TopKSparseAutoencoder":
        data = torch.load(path, map_location=device, weights_only=False)
        sae = cls(data["cfg"])
        sae.load_state_dict(data["state_dict"])
        return sae
