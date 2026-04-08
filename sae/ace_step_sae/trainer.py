"""Training loop for TopK Sparse Autoencoders on collected activations.

Follows the recipe from Gao et al. (2024) with:
  - TopK activation (no L1 penalty needed)
  - Auxiliary dead-feature loss
  - Decoder weight normalization
  - Linear warmup schedule
  - Periodic dead-feature counter resets
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ace_step_sae.config import SAEConfig
from ace_step_sae.model import TopKSparseAutoencoder

logger = logging.getLogger(__name__)


class SAETrainer:
    """Train a TopK SAE on pre-collected activations.

    Usage::

        trainer = SAETrainer(cfg, activations_tensor)
        trainer.train()
        trainer.sae.save("sae_layer_24.pt")
    """

    def __init__(
        self,
        cfg: SAEConfig,
        data: torch.Tensor,
        sae: TopKSparseAutoencoder | None = None,
    ):
        """
        Args:
            cfg: SAE configuration.
            data: Activation tensor ``[N, d_model]``.
            sae: Optional pre-initialized SAE (for resuming).
        """
        self.cfg = cfg
        self.data = data
        self.sae = sae or TopKSparseAutoencoder(cfg)

        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        self.sae = self.sae.to(device=cfg.device, dtype=dtype)

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(), lr=cfg.lr, betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._warmup_lambda
        )

        self.dataset = TensorDataset(data.float())
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=(cfg.device != "cpu"),
        )
        self._data_iter = None

        # Logging
        self.log_history: list[dict] = []
        self.output_dir = cfg.resolve_output_dir()
        self.global_step = 0

    def _warmup_lambda(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return step / max(1, self.cfg.warmup_steps)
        return 1.0

    def _next_batch(self) -> torch.Tensor:
        """Infinite iterator over shuffled batches."""
        if self._data_iter is None:
            self._data_iter = iter(self.dataloader)
        try:
            (batch,) = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            (batch,) = next(self._data_iter)
        dtype = torch.bfloat16 if self.cfg.dtype == "bfloat16" else torch.float32
        return batch.to(device=self.cfg.device, dtype=dtype)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> TopKSparseAutoencoder:
        """Run the full training loop.

        Returns:
            The trained SAE.
        """
        logger.info(
            f"Training SAE: d_model={self.cfg.d_model}, d_sae={self.cfg.d_sae}, "
            f"k={self.cfg.k}, steps={self.cfg.num_steps}, "
            f"data_size={len(self.data)}"
        )

        self.sae.train()
        start_time = time.time()

        for step in range(1, self.cfg.num_steps + 1):
            self.global_step = step
            batch = self._next_batch()

            out = self.sae(batch)
            loss = out["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Normalize decoder weights
            if self.cfg.normalize_decoder:
                self.sae.normalize_decoder_weights()

            # Periodically reset dead-feature counters
            if step % self.cfg.dead_feature_window == 0:
                self.sae.reset_dead_feature_tracking()

            # Logging
            if step % self.cfg.log_every == 0:
                metrics = {
                    "step": step,
                    "loss": out["loss"].item(),
                    "mse_loss": out["mse_loss"].item(),
                    "aux_loss": out["aux_loss"].item(),
                    "num_alive": self.sae.num_alive,
                    "num_dead": self.sae.num_dead,
                    "pct_alive": self.sae.num_alive / self.cfg.d_sae * 100,
                    "lr": self.scheduler.get_last_lr()[0],
                    "elapsed_s": time.time() - start_time,
                }
                self.log_history.append(metrics)
                logger.info(
                    f"[step {step:>7d}] loss={metrics['loss']:.6f}  "
                    f"mse={metrics['mse_loss']:.6f}  "
                    f"aux={metrics['aux_loss']:.6f}  "
                    f"alive={metrics['num_alive']}/{self.cfg.d_sae} "
                    f"({metrics['pct_alive']:.1f}%)"
                )

            # Checkpointing
            if step % self.cfg.save_every == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(self.global_step, final=True)
        self._save_log()

        self.sae.eval()
        return self.sae

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int, final: bool = False):
        tag = "final" if final else f"step_{step:07d}"
        path = self.output_dir / f"sae_{tag}.pt"
        self.sae.save(str(path))
        logger.info(f"Checkpoint saved: {path}")

    def _save_log(self):
        path = self.output_dir / "training_log.json"
        with open(path, "w") as f:
            json.dump(self.log_history, f, indent=2)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, n_batches: int = 10) -> dict[str, float]:
        """Compute eval metrics over ``n_batches`` random batches."""
        self.sae.eval()
        total_mse = 0.0
        total_loss = 0.0
        for _ in range(n_batches):
            batch = self._next_batch()
            out = self.sae(batch)
            total_mse += out["mse_loss"].item()
            total_loss += out["loss"].item()
        self.sae.train()
        return {
            "eval_mse": total_mse / n_batches,
            "eval_loss": total_loss / n_batches,
        }
