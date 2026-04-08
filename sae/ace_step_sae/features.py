"""Feature discovery and automated labeling for SAE features.

Implements the feature analysis pipeline from Singh, Cherep & Maes (2025):
  1. Find top-activating examples for each SAE feature.
  2. Compute activation statistics (frequency, mean magnitude).
  3. Optionally auto-label features using CLAP similarity.

For diffusion models we additionally track how features vary across
denoising timesteps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np

from ace_step_sae.config import SAEConfig
from ace_step_sae.model import TopKSparseAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class FeatureRecord:
    """Summary record for a single SAE feature."""

    feature_idx: int
    activation_frequency: float = 0.0  # fraction of inputs where it fires
    mean_activation: float = 0.0
    max_activation: float = 0.0
    top_example_indices: list[int] = field(default_factory=list)
    top_example_activations: list[float] = field(default_factory=list)
    label: str = ""
    label_confidence: float = 0.0
    is_alive: bool = True


class FeatureAnalyzer:
    """Analyze features discovered by a trained SAE.

    Usage::

        analyzer = FeatureAnalyzer(sae)
        records = analyzer.analyze(activations, top_k_examples=10)
        analyzer.save_records(records, "features.json")
    """

    def __init__(self, sae: TopKSparseAutoencoder):
        self.sae = sae
        self.cfg = sae.cfg

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze(
        self,
        activations: torch.Tensor,
        top_k_examples: int = 10,
        batch_size: int = 4096,
    ) -> list[FeatureRecord]:
        """Compute feature statistics over a dataset of activations.

        Args:
            activations: ``[N, d_model]`` tensor.
            top_k_examples: Number of top-activating examples to record
                per feature.
            batch_size: Process activations in batches to limit memory.

        Returns:
            List of ``FeatureRecord`` for every feature in the SAE.
        """
        self.sae.eval()
        d_sae = self.cfg.d_sae
        N = activations.shape[0]
        device = next(self.sae.parameters()).device
        dtype = next(self.sae.parameters()).dtype

        # Accumulators
        fire_count = torch.zeros(d_sae, device="cpu")
        activation_sum = torch.zeros(d_sae, device="cpu")
        activation_max = torch.full((d_sae,), -float("inf"), device="cpu")

        # Top-k tracking: maintain a running heap per feature
        # For memory efficiency, we keep only top_k_examples per feature
        topk_vals = torch.full(
            (d_sae, top_k_examples), -float("inf"), device="cpu"
        )
        topk_idxs = torch.zeros(
            (d_sae, top_k_examples), dtype=torch.long, device="cpu"
        )

        offset = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = activations[start:end].to(device=device, dtype=dtype)
            h = self.sae.encode(batch)  # [B, d_sae]
            h_cpu = h.float().cpu()
            B = h_cpu.shape[0]

            # Firing counts: positive activations
            fired = h_cpu > 0
            fire_count += fired.sum(dim=0)

            # Sum and max
            pos_h = h_cpu.clamp(min=0)
            activation_sum += pos_h.sum(dim=0)
            batch_max, _ = h_cpu.max(dim=0)
            activation_max = torch.max(activation_max, batch_max)

            # Update top-k examples per feature
            # For each feature, check if any in this batch beat the
            # current min of the top-k
            for feat_idx in range(d_sae):
                feat_vals = h_cpu[:, feat_idx]
                combined_vals = torch.cat(
                    [topk_vals[feat_idx], feat_vals]
                )
                combined_idxs = torch.cat(
                    [
                        topk_idxs[feat_idx],
                        torch.arange(offset, offset + B, dtype=torch.long),
                    ]
                )
                tk_vals, tk_pos = torch.topk(
                    combined_vals, top_k_examples, dim=0
                )
                topk_vals[feat_idx] = tk_vals
                topk_idxs[feat_idx] = combined_idxs[tk_pos]

            offset += B
            if start % (batch_size * 10) == 0:
                logger.info(f"Analyzed {end}/{N} vectors...")

        # Build records
        alive_mask = self.sae.get_alive_features().cpu()
        records = []
        for i in range(d_sae):
            freq = fire_count[i].item() / N if N > 0 else 0.0
            mean_act = (
                activation_sum[i].item() / max(fire_count[i].item(), 1)
            )
            valid_mask = topk_vals[i] > -float("inf")
            top_idxs = topk_idxs[i][valid_mask].tolist()
            top_vals = topk_vals[i][valid_mask].tolist()

            rec = FeatureRecord(
                feature_idx=i,
                activation_frequency=freq,
                mean_activation=mean_act,
                max_activation=activation_max[i].item(),
                top_example_indices=top_idxs,
                top_example_activations=top_vals,
                is_alive=alive_mask[i].item(),
            )
            records.append(rec)

        logger.info(
            f"Analysis complete: {sum(r.is_alive for r in records)}/{d_sae} "
            f"alive features"
        )
        return records

    # ------------------------------------------------------------------
    # CLAP-based auto-labeling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def label_with_clap(
        self,
        records: list[FeatureRecord],
        audio_paths: list[str],
        candidate_labels: list[str],
        clap_model=None,
        clap_processor=None,
        top_n: int = 5,
        device: str = "cuda",
    ) -> list[FeatureRecord]:
        """Auto-label features using CLAP audio-text alignment.

        For each feature, loads audio from the top-activating examples,
        encodes with CLAP, and computes cosine similarity against each
        candidate text label.

        Args:
            records: Feature records (must have ``top_example_indices``).
            audio_paths: List mapping example indices to audio file paths.
            candidate_labels: Text labels to score against (e.g.
                ``["piano", "drums", "bass", "violin", ...]``).
            clap_model: A CLAP model instance.
            clap_processor: CLAP processor for tokenization.
            top_n: Use top-N activating examples per feature.
            device: Device for CLAP inference.

        Returns:
            Updated records with ``label`` and ``label_confidence`` set.
        """
        if clap_model is None or clap_processor is None:
            logger.warning(
                "CLAP model/processor not provided — skipping labeling. "
                "Install transformers and load ClapModel to enable."
            )
            return records

        import torchaudio

        # Pre-encode candidate labels
        text_inputs = clap_processor(
            text=candidate_labels, return_tensors="pt", padding=True
        ).to(device)
        text_embeds = clap_model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)

        for rec in records:
            if not rec.is_alive or not rec.top_example_indices:
                continue

            indices = rec.top_example_indices[:top_n]
            audio_embeds_list = []

            for idx in indices:
                if idx >= len(audio_paths):
                    continue
                try:
                    waveform, sr = torchaudio.load(audio_paths[idx])
                    if sr != 48000:
                        waveform = torchaudio.functional.resample(
                            waveform, sr, 48000
                        )
                    # Mono and trim
                    waveform = waveform.mean(dim=0)[:48000 * 10]
                    audio_inputs = clap_processor(
                        audios=waveform.numpy(),
                        sampling_rate=48000,
                        return_tensors="pt",
                    ).to(device)
                    audio_embed = clap_model.get_audio_features(**audio_inputs)
                    audio_embeds_list.append(
                        F.normalize(audio_embed, dim=-1)
                    )
                except Exception as e:
                    logger.warning(f"Failed to load audio {audio_paths[idx]}: {e}")

            if not audio_embeds_list:
                continue

            # Average audio embedding
            avg_embed = torch.cat(audio_embeds_list, dim=0).mean(dim=0, keepdim=True)
            avg_embed = F.normalize(avg_embed, dim=-1)

            # Cosine similarity with each label
            sims = (avg_embed @ text_embeds.T).squeeze(0)
            best_idx = sims.argmax().item()
            rec.label = candidate_labels[best_idx]
            rec.label_confidence = sims[best_idx].item()

        labeled_count = sum(1 for r in records if r.label)
        logger.info(f"Labeled {labeled_count}/{len(records)} features with CLAP")
        return records

    # ------------------------------------------------------------------
    # Timestep analysis (diffusion-specific)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def analyze_by_timestep(
        self,
        activations: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int = 4096,
    ) -> dict[float, torch.Tensor]:
        """Compute per-timestep mean feature activations.

        This reveals which features are most active at different stages
        of the denoising process.

        Args:
            activations: ``[N, d_model]``.
            timesteps: ``[N]`` timestep for each activation vector.
            batch_size: Batch size for encoding.

        Returns:
            ``{timestep_value: mean_feature_activations [d_sae]}``.
        """
        self.sae.eval()
        device = next(self.sae.parameters()).device
        dtype = next(self.sae.parameters()).dtype

        unique_ts = timesteps.unique().sort().values.tolist()
        results = {}

        for t in unique_ts:
            mask = timesteps == t
            t_acts = activations[mask]
            if t_acts.shape[0] == 0:
                continue

            all_h = []
            for start in range(0, t_acts.shape[0], batch_size):
                end = min(start + batch_size, t_acts.shape[0])
                batch = t_acts[start:end].to(device=device, dtype=dtype)
                h = self.sae.encode(batch).float().cpu()
                all_h.append(h.clamp(min=0))

            all_h = torch.cat(all_h, dim=0)
            results[t] = all_h.mean(dim=0)

        return results

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_records(records: list[FeatureRecord], path: str):
        data = []
        for r in records:
            data.append({
                "feature_idx": r.feature_idx,
                "activation_frequency": r.activation_frequency,
                "mean_activation": r.mean_activation,
                "max_activation": r.max_activation,
                "top_example_indices": r.top_example_indices,
                "top_example_activations": r.top_example_activations,
                "label": r.label,
                "label_confidence": r.label_confidence,
                "is_alive": r.is_alive,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} feature records to {path}")

    @staticmethod
    def load_records(path: str) -> list[FeatureRecord]:
        with open(path) as f:
            data = json.load(f)
        records = []
        for d in data:
            records.append(FeatureRecord(**d))
        return records
