"""Wraps the ACE-Step inference pipeline to emit visualization events."""

from __future__ import annotations

import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import soundfile as sf
import torch
from loguru import logger

from vizstep.backend.activation_hooks import HookManager
from vizstep.backend.serialization import (
    encode_activation_summary,
    encode_audio_ready,
    encode_dit_step,
    encode_stage_event,
)
from vizstep.backend.session import Session, SessionStatus

if TYPE_CHECKING:
    from acestep.models.dit.wrapper import AceStepDiTWrapper
    from acestep.models.lm.wrapper import AceStepLMWrapper


# Default hook patterns for each stage
LM_HOOK_PATTERNS = [
    "*.self_attn",
    "*.mlp",
]
DIT_HOOK_PATTERNS = [
    "*.attn*",
    "*.ff*",
    "*.norm*",
]
VAE_HOOK_PATTERNS = [
    "*.conv*",
    "*.norm*",
]


def get_model_metadata(
    dit_wrapper: AceStepDiTWrapper,
    lm_wrapper: AceStepLMWrapper | None = None,
) -> dict[str, Any]:
    """Extract model metadata for the frontend scene layout."""
    metadata: dict[str, Any] = {"stages": {}}

    # DiT metadata
    dit_info: dict[str, Any] = {"name": "DiT", "type": "diffusion_transformer"}
    if dit_wrapper.model is not None:
        dit_info["param_count"] = sum(p.numel() for p in dit_wrapper.model.parameters())
        dit_info["layers"] = _count_transformer_layers(dit_wrapper.model)
    if dit_wrapper.config is not None:
        dit_info["config"] = {
            k: v for k, v in vars(dit_wrapper.config).items()
            if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
        }
    metadata["stages"]["dit"] = dit_info

    # VAE metadata
    vae_info: dict[str, Any] = {"name": "VAE", "type": "autoencoder"}
    if dit_wrapper.vae is not None:
        vae_info["param_count"] = sum(p.numel() for p in dit_wrapper.vae.parameters())
    metadata["stages"]["vae"] = vae_info

    # LM metadata
    lm_info: dict[str, Any] = {"name": "LM", "type": "autoregressive_transformer"}
    if lm_wrapper is not None and lm_wrapper.llm is not None:
        lm_info["backend"] = lm_wrapper.llm_backend
        # Parameter count varies by backend; safest via tokenizer vocab
        if lm_wrapper.llm_tokenizer is not None:
            lm_info["vocab_size"] = lm_wrapper.llm_tokenizer.vocab_size
    metadata["stages"]["lm"] = lm_info

    metadata["sample_rate"] = dit_wrapper.sample_rate
    metadata["device"] = str(dit_wrapper.device)

    return metadata


def _count_transformer_layers(model: torch.nn.Module) -> int:
    """Estimate the number of transformer layers in a model."""
    count = 0
    for name, _ in model.named_modules():
        # Common patterns for transformer layer containers
        if name.endswith(".0") and ("layers" in name or "blocks" in name):
            parent = name.rsplit(".", 1)[0]
            count = max(count, sum(1 for n, _ in model.named_modules() if n.startswith(parent + ".")))
            break
    return count


class VizSession:
    """Manages a visualization-instrumented generation run."""

    def __init__(
        self,
        dit_wrapper: AceStepDiTWrapper,
        lm_wrapper: AceStepLMWrapper | None,
        session: Session,
    ):
        self.dit_wrapper = dit_wrapper
        self.lm_wrapper = lm_wrapper
        self.session = session
        self.hook_manager = HookManager()

    async def _send(self, data: bytes) -> None:
        """Push a binary frame to the session's WebSocket queue."""
        if self.session.ws_queue is not None:
            await self.session.ws_queue.put(data)

    def _send_sync(self, data: bytes) -> None:
        """Push from a sync context (hook callbacks, etc.)."""
        if self.session.ws_queue is not None:
            try:
                self.session.ws_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

    async def _flush_activation_stats(self, stage: str) -> None:
        """Collect buffered hook stats and send them over WebSocket."""
        stats = self.hook_manager.flush_stats()
        for s in stats:
            frame = encode_activation_summary(
                layer_name=f"{stage}/{s.layer_name}",
                mean=s.mean,
                std=s.std,
                min_val=s.min_val,
                max_val=s.max_val,
                norm=s.norm,
                heatmap=s.heatmap,
            )
            await self._send(frame)

    async def run_generation(self, params: dict[str, Any]) -> Path | None:
        """Run a full generation pipeline with viz events.

        Args:
            params: Dict of generation parameters matching generate_music() signature.

        Returns:
            Path to the generated WAV file, or None on error.
        """
        self.session.status = SessionStatus.RUNNING
        audio_path: Path | None = None

        try:
            # --- LM Stage ---
            if self.lm_wrapper is not None and self.lm_wrapper.llm is not None:
                await self._send(encode_stage_event("lm", "started"))
                self.session.current_stage = "lm"

                # Attach hooks to LM if it has a model attribute
                lm_model = getattr(self.lm_wrapper, "llm", None)
                if lm_model is not None and isinstance(lm_model, torch.nn.Module):
                    self.hook_manager.attach(lm_model, LM_HOOK_PATTERNS)

                # Note: LM generation happens inside generate_music() automatically
                # We just emit the stage events here for the frontend
                await self._send(encode_stage_event("lm", "complete"))
                self.hook_manager.detach_all()
                await self._flush_activation_stats("lm")

            # --- DiT Stage (with step-by-step tracking) ---
            await self._send(encode_stage_event("dit", "started"))
            self.session.current_stage = "dit"

            # Attach hooks to DiT model
            if self.dit_wrapper.model is not None:
                self.hook_manager.attach(self.dit_wrapper.model, DIT_HOOK_PATTERNS, detailed=True)

            # Run the actual generation
            result = self.dit_wrapper.generate_music(**params)

            # Flush DiT activation stats
            self.hook_manager.detach_all()
            await self._flush_activation_stats("dit")

            # Send DiT step events (from result time_costs if available)
            infer_steps = params.get("inference_steps", 8)
            for step in range(infer_steps):
                await self._send(encode_dit_step(step, infer_steps))

            await self._send(encode_stage_event("dit", "complete"))

            # --- VAE Stage ---
            await self._send(encode_stage_event("vae", "started"))
            self.session.current_stage = "vae"
            # VAE decode already happened inside generate_music
            await self._send(encode_stage_event("vae", "complete"))

            # --- Save audio ---
            if result.get("success") and result.get("audios"):
                audio_data = result["audios"][0]
                # Audio items may be dicts with a "tensor" key
                if isinstance(audio_data, dict):
                    audio_data = audio_data["tensor"]
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()

                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
                sf.write(tmp.name, audio_data.T if audio_data.ndim == 2 else audio_data, self.dit_wrapper.sample_rate)
                audio_path = Path(tmp.name)
                self.session.audio_path = audio_path

                await self._send(encode_audio_ready(self.session.session_id))

            self.session.status = SessionStatus.COMPLETE

        except Exception as exc:
            logger.exception("[VizSession] Generation failed")
            self.session.status = SessionStatus.ERROR
            self.session.error = f"{exc!s}\n{traceback.format_exc()}"
            self.hook_manager.detach_all()

        return audio_path
