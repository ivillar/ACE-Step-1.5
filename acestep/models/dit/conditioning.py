"""Consolidated handler functions – conditioning module."""

import re
import traceback
from typing import Any

import torch
from loguru import logger

from acestep.constants import DEFAULT_DIT_INSTRUCTION, SAMPLE_RATE, SFT_GEN_PROMPT

__all__ = [
    "_prepare_batch",
    "infer_refer_latent",
    "infer_text_embeddings",
    "infer_lyric_embeddings",
    "preprocess_batch",
    "_build_chunk_masks_and_src_latents",
    "_prepare_precomputed_lm_hints",
    "_prepare_text_conditioning_inputs",
    "_prepare_target_latents_and_wavs",
    "_normalize_audio_code_hints",
    "_normalize_instructions",
    "_create_fallback_vocal_languages",
    "_encode_audio_to_latents",
    "prepare_batch_data",
    "_parse_audio_code_string",
    "_decode_audio_codes_to_latents",
    "convert_src_audio_to_codes",
]

# --- From conditioning_batch.py ---

def _prepare_batch(
    self,
    captions: list[str],
    lyrics: list[str],
    keys: list[str] | None = None,
    target_wavs: torch.Tensor | None = None,
    refer_audios: list[list[torch.Tensor]] | None = None,
    metas: list[str | dict[str, Any]] | None = None,
    vocal_languages: list[str] | None = None,
    repainting_start: list[float] | None = None,
    repainting_end: list[float] | None = None,
    instructions: list[str] | None = None,
    audio_code_hints: list[str | None] | None = None,
    audio_cover_strength: float = 1.0,
    cover_noise_strength: float = 0.0,
) -> dict[str, Any]:
    """Prepare model-ready conditioning batch tensors and metadata.

    Args:
        captions: Per-item captions.
        lyrics: Per-item lyric strings.
        keys: Optional per-item keys.
        target_wavs: Target audio tensor batch.
        refer_audios: Optional nested reference-audio tensors.
        metas: Optional per-item metadata strings/dicts.
        vocal_languages: Optional per-item vocal language codes.
        repainting_start: Optional repaint start times.
        repainting_end: Optional repaint end times.
        instructions: Optional per-item generation instructions.
        audio_code_hints: Optional per-item serialized audio-code hints.
        audio_cover_strength: Blend factor for cover/non-cover conditioning.

    Returns:
        Batch dictionary containing padded tensors and conditioning metadata
        consumed by ``preprocess_batch`` and downstream generation.
    """
    batch_size = len(captions)
    audio_code_hints = self._normalize_audio_code_hints(audio_code_hints, batch_size)

    if refer_audios is None:
        refer_audios = [[torch.zeros(2, 30 * self.sample_rate)] for _ in range(batch_size)]
    for ii, refer_audio_list in enumerate(refer_audios):
        if isinstance(refer_audio_list, list):
            for idx, _ in enumerate(refer_audio_list):
                refer_audio_list[idx] = refer_audio_list[idx].to(self.device).to(self._get_vae_dtype())
        elif isinstance(refer_audio_list, torch.Tensor):
            refer_audios[ii] = refer_audios[ii].to(self.device)

    if vocal_languages is None:
        vocal_languages = self._create_fallback_vocal_languages(batch_size)
    parsed_metas = self._parse_metas(metas)

    target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled = (
        self._prepare_target_latents_and_wavs(batch_size, target_wavs, audio_code_hints)
    )
    wav_lengths = torch.tensor([target_wavs.shape[-1]] * batch_size, dtype=torch.long)

    instructions = self._normalize_instructions(instructions, batch_size, DEFAULT_DIT_INSTRUCTION)
    chunk_masks, spans, is_covers, src_latents = self._build_chunk_masks_and_src_latents(
        batch_size,
        max_latent_length,
        instructions,
        audio_code_hints,
        target_wavs,
        target_latents,
        repainting_start,
        repainting_end,
        silence_latent_tiled,
    )
    precomputed_lm_hints_25hz = self._prepare_precomputed_lm_hints(
        batch_size, audio_code_hints, max_latent_length, silence_latent_tiled
    )
    (
        text_inputs,
        padded_text_token_idss,
        padded_text_attention_masks,
        padded_lyric_token_idss,
        padded_lyric_attention_masks,
        padded_non_cover_text_input_ids,
        padded_non_cover_text_attention_masks,
    ) = self._prepare_text_conditioning_inputs(
        batch_size,
        instructions,
        captions,
        lyrics,
        parsed_metas,
        vocal_languages,
        audio_cover_strength,
    )

    batch = {
        "keys": keys,
        "target_wavs": target_wavs.to(self.device),
        "refer_audioss": refer_audios,
        "wav_lengths": wav_lengths.to(self.device),
        "captions": captions,
        "lyrics": lyrics,
        "metas": parsed_metas,
        "vocal_languages": vocal_languages,
        "target_latents": target_latents,
        "src_latents": src_latents,
        "latent_masks": latent_masks,
        "chunk_masks": chunk_masks,
        "spans": spans,
        "text_inputs": text_inputs,
        "text_token_idss": padded_text_token_idss,
        "text_attention_masks": padded_text_attention_masks,
        "lyric_token_idss": padded_lyric_token_idss,
        "lyric_attention_masks": padded_lyric_attention_masks,
        "is_covers": is_covers,
        "precomputed_lm_hints_25Hz": precomputed_lm_hints_25hz,
        "non_cover_text_input_ids": padded_non_cover_text_input_ids,
        "non_cover_text_attention_masks": padded_non_cover_text_attention_masks,
    }
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(self.device)
            if torch.is_floating_point(batch[k]):
                batch[k] = batch[k].to(self.dtype)
    return batch

# --- From conditioning_embed.py ---

def infer_refer_latent(self, refer_audioss: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer packed reference-audio latents and order mask."""
    refer_audio_order_mask = []
    refer_audio_latents = []
    self._ensure_silence_latent_on_device()

    def _normalize_audio_2d(a: torch.Tensor) -> torch.Tensor:
        if not isinstance(a, torch.Tensor):
            raise TypeError(f"refer_audio must be a torch.Tensor, got {type(a)!r}")
        if a.dim() == 3 and a.shape[0] == 1:
            a = a.squeeze(0)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if a.dim() != 2:
            raise ValueError(f"refer_audio must be 1D/2D/3D(1,2,T); got shape={tuple(a.shape)}")
        if a.shape[0] == 1:
            a = torch.cat([a, a], dim=0)
        return a[:2]

    def _ensure_latent_3d(z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4 and z.shape[0] == 1:
            z = z.squeeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return z

    refer_encode_cache: dict[int, torch.Tensor] = {}
    for batch_idx, refer_audios in enumerate(refer_audioss):
        if len(refer_audios) == 1 and torch.all(refer_audios[0] == 0.0):
            refer_audio_latent = _ensure_latent_3d(self.silence_latent[:, :750, :])
            refer_audio_latents.append(refer_audio_latent)
            refer_audio_order_mask.append(batch_idx)
        else:
            for refer_audio in refer_audios:
                cache_key = refer_audio.data_ptr()
                if cache_key in refer_encode_cache:
                    refer_audio_latent = refer_encode_cache[cache_key].clone()
                else:
                    refer_audio = _normalize_audio_2d(refer_audio)
                    with torch.inference_mode():
                        refer_audio_latent = self.tiled_encode(refer_audio, offload_latent_to_cpu=True)
                    refer_audio_latent = refer_audio_latent.to(self.device).to(self.dtype)
                    if refer_audio_latent.dim() == 2:
                        refer_audio_latent = refer_audio_latent.unsqueeze(0)
                    refer_audio_latent = _ensure_latent_3d(refer_audio_latent.transpose(1, 2))
                    refer_encode_cache[cache_key] = refer_audio_latent
                refer_audio_latents.append(refer_audio_latent)
                refer_audio_order_mask.append(batch_idx)

    refer_audio_latents = torch.cat(refer_audio_latents, dim=0)
    refer_audio_order_mask = torch.tensor(refer_audio_order_mask, device=self.device, dtype=torch.long)
    return refer_audio_latents, refer_audio_order_mask

def infer_text_embeddings(self, text_token_idss):
    """Infer text-token embeddings via text encoder."""
    with torch.inference_mode():
        return self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state

def infer_lyric_embeddings(self, lyric_token_ids):
    """Infer lyric-token embeddings via text encoder embedding table."""
    with torch.inference_mode():
        return self.text_encoder.embed_tokens(lyric_token_ids)

def preprocess_batch(self, batch) -> tuple:
    """Preprocess an already prepared batch for DiT model input."""
    target_latents = batch["target_latents"]
    src_latents = batch["src_latents"]
    attention_mask = batch["latent_masks"]
    audio_codes = batch.get("audio_codes", None)
    audio_attention_mask = attention_mask

    dtype = target_latents.dtype
    device = target_latents.device

    keys = batch["keys"]
    with self._load_model_context("vae"):
        refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask = self.infer_refer_latent(
            batch["refer_audioss"]
        )
    if refer_audio_acoustic_hidden_states_packed.dtype != dtype:
        refer_audio_acoustic_hidden_states_packed = refer_audio_acoustic_hidden_states_packed.to(dtype)

    chunk_mask = batch["chunk_masks"]
    chunk_mask = chunk_mask.to(device).unsqueeze(-1).repeat(1, 1, target_latents.shape[2])
    spans = batch["spans"]

    text_token_idss = batch["text_token_idss"]
    text_attention_mask = batch["text_attention_masks"]
    lyric_token_idss = batch["lyric_token_idss"]
    lyric_attention_mask = batch["lyric_attention_masks"]
    text_inputs = batch["text_inputs"]

    logger.info("[preprocess_batch] Inferring prompt embeddings...")
    with self._load_model_context("text_encoder"):
        text_hidden_states = self.infer_text_embeddings(text_token_idss)
        logger.info("[preprocess_batch] Inferring lyric embeddings...")
        lyric_hidden_states = self.infer_lyric_embeddings(lyric_token_idss)

        is_covers = batch["is_covers"]
        precomputed_lm_hints_25hz = batch.get("precomputed_lm_hints_25Hz", None)
        non_cover_text_input_ids = batch.get("non_cover_text_input_ids", None)
        non_cover_text_attention_masks = batch.get("non_cover_text_attention_masks", None)
        non_cover_text_hidden_states = None
        if non_cover_text_input_ids is not None:
            logger.info("[preprocess_batch] Inferring non-cover text embeddings...")
            non_cover_text_hidden_states = self.infer_text_embeddings(non_cover_text_input_ids)

    return (
        keys,
        text_inputs,
        src_latents,
        target_latents,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        audio_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
        chunk_mask,
        spans,
        is_covers,
        audio_codes,
        lyric_token_idss,
        precomputed_lm_hints_25hz,
        non_cover_text_hidden_states,
        non_cover_text_attention_masks,
    )

# --- From conditioning_masks.py ---

def _build_chunk_masks_and_src_latents(
    self,
    batch_size: int,
    max_latent_length: int,
    instructions: list[str],
    audio_code_hints: list[str | None],
    target_wavs: torch.Tensor,
    target_latents: torch.Tensor,
    repainting_start: list[float] | None,
    repainting_end: list[float] | None,
    silence_latent_tiled: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[str, int, int]], torch.Tensor, torch.Tensor]:
    """Create chunk masks/spans and corresponding source latents."""
    chunk_masks = []
    spans = []
    is_covers = []
    repainting_ranges: dict[int, tuple[int, int]] = {}

    for i in range(batch_size):
        has_code_hint = audio_code_hints[i] is not None
        if repainting_start is not None and repainting_end is not None:
            start_sec = repainting_start[i] if repainting_start[i] is not None else 0.0
            end_sec = repainting_end[i]
            if end_sec is not None and end_sec > start_sec:
                left_padding_sec = max(0, -start_sec)
                adjusted_start_sec = start_sec + left_padding_sec
                adjusted_end_sec = end_sec + left_padding_sec
                start_latent = int(adjusted_start_sec * self.sample_rate // 1920)
                end_latent = int(adjusted_end_sec * self.sample_rate // 1920)
                start_latent = max(0, min(start_latent, max_latent_length - 1))
                end_latent = max(start_latent + 1, min(end_latent, max_latent_length))

                mask = torch.zeros(max_latent_length, dtype=torch.bool, device=self.device)
                mask[start_latent:end_latent] = True
                chunk_masks.append(mask)
                spans.append(("repainting", start_latent, end_latent))
                repainting_ranges[i] = (start_latent, end_latent)
                is_covers.append(False)
                continue

        chunk_masks.append(torch.ones(max_latent_length, dtype=torch.bool, device=self.device))
        spans.append(("full", 0, max_latent_length))
        instruction_i = instructions[i] if instructions and i < len(instructions) else ""
        instruction_lower = instruction_i.lower()
        is_cover = (
            "generate audio semantic tokens" in instruction_lower
            and "based on the given conditions" in instruction_lower
        ) or has_code_hint
        is_covers.append(is_cover)

    chunk_masks_tensor = torch.stack(chunk_masks)
    is_covers_tensor = torch.BoolTensor(is_covers).to(self.device)

    src_latents_list = []
    for i in range(batch_size):
        has_code_hint = audio_code_hints[i] is not None
        has_target_audio = has_code_hint or (target_wavs is not None and target_wavs[i].abs().sum() > 1e-6)
        if has_target_audio:
            if i in repainting_ranges:
                src_latent = target_latents[i].clone()
                start_latent, end_latent = repainting_ranges[i]
                src_latent[start_latent:end_latent] = silence_latent_tiled[start_latent:end_latent]
                src_latents_list.append(src_latent)
            else:
                src_latents_list.append(target_latents[i].clone())
        else:
            src_latents_list.append(silence_latent_tiled.clone())
    src_latents = torch.stack(src_latents_list)
    return chunk_masks_tensor, spans, is_covers_tensor, src_latents

# --- From conditioning_text.py ---

def _prepare_precomputed_lm_hints(
    self,
    batch_size: int,
    audio_code_hints: list[str | None],
    max_latent_length: int,
    silence_latent_tiled: torch.Tensor,
) -> torch.Tensor | None:
    """Decode audio-code hints into padded 25Hz latent hints."""
    precomputed_lm_hints_25hz_list = []
    for i in range(batch_size):
        if audio_code_hints[i] is not None:
            logger.info(f"[generate_music] Decoding audio codes for LM hints for item {i}...")
            hints = self._decode_audio_codes_to_latents(audio_code_hints[i])
            if hints is not None:
                if hints.shape[1] < max_latent_length:
                    pad_length = max_latent_length - hints.shape[1]
                    pad = self.silence_latent
                    if pad.dim() == 2:
                        pad = pad.unsqueeze(0)
                    if hints.dim() == 2:
                        hints = hints.unsqueeze(0)
                    pad_chunk = pad[:, :pad_length, :]
                    if pad_chunk.device != hints.device or pad_chunk.dtype != hints.dtype:
                        pad_chunk = pad_chunk.to(device=hints.device, dtype=hints.dtype)
                    hints = torch.cat([hints, pad_chunk], dim=1)
                elif hints.shape[1] > max_latent_length:
                    hints = hints[:, :max_latent_length, :]
                precomputed_lm_hints_25hz_list.append(hints[0])
            else:
                precomputed_lm_hints_25hz_list.append(None)
        else:
            precomputed_lm_hints_25hz_list.append(None)

    if any(h is not None for h in precomputed_lm_hints_25hz_list):
        return torch.stack([h if h is not None else silence_latent_tiled for h in precomputed_lm_hints_25hz_list])
    return None

def _prepare_text_conditioning_inputs(
    self,
    batch_size: int,
    instructions: list[str],
    captions: list[str],
    lyrics: list[str],
    parsed_metas: list[str],
    vocal_languages: list[str],
    audio_cover_strength: float,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Tokenize caption/lyric prompts and optional non-cover branch prompts."""
    actual_captions, actual_languages = self._extract_caption_and_language(parsed_metas, captions, vocal_languages)

    text_inputs = []
    text_token_idss = []
    text_attention_masks = []
    lyric_token_idss = []
    lyric_attention_masks = []

    for i in range(batch_size):
        instruction = self._format_instruction(
            instructions[i] if i < len(instructions) else DEFAULT_DIT_INSTRUCTION
        )
        actual_caption = actual_captions[i]
        actual_language = actual_languages[i]
        text_prompt = SFT_GEN_PROMPT.format(instruction, actual_caption, parsed_metas[i])

        if i == 0:
            logger.info(f"\n{'='*70}")
            logger.info("🔍 [DEBUG] DiT TEXT ENCODER INPUT (Inference)")
            logger.info(f"{'='*70}")
            logger.info(f"text_prompt:\n{text_prompt}")
            logger.info(f"{'='*70}")
            logger.info(f"lyrics_text:\n{self._format_lyrics(lyrics[i], actual_language)}")
            logger.info(f"{'='*70}\n")

        text_inputs_dict = self.text_tokenizer(
            text_prompt,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        text_token_ids = text_inputs_dict.input_ids[0]
        text_attention_mask = text_inputs_dict.attention_mask[0].bool()

        lyrics_text = self._format_lyrics(lyrics[i], actual_language)
        lyrics_inputs_dict = self.text_tokenizer(
            lyrics_text,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        lyric_token_ids = lyrics_inputs_dict.input_ids[0]
        lyric_attention_mask = lyrics_inputs_dict.attention_mask[0].bool()

        text_inputs.append(text_prompt + "\n\n" + lyrics_text)
        text_token_idss.append(text_token_ids)
        text_attention_masks.append(text_attention_mask)
        lyric_token_idss.append(lyric_token_ids)
        lyric_attention_masks.append(lyric_attention_mask)

    max_text_length = max(len(seq) for seq in text_token_idss)
    padded_text_token_idss = self._pad_sequences(text_token_idss, max_text_length, self.text_tokenizer.pad_token_id)
    padded_text_attention_masks = self._pad_sequences(text_attention_masks, max_text_length, 0)

    max_lyric_length = max(len(seq) for seq in lyric_token_idss)
    padded_lyric_token_idss = self._pad_sequences(lyric_token_idss, max_lyric_length, self.text_tokenizer.pad_token_id)
    padded_lyric_attention_masks = self._pad_sequences(lyric_attention_masks, max_lyric_length, 0)

    padded_non_cover_text_input_ids = None
    padded_non_cover_text_attention_masks = None
    if audio_cover_strength < 1.0:
        non_cover_text_input_ids = []
        non_cover_text_attention_masks = []
        for i in range(batch_size):
            text_prompt = SFT_GEN_PROMPT.format(
                self._format_instruction(DEFAULT_DIT_INSTRUCTION), actual_captions[i], parsed_metas[i]
            )
            text_inputs_dict = self.text_tokenizer(
                text_prompt,
                padding="longest",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            non_cover_text_input_ids.append(text_inputs_dict.input_ids[0])
            non_cover_text_attention_masks.append(text_inputs_dict.attention_mask[0].bool())
        padded_non_cover_text_input_ids = self._pad_sequences(
            non_cover_text_input_ids, max_text_length, self.text_tokenizer.pad_token_id
        )
        padded_non_cover_text_attention_masks = self._pad_sequences(non_cover_text_attention_masks, max_text_length, 0)

    return (
        text_inputs,
        padded_text_token_idss,
        padded_text_attention_masks,
        padded_lyric_token_idss,
        padded_lyric_attention_masks,
        padded_non_cover_text_input_ids,
        padded_non_cover_text_attention_masks,
    )

# --- From conditioning_target.py ---

def _prepare_target_latents_and_wavs(
    self,
    batch_size: int,
    target_wavs: torch.Tensor,
    audio_code_hints: list[str | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Encode target audio/codes to latents and pad batch tensors."""
    self._ensure_silence_latent_on_device()

    with torch.inference_mode():
        target_latents_list = []
        latent_lengths = []
        target_wavs_list = [target_wavs[i].clone() for i in range(batch_size)]
        if target_wavs.device != self.device:
            target_wavs = target_wavs.to(self.device)

        with self._load_model_context("vae"):
            _cached_wav_ref: torch.Tensor | None = None
            _cached_latent: torch.Tensor | None = None

            for i in range(batch_size):
                code_hint = audio_code_hints[i]
                if code_hint:
                    logger.info(f"[generate_music] Decoding audio codes for item {i}...")
                    decoded_latents = self._decode_audio_codes_to_latents(code_hint)
                    if decoded_latents is not None:
                        decoded_latents = decoded_latents.squeeze(0)
                        target_latents_list.append(decoded_latents)
                        latent_lengths.append(decoded_latents.shape[0])
                        frames_from_codes = max(1, int(decoded_latents.shape[0] * 1920))
                        target_wavs_list[i] = torch.zeros(2, frames_from_codes)
                        continue

                current_wav = target_wavs_list[i].to(self.device).unsqueeze(0)
                if self.is_silence(current_wav):
                    expected_latent_length = current_wav.shape[-1] // 1920
                    target_latent = self.silence_latent[0, :expected_latent_length, :]
                else:
                    if (
                        _cached_wav_ref is not None
                        and _cached_latent is not None
                        and _cached_wav_ref.shape == current_wav.shape
                        and torch.equal(_cached_wav_ref, current_wav)
                    ):
                        logger.info(
                            f"[generate_music] Reusing cached VAE latents for item {i} (same audio as previous item)"
                        )
                        target_latent = _cached_latent.clone()
                    else:
                        logger.info(f"[generate_music] Encoding target audio to latents for item {i}...")
                        target_latent = self._encode_audio_to_latents(current_wav.squeeze(0))
                        _cached_wav_ref = current_wav
                        _cached_latent = target_latent
                target_latents_list.append(target_latent)
                latent_lengths.append(target_latent.shape[0])

        max_target_frames = max(wav.shape[-1] for wav in target_wavs_list)
        padded_target_wavs = []
        for wav in target_wavs_list:
            if wav.shape[-1] < max_target_frames:
                pad_frames = max_target_frames - wav.shape[-1]
                wav = torch.nn.functional.pad(wav, (0, pad_frames), "constant", 0)
            padded_target_wavs.append(wav)
        target_wavs = torch.stack(padded_target_wavs)

        max_latent_length = max(latent.shape[0] for latent in target_latents_list)
        max_latent_length = max(128, max_latent_length)
        silence_latent_tiled = self.silence_latent[0, :max_latent_length, :]

        padded_latents = []
        for latent in target_latents_list:
            latent_length = latent.shape[0]
            if latent_length < max_latent_length:
                pad_length = max_latent_length - latent_length
                latent = torch.cat([latent, self.silence_latent[0, :pad_length, :]], dim=0)
            padded_latents.append(latent)

        target_latents = torch.stack(padded_latents)
        latent_masks = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(l, dtype=torch.long, device=self.device),
                        torch.zeros(max_latent_length - l, dtype=torch.long, device=self.device),
                    ]
                )
                for l in latent_lengths
            ]
        )
        return target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled

# --- From batch_prep.py ---

def _normalize_audio_code_hints(
    self, audio_code_hints: str | list[str] | None, batch_size: int
) -> list[str | None]:
    """Normalize ``audio_code_hints`` into a batch-length list."""
    if audio_code_hints is None:
        normalized: list[str | None] = [None] * batch_size
    elif isinstance(audio_code_hints, str):
        normalized = [audio_code_hints] * batch_size
    elif len(audio_code_hints) == 1 and batch_size > 1:
        normalized = audio_code_hints * batch_size
    elif len(audio_code_hints) != batch_size:
        normalized = list(audio_code_hints[:batch_size])
        while len(normalized) < batch_size:
            normalized.append(None)
    else:
        normalized = list(audio_code_hints)
    return [hint if isinstance(hint, str) and hint.strip() else None for hint in normalized]

def _normalize_instructions(
    self,
    instructions: str | list[str] | None,
    batch_size: int,
    default: str | None = None,
) -> list[str]:
    """Normalize instructions into a batch-length list."""
    if instructions is None:
        default_instruction = default or DEFAULT_DIT_INSTRUCTION
        return [default_instruction] * batch_size
    if isinstance(instructions, str):
        return [instructions] * batch_size
    if len(instructions) == 1:
        return instructions * batch_size
    if len(instructions) != batch_size:
        normalized = list(instructions[:batch_size])
        default_instruction = default or DEFAULT_DIT_INSTRUCTION
        while len(normalized) < batch_size:
            normalized.append(default_instruction)
        return normalized
    return list(instructions)

def _create_fallback_vocal_languages(self, batch_size: int) -> list[str]:
    """Create default vocal-language values for missing inputs."""
    return ["en"] * batch_size

def _encode_audio_to_latents(self, audio: torch.Tensor) -> torch.Tensor:
    """Encode audio to latents using tiled VAE encode path."""
    input_was_2d = audio.dim() == 2
    if input_was_2d:
        audio = audio.unsqueeze(0)

    with torch.inference_mode():
        latents = self.tiled_encode(audio, offload_latent_to_cpu=True)

    latents = latents.to(self.device).to(self.dtype)
    latents = latents.transpose(1, 2)
    if input_was_2d:
        latents = latents.squeeze(0)
    return latents

def prepare_batch_data(
    self,
    actual_batch_size,
    processed_src_audio,
    audio_duration,
    captions,
    lyrics,
    vocal_language,
    instruction,
    bpm,
    key_scale,
    time_signature,
):
    """Prepare repeated batch-level caption/instruction/metadata values."""
    pure_caption = self.extract_caption_from_sft_format(captions)
    captions_batch = [pure_caption] * actual_batch_size
    instructions_batch = [instruction] * actual_batch_size
    lyrics_batch = [lyrics] * actual_batch_size
    vocal_languages_batch = [vocal_language] * actual_batch_size

    calculated_duration = None
    if processed_src_audio is not None:
        calculated_duration = processed_src_audio.shape[-1] / SAMPLE_RATE
    elif audio_duration is not None and float(audio_duration) > 0:
        calculated_duration = float(audio_duration)

    metadata_dict: dict[str, str | int] = self._build_metadata_dict(
        bpm, key_scale, time_signature, calculated_duration
    )
    metas_batch = [metadata_dict.copy() for _ in range(actual_batch_size)]
    return captions_batch, instructions_batch, lyrics_batch, vocal_languages_batch, metas_batch

# --- From audio_codes.py ---

def _parse_audio_code_string(self, code_str: str) -> list[int]:
    """Extract integer audio codes from tokens like ``<|audio_code_123|>``."""
    if not code_str:
        return []
    try:
        max_audio_code = 63999
        codes = []
        clamped_count = 0
        for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str):
            code_value = int(x)
            clamped_value = max(0, min(code_value, max_audio_code))
            if clamped_value != code_value:
                clamped_count += 1
                logger.warning(
                    f"[_parse_audio_code_string] Clamped audio code value from {code_value} to {clamped_value}"
                )
            codes.append(clamped_value)
        if clamped_count > 0:
            logger.warning(
                f"[_parse_audio_code_string] Clamped {clamped_count} audio code value(s) "
                f"to valid range [0, {max_audio_code}]"
            )
        return codes
    except Exception as e:
        logger.debug(f"[_parse_audio_code_string] Failed to parse audio code string: {e}")
        return []

def _decode_audio_codes_to_latents(self, code_str: str) -> torch.Tensor | None:
    """Convert serialized audio-code string into 25Hz latents."""
    if self.model is None or not hasattr(self.model, "tokenizer") or not hasattr(self.model, "detokenizer"):
        return None

    code_ids = self._parse_audio_code_string(code_str)
    if len(code_ids) == 0:
        return None

    with self._load_model_context("model"):
        quantizer = self.model.tokenizer.quantizer
        detokenizer = self.model.detokenizer
        indices = torch.tensor(code_ids, device=self.device, dtype=torch.long)
        indices = indices.unsqueeze(0).unsqueeze(-1)

        quantized = quantizer.get_output_from_indices(indices)
        if quantized.dtype != self.dtype:
            quantized = quantized.to(self.dtype)
        lm_hints_25hz = detokenizer(quantized)
        return lm_hints_25hz

def convert_src_audio_to_codes(self, audio_file) -> str:
    """Convert uploaded source audio into serialized audio code tokens."""
    if audio_file is None:
        return "❌ Please upload source audio first"
    if self.model is None or self.vae is None:
        return "❌ Model not initialized. Please initialize the service first."

    try:
        processed_audio = self.process_src_audio(audio_file)
        if processed_audio is None:
            return "❌ Failed to process audio file"

        with torch.inference_mode():
            with self._load_model_context("vae"):
                if self.is_silence(processed_audio.unsqueeze(0)):
                    return "❌ Audio file appears to be silent"
                latents = self._encode_audio_to_latents(processed_audio)

            attention_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=self.device)
            with self._load_model_context("model"):
                hidden_states = latents.unsqueeze(0)
                _, indices, _ = self.model.tokenize(
                    hidden_states, self.silence_latent, attention_mask.unsqueeze(0)
                )
                indices_flat = indices.flatten().cpu().tolist()
                codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])
                logger.info(f"[convert_src_audio_to_codes] Generated {len(indices_flat)} audio codes")
                return codes_string
    except Exception as e:
        error_msg = f"❌ Error converting audio to codes: {str(e)}\n{traceback.format_exc()}"
        logger.exception("[convert_src_audio_to_codes] Error converting audio to codes")
        return error_msg
