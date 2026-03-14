"""PyTorch backend for 5Hz LM generation."""

import traceback
from typing import Any

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation.streamers import BaseStreamer

from acestep.models.lm.constrained_logits_processor import MetadataConstrainedLogitsProcessor

__all__ = [
    "_build_logits_processor",
    "_load_pytorch_model",
    "_apply_top_k_filter",
    "_apply_top_p_filter",
    "_sample_tokens",
    "_check_eos_token",
    "_update_constrained_processor_state",
    "_forward_pass",
    "_run_pt_single",
    "_run_pt",
    "_generate_with_constrained_decoding",
    "_generate_with_cfg_custom",
]


def _build_logits_processor(self, repetition_penalty: float) -> LogitsProcessorList:
    """Build logits processor list with repetition penalty if needed"""
    logits_processor = LogitsProcessorList()
    if repetition_penalty != 1.0:
        logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    return logits_processor


def _load_pytorch_model(self, model_path: str, device: str) -> tuple[bool, str]:
    """Load PyTorch model from path and return (success, status_message)"""
    try:
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if not self.offload_to_cpu:
            self.llm = self.llm.to(device).to(self.dtype)
        else:
            self.llm = self.llm.to("cpu").to(self.dtype)
        self.llm.eval()
        self.llm_backend = "pt"
        self.llm_initialized = True
        logger.info(f"5Hz LM initialized successfully using PyTorch backend on {device}")
        status_msg = f"✅ 5Hz LM initialized successfully\nModel: {model_path}\nBackend: PyTorch\nDevice: {device}"
        return True, status_msg
    except Exception as e:
        return False, f"❌ Error initializing 5Hz LM: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


def _apply_top_k_filter(self, logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Apply top-k filtering to logits"""
    if top_k is not None and top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    return logits


def _apply_top_p_filter(self, logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits"""
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Upcast to float32 for stable softmax/cumsum (critical for float16/MPS)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float(), dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    return logits


def _sample_tokens(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample tokens from logits with temperature.

    Upcasts to float32 for numerical stability (float16 logits can overflow
    during softmax, especially after CFG scaling).
    """
    if temperature > 0:
        # Upcast to float32 for stable softmax (critical for float16/MPS)
        logits = logits.float() / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        return torch.argmax(logits, dim=-1)


def _check_eos_token(self, tokens: torch.Tensor, eos_token_id: int, pad_token_id: int | None) -> bool:
    """Check if any token in the batch is EOS or pad token"""
    if torch.any(tokens == eos_token_id):
        return True
    if pad_token_id is not None and pad_token_id != eos_token_id:
        if torch.any(tokens == pad_token_id):
            return True
    return False


def _update_constrained_processor_state(self, constrained_processor: MetadataConstrainedLogitsProcessor | None, tokens: torch.Tensor):
    """Update constrained processor state with generated tokens"""
    if constrained_processor is not None:
        for b in range(tokens.shape[0]):
            constrained_processor.update_state(tokens[b].item())


def _forward_pass(
    self,
    model: Any,
    generated_ids: torch.Tensor,
    model_kwargs: dict[str, Any],
    past_key_values: Any | None,
    use_cache: bool,
) -> Any:
    """Perform forward pass with KV cache support"""
    if past_key_values is None:
        outputs = model(
            input_ids=generated_ids,
            **model_kwargs,
            use_cache=use_cache,
        )
    else:
        outputs = model(
            input_ids=generated_ids[:, -1:],
            past_key_values=past_key_values,
            **model_kwargs,
            use_cache=use_cache,
        )
    return outputs


def _run_pt_single(
    self,
    formatted_prompt: str,
    temperature: float,
    cfg_scale: float,
    negative_prompt: str,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    use_constrained_decoding: bool,
    constrained_decoding_debug: bool,
    target_duration: float | None,
    user_metadata: dict[str, str | None] | None,
    stop_at_reasoning: bool,
    skip_genres: bool,
    skip_caption: bool,
    skip_language: bool,
    generation_phase: str,
    caption: str,
    lyrics: str,
    cot_text: str,
) -> str:
    """Internal helper function for single-item PyTorch generation."""
    inputs = self.llm_tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )

    # Setup constrained processor
    constrained_processor = self._setup_constrained_processor(
        use_constrained_decoding=use_constrained_decoding,
        constrained_decoding_debug=constrained_decoding_debug,
        target_duration=target_duration,
        user_metadata=user_metadata,
        stop_at_reasoning=stop_at_reasoning,
        skip_genres=skip_genres,
        skip_caption=skip_caption,
        skip_language=skip_language,
        generation_phase=generation_phase,
        is_batch=False,
    )

    with self._load_model_context():
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Calculate max_new_tokens based on target_duration and generation phase
        max_new_tokens = self._compute_max_new_tokens(
            target_duration=target_duration,
            generation_phase=generation_phase,
            fallback_max=getattr(self.llm.config, "max_new_tokens", 4096),
        )

        # Build logits processor list (only for CFG and repetition penalty)
        logits_processor = self._build_logits_processor(repetition_penalty)

        if cfg_scale > 1.0:
            # Build unconditional prompt based on generation phase
            formatted_unconditional_prompt = self._build_unconditional_prompt(
                caption=caption,
                lyrics=lyrics,
                cot_text=cot_text,
                negative_prompt=negative_prompt,
                generation_phase=generation_phase,
                is_batch=False,
            )

            # Tokenize both prompts together to ensure same length (with left padding)
            # Left padding is important for generation tasks
            batch_texts = [formatted_prompt, formatted_unconditional_prompt]
            original_padding_side = self.llm_tokenizer.padding_side
            self.llm_tokenizer.padding_side = 'left'
            batch_inputs_tokenized = self.llm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            self.llm_tokenizer.padding_side = original_padding_side
            batch_inputs_tokenized = {k: v.to(self.device) for k, v in batch_inputs_tokenized.items()}

            # Extract batch inputs
            batch_input_ids = batch_inputs_tokenized['input_ids']
            batch_attention_mask = batch_inputs_tokenized.get('attention_mask', None)

            # Use custom CFG generation loop with constrained decoding
            outputs = self._generate_with_cfg_custom(
                batch_input_ids=batch_input_ids,
                batch_attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_scale=cfg_scale,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                streamer=None,
                constrained_processor=constrained_processor,
            )

            # Extract only the conditional output (first in batch)
            outputs = outputs[0:1]  # Keep only conditional output
        elif use_constrained_decoding:
            # Use custom constrained decoding loop for non-CFG
            outputs = self._generate_with_constrained_decoding(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                streamer=None,
                constrained_processor=constrained_processor,
            )
        else:
            # Generate without CFG using native generate() parameters
            with torch.inference_mode():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=True if temperature > 0 else False,
                    top_k=top_k if top_k is not None and top_k > 0 else None,
                    top_p=top_p if top_p is not None and 0.0 < top_p < 1.0 else None,
                    logits_processor=logits_processor if len(logits_processor) > 0 else None,
                    pad_token_id=self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
                    streamer=None,
                )

    # Decode the generated tokens
    # outputs is a tensor with shape [batch_size, seq_len], extract first sequence
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            generated_ids = outputs[0]
        else:
            generated_ids = outputs
    else:
        generated_ids = outputs[0]

    # Only decode the newly generated tokens (skip the input prompt)
    # Use the original input length (before batch processing for CFG)
    if cfg_scale > 1.0:
        # In CFG case, we need to use the conditional input length from batch_inputs_tokenized
        # Both sequences have the same length due to padding
        input_length = batch_inputs_tokenized['input_ids'].shape[1]
    else:
        input_length = inputs["input_ids"].shape[1]

    generated_ids = generated_ids[input_length:]

    # Move to CPU for decoding (tokenizer needs CPU tensors)
    if generated_ids.device.type != "cpu":
        generated_ids = generated_ids.cpu()

    output_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=False)
    return output_text


def _run_pt(
    self,
    formatted_prompts: str | list[str],
    temperature: float,
    cfg_scale: float,
    negative_prompt: str,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    use_constrained_decoding: bool = True,
    constrained_decoding_debug: bool = False,
    target_duration: float | None = None,
    user_metadata: dict[str, str | None] | None = None,
    stop_at_reasoning: bool = False,
    skip_genres: bool = True,
    skip_caption: bool = False,
    skip_language: bool = False,
    generation_phase: str = "cot",
    caption: str = "",
    lyrics: str = "",
    cot_text: str = "",
    seeds: list[int] | None = None,
) -> str | list[str]:
    """
    Unified PyTorch generation function supporting both single and batch modes.
    Accepts either a single formatted prompt (str) or a list of formatted prompts (List[str]).
    Returns a single string for single mode, or a list of strings for batch mode.
    Note: PyTorch backend processes batch items sequentially (doesn't support true batching efficiently).
    """
    # Determine if batch mode
    formatted_prompt_list, is_batch = self._normalize_batch_input(formatted_prompts)

    # For batch mode, process each item sequentially with different seeds.
    # Wrap the entire loop in a single _load_model_context() so the model
    # loads to GPU once and offloads once, instead of per-item.
    if is_batch:
        output_texts = []

        with self._load_model_context():
            for i, formatted_prompt in enumerate(formatted_prompt_list):
                # Set seed for this item if provided
                if seeds and i < len(seeds):
                    torch.manual_seed(seeds[i])
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seeds[i])
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        torch.mps.manual_seed(seeds[i])

                # Generate using single-item method with batch-mode defaults
                output_text = self._run_pt_single(
                    formatted_prompt=formatted_prompt,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    negative_prompt=negative_prompt,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_constrained_decoding=use_constrained_decoding,
                    constrained_decoding_debug=constrained_decoding_debug,
                    target_duration=target_duration,
                    user_metadata=None,
                    stop_at_reasoning=False,
                    skip_genres=True,
                    skip_caption=True,
                    skip_language=True,
                    generation_phase=generation_phase,
                    caption=caption,
                    lyrics=lyrics,
                    cot_text=cot_text,
                )

                output_texts.append(output_text)

        return output_texts

    # Single mode: process the formatted prompt
    formatted_prompt = formatted_prompt_list[0]

    return self._run_pt_single(
        formatted_prompt=formatted_prompt,
        temperature=temperature,
        cfg_scale=cfg_scale,
        negative_prompt=negative_prompt,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_constrained_decoding=use_constrained_decoding,
        constrained_decoding_debug=constrained_decoding_debug,
        target_duration=target_duration,
        user_metadata=user_metadata,
        stop_at_reasoning=stop_at_reasoning,
        skip_genres=skip_genres,
        skip_caption=skip_caption,
        skip_language=skip_language,
        generation_phase=generation_phase,
        caption=caption,
        lyrics=lyrics,
        cot_text=cot_text,
    )


def _generate_with_constrained_decoding(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    pad_token_id: int,
    streamer: BaseStreamer | None,
    constrained_processor: MetadataConstrainedLogitsProcessor | None = None,
) -> torch.Tensor:
    """
    Custom generation loop with constrained decoding support (non-CFG).
    This allows us to call update_state() after each token generation.
    """
    model = self.llm
    device = self.device

    # Initialize generated sequences
    generated_ids = input_ids.clone()
    if attention_mask is not None:
        attn_mask = attention_mask.clone()
    else:
        attn_mask = torch.ones_like(input_ids)

    # Prepare model inputs
    model_kwargs = {'attention_mask': attn_mask}

    # Past key values for KV cache
    past_key_values = None
    use_cache = hasattr(model, 'generation_config') and getattr(model.generation_config, 'use_cache', True)

    # Get EOS token ID
    eos_token_id = self.llm_tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = pad_token_id

    # Build logits processor for repetition penalty
    logits_processor = self._build_logits_processor(repetition_penalty)

    with torch.inference_mode():
        for _step in tqdm(range(max_new_tokens), desc="LLM Constrained Decoding", unit="token", disable=self.disable_tqdm):
            # Forward pass
            outputs = self._forward_pass(model, generated_ids, model_kwargs, past_key_values, use_cache)

            # Get logits for the last position
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Apply constrained processor FIRST (modifies logits based on FSM state)
            if constrained_processor is not None:
                next_token_logits = constrained_processor(generated_ids, next_token_logits)

            # Apply other logits processors (repetition penalty)
            for processor in logits_processor:
                next_token_logits = processor(generated_ids, next_token_logits)

            # Apply top-k and top-p filtering
            next_token_logits = self._apply_top_k_filter(next_token_logits, top_k)
            next_token_logits = self._apply_top_p_filter(next_token_logits, top_p)

            # Apply temperature and sample
            next_tokens = self._sample_tokens(next_token_logits, temperature)

            # Update constrained processor state
            self._update_constrained_processor_state(constrained_processor, next_tokens)

            # Check for EOS token
            should_stop = self._check_eos_token(next_tokens, eos_token_id, pad_token_id)

            # Append token to sequence
            next_tokens_unsqueezed = next_tokens.unsqueeze(1)
            generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones((input_ids.shape[0], 1), device=device, dtype=attn_mask.dtype)], dim=1)
            model_kwargs['attention_mask'] = attn_mask

            # Update KV cache
            if use_cache and hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values

            # Update streamer
            if streamer is not None:
                streamer.put(next_tokens_unsqueezed)

            if should_stop:
                break

    if streamer is not None:
        streamer.end()

    return generated_ids


def _generate_with_cfg_custom(
    self,
    batch_input_ids: torch.Tensor,
    batch_attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    temperature: float,
    cfg_scale: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    pad_token_id: int,
    streamer: BaseStreamer | None,
    constrained_processor: MetadataConstrainedLogitsProcessor | None = None,
) -> torch.Tensor:
    """
    Custom CFG generation loop that:
    1. Processes both conditional and unconditional sequences in parallel
    2. Applies CFG formula to logits
    3. Samples tokens only for conditional sequences
    4. Applies the same sampled tokens to both conditional and unconditional sequences
    5. Optionally applies constrained decoding via FSM-based logits processor

    Batch format: [cond_input, uncond_input]
    """
    model = self.llm
    device = self.device
    batch_size = batch_input_ids.shape[0] // 2  # Half are conditional, half are unconditional
    cond_start_idx = 0
    uncond_start_idx = batch_size

    # Initialize generated sequences
    generated_ids = batch_input_ids.clone()
    if batch_attention_mask is not None:
        attention_mask = batch_attention_mask.clone()
    else:
        attention_mask = torch.ones_like(batch_input_ids)

    # Prepare model inputs
    model_kwargs = {}
    if batch_attention_mask is not None:
        model_kwargs['attention_mask'] = attention_mask

    # Past key values for KV cache (if model supports it)
    past_key_values = None
    use_cache = hasattr(model, 'generation_config') and getattr(model.generation_config, 'use_cache', True)

    # Get EOS token ID for stopping condition
    eos_token_id = self.llm_tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = pad_token_id

    # Build logits processor for non-CFG operations (repetition penalty, top_k, top_p)
    logits_processor = self._build_logits_processor(repetition_penalty)

    with torch.inference_mode():
        for _step in tqdm(range(max_new_tokens), desc="LLM CFG Generation", unit="token", disable=self.disable_tqdm):
            # Forward pass for the entire batch (conditional + unconditional)
            outputs = self._forward_pass(model, generated_ids, model_kwargs, past_key_values, use_cache)

            # Get logits for the last position
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size*2, vocab_size]

            # Split conditional and unconditional logits
            cond_logits = next_token_logits[cond_start_idx:cond_start_idx+batch_size]
            uncond_logits = next_token_logits[uncond_start_idx:uncond_start_idx+batch_size]

            # Apply CFG formula: cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            # Upcast to float32 to prevent overflow in float16 (CFG scaling can exceed fp16 range)
            cfg_logits = uncond_logits.float() + cfg_scale * (cond_logits.float() - uncond_logits.float())

            # Apply constrained processor FIRST (modifies logits based on FSM state)
            if constrained_processor is not None:
                current_input_ids = generated_ids[cond_start_idx:cond_start_idx+batch_size]
                cfg_logits = constrained_processor(current_input_ids, cfg_logits)

            # Apply logits processors (repetition penalty, top-k, top-p)
            # Get current input_ids for repetition penalty (only conditional part)
            current_input_ids = generated_ids[cond_start_idx:cond_start_idx+batch_size]
            for processor in logits_processor:
                cfg_logits = processor(current_input_ids, cfg_logits)

            # Apply top-k and top-p filtering
            cfg_logits = self._apply_top_k_filter(cfg_logits, top_k)
            cfg_logits = self._apply_top_p_filter(cfg_logits, top_p)

            # Apply temperature and sample
            next_tokens = self._sample_tokens(cfg_logits, temperature)

            # Update constrained processor state AFTER sampling
            self._update_constrained_processor_state(constrained_processor, next_tokens)

            # Check for EOS token in conditional sequences BEFORE unsqueezing
            # Stop if any conditional sequence generates EOS token
            # next_tokens shape: [batch_size] (only conditional tokens)
            should_stop = self._check_eos_token(next_tokens, eos_token_id, pad_token_id)

            # Apply the same sampled tokens to both conditional and unconditional sequences
            next_tokens_unsqueezed = next_tokens.unsqueeze(1)
            generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed.repeat(2, 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size*2, 1), device=device, dtype=attention_mask.dtype)], dim=1)
            model_kwargs['attention_mask'] = attention_mask

            # Update past_key_values for next iteration
            if use_cache and hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values

            # Update streamer
            if streamer is not None:
                streamer.put(next_tokens_unsqueezed)  # Stream conditional tokens

            # Stop generation if EOS token detected
            if should_stop:
                break

    if streamer is not None:
        streamer.end()

    # Return the full batch (both conditional and unconditional)
    # The caller will extract only the conditional output
    return generated_ids
