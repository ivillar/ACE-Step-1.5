"""AceStepDiTWrapper — thin model wrapper for DiT generation.

Loads in __init__, generates in generate(). All functionality
preserved via method binding from consolidated handler modules.
"""

import os
import sys
import threading
import warnings

import torch

from acestep.constants import SAMPLE_RATE
from acestep.dit_modules import (
    codec,
    conditioning,
    generation,
    init,
    utils,
)
from acestep.env_utils import env_is_truthy

warnings.filterwarnings("ignore")


class AceStepDiTWrapper:
    """ACE-Step DiT model wrapper."""

    # --- Class-level constants (formerly on mixin classes) ---
    VAE_DECODE_MAX_CHUNK_SIZE = utils.VAE_DECODE_MAX_CHUNK_SIZE
    _MPS_DECODE_CHUNK_SIZE = codec._MPS_DECODE_CHUNK_SIZE
    _MPS_DECODE_OVERLAP = codec._MPS_DECODE_OVERLAP

    # --- init module ---
    _resolve_initialize_device = init._resolve_initialize_device
    _configure_initialize_runtime = init._configure_initialize_runtime
    _ensure_len_for_compile = init._ensure_len_for_compile
    _validate_quantization_setup = init._validate_quantization_setup
    _initialize_mlx_backends = init._initialize_mlx_backends
    _build_initialize_status_message = init._build_initialize_status_message
    _device_type = init._device_type
    get_available_checkpoints = init.get_available_checkpoints
    get_available_acestep_v15_models = init.get_available_acestep_v15_models
    is_flash_attention_available = init.is_flash_attention_available
    is_turbo_model = init.is_turbo_model
    _ensure_models_present = init._ensure_models_present
    _sync_model_code_if_needed = init._sync_model_code_if_needed
    _load_main_model_from_checkpoint = init._load_main_model_from_checkpoint
    _load_vae_model = init._load_vae_model
    _load_text_encoder_and_tokenizer = init._load_text_encoder_and_tokenizer
    initialize_service = init.initialize_service
    _empty_cache = init._empty_cache
    _synchronize = init._synchronize
    _memory_allocated = init._memory_allocated
    _max_memory_allocated = init._max_memory_allocated
    _is_on_target_device = init._is_on_target_device
    _get_affine_quantized_tensor_class = init._get_affine_quantized_tensor_class
    _is_quantized_tensor = init._is_quantized_tensor
    _has_quantized_params = init._has_quantized_params
    _ensure_silence_latent_on_device = init._ensure_silence_latent_on_device
    _move_module_recursive = init._move_module_recursive
    _move_quantized_param = init._move_quantized_param
    _recursive_to_device = init._recursive_to_device
    _load_model_context = init._load_model_context

    # --- generation module ---
    generate_music = generation.generate_music
    _resolve_generate_music_progress = generation._resolve_generate_music_progress
    _validate_generate_music_readiness = generation._validate_generate_music_readiness
    _has_non_empty_audio_codes = generation._has_non_empty_audio_codes
    _resolve_generate_music_task = generation._resolve_generate_music_task
    _prepare_generate_music_runtime = generation._prepare_generate_music_runtime
    _prepare_reference_and_source_audio = generation._prepare_reference_and_source_audio
    _prepare_generate_music_service_inputs = generation._prepare_generate_music_service_inputs
    _run_generate_music_service_with_progress = generation._run_generate_music_service_with_progress
    _prepare_generate_music_decode_state = generation._prepare_generate_music_decode_state
    _decode_generate_music_pred_latents = generation._decode_generate_music_pred_latents
    _build_generate_music_success_payload = generation._build_generate_music_success_payload
    service_generate = generation.service_generate
    _build_service_seed_list = generation._build_service_seed_list
    _normalize_service_generate_inputs = generation._normalize_service_generate_inputs
    _unpack_service_processed_data = generation._unpack_service_processed_data
    _resolve_service_seed_param = generation._resolve_service_seed_param
    _build_service_generate_kwargs = generation._build_service_generate_kwargs
    _execute_service_generate_diffusion = generation._execute_service_generate_diffusion
    _attach_service_generate_outputs = generation._attach_service_generate_outputs
    _mlx_run_diffusion = generation._mlx_run_diffusion

    # --- conditioning module ---
    _prepare_batch = conditioning._prepare_batch
    infer_refer_latent = conditioning.infer_refer_latent
    infer_text_embeddings = conditioning.infer_text_embeddings
    infer_lyric_embeddings = conditioning.infer_lyric_embeddings
    preprocess_batch = conditioning.preprocess_batch
    _build_chunk_masks_and_src_latents = conditioning._build_chunk_masks_and_src_latents
    _prepare_precomputed_lm_hints = conditioning._prepare_precomputed_lm_hints
    _prepare_text_conditioning_inputs = conditioning._prepare_text_conditioning_inputs
    _prepare_target_latents_and_wavs = conditioning._prepare_target_latents_and_wavs
    _normalize_audio_code_hints = conditioning._normalize_audio_code_hints
    _normalize_instructions = conditioning._normalize_instructions
    _create_fallback_vocal_languages = conditioning._create_fallback_vocal_languages
    _encode_audio_to_latents = conditioning._encode_audio_to_latents
    prepare_batch_data = conditioning.prepare_batch_data
    _parse_audio_code_string = conditioning._parse_audio_code_string
    _decode_audio_codes_to_latents = conditioning._decode_audio_codes_to_latents
    convert_src_audio_to_codes = conditioning.convert_src_audio_to_codes

    # --- codec module ---
    tiled_encode = codec.tiled_encode
    _tiled_encode_gpu = codec._tiled_encode_gpu
    _tiled_encode_offload_cpu = codec._tiled_encode_offload_cpu
    tiled_decode = codec.tiled_decode
    _tiled_decode_cpu_fallback = codec._tiled_decode_cpu_fallback
    _decode_on_cpu = codec._decode_on_cpu
    _tiled_decode_inner = codec._tiled_decode_inner
    _tiled_decode_gpu = codec._tiled_decode_gpu
    _tiled_decode_offload_cpu = codec._tiled_decode_offload_cpu
    _init_mlx_dit = codec._init_mlx_dit
    _init_mlx_vae = codec._init_mlx_vae
    _resolve_mlx_encode_fn = codec._resolve_mlx_encode_fn
    _mlx_vae_encode_sample = codec._mlx_vae_encode_sample
    _mlx_encode_single = codec._mlx_encode_single
    _resolve_mlx_decode_fn = codec._resolve_mlx_decode_fn
    _mlx_vae_decode = codec._mlx_vae_decode
    _mlx_decode_single = codec._mlx_decode_single
    _normalize_audio_to_stereo_48k = codec._normalize_audio_to_stereo_48k
    process_target_audio = codec.process_target_audio
    process_reference_audio = codec.process_reference_audio
    process_src_audio = codec.process_src_audio

    # --- utils module ---
    prepare_padding_info = utils.prepare_padding_info
    is_silence = utils.is_silence
    _get_system_memory_gb = utils._get_system_memory_gb
    _get_effective_mps_memory_gb = utils._get_effective_mps_memory_gb
    _get_auto_decode_chunk_size = utils._get_auto_decode_chunk_size
    _should_offload_wav_to_cpu = utils._should_offload_wav_to_cpu
    _vram_guard_reduce_batch = utils._vram_guard_reduce_batch
    _get_vae_dtype = utils._get_vae_dtype
    _create_default_meta = utils._create_default_meta
    _dict_to_meta_string = utils._dict_to_meta_string
    _parse_metas = utils._parse_metas
    prepare_metadata = utils.prepare_metadata
    _build_metadata_dict = utils._build_metadata_dict
    _get_project_root = utils._get_project_root
    _load_progress_estimates = utils._load_progress_estimates
    _save_progress_estimates = utils._save_progress_estimates
    _duration_bucket = utils._duration_bucket
    _update_progress_estimate = utils._update_progress_estimate
    _estimate_diffusion_per_step = utils._estimate_diffusion_per_step
    _start_diffusion_progress_estimator = utils._start_diffusion_progress_estimator
    _format_instruction = utils._format_instruction
    _format_lyrics = utils._format_lyrics
    _pad_sequences = utils._pad_sequences
    extract_caption_from_sft_format = utils.extract_caption_from_sft_format
    build_dit_inputs = utils.build_dit_inputs
    _get_text_hidden_states = utils._get_text_hidden_states
    _extract_caption_and_language = utils._extract_caption_and_language
    prepare_seeds = utils.prepare_seeds
    generate_instruction = utils.generate_instruction
    determine_task_type = utils.determine_task_type
    create_target_wavs = utils.create_target_wavs
    switch_to_training_preset = utils.switch_to_training_preset
    _resolve_custom_layers_config = utils._resolve_custom_layers_config
    _move_alignment_inputs_to_runtime = utils._move_alignment_inputs_to_runtime
    _sample_noise_like = utils._sample_noise_like
    _extract_lyric_segment = utils._extract_lyric_segment
    _lyric_timestamp_error = utils._lyric_timestamp_error
    _lyric_score_error = utils._lyric_score_error
    get_lyric_score = utils.get_lyric_score
    _calculate_single_lyric_score = utils._calculate_single_lyric_score
    get_lyric_timestamp = utils.get_lyric_timestamp

    # --- LoRA (bound from lora sub-package via utils module) ---
    _ensure_lora_registry = utils._ensure_lora_registry
    _sync_lora_state_from_service = utils._sync_lora_state_from_service
    _debug_lora_registry_snapshot = utils._debug_lora_registry_snapshot
    _collect_adapter_names = utils._collect_adapter_names
    _rebuild_lora_registry = utils._rebuild_lora_registry
    _apply_scale_to_adapter = utils._apply_scale_to_adapter
    add_lora = utils.add_lora
    add_voice_lora = utils.add_voice_lora
    load_lora = utils.load_lora
    remove_lora = utils.remove_lora
    unload_lora = utils.unload_lora
    set_use_lora = utils.set_use_lora
    set_lora_scale = utils.set_lora_scale
    set_active_lora_adapter = utils.set_active_lora_adapter
    get_lora_status = utils.get_lora_status

    def __init__(self):
        """Initialize runtime model handles, feature flags, and generation state."""
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32

        # VAE for audio encoding/decoding
        self.vae = None

        # Text encoder and tokenizer
        self.text_encoder = None
        self.text_tokenizer = None

        # Silence latent for initialization
        self.silence_latent = None

        # Sample rate
        self.sample_rate = SAMPLE_RATE

        # Reward model (temporarily disabled)
        self.reward_model = None

        # Batch size
        self.batch_size = 2

        # Custom layers config
        self.custom_layers_config = {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
        self.offload_to_cpu = False
        self.offload_dit_to_cpu = False
        self.compiled = False
        self.current_offload_cost = 0.0
        self.disable_tqdm = (
            env_is_truthy("ACESTEP_DISABLE_TQDM")
            or not getattr(sys.stderr, 'isatty', lambda: False)()
        )
        self.debug_stats = env_is_truthy("ACESTEP_DEBUG_STATS")
        self._last_diffusion_per_step_sec: float | None = None
        self._progress_estimates_lock = threading.Lock()
        self._progress_estimates = {"records": []}
        self._progress_estimates_path = os.path.join(
            self._get_project_root(),
            ".cache",
            "acestep",
            "progress_estimates.json",
        )
        self._load_progress_estimates()
        self.last_init_params = None

        # Quantization state
        self.quantization = None

        # LoRA state
        self.lora_loaded = False
        self.use_lora = False
        self.lora_scale = 1.0
        self._base_decoder = None
        self._active_loras = {}
        self._lora_adapter_registry = {}
        self._lora_active_adapter = None

        # MLX DiT acceleration (macOS Apple Silicon only)
        self.mlx_decoder = None
        self.use_mlx_dit = False
        self.mlx_dit_compiled = False

        # MLX VAE acceleration (macOS Apple Silicon only)
        self.mlx_vae = None
        self.use_mlx_vae = False


