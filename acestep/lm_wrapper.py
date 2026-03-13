"""AceStepLMWrapper — thin model wrapper for 5Hz LM generation.

All functionality preserved via method binding from extracted
modules (lm_core, lm_backends/*, lm_tasks).
"""

import sys

import torch

from acestep import lm_core, lm_tasks
from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor
from acestep.env_utils import env_is_truthy
from acestep.lm_backends import mlx as mlx_backend
from acestep.lm_backends import pt as pt_backend
from acestep.lm_backends import vllm as vllm_backend

# Module-level constants (from original LLMHandler)
VRAM_SAFE_FREE_GB = lm_core.VRAM_SAFE_FREE_GB
IS_HUGGINGFACE_SPACE = lm_core.IS_HUGGINGFACE_SPACE


class AceStepLMWrapper:
    """5Hz LM wrapper for audio code generation."""

    STOP_REASONING_TAG = lm_core.STOP_REASONING_TAG
    IS_HUGGINGFACE_SPACE = lm_core.IS_HUGGINGFACE_SPACE

    # --- Core methods (from lm_core) ---
    unload = lm_core.unload
    _cleanup_torch_distributed_state = lm_core._cleanup_torch_distributed_state
    _get_checkpoint_dir = lm_core._get_checkpoint_dir
    get_available_5hz_lm_models = lm_core.get_available_5hz_lm_models
    get_gpu_memory_utilization = lm_core.get_gpu_memory_utilization
    _compute_max_new_tokens = lm_core._compute_max_new_tokens
    _has_meaningful_negative_prompt = lm_core._has_meaningful_negative_prompt
    _setup_constrained_processor = lm_core._setup_constrained_processor
    _build_unconditional_prompt = lm_core._build_unconditional_prompt
    _normalize_batch_input = lm_core._normalize_batch_input
    initialize = lm_core.initialize
    has_all_metas = lm_core.has_all_metas
    _format_metadata_as_cot = lm_core._format_metadata_as_cot
    generate_with_stop_condition = lm_core.generate_with_stop_condition
    build_formatted_prompt = lm_core.build_formatted_prompt
    build_formatted_prompt_with_cot = lm_core.build_formatted_prompt_with_cot
    generate_from_formatted_prompt = lm_core.generate_from_formatted_prompt
    parse_lm_output = lm_core.parse_lm_output
    _load_model_context = lm_core._load_model_context

    # --- vLLM backend ---
    _initialize_5hz_lm_vllm = vllm_backend._initialize_5hz_lm_vllm
    _run_vllm = vllm_backend._run_vllm

    # --- PyTorch backend ---
    _build_logits_processor = pt_backend._build_logits_processor
    _load_pytorch_model = pt_backend._load_pytorch_model
    _apply_top_k_filter = pt_backend._apply_top_k_filter
    _apply_top_p_filter = pt_backend._apply_top_p_filter
    _sample_tokens = pt_backend._sample_tokens
    _check_eos_token = pt_backend._check_eos_token
    _update_constrained_processor_state = pt_backend._update_constrained_processor_state
    _forward_pass = pt_backend._forward_pass
    _run_pt_single = pt_backend._run_pt_single
    _run_pt = pt_backend._run_pt
    _generate_with_constrained_decoding = pt_backend._generate_with_constrained_decoding
    _generate_with_cfg_custom = pt_backend._generate_with_cfg_custom

    # --- MLX backend ---
    _is_mlx_available = mlx_backend._is_mlx_available
    _load_mlx_model = mlx_backend._load_mlx_model
    _make_mlx_cache = mlx_backend._make_mlx_cache
    _run_mlx_batch_native = mlx_backend._run_mlx_batch_native
    _run_mlx_single_native = mlx_backend._run_mlx_single_native
    _run_mlx_single = mlx_backend._run_mlx_single
    _run_mlx = mlx_backend._run_mlx

    # --- Task methods ---
    build_formatted_prompt_for_understanding = lm_tasks.build_formatted_prompt_for_understanding
    understand_audio_from_codes = lm_tasks.understand_audio_from_codes
    _extract_lyrics_from_output = lm_tasks._extract_lyrics_from_output
    build_formatted_prompt_for_inspiration = lm_tasks.build_formatted_prompt_for_inspiration
    create_sample_from_query = lm_tasks.create_sample_from_query
    build_formatted_prompt_for_format = lm_tasks.build_formatted_prompt_for_format
    format_sample_from_input = lm_tasks.format_sample_from_input
    get_hf_model_for_scoring = lm_tasks.get_hf_model_for_scoring

    def __init__(self, persistent_storage_path: str | None = None):
        """Initialize LM wrapper with default values."""
        self.llm = None
        self.llm_tokenizer = None
        self.llm_initialized = False
        self.llm_backend = None
        self.max_model_len = 4096
        self.device = "cpu"
        self.dtype = torch.float32
        self.offload_to_cpu = False
        self.disable_tqdm = (
            env_is_truthy("ACESTEP_DISABLE_TQDM")
            or not (hasattr(sys.stderr, 'isatty') and sys.stderr.isatty())
        )

        # HuggingFace Space persistent storage support
        if persistent_storage_path is None and self.IS_HUGGINGFACE_SPACE:
            persistent_storage_path = "/data"
        self.persistent_storage_path = persistent_storage_path

        # Shared constrained decoding processor
        self.constrained_processor: MetadataConstrainedLogitsProcessor | None = None

        # Shared HuggingFace model for perplexity calculation
        self._hf_model_for_scoring = None

        # MLX model reference (used when llm_backend == "mlx")
        self._mlx_model = None
        self._mlx_model_path = None


