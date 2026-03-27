"""Unified LLM backend abstraction.

Provides:
- Factory ``select_backend()`` with fallback stub (``VRM_BACKEND_ALLOW_STUB=1``).
- ``HuggingFaceBackend``: real model loading, VRAM-proportional multi-GPU split
  via ``model_splitter``, sequential block inference with ``TransferManager``.
  Supports KV-cache for efficient multi-GPU auto-regressive generation.
- ``vLLMBackend`` / ``OllamaBackend``: real integration if lib present, stub otherwise.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import os
import logging
logger = logging.getLogger(__name__)

from core.logger import LoggerAdapter

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# Defensive torch import (optional dependency)
try:
    import torch as _torch  # noqa: F401
    import torch.nn as _nn  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore
    _nn = None  # type: ignore
    _HAS_TORCH = False

# Fused Triton sampling (optional — falls back to PyTorch)
try:
    from core.triton_sampling import fused_sample as _fused_sample, has_triton
    _HAS_FUSED_SAMPLE = True
except ImportError:
    _fused_sample = None
    _HAS_FUSED_SAMPLE = False

# ---------------------------------------------------------------------------
# Early patch: transformers.modeling_utils.no_init_weights
# auto_gptq 0.7.x imports this at module level; it was removed in
# transformers >= 5.x.  Patch BEFORE any auto_gptq import happens.
# ---------------------------------------------------------------------------
try:
    import contextlib as _contextlib
    import transformers.modeling_utils as _tmu
    if not hasattr(_tmu, 'no_init_weights'):
        @_contextlib.contextmanager
        def _no_init_weights(_enable=True):
            yield
        _tmu.no_init_weights = _no_init_weights
except Exception:
    pass


# ---------------------------------------------------------------------------
# Model component extraction patterns (for KV-cache multi-GPU)
# ---------------------------------------------------------------------------
_EMBED_PATTERNS = [
    ["transformer", "wte"],         # GPT-2, GPT-J
    ["model", "embed_tokens"],      # Llama, Mistral, Falcon
    ["embed_tokens"],               # Some variants
]
_POS_EMBED_PATTERNS = [
    ["transformer", "wpe"],         # GPT-2
]
_FINAL_NORM_PATTERNS = [
    ["transformer", "ln_f"],        # GPT-2
    ["model", "norm"],              # Llama, Mistral
    ["norm"],                       # Some variants
    ["transformer", "norm"],        # Falcon
]
_LM_HEAD_PATTERNS = [
    ["lm_head"],                    # Most causal LM models
]
_DROP_PATTERNS = [
    ["transformer", "drop"],        # GPT-2 dropout
]
_ROTARY_EMBED_PATTERNS = [
    ["model", "rotary_emb"],        # Llama, Mistral (transformers >= 4.46)
    ["rotary_emb"],                 # Some variants
]


def _get_submodule(model: Any, path: list) -> Optional[Any]:
    """Traverse a model to find a submodule by attribute path."""
    obj = model
    try:
        for attr in path:
            obj = getattr(obj, attr)
        return obj
    except (AttributeError, TypeError):
        return None


def _extract_model_components(model: Any) -> dict:
    """Extract embedding, final norm, and lm_head from a HuggingFace model.

    Returns a dict with keys: embed, pos_embed, final_norm, lm_head, drop.
    Values are None if not found.
    """
    components = {"embed": None, "pos_embed": None, "final_norm": None,
                  "lm_head": None, "drop": None, "rotary_emb": None}
    for path in _EMBED_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["embed"] = mod
            break
    for path in _POS_EMBED_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["pos_embed"] = mod
            break
    for path in _FINAL_NORM_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["final_norm"] = mod
            break
    for path in _LM_HEAD_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["lm_head"] = mod
            break
    for path in _DROP_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["drop"] = mod
            break
    for path in _ROTARY_EMBED_PATTERNS:
        mod = _get_submodule(model, path)
        if mod is not None:
            components["rotary_emb"] = mod
            break
    return components


class KVCacheBlock(_nn.Module if _HAS_TORCH else object):
    """Wraps a list of transformer layers with KV-cache support.

    Replaces nn.Sequential for multi-GPU pipeline-parallel inference.
    Each layer is called with (hidden_states, past_key_value, use_cache)
    and returns (hidden_states, present_key_value).
    """

    def __init__(self, layers: list):
        if _HAS_TORCH:
            super().__init__()
            self.layers = _nn.ModuleList(layers)
        else:
            self.layers = layers

    def forward(self, hidden_states, past_key_values=None, use_cache=False,
                attention_mask=None, position_ids=None, position_embeddings=None):
        """Forward pass with KV-cache support.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            past_key_values: List of per-layer KV cache tuples, or None
            use_cache: Whether to return new KV cache
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            position_embeddings: Pre-computed (cos, sin) rotary embeddings

        Returns:
            (hidden_states, presents) where presents is a list of KV tuples
            if use_cache=True, else None.
        """
        # Detect if past_key_values is a Cache object (DynamicCache etc.)
        _is_cache_obj = hasattr(past_key_values, 'update')
        presents = [] if (use_cache and not _is_cache_obj) else None

        for i, layer in enumerate(self.layers):
            # For Cache objects, pass the whole cache (each layer uses its layer_idx);
            # for legacy tuple format, index per-layer
            if _is_cache_obj:
                layer_past = past_key_values
            else:
                layer_past = past_key_values[i] if past_key_values else None

            # Base kwargs
            layer_kwargs = {
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            # Compute or forward position_embeddings for rotary models
            # (Llama, Mistral, Qwen2 with transformers >= 4.46)
            if position_embeddings is not None:
                # Pre-computed from model-level rotary_emb — use directly
                layer_kwargs["position_embeddings"] = position_embeddings
            elif position_ids is not None and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                # Older transformers: rotary_emb lives on each attention layer
                try:
                    pos_emb = layer.self_attn.rotary_emb(hidden_states, position_ids)
                    layer_kwargs["position_embeddings"] = pos_emb
                except Exception:
                    pass

            # Try calling with KV cache kwargs — singular 'past_key_value' FIRST.
            # Most modern HF layers (Llama, Mistral, Qwen2) use singular + **kwargs,
            # so passing plural first would silently be absorbed by **kwargs without
            # actually setting the cache parameter.
            try:
                output = layer(
                    hidden_states,
                    past_key_value=layer_past,
                    **layer_kwargs
                )
            except TypeError:
                try:
                    # Try plural 'past_key_values' for models that use it
                    output = layer(
                        hidden_states,
                        past_key_values=layer_past,
                        **layer_kwargs
                    )
                except TypeError:
                    try:
                        # Try GPT-2 signature (layer_past)
                        output = layer(
                            hidden_states,
                            layer_past=layer_past,
                            use_cache=use_cache,
                            attention_mask=attention_mask,
                        )
                    except TypeError:
                        # Fallback: Qwen and newer architectures sometimes require kwargs via **kwargs
                        try:
                            output = layer(
                                hidden_states, 
                                **layer_kwargs
                            )
                        except TypeError:
                            # Final fallback
                            output = layer(hidden_states)

            # Parse output: tuple (hidden_states, present_kv, ...) or just tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
                if presents is not None and len(output) > 1 and output[1] is not None:
                    presents.append(output[1])
                elif presents is not None:
                    presents.append(None)
            else:
                hidden_states = output
                if presents is not None:
                    presents.append(None)

        # For Cache objects, they are mutated in-place — return the same object
        if _is_cache_obj:
            return hidden_states, past_key_values
        return hidden_states, presents


def _vllm_available() -> bool:
    """Check if vLLM can actually run (installed + CUDA present)."""
    if _MINIMAL:
        return False
    if not _HAS_TORCH or not getattr(_torch, 'cuda', None) or not _torch.cuda.is_available():
        return False
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _gpus_are_homogeneous() -> bool:
    """Check if all CUDA GPUs share the same architecture (CC major).

    Tensor Parallelism requires frequent all-reduce between GPUs.
    Without P2P (heterogeneous GPUs, different PCIe root), all-reduce
    falls back to CPU-staged NCCL which is ~7x slower than pipeline parallel.
    """
    if not _HAS_TORCH or not _torch.cuda.is_available():
        return True
    n = _torch.cuda.device_count()
    if n < 2:
        return True
    majors = set()
    for i in range(n):
        props = _torch.cuda.get_device_properties(i)
        majors.add(props.major)
    return len(majors) == 1


def _compute_vllm_config(num_gpus: int) -> dict:
    """Build vLLM engine kwargs from hetero_config (compute-aware).

    Returns dict with keys: tensor_parallel_size, gpu_memory_utilization,
    dtype_str, and optionally max_model_len.

    Key decision: if GPUs are heterogeneous (different CC major), TP is
    downgraded to 1 on the largest GPU — tensor parallel all-reduce without
    P2P goes through CPU and is much slower than pipeline parallelism.
    """
    result = {"tensor_parallel_size": num_gpus, "gpu_memory_utilization": 0.90}

    # Heterogeneous GPUs: TP=1 on the biggest GPU (vLLM TP all-reduce
    # without P2P is ~7x slower than accelerate pipeline parallel)
    if num_gpus > 1 and not _gpus_are_homogeneous():
        logger.info(
            "Heterogeneous GPUs detected (different architectures) — "
            "vLLM will use TP=1 on largest GPU for best throughput"
        )
        result["tensor_parallel_size"] = 1

    try:
        from core.hetero_config import auto_configure
        config = auto_configure(strategy="balanced")
        if config.gpus:
            # When TP=1, use only the GPU with the most VRAM
            if result["tensor_parallel_size"] == 1 and len(config.gpus) >= 2:
                best_gpu = max(config.gpus, key=lambda g: g.total_vram_gb)
                result["target_gpu"] = best_gpu.index
                vram_gb = best_gpu.total_vram_gb
                logger.info(
                    f"vLLM TP=1 targeting GPU {best_gpu.index} "
                    f"({best_gpu.name}, {vram_gb:.1f} GB)"
                )
            else:
                vram_gb = min(g.total_vram_gb for g in config.gpus)

            # Utilisation: leave 1.5 GB headroom
            headroom_gb = 1.5
            if vram_gb > headroom_gb:
                result["gpu_memory_utilization"] = round(
                    (vram_gb - headroom_gb) / vram_gb, 2
                )
            # Optimal dtype string for vLLM
            best_cc = max((g.compute_capability for g in config.gpus), default=(0, 0))
            if best_cc >= (8, 0):
                result["dtype_str"] = "bfloat16"
            else:
                result["dtype_str"] = "float16"
            # Log tier info
            primary = max(config.gpus, key=lambda g: g.effective_compute)
            arch = primary.profile.architecture if primary.profile else "unknown"
            logger.info(
                f"vLLM compute-aware: primary GPU {primary.index} "
                f"({primary.name}, {arch}), TP={result['tensor_parallel_size']}, "
                f"gpu_util={result['gpu_memory_utilization']}"
            )
    except Exception as e:
        logger.debug(f"hetero_config unavailable for vLLM config: {e}")
    return result


def select_backend(model_name: str, cache_dir: str = None, backend: str = "auto", num_gpus: int = 1):
    logger.info(f"Sélection du backend pour {model_name} (demandé: {backend}, gpus: {num_gpus})")

    # vLLM: preferred when explicitly requested or auto + CUDA available
    if backend == "vllm" or (backend == "auto" and _vllm_available()):
        try:
            from core.backends_vllm import vLLMBackend
            vllm_cfg = _compute_vllm_config(num_gpus)
            tp = vllm_cfg["tensor_parallel_size"]
            target_gpu = vllm_cfg.get("target_gpu")

            # If heterogeneous GPUs forced TP=1 but model needs more VRAM
            # than the single largest GPU, fall through to accelerate
            if tp == 1 and num_gpus > 1:
                logger.info(
                    f"vLLM TP=1 on GPU {target_gpu or 0} "
                    f"(heterogeneous setup — 2nd GPU available via accelerate fallback "
                    f"if model doesn't fit)"
                )

            logger.info(f"Utilisation du backend vLLM pour {model_name} (TP={tp})")
            return vLLMBackend(
                model_name,
                cache_dir=cache_dir,
                tensor_parallel_size=tp,
                gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.90),
                dtype_str=vllm_cfg.get("dtype_str"),
                target_gpu=target_gpu,
            )
        except ImportError:
            logger.info("Fallback sur HuggingFaceBackend car vLLM n'est pas disponible.")
            return HuggingFaceBackend(model_name, cache_dir=cache_dir)

    # auto without vLLM → HuggingFace (accelerate)
    if backend == "auto":
        logger.info(f"Utilisation du backend HuggingFace pour {model_name}")
        return HuggingFaceBackend(model_name, cache_dir=cache_dir)

    if backend == "llamacpp":
        from core.backends_llamacpp import LlamaCppBackend
        return LlamaCppBackend(model_name, cache_dir=cache_dir)

    if backend == "ollama":
        from core.backends_ollama import OllamaBackend
        return OllamaBackend(model_name)
        
    if backend == "webgpu":
        try:
            from core.backends_webgpu import WebGPUBackend
            return WebGPUBackend()
        except ImportError as e:
            logger.error(f"Failed to import WebGPUBackend: {e}")
            raise ValueError("WebGPU support requires extra dependencies. Ensure they are installed.")
        
    if backend == "huggingface":
        return HuggingFaceBackend(model_name, cache_dir=cache_dir)
        
    raise ValueError(f"Backend inconnu ou non supporté : {backend}")


class BaseLLMBackend(ABC):
    """Interface générique pour tous les backends LLM."""

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        ...

    @abstractmethod
    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None) -> List[Any]:
        """Découpe le modèle en blocs selon le nombre de GPUs."""
        ...

    @abstractmethod
    def infer(self, inputs: Any) -> Any:
        ...

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Text generation — backends can override for streaming support."""
        raise NotImplementedError("generate() not implemented for this backend")

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Yield tokens one at a time for real streaming.

        Default implementation falls back to generate() then yields word-by-word.
        Backends with true streaming support should override this.
        """
        text = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        for i, word in enumerate(text.split(' ')):
            yield word if i == 0 else ' ' + word

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        """Batch generation — process multiple prompts in a single forward pass.

        Default implementation falls back to sequential generate() calls.
        Backends with true padding-based batching should override this.
        """
        return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]

# ------------------- HuggingFace Backend -------------------
class HuggingFaceBackend(BaseLLMBackend):
    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model = None
        self.model_name: Optional[str] = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.blocks: Optional[List[Any]] = None
        self.block_devices: Optional[List[int]] = None  # GPU id per block
        self.log = LoggerAdapter("backend.hf")
        self.hmem = None  # reference injectée optionnelle
        self.transfer_manager = None  # injecté par le pipeline
        # Model components for multi-GPU KV-cache forward
        self._components: Optional[dict] = None  # embed, pos_embed, final_norm, lm_head

    def _build_compute_aware_memory_map(self) -> Optional[dict]:
        """Build a max_memory dict for accelerate that favors faster GPUs.

        Uses hetero_config to detect GPU compute capabilities and assigns
        more VRAM budget to higher-compute GPUs. This causes accelerate
        to place more model layers on the best GPU (e.g. Blackwell 5070 Ti
        gets priority over Ampere 3090).

        Returns None if detection fails or only 1 GPU (let accelerate decide).
        """
        if not _HAS_TORCH or not _torch.cuda.is_available():
            return None
        num_gpus = _torch.cuda.device_count()
        if num_gpus < 2:
            return None

        try:
            from core.hetero_config import auto_configure
            config = auto_configure(strategy="balanced")
            if len(config.gpus) < 2:
                return None

            max_memory = {}
            quant_mode = self._get_quantization_mode()
            for gpu in config.gpus:
                # Use FREE VRAM (not total) to account for driver/OS usage.
                # max_memory is a per-GPU cap — accelerate handles placement.
                base_vram = gpu.free_vram_gb if gpu.free_vram_gb > 0 else gpu.total_vram_gb
                # NF4: 60% (fp16 loading transient = 4x planned NF4 size)
                # INT8: 85% (fp16→int8 but less headroom needed)
                # Non-quantized: aggressive 92% to minimize CPU offload.
                if quant_mode == "nf4":
                    reserve = 0.60
                elif quant_mode == "int8":
                    reserve = 0.85
                else:
                    reserve = 0.92
                budget_gb = max(2.0, base_vram * reserve)
                max_memory[gpu.index] = f"{budget_gb:.1f}GiB"

            # CPU overflow: accelerate offloads excess layers to RAM.
            # For quantized models, llm_int8_enable_fp32_cpu_offload=True
            # keeps CPU-dispatched modules in fp32 (not quantized).
            max_memory["cpu"] = "48GiB"

            # Log the decision
            primary = max(config.gpus, key=lambda g: g.effective_compute)
            arch = primary.profile.architecture if primary.profile else "unknown"
            self.log.info(
                f"Compute-aware placement: primary GPU {primary.index} "
                f"({primary.name}, {arch}, "
                f"{primary.profile.fp16_tflops if primary.profile else '?'} TFLOPS)"
            )
            for gpu in config.gpus:
                cap_str = ""
                if gpu.profile and gpu.profile.architecture in ("Blackwell",):
                    cap_str = " [NVFP4, FP8, MicroTensor]"
                elif gpu.profile and gpu.profile.architecture in ("Ada",):
                    cap_str = " [FP8]"
                self.log.info(
                    f"  GPU {gpu.index}: {gpu.name} — {max_memory[gpu.index]} budget, "
                    f"tier {gpu.tier}, ratio {gpu.split_ratio:.1%}{cap_str}"
                )

            return max_memory
        except Exception as e:
            self.log.debug(f"Compute-aware memory map failed, using default: {e}")
            return None

    def _detect_optimal_dtype(self):
        """Detect the best torch dtype based on GPU capabilities.

        - Blackwell (CC >= 12.0): bfloat16 natively, NVFP4 via quantization
        - Ada (CC >= 8.9): bfloat16
        - Ampere (CC >= 8.0): bfloat16
        - Older: float16
        """
        if not _HAS_TORCH or not _torch.cuda.is_available():
            return None
        try:
            best_cc = (0, 0)
            best_arch = ""
            for i in range(_torch.cuda.device_count()):
                props = _torch.cuda.get_device_properties(i)
                cc = (props.major, props.minor)
                if cc > best_cc:
                    best_cc = cc
                    try:
                        from core.hetero_config import lookup_gpu_profile
                        profile = lookup_gpu_profile(props.name)
                        best_arch = profile.architecture if profile else ""
                    except Exception:
                        pass

            if best_cc >= (12, 0):
                self.log.info(
                    f"Blackwell GPU detected (CC {best_cc[0]}.{best_cc[1]}) — "
                    f"using bfloat16 (NVFP4 quantization available via BitsAndBytes)"
                )
                return _torch.bfloat16
            elif best_cc >= (8, 0):
                self.log.info(
                    f"Ampere+ GPU detected (CC {best_cc[0]}.{best_cc[1]}) — using bfloat16"
                )
                return _torch.bfloat16
            else:
                self.log.info(f"GPU CC {best_cc[0]}.{best_cc[1]} — using float16")
                return _torch.float16
        except Exception:
            return None

    def _ensure_gptq_imports(self):
        """Make GPTQ model loading work despite auto_gptq/optimum incompatibility.

        Problem: optimum >= 2.1 imports from ``gptqmodel`` (not auto_gptq).
        The user has ``auto_gptq`` 0.7.x which is incompatible with
        transformers >= 5.x.  This means ``optimum.gptq.quantizer.GPTQQuantizer``
        cannot be instantiated — it tries to use ``METHOD``, ``FORMAT``,
        ``QuantizeConfig`` etc. from gptqmodel which isn't installed.

        Solution: monkey-patch ``GptqHfQuantizer.__init__`` in transformers to
        skip the broken ``GPTQQuantizer.from_dict()`` call.  For **inference**
        (loading pre-quantized GPTQ models), the optimum quantizer object is
        NOT needed — transformers handles weight loading via its own
        ``GptqHfQuantizer`` lifecycle methods which only need the config dict.
        """
        import sys

        # ── Patch transformers.modeling_utils.no_init_weights ───────────
        try:
            import contextlib
            import transformers.modeling_utils as _tmu
            if not hasattr(_tmu, 'no_init_weights'):
                @contextlib.contextmanager
                def _no_init_weights(_enable=True):
                    yield
                _tmu.no_init_weights = _no_init_weights
        except Exception:
            pass

        # ── Monkey-patch GptqHfQuantizer.__init__ ───────────────────────
        try:
            from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

            _original_init = GptqHfQuantizer.__init__

            def _safe_gptq_init(self_inner, quantization_config, **kwargs):
                """GptqHfQuantizer.__init__ that survives broken optimum."""
                try:
                    _original_init(self_inner, quantization_config, **kwargs)
                except (NameError, ImportError, AttributeError) as exc:
                    # optimum's GPTQQuantizer failed (missing gptqmodel symbols).
                    # Store the config and set optimum_quantizer = None.
                    # transformers will still load GPTQ weights correctly
                    # using its own quantization hooks.
                    from transformers.quantizers.base import HfQuantizer
                    HfQuantizer.__init__(self_inner, quantization_config, **kwargs)
                    self_inner.optimum_quantizer = None
                    logger.warning(
                        "GPTQ: bypassed broken optimum GPTQQuantizer (%s). "
                        "Inference will use transformers native GPTQ loading.", exc
                    )

            GptqHfQuantizer.__init__ = _safe_gptq_init

            # ── Monkey-patch validate_environment ───────────────────────
            # transformers 5.x requires gptqmodel (not auto_gptq).
            # For inference with pre-quantized models, auto_gptq is enough.
            _original_validate = GptqHfQuantizer.validate_environment

            def _safe_validate_environment(self_inner, *args, **kwargs):
                try:
                    _original_validate(self_inner, *args, **kwargs)
                except ImportError:
                    # gptqmodel not installed, but auto_gptq is available
                    # for inference (weight deserialization via QuantLinear).
                    try:
                        import auto_gptq  # noqa: F401
                        logger.warning(
                            "GPTQ: gptqmodel not installed, using auto_gptq for inference"
                        )
                    except ImportError:
                        raise ImportError(
                            "GPTQ quantized model requires either gptqmodel or auto-gptq. "
                            "Install with: pip install auto-gptq (BUILD_CUDA_EXT=0 if no CUDA toolkit)"
                        )

            GptqHfQuantizer.validate_environment = _safe_validate_environment

            # ── Monkey-patch _process_model_before_weight_loading ────────
            # Calls self.optimum_quantizer.convert_model() which fails
            # when optimum_quantizer is None.  For pre-quantized models
            # the weights are already quantized — convert_model is a no-op.
            _original_before = GptqHfQuantizer._process_model_before_weight_loading

            def _safe_before_weight_loading(self_inner, model, **kwargs):
                if getattr(self_inner, 'optimum_quantizer', None) is not None:
                    return _original_before(self_inner, model, **kwargs)
                # Skip convert_model — weights are already GPTQ-quantized
                logger.info("GPTQ: skipping convert_model (no optimum_quantizer)")

            GptqHfQuantizer._process_model_before_weight_loading = _safe_before_weight_loading

            # ── Monkey-patch _process_model_after_weight_loading ─────────
            # Calls self.optimum_quantizer.post_init_model() for pre-quantized
            # models.  Without it, we do a manual post-init (set is_quantized etc.)
            _original_after = GptqHfQuantizer._process_model_after_weight_loading

            def _safe_after_weight_loading(self_inner, model, **kwargs):
                if getattr(self_inner, 'optimum_quantizer', None) is not None:
                    return _original_after(self_inner, model, **kwargs)
                # Manual post-init for pre-quantized models
                model.is_quantized = True
                model.quantization_method = "gptq"
                logger.info("GPTQ: manual post-init (no optimum_quantizer)")

            GptqHfQuantizer._process_model_after_weight_loading = _safe_after_weight_loading

            self.log.info("GPTQ: patched GptqHfQuantizer to bypass broken optimum")
        except ImportError:
            self.log.warning("GPTQ: transformers.quantizers.quantizer_gptq not found")

    def _get_quantization_mode(self) -> str:
        """Return the active quantization mode.

        Returns one of: 'nvfp4', 'nf4', 'int8', or '' (no quantization).
        Checks VRM_QUANTIZATION env var and verifies dependencies.

        nvfp4: Native Blackwell FP4 via torchao (CC >= 10.0, sm100+).
               Uses cublas scaled_mm with float4_e2m1fn_x2 dtype.
               ~62% VRAM reduction vs BF16. Requires torchao >= 0.16.
        nf4: ~0.5 bytes/param (4-bit NormalFloat via BnB), all CUDA GPUs.
        int8: ~1.0 bytes/param (LLM.int8() via BnB), CC >= 7.5.
        """
        quant = os.environ.get("VRM_QUANTIZATION", "").lower()
        if quant not in ("nvfp4", "nf4", "int8"):
            return ""
        if not _HAS_TORCH or not _torch.cuda.is_available():
            return ""

        # NVFP4: real Blackwell FP4 via torchao (not BnB)
        if quant == "nvfp4":
            if self._has_blackwell_gpu():
                try:
                    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: F401
                    return "nvfp4"
                except ImportError:
                    self.log.warning(
                        "VRM_QUANTIZATION=nvfp4 requested but torchao NVFP4 not available. "
                        "Install with: pip install torchao>=0.16.0. "
                        "Falling back to BnB NF4."
                    )
            else:
                self.log.warning(
                    "VRM_QUANTIZATION=nvfp4 requested but no Blackwell GPU (CC >= 10.0) found. "
                    "NVFP4 requires sm100+ (RTX 50xx / Blackwell). "
                    "Falling back to BnB NF4."
                )
            # Fallback: nvfp4 -> nf4 on non-Blackwell or missing torchao
            quant = "nf4"

        # BnB quantization (nf4 / int8)
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            self.log.warning(
                "VRM_QUANTIZATION=%s requested but bitsandbytes not installed. "
                "Install with: pip install bitsandbytes>=0.43.0", quant
            )
            return ""
        if quant == "nf4":
            return "nf4"
        return "int8"

    def _should_use_nvfp4(self) -> bool:
        """Check if NF4 quantization should be used (backward compat)."""
        return self._get_quantization_mode() in ("nf4", "nvfp4")

    @staticmethod
    def _has_blackwell_gpu() -> bool:
        """Check if any visible GPU has Blackwell architecture (CC >= 10.0)."""
        if not _HAS_TORCH or not _torch.cuda.is_available():
            return False
        for i in range(_torch.cuda.device_count()):
            cc = _torch.cuda.get_device_capability(i)
            if cc[0] >= 10:
                return True
        return False

    @staticmethod
    def _nvfp4_filter_fn(module, fqn: str) -> bool:
        """Filter function for torchao quantize_(): exclude lm_head.

        lm_head uses aten.expand which NVFP4Tensor doesn't implement.
        Only quantize nn.Linear layers that are NOT the output head.
        """
        if not _HAS_TORCH:
            return False
        if not isinstance(module, _torch.nn.Linear):
            return False
        # Exclude lm_head, output projection, and embedding layers
        excluded = ("lm_head", "embed_tokens", "wte", "wpe")
        for name in excluded:
            if name in fqn:
                return False
        return True

    def _apply_nvfp4_quantization(self, model):
        """Apply real NVFP4 Blackwell quantization via torchao.

        Uses NVFP4DynamicActivationNVFP4WeightConfig (Dynamic W+A mode)
        which calls torch._scaled_mm with float4_e2m1fn_x2 dtype —
        the real cublas Blackwell FP4 kernel.

        Weight-Only mode is NOT used: it dequantizes every forward pass
        (prototype limitation in torchao 0.16), giving ~0.9 tok/s vs
        ~11 tok/s for Dynamic W+A.

        The model must be on CPU before calling this method.
        After quantization, the model is moved to the best Blackwell GPU.
        """
        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )

        self.log.info(
            "NVFP4 Blackwell: quantizing with Dynamic W+A mode "
            "(cublas scaled_mm FP4 kernel, excluding lm_head)"
        )
        import time
        t0 = time.monotonic()
        nvfp4_config = NVFP4DynamicActivationNVFP4WeightConfig()
        quantize_(model, nvfp4_config, filter_fn=self._nvfp4_filter_fn)
        quant_time = time.monotonic() - t0
        self.log.info("NVFP4 quantization completed in %.1fs", quant_time)

        # Move to best Blackwell GPU
        best_gpu = 0
        best_free = 0.0
        for i in range(_torch.cuda.device_count()):
            cc = _torch.cuda.get_device_capability(i)
            if cc[0] >= 10:
                free_bytes = _torch.cuda.mem_get_info(i)[0]
                if free_bytes > best_free:
                    best_free = free_bytes
                    best_gpu = i

        self.log.info(
            "NVFP4: moving model to cuda:%d (%.1f GiB free)",
            best_gpu, best_free / (1024 ** 3),
        )
        t0 = time.monotonic()
        model = model.to(f"cuda:{best_gpu}")
        move_time = time.monotonic() - t0
        self.log.info("NVFP4: GPU transfer in %.1fs", move_time)

        # Replace NVFP4Tensor layers with DirectFP4Linear to bypass
        # torchao __torch_dispatch__ overhead (~7% faster inference)
        try:
            from core.nvfp4_direct import replace_with_direct_fp4
            t0 = time.monotonic()
            n_replaced = replace_with_direct_fp4(model, verbose=False)
            bypass_time = time.monotonic() - t0
            if n_replaced > 0:
                self.log.info(
                    "NVFP4 DirectFP4 bypass: replaced %d layers in %.2fs",
                    n_replaced, bypass_time,
                )
        except Exception as e:
            self.log.warning(
                "NVFP4 DirectFP4 bypass unavailable, using torchao dispatch: %s", e
            )

        return model

    def _build_bnb_nf4_config(self):
        """Build BitsAndBytesConfig for NF4 (4-bit NormalFloat) quantization.

        Works on all CUDA GPUs (CC >= 7.0). Uses bfloat16 compute dtype.
        A 14B model goes from ~28 GB (BF16) to ~7 GB (NF4).
        """
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.log.info(
            "NF4 quantization enabled (4-bit NormalFloat + double quant) — "
            "model size reduced ~75%%"
        )
        return bnb_config

    _build_bnb_nvfp4_config = _build_bnb_nf4_config  # backward compat alias

    def _build_bnb_int8_config(self):
        """Build BitsAndBytesConfig for INT8 (LLM.int8()) quantization.

        Works on all CUDA GPUs with CC >= 7.5 (Turing+).
        A 14B model goes from ~28 GB (BF16) to ~14 GB (INT8).
        """
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.log.info(
            "INT8 quantization enabled (LLM.int8()) — model size reduced ~50%%"
        )
        return bnb_config

    @staticmethod
    def _patch_bnb_for_transformers():
        """Monkey-patch bitsandbytes to accept extra kwargs from transformers 5.x.

        transformers >= 5.0 passes _is_hf_initialized to tensor __new__()
        but bitsandbytes 0.49.x doesn't accept it -> TypeError.
        Wrap __new__ to strip unknown kwargs.
        """
        try:
            import bitsandbytes as bnb
            for cls in (bnb.nn.Params4bit, bnb.nn.Int8Params):
                orig_new = cls.__new__
                if getattr(orig_new, '_vrm_patched', False):
                    continue
                import inspect
                sig = inspect.signature(orig_new)
                if '_is_hf_initialized' not in sig.parameters and '**' not in str(sig):
                    def make_wrapper(original, klass):
                        def _patched_new(cls_, *args, **kwargs):
                            kwargs.pop('_is_hf_initialized', None)
                            return original(cls_, *args, **kwargs)
                        _patched_new._vrm_patched = True
                        return _patched_new
                    cls.__new__ = make_wrapper(orig_new, cls)
        except ImportError:
            pass

    def load_model(self, model_name: str, **kwargs):
        # Pop VRAMancer-specific kwargs that should NOT be forwarded to from_pretrained
        _pipeline_num_gpus = kwargs.pop("num_gpus", None)

        if os.environ.get("VRM_MINIMAL_TEST") == "1" or model_name.startswith("stub"):
            self.model_name = model_name
            self.model = "STUB_MODEL"
            self.tokenizer = "STUB_TOKENIZER"
            return self.model

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        quant_mode = ""  # set properly in else: branch below
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
        if model_name.endswith("-AWQ") or "-awq" in model_name.lower():
            # AWQ: use autoawq native loader (transformers 5.3 requires gptqmodel
            # which has build issues). autoawq fuses QKV/MLP layers for faster
            # INT4 GEMM kernels (~3-4x faster than BnB kgemm_4bit_inference_naive).
            try:
                from awq import AutoAWQForCausalLM as _AWQ
                self.log.info("AWQ model detected — loading via autoawq (fused layers)")
                # Pick best GPU for single-GPU AWQ
                if _HAS_TORCH and _torch.cuda.is_available():
                    best_gpu = 0
                    best_free = 0.0
                    for i in range(_torch.cuda.device_count()):
                        free_bytes = _torch.cuda.mem_get_info(i)[0]
                        if free_bytes > best_free:
                            best_free = free_bytes
                            best_gpu = i
                    fuse_layers = True
                else:
                    best_gpu = 0
                    fuse_layers = False
                awq_model = _AWQ.from_quantized(
                    model_name,
                    fuse_layers=fuse_layers,
                    trust_remote_code=(os.environ.get("VRM_TRUST_REMOTE_CODE") == "1"),
                )
                self.model = awq_model.model
                if _HAS_TORCH and _torch.cuda.is_available() and best_gpu != 0:
                    self.model = self.model.to(f"cuda:{best_gpu}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    self.log.warning(f"Tokenizer load failed: {e}")
                return self.model
            except ImportError:
                self.log.warning("autoawq not installed — falling back to transformers AWQ loader")
                kwargs["low_cpu_mem_usage"] = True
        elif "gptq" in model_name.lower():
            # GPTQ: pre-quantized model — weights are already INT4/INT8.
            # No runtime quantization needed (no BnB, no fp16 transient).
            # The model's config.json contains its own quantization_config,
            # so we do NOT pass our own — just disable exllama kernels via
            # env var so auto_gptq uses PyTorch dequant (no CUDA compilation).
            # Requires: pip install optimum auto-gptq (BUILD_CUDA_EXT=0)
            self.log.info("GPTQ model detected — loading pre-quantized (no BnB needed)")
            os.environ["DISABLE_EXLLAMA"] = "1"
            os.environ["DISABLE_EXLLAMAV2"] = "1"
            self._ensure_gptq_imports()
            kwargs["low_cpu_mem_usage"] = True
            kwargs["device_map"] = "auto"
            num_gpus = _torch.cuda.device_count() if _HAS_TORCH and _torch.cuda.is_available() else 1
            if num_gpus >= 2:
                max_memory = self._build_compute_aware_memory_map()
                if max_memory:
                    kwargs["max_memory"] = max_memory
            # Optimal dtype for non-quantized layers (embed, norm, lm_head)
            best_dtype = self._detect_optimal_dtype()
            if best_dtype is not None:
                try:
                    import transformers as _tf
                    _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
                    if _tf_ver >= (4, 46):
                        kwargs["dtype"] = best_dtype
                    else:
                        kwargs["torch_dtype"] = best_dtype
                except Exception:
                    kwargs["torch_dtype"] = best_dtype
        else:
            quant_mode = self._get_quantization_mode()
            # Use pipeline's num_gpus decision (respects _auto_select_num_gpus)
            # rather than torch.cuda.device_count() which ignores single-GPU bypass.
            if _pipeline_num_gpus is not None:
                num_gpus = _pipeline_num_gpus
            else:
                num_gpus = _torch.cuda.device_count() if _HAS_TORCH and _torch.cuda.is_available() else 1

            # NVFP4 Blackwell: load on CPU, quantize post-load via torchao,
            # then move to GPU. This is fundamentally different from BnB
            # which quantizes during from_pretrained.
            if quant_mode == "nvfp4":
                kwargs["device_map"] = "cpu"
                kwargs["low_cpu_mem_usage"] = True
                if "torch_dtype" not in kwargs and "dtype" not in kwargs:
                    kwargs["torch_dtype"] = _torch.bfloat16
                self.log.info(
                    "NVFP4 Blackwell: loading model on CPU for post-load "
                    "quantization via torchao"
                )
            elif quant_mode and num_gpus >= 2:
                kwargs["device_map"] = "auto"
                # BnB multi-GPU is broken upstream: accelerate AlignDevicesHook
                # doesn't handle cross-GPU residual connections with NF4/INT8.
                # Error: "Expected all tensors on same device, cuda:0 and cuda:1"
                # (confirmed with raw transformers + accelerate + BnB, no VRAMancer)
                #
                # Workaround: force single-GPU placement. With device_map={"": gpu}
                # + torch_dtype for streaming load, BnB quantizes each layer
                # in-place without buffering all fp16 weights. A 14B NF4 model
                # uses only ~10.8 GB final on a 24 GB GPU.
                best_gpu = 0
                best_free = 0.0
                for i in range(num_gpus):
                    free_bytes = _torch.cuda.mem_get_info(i)[0]
                    if free_bytes > best_free:
                        best_free = free_bytes
                        best_gpu = i
                kwargs["device_map"] = {"": best_gpu}
                kwargs["low_cpu_mem_usage"] = True
                self.log.info(
                    "BnB %s: single-GPU %d (%.1fGiB free) — "
                    "multi-GPU BnB broken upstream (accelerate hooks)",
                    quant_mode.upper(), best_gpu, best_free / (1024 ** 3),
                )
            else:
                kwargs["device_map"] = "auto"
                # Non-quantized OR quantized single-GPU
                if quant_mode and num_gpus <= 1:
                    # Quantized single-GPU: use device_map={"": gpu_id}
                    # to place the ENTIRE model on a single GPU.
                    # device_map="auto" installs accelerate AlignDevicesHook
                    # pre_forward hooks that move weights CPU↔GPU during
                    # inference — these hang with BnB 0.49 + accelerate 1.13.
                    # With {"": gpu_id}, weights load directly onto GPU,
                    # no hooks, no CPU offloading. Model must fit in VRAM.
                    best_gpu = 0
                    best_free = 0.0
                    total_gpus = _torch.cuda.device_count()
                    for i in range(total_gpus):
                        free_bytes = _torch.cuda.mem_get_info(i)[0]
                        if free_bytes > best_free:
                            best_free = free_bytes
                            best_gpu = i
                    kwargs["device_map"] = {"": best_gpu}
                    kwargs["low_cpu_mem_usage"] = True
                    self.log.info(
                        "BnB %s single-GPU %d: %.1fGiB free (no CPU offload)",
                        quant_mode.upper(), best_gpu,
                        best_free / (1024 ** 3),
                    )
                else:
                    max_memory = self._build_compute_aware_memory_map()
                    if max_memory:
                        kwargs["max_memory"] = max_memory

            # Quantization config (BnB only — NVFP4 is post-load)
            if quant_mode == "nf4" and "quantization_config" not in kwargs:
                self._patch_bnb_for_transformers()
                kwargs["quantization_config"] = self._build_bnb_nf4_config()
                # Set torch_dtype (NOT 'dtype') for non-quantized layers.
                # In transformers 5.3, 'dtype' bypasses BnB quantization
                # (loads everything in that dtype), while 'torch_dtype'
                # (deprecated but functional) controls only non-quantized
                # layers and lets BnB quantize normally.
                if "torch_dtype" not in kwargs and "dtype" not in kwargs:
                    kwargs["torch_dtype"] = _torch.float16
            elif quant_mode == "int8" and "quantization_config" not in kwargs:
                self._patch_bnb_for_transformers()
                kwargs["quantization_config"] = self._build_bnb_int8_config()
                if "torch_dtype" not in kwargs and "dtype" not in kwargs:
                    kwargs["torch_dtype"] = _torch.float16
            # Auto-select optimal dtype based on best available GPU
            # (skip for nvfp4 — dtype already set above)
            elif quant_mode != "nvfp4" and "torch_dtype" not in kwargs and "dtype" not in kwargs:
                best_dtype = self._detect_optimal_dtype()
                if best_dtype is not None:
                    # Newer transformers (>= 4.46) use 'dtype', older use 'torch_dtype'
                    try:
                        import transformers as _tf
                        _tf_ver = tuple(int(x) for x in _tf.__version__.split(".")[:2])
                        if _tf_ver >= (4, 46):
                            kwargs["dtype"] = best_dtype
                        else:
                            kwargs["torch_dtype"] = best_dtype
                    except Exception:
                        kwargs["torch_dtype"] = best_dtype

        if "trust_remote_code" not in kwargs:
            kwargs["trust_remote_code"] = (os.environ.get("VRM_TRUST_REMOTE_CODE") == "1")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except Exception as e:
            # Log full traceback for GPTQ/quantized model debugging
            import traceback
            self.log.error("from_pretrained failed:\n%s", traceback.format_exc())
            raise

        # NVFP4 post-load quantization: model is on CPU, quantize then move to GPU
        if quant_mode == "nvfp4":
            self.model = self._apply_nvfp4_quantization(self.model)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.log.warning(f"Tokenizer load failed: {e}")
        return self.model

    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None):
        from core.model_splitter import split_model_into_blocks, assign_blocks_to_gpus

        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")

        # If accelerate already distributed the model via device_map="auto",
        # keep its dispatch hooks and use native model.generate() for inference.
        # This is the most reliable multi-GPU path: accelerate + transformers
        # handle attention masks, rotary embeddings, KV cache, and device
        # transfers internally.
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            dev_map = self.model.hf_device_map
            devices_used = sorted(set(str(v) for v in dev_map.values()))
            layers_per_dev = {}
            for _k, _v in dev_map.items():
                d = str(_v)
                layers_per_dev[d] = layers_per_dev.get(d, 0) + 1
            self.log.info(f"Model distributed by accelerate across {len(devices_used)} device(s): {layers_per_dev}")
            self.log.info("Using native accelerate inference (reliable multi-GPU)")
            self.blocks = None
            self._components = None
            self.block_devices = None
            return []

        # No accelerate device_map — use manual VRAMancer split
        if self.model is not None:
            try:
                import accelerate
                accelerate.hooks.remove_hook_from_module(self.model, recurse=True)
                self.log.info("Hooks Accelerate retirés pour éviter les conflits de devices")
            except Exception as e:
                self.log.debug(f"Impossible de retirer les hooks Accelerate: {e}")

        self.log.debug(f"Découpage en blocs sur {num_gpus} GPUs")

        from core.model_splitter import _extract_layers, _split_by_vram, _get_free_vram_per_gpu
        layers = _extract_layers(self.model)
        if layers is None:
            # Fallback: wrap entire model as a single block
            self.blocks = [_nn.Sequential(self.model)]
            self.block_devices = [0]
            self._components = None
            return self.blocks
        if num_gpus <= 1:
            self.blocks = [_nn.Sequential(*layers)]
            self.block_devices = [0]
            self._components = None
            return self.blocks

        # Extract model components (embedding, final norm, lm_head) for
        # proper multi-GPU KV-cache forward
        self._components = _extract_model_components(self.model)
        if self._components["embed"] is None or self._components["lm_head"] is None:
            self.log.warning("Could not extract model components — "
                             "multi-GPU KV-cache disabled, using whole-model blocks")
            self._components = None

        # Profiler-based ou VRAM-proportional
        if vram_per_gpu:
            vram = vram_per_gpu[:num_gpus]
        else:
            vram = _get_free_vram_per_gpu(num_gpus)

        # Create KVCacheBlock instances instead of nn.Sequential
        raw_blocks = _split_by_vram(layers, vram)
        if self._components is not None and _HAS_TORCH:
            self.blocks = []
            for seq_block in raw_blocks:
                layer_list = list(seq_block.children())
                self.blocks.append(KVCacheBlock(layer_list))
        else:
            self.blocks = raw_blocks

        if _HAS_TORCH and _torch.cuda.is_available():
            from core.utils import _get_logical_mapping
            mapping = _get_logical_mapping()
            self.block_devices = [mapping.get(i, i) for i in range(len(self.blocks))]
        else:
            self.block_devices = list(range(len(self.blocks)))
            
        # Move blocks to their devices
        try:
            self.blocks = assign_blocks_to_gpus(self.blocks)
            # Move embedding and head to first/last GPU
            if self._components is not None and _HAS_TORCH:
                first_dev = f"cuda:{self.block_devices[0]}" if _torch.cuda.is_available() else "cpu"
                last_dev = f"cuda:{self.block_devices[-1]}" if _torch.cuda.is_available() else "cpu"
                if self._components["embed"] is not None:
                    self._components["embed"] = self._components["embed"].to(first_dev)
                if self._components["pos_embed"] is not None:
                    self._components["pos_embed"] = self._components["pos_embed"].to(first_dev)
                if self._components.get("rotary_emb") is not None:
                    self._components["rotary_emb"] = self._components["rotary_emb"].to(first_dev)
                if self._components["final_norm"] is not None:
                    self._components["final_norm"] = self._components["final_norm"].to(last_dev)
                
                # Clone lm_head if its weights are tied to embed
                if self._components["lm_head"] is not None:
                    if self._components["embed"] is not None:
                        import copy
                        import torch.nn as nn
                        if hasattr(self._components["lm_head"], "weight") and hasattr(self._components["embed"], "weight"):
                            if self._components["lm_head"].weight is self._components["embed"].weight or \
                               self._components["lm_head"].weight.data_ptr() == self._components["embed"].weight.data_ptr():
                                new_head = copy.deepcopy(self._components["lm_head"])
                                new_head.weight = nn.Parameter(self._components["lm_head"].weight.clone())
                                self._components["lm_head"] = new_head
                    self._components["lm_head"] = self._components["lm_head"].to(last_dev)
        except Exception as e:
            self.log.warning(f"Placement GPU échoué (CPU fallback): {e}")

        block_sizes = []
        for b in self.blocks:
            if hasattr(b, 'layers'):
                block_sizes.append(len(b.layers))
            elif hasattr(b, 'children'):
                block_sizes.append(len(list(b.children())))
            else:
                block_sizes.append(1)
        self.log.info(f"Modèle découpé en {len(self.blocks)} blocs: {block_sizes}")
        return self.blocks

    def _transfer_to_device(self, tensor: Any, src_gpu: int, dst_gpu: int) -> Any:
        """Transfer tensor between GPUs using TransferManager or fallback."""
        if src_gpu == dst_gpu:
            return tensor
        if self.transfer_manager:
            try:
                if _HAS_TORCH and _torch.is_tensor(tensor) and tensor.is_cuda:
                    self.transfer_manager.send_activation(src_gpu, dst_gpu, tensor)
            except Exception as e:
                self.log.debug(f"TransferManager send failed: {e}")
        # Always do the actual .to() move
        try:
            return tensor.to(f"cuda:{dst_gpu}")
        except Exception:
            return tensor

    def infer(self, inputs: Any, past_key_values: Optional[List] = None,
              use_cache: bool = False) -> Any:
        """Forward pass with optional KV-cache for multi-GPU pipeline.

        If the model is split into KVCacheBlocks with extracted components,
        performs: embed → blocks (with KV-cache) → final_norm → lm_head.
        Otherwise falls back to direct model forward.

        Args:
            inputs: input_ids tensor [batch, seq_len] or hidden_states
            past_key_values: nested list of per-block, per-layer KV tuples
            use_cache: whether to return updated KV cache

        Returns:
            If use_cache: (logits, all_past_key_values)
            Else: logits
        """
        if self.blocks is None:
            # No split — direct model forward
            if self.model is None:
                raise RuntimeError("Modèle non chargé")
            if use_cache:
                out = self.model(inputs, past_key_values=past_key_values,
                                 use_cache=True)
                logits = out.logits if hasattr(out, "logits") else out
                pkv = getattr(out, "past_key_values", None)
                return logits, pkv
            if isinstance(self.model, str): return self.model; out = self.model(inputs)
            if hasattr(out, "logits"):
                return out.logits
            return out

        # Multi-GPU with KV-cache components
        if self._components is not None and _HAS_TORCH:
            return self._infer_with_kv_cache(inputs, past_key_values, use_cache)

        # Legacy path: nn.Sequential blocks without KV-cache
        x = inputs
        self.log.debug("Début inférence séquentielle sur %d blocs", len(self.blocks))
        for idx, block in enumerate(self.blocks):
            if self.block_devices and idx > 0:
                prev_gpu = self.block_devices[idx - 1]
                curr_gpu = self.block_devices[idx]
                if prev_gpu != curr_gpu:
                    x = self._transfer_to_device(x, prev_gpu, curr_gpu)

            out = block(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
                
        if use_cache:
            return x, None
        return x

    def _infer_with_kv_cache(self, inputs: Any, past_key_values: Optional[List] = None, use_cache: bool = False) -> Any:
        import traceback
        try:
            return self.__infer_with_kv_cache_impl(inputs, past_key_values, use_cache)
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("EXCEPTION IN INFER WITH KV CACHE:")
            raise e

    def __infer_with_kv_cache_impl(self, inputs: Any, past_key_values: Optional[List] = None, use_cache: bool = False) -> Any:
        import torch as pt
        all_presents = [] if use_cache else None
        
        comp = self._components
        first_gpu_dev = f"cuda:{self.block_devices[0]}" if (self.block_devices and pt.cuda.is_available()) else "cpu"
        
        if getattr(self, "_comp_devices", None) is None:
            self._comp_devices = {}

        # 1. Embeddings (Dynamic device resolution to support Accelerate)
        if "embed" not in self._comp_devices:
            try:
                self._comp_devices["embed"] = next(comp["embed"].parameters()).device
            except StopIteration:
                self._comp_devices["embed"] = first_gpu_dev
                
        embed_dev = self._comp_devices["embed"]
        
        inputs = inputs.to(embed_dev)
        hidden_states = comp["embed"](inputs)
        # Ensure hidden_states stays on embed_dev
        if hasattr(hidden_states, "device") and hidden_states.device != embed_dev:
            hidden_states = hidden_states.to(embed_dev)
        # Ensure hidden_states stays on embed_dev
        if hasattr(hidden_states, "device") and hidden_states.device != embed_dev:
            hidden_states = hidden_states.to(embed_dev)
        
        if comp["pos_embed"] is not None:
            if "pos_embed" not in self._comp_devices:
                try:
                    self._comp_devices["pos_embed"] = next(comp["pos_embed"].parameters()).device
                except StopIteration:
                    self._comp_devices["pos_embed"] = first_gpu_dev
            pos_dev = self._comp_devices["pos_embed"]
            
            seq_len = inputs.shape[1]
            past_len = 0
            if hasattr(past_key_values, 'get_seq_length'):
                past_len = past_key_values.get_seq_length()
            elif past_key_values and past_key_values[0] and past_key_values[0][0] is not None:
                past_len = past_key_values[0][0][0].shape[-2]
            pos = pt.arange(past_len, past_len + seq_len, dtype=pt.long, device=pos_dev).unsqueeze(0)
            
            pos_emb = comp["pos_embed"](pos)
            if hidden_states.device != pos_emb.device:
                hidden_states = hidden_states.to(pos_emb.device)
            hidden_states = hidden_states + pos_emb
            
        if comp["drop"] is not None:
            hidden_states = comp["drop"](hidden_states)

        # Use DynamicCache for modern transformers (Llama, Mistral, Qwen, etc.)
        # Individual layers don't create a cache — it must be provided externally.
        _dynamic_cache = None
        if use_cache:
            try:
                from transformers.cache_utils import DynamicCache as _DynCache
                if past_key_values is None:
                    _dynamic_cache = _DynCache()
                elif isinstance(past_key_values, _DynCache):
                    _dynamic_cache = past_key_values
            except ImportError:
                pass

        # Pre-compute past sequence length BEFORE any block modifies the cache
        _step_past_len = 0
        if _dynamic_cache is not None:
            _step_past_len = _dynamic_cache.get_seq_length()
        elif past_key_values:
            try:
                _step_past_len = past_key_values[0][0][0].shape[-2]
            except (IndexError, TypeError, AttributeError):
                pass

        # Build 4D causal attention mask.
        # When calling decoder layers directly (bypassing MistralModel.forward()),
        # no causal mask is created — tokens can attend to future positions.
        # Shape: [batch, 1, seq_len, past_len + seq_len]
        _seq_len = hidden_states.shape[1]
        _total_len = _step_past_len + _seq_len
        _causal_mask = None
        if _seq_len > 1:
            # Prefill: build full causal mask (only once per generation)
            try:
                _causal_mask = pt.full(
                    (_total_len, _total_len), float("-inf"),
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )
                _causal_mask = pt.triu(_causal_mask, diagonal=1)
                _causal_mask = _causal_mask[_step_past_len:_step_past_len + _seq_len, :_total_len]
                _causal_mask = _causal_mask.unsqueeze(0).unsqueeze(0)
            except Exception:
                _causal_mask = None
        # Decode (seq_len=1): single token attends to all prior — no mask needed.
        # This eliminates an O(N^2) allocation per token.

        # 2. Process blocks sequentially
        if "blocks" not in self._comp_devices:
            self._comp_devices["blocks"] = {}
            
        # Lazily create a transfer stream for async cross-GPU copies
        if not hasattr(self, '_transfer_streams'):
            self._transfer_streams = {}

        for idx, block in enumerate(self.blocks):
            # Dynamic device resolution for the block
            if idx not in self._comp_devices["blocks"]:
                try:
                    self._comp_devices["blocks"][idx] = next(block.parameters()).device
                except StopIteration:
                    self._comp_devices["blocks"][idx] = None
            block_dev = self._comp_devices["blocks"][idx]
            
            if block_dev is not None and hidden_states.device != block_dev:
                # Async cross-GPU transfer via dedicated stream
                _dst_idx = block_dev.index if hasattr(block_dev, 'index') else None
                if _dst_idx is not None and pt.cuda.is_available():
                    if _dst_idx not in self._transfer_streams:
                        self._transfer_streams[_dst_idx] = pt.cuda.Stream(device=_dst_idx)
                    _ts = self._transfer_streams[_dst_idx]
                    with pt.cuda.stream(_ts):
                        hidden_states = hidden_states.to(block_dev, non_blocking=True)
                    # Sync before compute on the target device's default stream
                    pt.cuda.current_stream(block_dev).wait_stream(_ts)
                else:
                    hidden_states = hidden_states.to(block_dev)
            elif self.block_devices and idx > 0:
                # Fallback to static mapping if dynamic fails
                prev_gpu = self.block_devices[idx - 1]
                curr_gpu = self.block_devices[idx]
                if prev_gpu != curr_gpu:
                    hidden_states = self._transfer_to_device(hidden_states, prev_gpu, curr_gpu)

            # Cache: DynamicCache is shared across all blocks; legacy is per-block
            if _dynamic_cache is not None:
                block_past = _dynamic_cache
                past_length = _step_past_len
            else:
                block_past = past_key_values[idx] if past_key_values else None
                past_length = 0
                if block_past is not None:
                    try:
                        past_length = block_past[0][0].shape[-2]
                    except (IndexError, TypeError, AttributeError):
                        pass

            seq_length = hidden_states.shape[1]
            position_ids = pt.arange(past_length, past_length + seq_length, dtype=pt.long, device=hidden_states.device).unsqueeze(0)

            # Compute rotary position_embeddings from model-level rotary_emb
            # (required for Mistral/Llama with transformers >= 4.46)
            position_embeddings = None
            if comp.get("rotary_emb") is not None:
                try:
                    # Find rotary_emb device from parameters OR buffers
                    # (MistralRotaryEmbedding uses inv_freq as a buffer, not a parameter)
                    rotary_dev = None
                    for p in comp["rotary_emb"].parameters():
                        rotary_dev = p.device; break
                    if rotary_dev is None:
                        for b in comp["rotary_emb"].buffers():
                            rotary_dev = b.device; break
                    if rotary_dev is None:
                        rotary_dev = hidden_states.device
                    r_pos = position_ids.to(rotary_dev)
                    r_hid = hidden_states.to(rotary_dev)
                    pe = comp["rotary_emb"](r_hid, r_pos)
                    # Move cos/sin to block device if needed
                    position_embeddings = tuple(t.to(hidden_states.device) for t in pe)
                except Exception as _re:
                    import logging as _lg
                    _lg.getLogger(__name__).warning("rotary_emb failed: %s", _re)

            # Move causal mask to block device if needed
            _block_mask = _causal_mask
            if _block_mask is not None and _block_mask.device != hidden_states.device:
                _block_mask = _block_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

            hidden_states, presents = block(
                hidden_states,
                past_key_values=block_past,
                use_cache=use_cache,
                attention_mask=_block_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            if use_cache and _dynamic_cache is None:
                all_presents.append(presents)
                
        # 3. Final layer norm (Dynamic device)
        if comp["final_norm"] is not None:
            if "final_norm" not in self._comp_devices:
                try:
                    self._comp_devices["final_norm"] = next(comp["final_norm"].parameters()).device
                except StopIteration:
                    self._comp_devices["final_norm"] = first_gpu_dev
            norm_dev = self._comp_devices["final_norm"]
            if norm_dev is not None:
                hidden_states = hidden_states.to(norm_dev)
            hidden_states = comp["final_norm"](hidden_states)
            
        # 4. LM head → logits (Dynamic device)
        if comp["lm_head"] is not None:
            if "lm_head" not in self._comp_devices:
                try:
                    self._comp_devices["lm_head"] = next(comp["lm_head"].parameters()).device
                except StopIteration:
                    self._comp_devices["lm_head"] = first_gpu_dev
            head_dev = self._comp_devices["lm_head"]
            if head_dev is not None:
                hidden_states = hidden_states.to(head_dev)
            logits = comp["lm_head"](hidden_states)
        elif comp["embed"] is not None:
            # tied weights fallback if head was missing
            # use embed_dev
            if embed_dev is not None:
                hidden_states = hidden_states.to(embed_dev)
            logits = comp["embed"](hidden_states)
        
        if use_cache:
            return logits, _dynamic_cache if _dynamic_cache is not None else all_presents
        return logits

    def _get_device(self):
        if hasattr(self, "_cached_device"):
            return self._cached_device
        if self.model is not None:
            import torch as _pt
            try:
                self._cached_device = next(self.model.parameters()).device
                return self._cached_device
            except StopIteration:
                self._cached_device = getattr(self.model, "device", _pt.device("cpu"))
                return self._cached_device
        import torch as _pt
        return _pt.device("cpu")

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text from a prompt using the loaded model and tokenizer.

        Three code paths:
          1. No split (single GPU/CPU) → native model.generate() with KV cache
          2. Multi-GPU pipeline with KV-cache → embed → blocks → head, incremental
          3. Fallback → auto-regressive loop without KV-cache
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non disponible")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.get("attention_mask", None)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Path 1: No split — use native generate() (already uses KV cache)
        if self.blocks is None or len(self.blocks) <= 1:
            if self.model is not None:
                # Move correctly to model's execution device
                device = self._get_device()
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
            new_tokens = out_ids[0][input_ids.shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Path 2: Multi-GPU pipeline-parallel with KV-cache
        do_sample = kwargs.get('do_sample', None)
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 0)
        top_p = kwargs.get('top_p', 1.0)

        # Determine if we should sample or use greedy argmax.
        # Greedy when: do_sample=False explicitly, or no sampling params changed.
        _greedy = (do_sample is False) or (
            do_sample is None and temperature == 1.0
            and top_k == 0 and top_p == 1.0
        )

        generated = input_ids
        past_blocks = None  # KV-cache per block

        for step in range(max_new_tokens):
            # Incremental: only feed last token after first step
            if past_blocks is not None:
                step_input = generated[:, -1:]
            else:
                step_input = generated

            # Forward with KV-cache
            result = self.infer(step_input, past_key_values=past_blocks, use_cache=True)
            if isinstance(result, tuple):
                logits, past_blocks = result
            else:
                logits = result
                past_blocks = None

            # Take last token logits
            if logits.dim() >= 2:
                next_logits = logits[:, -1, :]
            else:
                next_logits = logits

            # Sampling (optimized: fused Triton kernel when available)
            if _greedy:
                next_token = _torch.argmax(next_logits, dim=-1, keepdim=True)
            elif _HAS_FUSED_SAMPLE:
                next_token = _fused_sample(
                    next_logits, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                )
            else:
                if temperature > 0 and temperature != 1.0:
                    next_logits = next_logits / temperature

                if top_k > 0 and top_k < next_logits.size(-1):
                    topk_vals, _ = _torch.topk(next_logits, top_k)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = _torch.sort(next_logits, descending=True)
                    cumulative_probs = _torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove_mask = cumulative_probs - _torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove_mask] = float('-inf')
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = _torch.softmax(next_logits, dim=-1)
                next_token = _torch.multinomial(probs, num_samples=1)

            # Move back to first device for concatenation
            next_token = next_token.to(generated.device)
            generated = _torch.cat([generated, next_token], dim=-1)

            # Stop on EOS
            if self.tokenizer.eos_token_id is not None:
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        new_tokens = generated[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Yield tokens one at a time for real streaming.

        Uses KV-cache for both single-GPU (native model forward) and
        multi-GPU (pipeline-parallel via infer()) paths.
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non disponible")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        if _HAS_TORCH and _torch.cuda.is_available() and self.block_devices:
            try:
                device = f"cuda:{self.block_devices[0]}"
                input_ids = input_ids.to(device)
                if self.model is not None and not self.blocks:
                    if not (hasattr(self.model, 'hf_device_map') and self.model.hf_device_map):
                        self.model = self.model.to(device)
            except Exception as e:
                logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)
        elif _HAS_TORCH and self.model is not None:
            try:
                device = self._get_device()
                input_ids = input_ids.to(device)
            except Exception:
                pass

        generated = input_ids
        past_key_values = None
        prev_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        use_multi_gpu = (self.blocks is not None and len(self.blocks) > 1
                         and self._components is not None)

        for step in range(max_new_tokens):
            if past_key_values is not None and step > 0:
                step_input = generated[:, -1:]
            else:
                step_input = generated

            if use_multi_gpu:
                # Multi-GPU path with KV-cache
                result = self.infer(step_input, past_key_values=past_key_values,
                                    use_cache=True)
                if isinstance(result, tuple):
                    logits, past_key_values = result
                else:
                    logits = result
                    past_key_values = None
            else:
                # Single-GPU path with native model forward
                try:
                    model_output = self.model(
                        step_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = model_output.logits if hasattr(model_output, 'logits') else model_output
                    past_key_values = getattr(model_output, 'past_key_values', None)
                except (TypeError, AttributeError):
                    logits = self.infer(generated)
                    past_key_values = None

            if logits.dim() >= 2:
                next_logits = logits[:, -1, :]
            else:
                next_logits = logits

            next_token = _torch.argmax(next_logits, dim=-1, keepdim=True)
            next_token = next_token.to(generated.device)
            generated = _torch.cat([generated, next_token], dim=-1)

            # Decode incrementally
            cur_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = cur_text[len(prev_text):]
            prev_text = cur_text

            if new_text:
                yield new_text

            if self.tokenizer.eos_token_id is not None:
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        """True batched generation — pads prompts and runs a single forward pass.

        For single-GPU (no split), uses HuggingFace generate() with padding.
        For multi-GPU split models, falls back to sequential (blockwise
        forward doesn't support variable-length batch dims easily).
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non disponible")

        if not prompts:
            return []

        # Multi-GPU split: fallback to sequential (block-wise forward is
        # per-sample due to variable activation shapes)
        if self.blocks and len(self.blocks) > 1:
            return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]

        # Single model — true batch with left-padding
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Move to device
        if _HAS_TORCH and _torch.cuda.is_available() and self.block_devices:
            try:
                device = f"cuda:{self.block_devices[0]}"
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                if not next(self.model.parameters()).is_cuda:
                    self.model = self.model.to(device)
            except Exception as e:
                logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        out_ids = self.model.generate(input_ids, **gen_kwargs)

        results = []
        for i in range(len(prompts)):
            # Decode only newly generated tokens
            new_tokens = out_ids[i][input_ids.shape[1]:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        return results

# ------------------- vLLM Backend -------------------
# ------------------- External backends (extracted) -------------------
__all__ = [
    'select_backend', 'BaseLLMBackend', 'HuggingFaceBackend',
    'KVCacheBlock',
    '_extract_model_components', '_get_submodule',
]
