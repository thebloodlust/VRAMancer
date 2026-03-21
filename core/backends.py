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
            is_quantized = self._should_use_nvfp4()
            for gpu in config.gpus:
                # Use FREE VRAM (not total) to account for driver/OS usage.
                # max_memory is a per-GPU cap — accelerate handles placement.
                base_vram = gpu.free_vram_gb if gpu.free_vram_gb > 0 else gpu.total_vram_gb
                # BnB 4-bit: accelerate plans NF4 sizes, but during loading
                # safetensor shards are held in fp16 transiently.  If one GPU's
                # budget >= total model NF4 size, accelerate puts EVERYTHING
                # on that GPU → fp16 transient causes OOM.  Use 60% so no
                # single GPU can hold the whole model, FORCING a spread.
                # Non-quantized: 80% headroom for KV cache + activations.
                reserve = 0.60 if is_quantized else 0.80
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

    def _should_use_nvfp4(self) -> bool:
        """Check if NVFP4 quantization should be used.

        Conditions:
          - VRM_QUANTIZATION=nvfp4 env var set (explicit opt-in)
          - At least one Blackwell GPU (CC >= 12.0)
          - bitsandbytes is installed
        """
        quant = os.environ.get("VRM_QUANTIZATION", "").lower()
        if quant != "nvfp4":
            return False
        if not _HAS_TORCH or not _torch.cuda.is_available():
            return False
        has_blackwell = False
        for i in range(_torch.cuda.device_count()):
            props = _torch.cuda.get_device_properties(i)
            if props.major >= 12:
                has_blackwell = True
                break
        if not has_blackwell:
            self.log.warning("VRM_QUANTIZATION=nvfp4 requested but no Blackwell GPU (CC>=12) found")
            return False
        try:
            import bitsandbytes  # noqa: F401
            return True
        except ImportError:
            self.log.warning(
                "VRM_QUANTIZATION=nvfp4 requested but bitsandbytes not installed. "
                "Install with: pip install bitsandbytes>=0.43.0"
            )
            return False

    def _build_bnb_nvfp4_config(self):
        """Build BitsAndBytesConfig for NVFP4 (4-bit NormalFloat) quantization.

        Uses NF4 quantization with bfloat16 compute dtype — the Blackwell
        native FP4 format. A 7B model goes from ~14 GB (BF16) to ~4 GB (NF4).
        """
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Allow small modules (embed, lm_head) to live on CPU in fp32
            # if they don't fit on GPUs.  Without this BnB crashes when
            # accelerate dispatches ANY module to CPU.
            llm_int8_enable_fp32_cpu_offload=True,
        )
        self.log.info(
            "NVFP4 quantization enabled (NF4 + double quant + bfloat16 compute) — "
            "model size reduced ~75%%"
        )
        return bnb_config

    def load_model(self, model_name: str, **kwargs):
        if os.environ.get("VRM_MINIMAL_TEST") == "1" or model_name.startswith("stub"):
            self.model_name = model_name
            self.model = "STUB_MODEL"
            self.tokenizer = "STUB_TOKENIZER"
            return self.model

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
        if model_name.endswith("-AWQ"):
            kwargs["low_cpu_mem_usage"] = True
        elif "gptq" in model_name.lower():
            # GPTQ: pre-quantized model — weights are already INT4/INT8.
            # No runtime quantization needed (no BnB, no fp16 transient).
            # Uses transformers' built-in GPTQConfig with use_exllama=False
            # so no compiled CUDA kernels (auto-gptq) are needed.
            # Requires: pip install optimum
            self.log.info("GPTQ model detected — loading pre-quantized (no BnB needed)")
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
                kwargs["torch_dtype"] = best_dtype
            # Explicit GPTQConfig: disable exllama kernels to avoid
            # needing auto-gptq compiled CUDA extensions.
            # Pure PyTorch dequantization — slower but works everywhere.
            if "quantization_config" not in kwargs:
                try:
                    from transformers import GPTQConfig
                    kwargs["quantization_config"] = GPTQConfig(
                        bits=4,
                        use_exllama=False,
                    )
                    self.log.info("GPTQ: using PyTorch dequant (no exllama kernels)")
                except ImportError:
                    self.log.info("GPTQ: GPTQConfig not available, relying on model config")
        else:
            is_quantized = self._should_use_nvfp4()
            num_gpus = _torch.cuda.device_count() if _HAS_TORCH and _torch.cuda.is_available() else 1

            kwargs["device_map"] = "auto"

            if is_quantized and num_gpus >= 2:
                # BnB 4-bit multi-GPU: during from_pretrained, weights are
                # loaded in fp16 onto each GPU BEFORE being quantized to NF4.
                # infer_auto_device_map plans using NF4 sizes (~0.56 B/param)
                # but loading uses fp16 (2 B/param) — a ~3.6x mismatch.
                # If max_memory is too generous, one GPU gets assigned the
                # entire model in NF4 but can't hold it in fp16 during loading.
                #
                # Fix: set max_memory to ~25% of free VRAM.  This ensures the
                # fp16 equivalent of assigned NF4 modules fits during loading.
                # Overflow modules go to CPU in fp32, kept functional via
                # llm_int8_enable_fp32_cpu_offload=True in the BnB config.
                max_memory = {}
                for i in range(num_gpus):
                    free_bytes = _torch.cuda.mem_get_info(i)[0]
                    free_gb = free_bytes / (1024 ** 3)
                    # 25% of free: NF4 planned → fp16 actual ≈ 4x → fits in free
                    budget = max(2.0, free_gb * 0.25)
                    max_memory[i] = f"{budget:.1f}GiB"
                max_memory["cpu"] = "48GiB"
                kwargs["max_memory"] = max_memory
                kwargs["low_cpu_mem_usage"] = True
                self.log.info(
                    "BnB 4-bit multi-GPU: max_memory=%s "
                    "(conservative for fp16 loading transient)",
                    {k: v for k, v in max_memory.items() if k != "cpu"},
                )
            else:
                # Non-quantized: compute-aware placement with generous budgets
                max_memory = self._build_compute_aware_memory_map()
                if max_memory:
                    kwargs["max_memory"] = max_memory

            # NVFP4 quantization: 4-bit NormalFloat for Blackwell GPUs
            if is_quantized and "quantization_config" not in kwargs:
                kwargs["quantization_config"] = self._build_bnb_nvfp4_config()
            # Auto-select optimal dtype based on best available GPU
            elif "torch_dtype" not in kwargs and "dtype" not in kwargs:
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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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
            
            # --- WEB GPU INTERCEPT (Mobile/Edge Node routing) ---
            if getattr(self, "webgpu_manager", None) is not None and len(self.webgpu_manager.clients) > 0 and idx == len(self.blocks) // 2:
                import torch as pt
                import asyncio
                self.log.info(f" [WebGPU] Offloading block {idx} to Edge Swarm (Mobile)...")
                tensor_bytes = x.detach().cpu().numpy().tobytes()
                # Compat with 'dispatch_layer' or 'submit_tensor'
                submit_fn = getattr(self.webgpu_manager, "dispatch_layer", getattr(self.webgpu_manager, "submit_tensor", None))
                if submit_fn:
                    future = submit_fn(layer_id=idx, tensor_data=tensor_bytes)
                    try:
                        # Since we are in sync land, we have to block on future (or if threadsafe wait)
                        if asyncio.iscoroutine(future):
                            try:
                                loop = asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                                    res_bytes = pool.submit(asyncio.run, future).result()
                            except RuntimeError:
                                res_bytes = asyncio.run(future)
                        else:
                            # Might be concurrent.futures.Future or asyncio.Future
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            if hasattr(future, "result") and not isinstance(future, asyncio.Future):
                                res_bytes = future.result(timeout=5.0)
                            else:
                                res_bytes = loop.run_until_complete(future)
                                
                        if res_bytes:
                            import numpy as np
                            # Reconstruct tensor (assuming same shape and dtype float32/16)
                            dt = x.cpu().numpy().dtype
                            arr = np.frombuffer(res_bytes, dtype=dt).reshape(x.shape)
                            x = pt.tensor(arr).to(x.device)
                            self.log.info(f" [WebGPU] Block {idx} returned from Edge.")
                            continue
                    except Exception as e:
                        self.log.warning(f" [WebGPU] Edge Swarm failed for block {idx}, reverting to local GPU: {e}")
            # ----------------------------------------------------

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
        try:
            _causal_mask = pt.full((_total_len, _total_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype)
            _causal_mask = pt.triu(_causal_mask, diagonal=1)
            # Slice to [seq_len, total_len] for the current step
            _causal_mask = _causal_mask[_step_past_len:_step_past_len + _seq_len, :_total_len]
            # Expand to 4D: [batch, 1, seq_len, total_len]
            _causal_mask = _causal_mask.unsqueeze(0).unsqueeze(0)
        except Exception:
            _causal_mask = None

        # 2. Process blocks sequentially
        if "blocks" not in self._comp_devices:
            self._comp_devices["blocks"] = {}
            
        for idx, block in enumerate(self.blocks):
            # Dynamic device resolution for the block
            if idx not in self._comp_devices["blocks"]:
                try:
                    self._comp_devices["blocks"][idx] = next(block.parameters()).device
                except StopIteration:
                    self._comp_devices["blocks"][idx] = None
            block_dev = self._comp_devices["blocks"][idx]
            
            if block_dev is not None and hidden_states.device != block_dev:
                hidden_states = hidden_states.to(block_dev)
            elif self.block_devices and idx > 0:
                # Fallback to static mapping if dynamic fails
                prev_gpu = self.block_devices[idx - 1]
                curr_gpu = self.block_devices[idx]
                if prev_gpu != curr_gpu:
                    hidden_states = self._transfer_to_device(hidden_states, prev_gpu, curr_gpu)
            
            # --- WEB GPU INTERCEPT (Mobile/Edge Node routing) ---
            if getattr(self, "webgpu_manager", None) is not None and len(self.webgpu_manager.clients) > 0 and idx == len(self.blocks) // 2:
                import asyncio
                import numpy as np
                self.log.info(f" [WebGPU] Offloading KV-cache block {idx} to Edge Swarm (Mobile)...")
                tensor_bytes = hidden_states.detach().cpu().numpy().tobytes()
                submit_fn = getattr(self.webgpu_manager, "dispatch_layer", getattr(self.webgpu_manager, "submit_tensor", None))
                if submit_fn:
                    future = submit_fn(layer_id=idx, tensor_data=tensor_bytes)
                    try:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        if asyncio.iscoroutine(future):
                            try:
                                asyncio.get_running_loop()
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                                    res_bytes = pool.submit(asyncio.run, future).result()
                            except RuntimeError:
                                res_bytes = asyncio.run(future)
                        else:
                            if hasattr(future, "result") and not isinstance(future, asyncio.Future):
                                res_bytes = future.result(timeout=5.0)
                            else:
                                res_bytes = loop.run_until_complete(future)
                        
                        if res_bytes:
                            dt = hidden_states.cpu().numpy().dtype
                            arr = np.frombuffer(res_bytes, dtype=dt)
                            # the dummy network might return a smaller/different size, just handle basic shape if exact
                            if arr.size == hidden_states.numel():
                                arr = arr.reshape(hidden_states.shape)
                                hidden_states = pt.tensor(arr).to(hidden_states.device)
                                self.log.info(f" [WebGPU] Block {idx} returned from Edge.")
                    except Exception as e:
                        self.log.warning(f" [WebGPU] Edge Swarm failed for block {idx}, reverting to local GPU: {e}")
            # ----------------------------------------------------

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
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 1.0)

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

            # Sampling
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

            if temperature > 0 and (temperature != 1.0 or top_p < 1.0 or top_k > 0):
                probs = _torch.softmax(next_logits, dim=-1)
                next_token = _torch.multinomial(probs, num_samples=1)
            else:
                next_token = _torch.argmax(next_logits, dim=-1, keepdim=True)

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
