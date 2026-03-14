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
                  "lm_head": None, "drop": None}
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
                attention_mask=None, position_ids=None):
        """Forward pass with KV-cache support.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            past_key_values: List of per-layer KV cache tuples, or None
            use_cache: Whether to return new KV cache
            attention_mask: Optional attention mask
            position_ids: Optional position IDs

        Returns:
            (hidden_states, presents) where presents is a list of KV tuples
            if use_cache=True, else None.
        """
        presents = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values else None

            # Base kwargs
            layer_kwargs = {
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }

            # Llama >= 4.38 / Qwen2 require precomputed position_embeddings
            if position_ids is not None and hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                try:
                    pos_emb = layer.self_attn.rotary_emb(hidden_states, position_ids)
                    layer_kwargs["position_embeddings"] = pos_emb
                except Exception:
                    pass

            # Try calling with KV cache kwargs (different HF model signatures)
            try:
                # Try Modern HF signature first (Llama > 4.36, Qwen2, past_key_values plural)
                output = layer(
                    hidden_states,
                    past_key_values=layer_past,
                    **layer_kwargs
                )
            except TypeError:
                try:
                    # Try Legacy Llama/Mistral signature (past_key_value singular)
                    output = layer(
                        hidden_states,
                        past_key_value=layer_past,
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
                if use_cache and len(output) > 1 and output[1] is not None:
                    presents.append(output[1])
                elif use_cache:
                    presents.append(None)
            else:
                hidden_states = output
                if use_cache:
                    presents.append(None)

        return hidden_states, presents


def select_backend(model_name: str, cache_dir: str = None, backend: str = "auto", num_gpus: int = 1):
    logger.info(f"Sélection du backend pour {model_name} (demandé: {backend}, gpus: {num_gpus})")
    
    # FORCER vLLM DANS TOUS LES CAS (auto ou vllm)
    if backend == "vllm" or backend == "auto":
        import vllm  # Lèvera une erreur stricte s'il n'est pas installé
        logger.info(f"Utilisation du backend vLLM pour le modèle {model_name}.")
        from core.backends_vllm import vLLMBackend
        return vLLMBackend(model_name, cache_dir=cache_dir, tensor_parallel_size=num_gpus)

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
        from core.backends import HuggingFaceBackend
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

    def load_model(self, model_name: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
        if model_name.endswith("-AWQ"):
            kwargs["low_cpu_mem_usage"] = True
        else:
            # Force 8-bit quantization for large models to fit in 16GB VRAM
            # Desactivated by default to avoid issues with new models (Qwen2, Llama3)
            # kwargs["load_in_8bit"] = True
            kwargs["device_map"] = "auto"
            # kwargs["torch_dtype"] = "float16" # Enable this if VRAM is an issue without load_in_8bit

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
                if self._components["final_norm"] is not None:
                    self._components["final_norm"] = self._components["final_norm"].to(last_dev)
                if self._components["lm_head"] is not None:
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
            out = self.model(inputs)
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
        import torch as pt
        all_presents = [] if use_cache else None
        
        comp = self._components
        first_gpu_dev = f"cuda:{self.block_devices[0]}" if (self.block_devices and pt.cuda.is_available()) else "cpu"
        
        # 1. Embeddings (on first GPU)
        inputs = inputs.to(first_gpu_dev)
        hidden_states = comp["embed"](inputs)
        
        if comp["pos_embed"] is not None:
            seq_len = inputs.shape[1]
            past_len = past_key_values[0][0][0].shape[-2] if past_key_values and past_key_values[0] else 0
            pos = pt.arange(past_len, past_len + seq_len, dtype=pt.long, device=first_gpu_dev).unsqueeze(0)
            hidden_states = hidden_states + comp["pos_embed"](pos)
            
        if comp["drop"] is not None:
            hidden_states = comp["drop"](hidden_states)

        # 2. Process blocks sequentially
        for idx, block in enumerate(self.blocks):
            if self.block_devices and idx > 0:
                prev_gpu = self.block_devices[idx - 1]
                curr_gpu = self.block_devices[idx]
                if prev_gpu != curr_gpu:
                    hidden_states = self._transfer_to_device(hidden_states, prev_gpu, curr_gpu)
            
            block_past = past_key_values[idx] if past_key_values else None
            
            seq_length = hidden_states.shape[1]
            past_length = 0
            if block_past is not None and len(block_past) > 0 and block_past[0] is not None:
                past_length = block_past[0][0].shape[-2]
                
            position_ids = pt.arange(past_length, past_length + seq_length, dtype=pt.long, device=hidden_states.device).unsqueeze(0)

            hidden_states, presents = block(
                hidden_states,
                past_key_values=block_past,
                use_cache=use_cache,
                position_ids=position_ids
            )
            
            if use_cache:
                all_presents.append(presents)
                
        # 3. Final layer norm (on last GPU)
        if comp["final_norm"] is not None:
            hidden_states = comp["final_norm"](hidden_states)
            
        # 4. LM head → logits (on last GPU)
        logits = comp["lm_head"](hidden_states)
        
        if use_cache:
            return logits, all_presents
        return logits

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
                device = getattr(self.model, "device", _torch.device("cpu"))
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
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

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
                next_logits = logits[:, -1, :]                curl -X POST "http://192.168.1.21:5000/api/models/load" \
                     -H "Content-Type: application/json" \
                     -H "Authorization: Bearer testtoken" \
                     -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "num_gpus": 2, "backend": "huggingface"}'                    curl -X POST http://192.168.1.21:5000/api/models/load \
                         -H "Content-Type: application/json" \
                         -H "Authorization: Bearer testtoken" \
                         -d '{"model": "Qwen/Qwen2.5-14B-Instruct", "gpu_memory_utilization": 0.50}'
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

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

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
                    self.model = self.model.to(device)
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
            except Exception:
                pass

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
# vLLMBackend and OllamaBackend are extracted to separate files for
# maintainability. Re-imported here for backward compatibility.
# Note: BaseLLMBackend must be defined above before these imports.
from core.backends_vllm import vLLMBackend  # noqa: E402
from core.backends_ollama import OllamaBackend  # noqa: E402

__all__ = [
    'select_backend', 'BaseLLMBackend', 'HuggingFaceBackend',
    'vLLMBackend', 'OllamaBackend', 'KVCacheBlock',
    '_extract_model_components', '_get_submodule',
]
