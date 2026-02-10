"""Unified LLM backend abstraction.

Provides:
- Factory ``select_backend()`` with fallback stub (``VRM_BACKEND_ALLOW_STUB=1``).
- ``HuggingFaceBackend``: real model loading, VRAM-proportional multi-GPU split
  via ``model_splitter``, sequential block inference with ``TransferManager``.
- ``vLLMBackend`` / ``OllamaBackend``: real integration if lib present, stub otherwise.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import hashlib
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


def select_backend(backend_name: str = "auto") -> "BaseLLMBackend":
    backend_name = (backend_name or "auto").lower()
    allow_stub = os.environ.get("VRM_BACKEND_ALLOW_STUB")
    if backend_name == "huggingface":
        return HuggingFaceBackend()
    if backend_name == "vllm":
        try:
            import vllm  # noqa: F401
            return vLLMBackend(real=True)
        except ImportError:
            if allow_stub:
                return vLLMBackend(real=False)
            raise RuntimeError("vLLM non installé (export VRM_BACKEND_ALLOW_STUB=1 pour stub)")
    if backend_name == "ollama":
        try:
            import ollama  # noqa: F401
            return OllamaBackend(real=True)
        except ImportError:
            if allow_stub:
                return OllamaBackend(real=False)
            raise RuntimeError("Ollama non installé (export VRM_BACKEND_ALLOW_STUB=1 pour stub)")
    # auto
    try:
        import vllm  # noqa: F401
        return vLLMBackend(real=True)
    except ImportError:
        if allow_stub:
            return vLLMBackend(real=False)
    try:
        import ollama  # noqa: F401
        return OllamaBackend(real=True)
    except ImportError:
        if allow_stub:
            return OllamaBackend(real=False)
    return HuggingFaceBackend()


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
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.tokenizer = None
        self.blocks: Optional[List[Any]] = None
        self.block_devices: Optional[List[int]] = None  # GPU id per block
        self.log = LoggerAdapter("backend.hf")
        self.hmem = None  # reference injectée optionnelle
        self.transfer_manager = None  # injecté par le pipeline

    def load_model(self, model_name: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
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
        # split_model_into_blocks accepte un model_name (str) — on passe
        # le modèle déjà chargé en utilisant la version qui prend des layers
        # directement via _extract_layers + _split_by_vram.
        from core.model_splitter import _extract_layers, _split_by_vram, _get_free_vram_per_gpu
        layers = _extract_layers(self.model)
        if layers is None:
            # Fallback: wrap entire model as a single block
            self.blocks = [_nn.Sequential(self.model)]
            self.block_devices = [0]
            return self.blocks
        if num_gpus <= 1:
            self.blocks = [_nn.Sequential(*layers)]
            self.block_devices = [0]
            return self.blocks
        # Profiler-based ou VRAM-proportional
        if vram_per_gpu:
            vram = vram_per_gpu[:num_gpus]
        else:
            vram = _get_free_vram_per_gpu(num_gpus)
        self.blocks = _split_by_vram(layers, vram)
        self.block_devices = list(range(len(self.blocks)))
        # Move blocks to their devices
        try:
            self.blocks = assign_blocks_to_gpus(self.blocks)
        except Exception as e:
            self.log.warning(f"Placement GPU échoué (CPU fallback): {e}")
        self.log.info(f"Modèle découpé en {len(self.blocks)} blocs: "
                      f"{[len(list(b.children())) for b in self.blocks]}")
        return self.blocks

    def infer(self, inputs: Any):
        """Forward séquentiel sur les blocs avec transfert inter-GPU.

        Si ``self.blocks`` n'est pas initialisé (pas de split), utilise
        directement ``self.model.forward()``.
        """
        if self.blocks is None:
            # Pas de split — inférence directe
            if self.model is None:
                raise RuntimeError("Modèle non chargé")
            out = self.model(inputs)
            if hasattr(out, "logits"):
                return out.logits
            return out

        x = inputs
        self.log.debug("Début inférence séquentielle sur %d blocs", len(self.blocks))

        for idx, block in enumerate(self.blocks):
            # Transfer activation to the right GPU if needed
            if self.transfer_manager and self.block_devices and idx > 0:
                prev_gpu = self.block_devices[idx - 1]
                curr_gpu = self.block_devices[idx]
                if prev_gpu != curr_gpu:
                    try:
                        if _HAS_TORCH and _torch.is_tensor(x) and x.is_cuda:
                            result = self.transfer_manager.send_activation(
                                prev_gpu, curr_gpu, x
                            )
                            x = x.to(f"cuda:{curr_gpu}")
                    except Exception as e:
                        self.log.warning(f"Transfer GPU {prev_gpu}→{curr_gpu} failed: {e}")
                        try:
                            x = x.to(f"cuda:{curr_gpu}")
                        except Exception:
                            pass

            out = block(x)
            # Handle HuggingFace model outputs
            if hasattr(out, "logits"):
                out = out.logits
            elif hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            x = out

            # Hook accès hiérarchique memory
            if self.hmem:
                try:
                    from core.memory_block import MemoryBlock
                    mb = MemoryBlock(size_mb=getattr(block, 'size_mb', 128),
                                     gpu_id=self.block_devices[idx] if self.block_devices else 0,
                                     status="allocated")
                    bid = hashlib.sha1(str(id(block)).encode()).hexdigest()
                    if bid not in self.hmem.registry:
                        mb.id = bid
                        self.hmem.register_block(mb, "L1")
                    else:
                        mb.id = bid
                    self.hmem.touch(mb)
                    self.hmem.promote_policy(mb)
                except Exception:
                    pass

        self.log.debug("Fin inférence")
        return x

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text from a prompt using the loaded model and tokenizer.

        Three code paths:
          1. No split (single GPU/CPU) → native model.generate() with KV cache
          2. Multi-GPU pipeline → block-by-block forward with inter-GPU transfer
          3. Fallback → sequential auto-regressive loop through infer()
        """
        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non disponible")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Move to first device if GPU available
        if _HAS_TORCH and _torch.cuda.is_available() and self.block_devices:
            try:
                device = f"cuda:{self.block_devices[0]}"
                input_ids = input_ids.to(device)
                if self.model is not None and not self.blocks:
                    self.model = self.model.to(device)
            except Exception:
                pass

        # Path 1: No split — use native generate() (already uses KV cache)
        if self.blocks is None or len(self.blocks) <= 1:
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
            return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        # Path 2: Multi-GPU pipeline-parallel auto-regressive loop
        # Each step: embedding → block_0(GPU_0) → transfer → block_1(GPU_1) → ... → lm_head → sample
        generated = input_ids
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 50)
        top_p = kwargs.get('top_p', 1.0)

        for step in range(max_new_tokens):
            # Forward through blocks using infer() which handles inter-GPU transfers
            logits = self.infer(generated)

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

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Yield tokens one at a time for real streaming.

        Uses the same auto-regressive loop as generate() but yields each
        decoded token as it is produced, enabling true SSE streaming.
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

        for step in range(max_new_tokens):
            if past_key_values is not None and step > 0:
                step_input = generated[:, -1:]
            else:
                step_input = generated

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

            # Decode incrementally: diff between current and previous text
            cur_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = cur_text[len(prev_text):]
            prev_text = cur_text

            if new_text:
                yield new_text

            # Stop on EOS
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

# ------------------- vLLM Backend (squelette) -------------------
class vLLMBackend(BaseLLMBackend):
    def __init__(self, real: bool = True):
        self.model = None
        self.log = LoggerAdapter("backend.vllm" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        if not self.real:
            self.model = {"name": model_name, "stub": True}
            return self.model
        try:
            from vllm import LLM
            self.model = LLM(model=model_name, **kwargs)
            return self.model
        except Exception as e:
            if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
                self.log.warning(f"Fallback stub vLLM: {e}")
                self.real = False
                self.model = {"name": model_name, "stub": True}
                return self.model
            raise

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        if not self.real:
            if _HAS_TORCH:
                return _torch.zeros_like(inputs)
            return inputs
        # Intégration réelle: utilisation API LLM.generate (texte)
        try:
            if isinstance(inputs, str):
                # Hypothèse: self.model = vllm.LLM
                out = self.model.generate([inputs], sampling_params=None)
                # vLLM renvoie une liste de RequestOutput
                return out[0].outputs[0].text if out and out[0].outputs else ""
            return inputs
        except Exception as e:
            self.log.warning(f"Infer vLLM échec: {e}")
            return ""

# ------------------- Ollama Backend (squelette) -------------------
class OllamaBackend(BaseLLMBackend):
    def __init__(self, real: bool = True):
        self.model = None
        self.log = LoggerAdapter("backend.ollama" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        if not self.real:
            self.model = {"name": model_name, "stub": True}
            return self.model
        try:
            import requests  # noqa: F401
            self.model = model_name
            return self.model
        except Exception as e:
            if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
                self.log.warning(f"Fallback stub Ollama: {e}")
                self.real = False
                self.model = {"name": model_name, "stub": True}
                return self.model
            raise

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle Ollama non chargé.")
        if not self.real:
            return {"text": "stub-ollama-output", "len_in": getattr(inputs, 'shape', '?')}
        # Intégration HTTP simple : POST /api/generate (selon spec publique Ollama locale)
        try:
            import requests
            prompt = inputs if isinstance(inputs, str) else str(inputs)
            resp = requests.post("http://localhost:11434/api/generate", json={"model": self.model, "prompt": prompt}, timeout=30)
            if resp.status_code == 200:
                # Ollama stream chunk by chunk; ici hypothèse d'une réponse jointe (simplifié)
                data = resp.json()
                return {"text": data.get('response',''), "model": self.model}
            return {"error": resp.status_code}
        except Exception as e:
            self.log.warning(f"Infer Ollama échec: {e}")
            return {"error": str(e)}
