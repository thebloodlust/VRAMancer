"""llama.cpp backend integration for VRAMancer.

Uses llama-cpp-python to leverage llama.cpp's optimized CUDA kernels
for GGUF model inference. This provides:
  - Native quantized GEMV via dp4a (INT8 dot product, ~2-3x faster than BnB NF4)
  - GGUF format support (Q4_K_M, Q5_K_M, Q6_K, Q8_0, etc.)
  - Multi-GPU layer splitting (n_gpu_layers=-1 for full offload)
  - Flash attention support
  - Native KV cache management (paged attention built-in)

Install: pip install llama-cpp-python
  With CUDA:  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
  With ROCm:  CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python

Usage:
  # Via API
  POST /api/models/load {"model": "/path/to/model.gguf", "backend": "llamacpp"}

  # Via CLI
  python -m core.production_api --model /path/to/model.gguf --backend llamacpp

  # Via pipeline
  pipeline = InferencePipeline(backend_name="llamacpp")
  pipeline.load("/path/to/model.gguf", n_gpu_layers=-1)
"""

import os
import logging
from typing import Any, List, Optional

from core.backends import BaseLLMBackend

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except ImportError:
    _HAS_LLAMA_CPP = False


class LlamaCppBackend(BaseLLMBackend):
    """llama.cpp backend via llama-cpp-python.

    Leverages GGUF quantized models with optimized CUDA kernels (dp4a INT8
    dot product) for 2-3x throughput improvement over PyTorch+BnB NF4.

    The model path can be:
      - A local .gguf file path
      - A HuggingFace repo ID with a GGUF file (auto-downloaded)
    """

    def __init__(self, model_name: str = None, cache_dir: str = None):
        self.model_name: Optional[str] = model_name
        self.cache_dir = cache_dir
        self.model = None          # Llama instance
        self.tokenizer = None      # Proxy tokenizer for pipeline compatibility
        self.backend_type = "llamacpp"
        self.is_loaded = False
        self._n_gpu_layers = -1    # -1 = offload all layers to GPU
        self._n_ctx = 4096
        self._flash_attn = True
        self._verbose = False

    def load_model(self, model_name: str, **kwargs) -> Any:
        """Load a GGUF model via llama-cpp-python.

        Parameters
        ----------
        model_name : str
            Path to .gguf file, or HuggingFace repo ID (e.g. "TheBloke/Mistral-7B-v0.1-GGUF").
        n_gpu_layers : int
            Number of layers to offload to GPU. -1 = all (default).
        n_ctx : int
            Context window size (default: 4096).
        flash_attn : bool
            Enable flash attention (default: True).
        num_gpus : int
            Number of GPUs (controls split_mode for multi-GPU).
        """
        self.model_name = model_name
        self._n_gpu_layers = int(kwargs.get("n_gpu_layers", -1))
        self._n_ctx = int(kwargs.get("n_ctx", 4096))
        self._flash_attn = bool(kwargs.get("flash_attn", True))
        num_gpus = int(kwargs.get("num_gpus", 1))

        # Stub mode for testing
        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            self.model = "LLAMACPP_STUB"
            self.tokenizer = _StubTokenizer()
            self.is_loaded = True
            logger.info("llama.cpp backend loaded in stub mode")
            return self.model

        if not _HAS_LLAMA_CPP:
            raise ImportError(
                "llama-cpp-python is not installed. Install with:\n"
                '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python\n'
                "Or for ROCm:\n"
                '  CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python'
            )

        # Resolve model path: local file or HuggingFace download
        model_path = self._resolve_model_path(model_name)

        # Multi-GPU: llama.cpp uses split_mode
        # 0 = single GPU, 1 = split layers across GPUs, 2 = split rows
        split_mode = 1 if num_gpus > 1 else None
        tensor_split = None

        if num_gpus > 1:
            # Auto-detect VRAM proportions for heterogeneous split
            tensor_split = self._compute_tensor_split(num_gpus)
            # Detect P2P capability — row-split (mode 2) only benefits with P2P
            split_mode = self._select_split_mode(num_gpus)
            logger.info(
                "llama.cpp multi-GPU: %d GPUs, split_mode=%d, tensor_split=%s",
                num_gpus, split_mode, tensor_split
            )

        logger.info(
            "Loading GGUF model: %s (n_gpu_layers=%d, n_ctx=%d, flash_attn=%s)",
            model_path, self._n_gpu_layers, self._n_ctx, self._flash_attn
        )

        llama_kwargs = dict(
            model_path=model_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            flash_attn=self._flash_attn,
            verbose=self._verbose,
        )
        if split_mode is not None:
            llama_kwargs["split_mode"] = split_mode
        if tensor_split is not None:
            llama_kwargs["tensor_split"] = tensor_split

        self.model = Llama(**llama_kwargs)
        self.tokenizer = _LlamaCppTokenizerProxy(self.model)
        self.is_loaded = True

        logger.info("llama.cpp model loaded successfully: %s", model_name)
        return self.model

    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None) -> List[Any]:
        """llama.cpp handles multi-GPU internally via tensor_split."""
        if self.model is None:
            raise RuntimeError("Modèle llama.cpp non chargé.")
        logger.info("llama.cpp gère le split GPU en interne (%d GPUs).", num_gpus)
        return [self.model]

    def infer(self, inputs: Any) -> Any:
        """Raw tensor inference — not applicable for llama.cpp."""
        if not self.is_loaded:
            raise RuntimeError("Modèle llama.cpp non chargé.")
        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            return "llamacpp_infer_stub"
        # llama.cpp doesn't expose raw tensor inference
        # Convert to text generate as fallback
        prompt = inputs if isinstance(inputs, str) else str(inputs)
        return self.generate(prompt)

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text from a GGUF model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Modèle llama.cpp non chargé.")

        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            return f"[llamacpp-stub] {prompt[:50]}..."

        max_tokens = int(kwargs.get("max_tokens", max_new_tokens))
        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.95))
        top_k = int(kwargs.get("top_k", 40))
        repeat_penalty = float(kwargs.get("repeat_penalty", 1.1))

        if kwargs.get("stream", False):
            # Collect streamed tokens
            return "".join(self.generate_stream(
                prompt, max_new_tokens=max_new_tokens, **kwargs
            ))

        result = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            echo=False,
        )

        if result and "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("text", "")
        return ""

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Stream tokens from llama.cpp one at a time."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Modèle llama.cpp non chargé.")

        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            yield "[llamacpp-"
            yield "stub] "
            yield prompt[:20]
            return

        max_tokens = int(kwargs.get("max_tokens", max_new_tokens))
        temperature = float(kwargs.get("temperature", 0.7))
        top_p = float(kwargs.get("top_p", 0.95))
        top_k = int(kwargs.get("top_k", 40))
        repeat_penalty = float(kwargs.get("repeat_penalty", 1.1))

        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            echo=False,
            stream=True,
        ):
            if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                token = chunk["choices"][0].get("text", "")
                if token:
                    yield token

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        """Batch generation — sequential for llama.cpp (no native batching in python bindings)."""
        return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]

    # ---- Private helpers ----

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model_name to a local .gguf file path.

        If model_name is already a path to a .gguf file, return it.
        If it's a HuggingFace repo ID, download the GGUF file.
        """
        # Direct file path
        if os.path.isfile(model_name) and model_name.endswith(".gguf"):
            return model_name

        # Check in cache dir
        if self.cache_dir:
            candidate = os.path.join(self.cache_dir, os.path.basename(model_name))
            if os.path.isfile(candidate):
                return candidate

        # Try HuggingFace Hub download
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            # If model_name contains a filename (e.g. "user/repo/file.gguf")
            parts = model_name.split("/")
            if len(parts) >= 3 and parts[-1].endswith(".gguf"):
                repo_id = "/".join(parts[:2])
                filename = "/".join(parts[2:])
                logger.info("Downloading GGUF from HF: %s/%s", repo_id, filename)
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                )

            # Repo ID without specific file — find the best GGUF
            if len(parts) == 2:
                repo_id = model_name
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    raise FileNotFoundError(
                        f"No .gguf files found in HuggingFace repo: {repo_id}"
                    )

                # Prefer Q4_K_M > Q5_K_M > Q4_K_S > first available
                preferred = None
                for pattern in ["Q4_K_M", "Q5_K_M", "Q4_K_S", "q4_k_m", "q5_k_m"]:
                    for f in gguf_files:
                        if pattern in f:
                            preferred = f
                            break
                    if preferred:
                        break

                target_file = preferred or gguf_files[0]
                logger.info(
                    "Downloading GGUF from HF: %s/%s (%d GGUF files available)",
                    repo_id, target_file, len(gguf_files)
                )
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=target_file,
                    cache_dir=self.cache_dir,
                )
        except ImportError:
            pass  # huggingface_hub not available
        except Exception as e:
            logger.warning("HuggingFace GGUF download failed: %s", e)

        raise FileNotFoundError(
            f"Cannot resolve GGUF model: {model_name}. "
            "Provide a local .gguf file path or a HuggingFace repo ID "
            "(e.g. 'bartowski/Qwen2.5-7B-Instruct-GGUF')"
        )

    def _compute_tensor_split(self, num_gpus: int) -> Optional[List[float]]:
        """Compute VRAM-proportional tensor_split for heterogeneous GPUs.

        Weights by both free VRAM and compute capability (faster GPUs get
        more layers). Falls back to torch.cuda.mem_get_info() if
        hetero_config is unavailable.
        """
        vram_values = self._get_vram_per_gpu(num_gpus)
        if vram_values is None:
            return None

        # Apply compute weight: scale VRAM by relative FP16 throughput
        compute_weights = self._get_compute_weights(num_gpus)
        weighted = [v * w for v, w in zip(vram_values, compute_weights)]

        total = sum(weighted)
        if total <= 0:
            return None

        tensor_split = [w / total for w in weighted]
        logger.info("Tensor split (VRAM×compute weighted): %s", tensor_split)
        return tensor_split

    def _get_vram_per_gpu(self, num_gpus: int) -> Optional[List[float]]:
        """Get free VRAM per GPU in GB, with fallback chain."""
        # Try hetero_config first (rich GPU database)
        try:
            from core.hetero_config import auto_configure
            config = auto_configure(strategy="balanced")
            if config and len(config.gpus) >= num_gpus:
                vals = []
                for gpu in config.gpus[:num_gpus]:
                    v = gpu.free_vram_gb if gpu.free_vram_gb > 0 else gpu.total_vram_gb
                    vals.append(v)
                if all(v > 0 for v in vals):
                    return vals
        except Exception as e:
            logger.debug("hetero_config unavailable: %s", e)

        # Fallback: direct torch.cuda.mem_get_info
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus:
                vals = []
                for i in range(num_gpus):
                    free, total = torch.cuda.mem_get_info(i)
                    vals.append(free / (1024 ** 3))
                return vals
        except Exception as e:
            logger.debug("torch.cuda.mem_get_info unavailable: %s", e)

        # Fallback: pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count >= num_gpus:
                vals = []
                for i in range(num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vals.append(info.free / (1024 ** 3))
                return vals
        except Exception:
            pass

        logger.warning("Cannot detect GPU VRAM for tensor_split")
        return None

    def _get_compute_weights(self, num_gpus: int) -> List[float]:
        """Return relative compute weight per GPU (1.0 = baseline).

        Uses SM count × clock as a proxy for FP16 throughput.
        Falls back to equal weights if unavailable.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return [1.0] * num_gpus
            scores = []
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                score = props.multi_processor_count * props.clock_rate
                scores.append(score)
            # Normalize: fastest GPU = 1.0
            max_s = max(scores) if scores else 1
            return [s / max_s for s in scores]
        except Exception:
            return [1.0] * num_gpus

    def _select_split_mode(self, num_gpus: int) -> int:
        """Select llama.cpp split_mode: 1=layer split, 2=row split.

        Row split (mode 2) benefits from P2P/NVLink. Fall back to layer split
        (mode 1) when P2P is blocked (e.g. Proxmox VM with IOMMU).
        """
        if os.environ.get("VRM_TRANSFER_P2P", "").lower() in ("0", "false"):
            return 1  # forced no-P2P

        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus:
                for i in range(num_gpus):
                    for j in range(num_gpus):
                        if i != j and torch.cuda.can_device_access_peer(i, j):
                            logger.info("P2P available between GPU %d and %d → row split", i, j)
                            return 2
        except Exception:
            pass
        return 1  # default: layer split (safe without P2P)


class _LlamaCppTokenizerProxy:
    """Minimal tokenizer proxy for pipeline compatibility.

    The inference pipeline accesses backend.tokenizer.encode() and .decode().
    llama-cpp-python exposes tokenize()/detokenize() — this proxy bridges them.
    """

    def __init__(self, llama_model):
        self._model = llama_model

    def encode(self, text: str, return_tensors: str = None, **kwargs):
        """Tokenize text. Returns list of token IDs (or tensor if requested)."""
        tokens = self._model.tokenize(text.encode("utf-8"), add_bos=True)
        if return_tensors == "pt":
            try:
                import torch
                return torch.tensor([tokens])
            except ImportError:
                pass
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        """Decode token IDs to text."""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self._model.detokenize(token_ids).decode("utf-8", errors="replace")

    def __call__(self, text, return_tensors=None, **kwargs):
        """Make compatible with tokenizer(text, return_tensors="pt") pattern."""
        tokens = self.encode(text, return_tensors=return_tensors, **kwargs)
        if return_tensors == "pt":
            return type('TokenizerOutput', (), {'input_ids': tokens})()
        return tokens


class _StubTokenizer:
    """Minimal stub tokenizer for VRM_MINIMAL_TEST mode."""

    def encode(self, text, return_tensors=None, **kwargs):
        tokens = list(range(len(text.split())))
        if return_tensors == "pt":
            try:
                import torch
                return torch.tensor([tokens])
            except ImportError:
                pass
        return tokens

    def decode(self, token_ids, **kwargs):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return " ".join(str(t) for t in (token_ids[0] if isinstance(token_ids[0], list) else token_ids))

    def __call__(self, text, return_tensors=None, **kwargs):
        tokens = self.encode(text, return_tensors=return_tensors)
        if return_tensors == "pt":
            return type('TokenizerOutput', (), {'input_ids': tokens})()
        return tokens
