"""HuggingFace Hub GGUF utility loader.

**Note** : ce module fournit des helpers de téléchargement/cache GGUF.
Le backend d'inférence GGUF de production est `core/backends_llamacpp.py`
(classe `LlamaCppBackend`, enregistrée dans `select_backend()`).
"""
import gc
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

log = logging.getLogger("vramancer.llama_backend")

CACHE_DIR = Path.home() / ".cache" / "vramancer" / "gguf"

# Maps HuggingFace repo → suggested filename for common models
KNOWN_FILES = {
    "unsloth/Qwen3-Coder-Next-GGUF": "Qwen3-Coder-Next-UD-Q3_K_XL.gguf",
    "bartowski/Qwen_Qwen3-Coder-Next-GGUF": "Qwen3-Coder-Next-Q3_K_M.gguf",
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
}


def download_gguf(repo_id: str, filename: str) -> str:
    """Download a GGUF file from HuggingFace hub; return local path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        log.info("Downloading %s / %s …", repo_id, filename)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CACHE_DIR),
        )
        log.info("GGUF cached at %s", path)
        return path
    except Exception as exc:
        raise RuntimeError(f"Failed to download {repo_id}/{filename}: {exc}") from exc


def _get_tensor_split(num_gpus: int) -> Optional[List[float]]:
    """Return VRAM-proportional split list for llama.cpp tensor_split."""
    if num_gpus <= 1:
        return None
    try:
        import torch
        vrams = []
        for i in range(min(num_gpus, torch.cuda.device_count())):
            _, total = torch.cuda.mem_get_info(i)
            vrams.append(round(total / 1e9, 2))
        if len(vrams) >= 2:
            log.info("llama tensor_split: %s GB", vrams)
            return vrams
    except Exception as exc:
        log.warning("Could not probe VRAM for tensor_split: %s", exc)
    return None


def _detect_chat_format(repo_id: str) -> Optional[str]:
    """Guess the right chat format from the repo name."""
    repo_lower = repo_id.lower()
    if "qwen" in repo_lower:
        return "chatml"
    if "mistral" in repo_lower or "mixtral" in repo_lower:
        return "mistral-instruct"
    if "llama" in repo_lower:
        return "llama-3"
    if "deepseek" in repo_lower:
        return "chatml"
    return None  # llama-cpp-python will try to auto-detect from GGUF metadata


class LlamaBackend:
    """Thin wrapper around llama-cpp-python for GGUF inference.

    Supports:
      - Multi-GPU tensor split
      - Streaming via llama-cpp native stream=True
      - Chat completion using native chat template from GGUF metadata
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        tensor_split: Optional[List[float]] = None,
        n_ctx: int = 16384,
        chat_format: Optional[str] = None,
        verbose: bool = False,
    ):
        from llama_cpp import Llama

        log.info(
            "LlamaBackend: loading %s  gpu_layers=%d  tensor_split=%s  n_ctx=%d",
            model_path, n_gpu_layers, tensor_split, n_ctx,
        )
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=verbose,
        )
        self.model_path = model_path
        self.n_ctx = n_ctx
        log.info("LlamaBackend ready")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        filename: str,
        num_gpus: int = 2,
        n_ctx: int = 16384,
        verbose: bool = False,
    ) -> "LlamaBackend":
        """Download GGUF from HuggingFace and load with auto tensor split."""
        # Use pre-existing local path if the file is already on disk
        local_path = _find_cached(repo_id, filename)
        if local_path is None:
            local_path = download_gguf(repo_id, filename)

        tensor_split = _get_tensor_split(num_gpus)
        chat_format = _detect_chat_format(repo_id)

        return cls(
            model_path=local_path,
            n_gpu_layers=-1,
            tensor_split=tensor_split,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=verbose,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )
        return out["choices"][0]["text"]

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        for chunk in self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    def chat_stream(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Iterator[str]:
        """Stream using the native chat completion (respects GGUF chat template)."""
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield text

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> str:
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return out["choices"][0]["message"]["content"]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self):
        try:
            del self.llm
        except Exception:
            log.debug("llama_cpp model cleanup failed", exc_info=True)
        gc.collect()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_cached(repo_id: str, filename: str) -> Optional[str]:
    """Check if the GGUF is already in the HuggingFace hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(repo_id=repo_id, filename=filename)
        if path and os.path.isfile(path):
            log.info("GGUF already cached: %s", path)
            return path
    except Exception:
        pass
    return None
