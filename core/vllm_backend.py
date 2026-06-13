"""**Legacy alias** — superseded by `core/backends_vllm.py`.

`core/backends.py:select_backend()` imports from `core.backends_vllm`.
Ce fichier est conservé pour compatibilité d'import ascendante. Ne pas étendre.

vLLM backend for high-throughput serving.

Supports:
- NVFP4 on Blackwell GPUs (RTX 50xx, H100)
- Continuous batching via PagedAttention
- Tensor parallel across multiple GPUs
- FP8, GPTQ, AWQ, NVFP4 quantization formats
"""
import logging
from typing import Iterator, List, Optional

log = logging.getLogger("vramancer.vllm_backend")


class VLLMBackend:
    """Wrapper around vLLM LLM for OpenAI-compatible serving."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,  # "fp8", "nvfp4", "gptq", "awq"
        dtype: str = "auto",
        verbose: bool = False,
    ):
        try:
            from vllm import LLM, SamplingParams as _SP
        except ImportError:
            raise RuntimeError(
                "vLLM not installed. Run: pip install vllm"
            )

        log.info(
            "VLLMBackend: loading %s  tp=%d  quant=%s  gpu_mem=%.0f%%  max_len=%s",
            model, tensor_parallel_size, quantization,
            gpu_memory_utilization * 100, max_model_len,
        )

        kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=True,
        )
        if quantization:
            kwargs["quantization"] = quantization
        if max_model_len:
            kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**kwargs)
        self.model_name = model
        self._SamplingParams = _SP
        log.info("VLLMBackend ready")

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model: str,
        num_gpus: int = 2,
        quantization: str = "",
        max_model_len: int = 16384,
    ) -> "VLLMBackend":
        """Auto-configure based on GPU count and quantization."""
        # For NVFP4 on mixed Ampere+Blackwell, reduce context to fit in VRAM
        _max_len = max_model_len
        if quantization == "nvfp4":
            _max_len = min(max_model_len, 8192)
            log.info("NVFP4: capping max_model_len=%d to fit in VRAM", _max_len)

        return cls(
            model=model,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.92,
            max_model_len=_max_len,
            quantization=quantization or None,
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
        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> str:
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        outputs = self.llm.chat(messages, params)
        return outputs[0].outputs[0].text

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """vLLM streaming via RequestOutput iteration."""
        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )
        prev_len = 0
        for output in self.llm.generate([prompt], params, use_tqdm=False):
            text = output.outputs[0].text
            new = text[prev_len:]
            if new:
                yield new
            prev_len = len(text)

    def chat_stream(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> Iterator[str]:
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prev_len = 0
        for output in self.llm.chat(messages, params, use_tqdm=False):
            text = output.outputs[0].text
            new = text[prev_len:]
            if new:
                yield new
            prev_len = len(text)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def shutdown(self):
        try:
            import torch, gc
            del self.llm
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            log.debug("vLLM model cleanup failed", exc_info=True)
