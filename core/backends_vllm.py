"""vLLM backend integration for VRAMancer.

Extracted from core/backends.py for maintainability.
Uses the vLLM engine with SamplingParams when available,
falls back to stub mode with VRM_BACKEND_ALLOW_STUB=1.
"""

import os
from typing import Any, List, Optional

from core.backends import BaseLLMBackend, _HAS_TORCH
from core.logger import LoggerAdapter

if _HAS_TORCH:
    import torch as _torch
else:
    _torch = None  # type: ignore


class vLLMBackend(BaseLLMBackend):
    """vLLM backend with real generate/stream support.

    When vllm is installed, uses the real LLM engine with SamplingParams.
    Falls back to stub mode if VRM_BACKEND_ALLOW_STUB=1.
    """
    def __init__(self, real: bool = True):
        self.model = None
        self.model_name: Optional[str] = None
        self.log = LoggerAdapter("backend.vllm" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
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
        # vLLM handles tensor parallelism internally
        return [self.model]

    def infer(self, inputs: Any):
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        if not self.real:
            if isinstance(inputs, str):
                return f"[stub-vllm] {inputs[:50]}"
            if _HAS_TORCH and _torch.is_tensor(inputs):
                return _torch.zeros_like(inputs)
            return inputs
        try:
            if isinstance(inputs, str):
                out = self.model.generate([inputs], sampling_params=None)
                return out[0].outputs[0].text if out and out[0].outputs else ""
            return inputs
        except Exception as e:
            self.log.warning(f"Infer vLLM échec: {e}")
            return ""

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Generate text using vLLM with proper SamplingParams."""
        if self.model is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        if not self.real:
            return f"[stub-vllm] {prompt[:50]}..."
        try:
            from vllm import SamplingParams
            params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                top_k=kwargs.get('top_k', -1),  # vLLM uses -1 for no top_k
            )
            outputs = self.model.generate([prompt], sampling_params=params)
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text
            return ""
        except Exception as e:
            self.log.error(f"vLLM generate failed: {e}")
            raise

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Stream tokens from vLLM.

        Uses vLLM's async generate if available, otherwise falls back
        to generating the full text and yielding word-by-word.
        """
        if not self.real:
            text = f"[stub-vllm] {prompt[:50]}..."
            for word in text.split(' '):
                yield word + ' '
            return

        # vLLM doesn't have a simple sync streaming API in the offline LLM class.
        # Generate full output and yield incrementally.
        text = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        for i, word in enumerate(text.split(' ')):
            yield word if i == 0 else ' ' + word
