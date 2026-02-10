"""Thread-safe Pipeline Registry for VRAMancer.

Centralizes model lifecycle (load / shutdown / generate / infer)
behind a reentrant lock so concurrent Flask requests are safe.
"""
from __future__ import annotations

import threading
from typing import Any, Optional


class PipelineRegistry:
    """Thread-safe singleton registry for the inference pipeline.

    All access to the pipeline goes through this class, which holds a
    reentrant lock to protect load/shutdown/access operations.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._pipeline = None

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def get(self):
        """Return the current pipeline (may be None)."""
        with self._lock:
            return self._pipeline

    def is_loaded(self) -> bool:
        with self._lock:
            return self._pipeline is not None and self._pipeline.is_loaded()

    def load(self, model_name: str, backend: str = "auto",
             num_gpus: Optional[int] = None, verbose: bool = False):
        """Load a model, shutting down any existing pipeline first."""
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.shutdown()
                except Exception:
                    pass
            from core.inference_pipeline import InferencePipeline
            self._pipeline = InferencePipeline(
                backend_name=backend, verbose=verbose
            )
            self._pipeline.load(model_name, num_gpus=num_gpus)

    def shutdown(self):
        with self._lock:
            if self._pipeline:
                try:
                    self._pipeline.shutdown()
                except Exception:
                    pass
                self._pipeline = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        with self._lock:
            p = self._pipeline
        if p is None:
            raise RuntimeError("No model loaded")
        return p.generate(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """Yield tokens for streaming. Falls back to word-level split."""
        with self._lock:
            p = self._pipeline
        if p is None:
            raise RuntimeError("No model loaded")
        backend = getattr(p, 'backend', None)
        if backend and hasattr(backend, 'generate_stream'):
            yield from backend.generate_stream(prompt, **kwargs)
        else:
            text = p.generate(prompt, **kwargs)
            for i, word in enumerate(text.split(' ')):
                yield word if i == 0 else ' ' + word

    def infer(self, input_tensor):
        with self._lock:
            p = self._pipeline
        if p is None:
            raise RuntimeError("No model loaded")
        return p.infer(input_tensor)

    def status(self) -> dict:
        with self._lock:
            if self._pipeline:
                return self._pipeline.status()
        return {'loaded': False, 'message': 'Pipeline not initialized'}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> Optional[str]:
        with self._lock:
            if self._pipeline:
                return self._pipeline.model_name
        return None

    @property
    def backend(self):
        with self._lock:
            if self._pipeline:
                return self._pipeline.backend
        return None

    @property
    def num_gpus(self) -> int:
        with self._lock:
            if self._pipeline:
                return self._pipeline.num_gpus
        return 0

    @property
    def blocks(self) -> list:
        with self._lock:
            if self._pipeline:
                return self._pipeline.blocks
        return []

    def get_tokenizer(self):
        """Try to get a tokenizer from the pipeline or its backend."""
        with self._lock:
            p = self._pipeline
        if p is None:
            return None
        tok = getattr(p, 'tokenizer', None)
        if tok is None:
            tok = getattr(getattr(p, 'backend', None), 'tokenizer', None)
        return tok

    def get_nodes(self) -> dict:
        with self._lock:
            if (self._pipeline
                    and hasattr(self._pipeline, 'discovery')
                    and self._pipeline.discovery):
                return self._pipeline.get_nodes()
        return {}


__all__ = ["PipelineRegistry"]
