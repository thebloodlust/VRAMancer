"""Thread-safe Pipeline Registry for VRAMancer.

Centralizes model lifecycle (load / shutdown / generate / infer)
behind a reentrant lock so concurrent Flask requests are safe.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

_logger = logging.getLogger("vramancer.registry")


class PipelineRegistry:
    """Thread-safe singleton registry for the inference pipeline.

    All access to the pipeline goes through this class, which holds a
    reentrant lock to protect load/shutdown/access operations.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._pipeline = None
        self.discovery = None

        # Cluster discovery is opt-in: starting it broadcasts UDP packets and
        # spawns background threads, which is undesirable in tests / CLI tools.
        # Enable explicitly with VRM_CLUSTER_AUTO_DISCOVER=1.
        if os.environ.get("VRM_CLUSTER_AUTO_DISCOVER", "").lower() in ("1", "true", "yes"):
            try:
                from experimental.cluster_discovery import ClusterDiscovery
                self.discovery = ClusterDiscovery(heartbeat_interval=5)
                self.discovery.start()
            except ImportError:
                pass

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

    def health_extra(self) -> dict:
        """T7.6 auto-heal state, surfaced by /health (degraded / reason)."""
        with self._lock:
            p = self._pipeline
        if p is None:
            return {"degraded": False, "degraded_reason": None}
        return {
            "degraded": getattr(p, "degraded", False),
            "degraded_reason": getattr(p, "degraded_reason", None),
        }

    def load(self, model_name: str, backend: str = "auto", num_gpus: Optional[int] = None, verbose: bool = False, **kwargs):
        """Load a model, shutting down any existing pipeline first."""
        with self._lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.shutdown()
                except Exception:
                    _logger.debug("Pipeline shutdown failed during reset", exc_info=True)
            from core.inference_pipeline import InferencePipeline
            self._pipeline = InferencePipeline(
                backend_name=backend, verbose=verbose
            )
            self._pipeline.load(model_name, num_gpus=num_gpus, **kwargs)

            # Auto-start continuous batcher for API serving
            if os.environ.get("VRM_CONTINUOUS_BATCHING", "").strip() == "1":
                if self._pipeline.continuous_batcher is not None:
                    self._pipeline.continuous_batcher.start()
                    _logger.info(
                        "Continuous batcher auto-started (VRM_CONTINUOUS_BATCHING=1)"
                    )

    def shutdown(self):
        with self._lock:
            if self._pipeline:
                try:
                    self._pipeline.shutdown()
                except Exception:
                    _logger.debug("Pipeline shutdown failed during cleanup", exc_info=True)
                self._pipeline = None
            # Stop cluster discovery threads
            if self.discovery:
                try:
                    self.discovery.stop()
                except Exception:
                    _logger.debug("Discovery stop failed", exc_info=True)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            from experimental.wake_on_inference import get_woi_manager
            get_woi_manager().wake_all()
        except ImportError:
            pass

        with self._lock:
            p = self._pipeline
        if p is None:
            raise RuntimeError("No model loaded")
        return p.generate(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """Yield tokens for streaming. Falls back to word-level split."""
        try:
            from experimental.wake_on_inference import get_woi_manager
            get_woi_manager().wake_all()
        except ImportError:
            pass

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
            nodes = {}
            if self.discovery:
                nodes.update(self.discovery.get_nodes())
            if (self._pipeline
                    and hasattr(self._pipeline, 'discovery')
                    and self._pipeline.discovery):
                nodes.update(self._pipeline.get_nodes())
            return nodes


__all__ = ["PipelineRegistry"]
