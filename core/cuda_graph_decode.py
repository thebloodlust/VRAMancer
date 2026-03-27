"""CUDA Graph capture for single-token decode steps.

Wraps a model's forward pass in a `torch.cuda.CUDAGraph`, replaying
the captured kernel launch sequence instead of re-dispatching Python
each step.  This eliminates CPU-side overhead (~10-30 % speedup on
small batch sizes where the GPU is underutilised).

Limitations:
  - Static shapes only: one graph is captured per (batch_size,) key.
  - KV cache must use a **static buffer** (pre-allocated max length)
    with a position mask — not the default HuggingFace growing tuple.
  - NCCL / P2P ops inside the captured region are not supported.

Usage inside VRAMancer:
  runner = CUDAGraphRunner(model, max_batch_size=8)
  for step in range(max_new_tokens):
      logits = runner.forward(input_ids_1token, past_key_values, ...)
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple

_logger = logging.getLogger("vramancer.cuda_graph")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False


class CUDAGraphRunner:
    """Per-batch-size CUDA graph cache for single-token decode.

    On the first call for a given batch size the runner performs a
    warmup forward, captures the graph, then replays it on every
    subsequent call — bypassing Python dispatch entirely.
    """

    def __init__(
        self,
        model: Any,
        max_cache_entries: int = 4,
        warmup_steps: int = 3,
        enabled: bool = True,
    ):
        self.model = model
        self.max_cache_entries = max_cache_entries
        self.warmup_steps = warmup_steps
        self.enabled = enabled and _HAS_TORCH and torch.cuda.is_available()

        # graph_key → (graph, static_input, static_output)
        self._graphs: Dict[int, Tuple[Any, Any, Any]] = {}
        self._call_counts: Dict[int, int] = {}

        if not self.enabled:
            _logger.debug("CUDAGraphRunner disabled (no CUDA)")

    def forward(
        self,
        input_ids: "torch.Tensor",
        **model_kwargs,
    ) -> "torch.Tensor":
        """Run one decode step, replaying graph if available.

        Parameters
        ----------
        input_ids : Tensor [B, 1]
            The single new token(s) for this decode step.
        **model_kwargs
            Forwarded to model (past_key_values, attention_mask, etc.).

        Returns
        -------
        logits : Tensor [B, 1, V]
        """
        if not self.enabled:
            return self._eager_forward(input_ids, **model_kwargs)

        bs = input_ids.size(0)

        # Count calls to decide when to capture
        self._call_counts[bs] = self._call_counts.get(bs, 0) + 1

        # Not enough warmup yet — run eager
        if self._call_counts[bs] <= self.warmup_steps:
            return self._eager_forward(input_ids, **model_kwargs)

        # Graph already captured for this batch size — replay
        if bs in self._graphs:
            return self._replay(bs, input_ids, **model_kwargs)

        # Capture now
        if len(self._graphs) < self.max_cache_entries:
            return self._capture_and_run(bs, input_ids, **model_kwargs)

        # Cache full — eager fallback
        return self._eager_forward(input_ids, **model_kwargs)

    def _eager_forward(self, input_ids, **kwargs):
        with torch.no_grad():
            out = self.model(input_ids, **kwargs)
            return out.logits if hasattr(out, "logits") else out[0]

    def _capture_and_run(self, bs: int, input_ids, **kwargs):
        """Capture a CUDA graph for batch_size=bs."""
        _logger.info("Capturing CUDA graph for batch_size=%d", bs)

        # Static input buffer (stays on same memory address)
        static_input = input_ids.clone()

        # Warmup inside capture stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                with torch.no_grad():
                    out = self.model(static_input, **kwargs)

        torch.cuda.current_stream().wait_stream(stream)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                static_out = self.model(static_input, **kwargs)
                static_logits = static_out.logits if hasattr(static_out, "logits") else static_out[0]

        self._graphs[bs] = (graph, static_input, static_logits)
        _logger.info("CUDA graph captured for batch_size=%d", bs)

        # First real run via replay
        static_input.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    def _replay(self, bs: int, input_ids, **kwargs):
        """Replay a previously captured graph."""
        graph, static_input, static_logits = self._graphs[bs]
        static_input.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    def reset(self):
        """Release all captured graphs."""
        self._graphs.clear()
        self._call_counts.clear()

    @property
    def captured_sizes(self) -> list:
        return list(self._graphs.keys())
