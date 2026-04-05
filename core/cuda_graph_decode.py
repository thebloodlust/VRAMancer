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


class _GraphState:
    """Internal state for a captured CUDA graph (one per batch size)."""
    __slots__ = (
        "graph", "static_input", "static_logits",
        "static_mask", "static_pos", "static_cache_pos", "cur_pos",
    )

    def __init__(self, graph, static_input, static_logits,
                 static_mask, static_pos, static_cache_pos, cur_pos: int):
        self.graph = graph
        self.static_input = static_input
        self.static_logits = static_logits
        self.static_mask = static_mask
        self.static_pos = static_pos
        self.static_cache_pos = static_cache_pos
        self.cur_pos = cur_pos


class CUDAGraphRunner:
    """Per-batch-size CUDA graph cache for single-token decode.

    On the first call for a given batch size the runner performs a
    warmup forward, captures the graph, then replays it on every
    subsequent call — bypassing Python dispatch entirely.

    For correct decode, the model's KV cache must use **static buffers**
    (pre-allocated at max sequence length) with in-place updates, not
    HuggingFace's default growing-tuple DynamicCache.  The caller must
    provide ``past_key_values`` that writes KV in-place (e.g. StaticCache
    from transformers >= 4.38 or VRAMancer's StaticKVCache).

    Static buffers for ``attention_mask``, ``position_ids`` and
    ``cache_position`` are managed internally and updated between replays.
    """

    def __init__(
        self,
        model: Any,
        max_cache_entries: int = 4,
        warmup_steps: int = 3,
        enabled: bool = True,
        max_seq_len: int = 2048,
    ):
        self.model = model
        self.max_cache_entries = max_cache_entries
        self.warmup_steps = warmup_steps
        self.max_seq_len = max_seq_len
        self.enabled = enabled and _HAS_TORCH and torch.cuda.is_available()

        # graph_key → _GraphState
        self._graphs: Dict[int, "_GraphState"] = {}
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

        # Guard: CUDA graphs require static KV cache buffers.
        # DynamicCache grows in-place which breaks the captured graph's
        # fixed memory addresses — silently producing garbage output.
        pkv = model_kwargs.get("past_key_values")
        if pkv is not None:
            pkv_type = type(pkv).__name__
            if pkv_type == "DynamicCache":
                _logger.warning(
                    "DynamicCache detected — incompatible with CUDA graphs. "
                    "Falling back to eager. Use StaticCache instead."
                )
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
        """Capture a CUDA graph for batch_size=bs.

        Creates static buffers for input_ids, attention_mask, position_ids
        and cache_position so that replay can update them without breaking
        the graph's fixed memory addresses.
        """
        _logger.info("Capturing CUDA graph for batch_size=%d", bs)

        device = input_ids.device

        # Static input buffer (stays on same memory address)
        static_input = input_ids.clone()

        # Static attention_mask — pre-allocate at max_seq_len, filled with 1s
        # Graph captures the address; we update the contents between replays.
        cur_seq_len = kwargs.get("attention_mask", input_ids).shape[-1] if "attention_mask" in kwargs else 1
        static_mask = torch.ones(bs, self.max_seq_len, dtype=torch.long, device=device)
        # Zero out positions beyond current sequence
        if cur_seq_len < self.max_seq_len:
            static_mask[:, cur_seq_len:] = 0

        # Static position_ids for the new token
        static_pos = torch.tensor([[cur_seq_len - 1]], device=device).expand(bs, 1).clone()

        # Static cache_position
        static_cache_pos = torch.tensor([cur_seq_len - 1], dtype=torch.long, device=device)

        # Build static kwargs — replace dynamic tensors with static buffers
        static_kwargs = dict(kwargs)
        static_kwargs["attention_mask"] = static_mask[:, :cur_seq_len]
        if "position_ids" in kwargs or True:  # always provide for graph safety
            static_kwargs["position_ids"] = static_pos
        if "cache_position" in kwargs:
            static_kwargs["cache_position"] = static_cache_pos

        # Warmup inside capture stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(2):
                with torch.no_grad():
                    out = self.model(static_input, **static_kwargs)

        torch.cuda.current_stream().wait_stream(stream)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                static_out = self.model(static_input, **static_kwargs)
                static_logits = static_out.logits if hasattr(static_out, "logits") else static_out[0]

        state = _GraphState(
            graph=graph,
            static_input=static_input,
            static_logits=static_logits,
            static_mask=static_mask,
            static_pos=static_pos,
            static_cache_pos=static_cache_pos,
            cur_pos=cur_seq_len,
        )
        self._graphs[bs] = state
        _logger.info("CUDA graph captured for batch_size=%d (seq_pos=%d)", bs, cur_seq_len)

        # First real run via replay
        static_input.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    def _replay(self, bs: int, input_ids, **kwargs):
        """Replay a previously captured graph, advancing KV position."""
        state = self._graphs[bs]

        # Advance position by 1 (one new token per decode step)
        state.cur_pos += 1
        if state.cur_pos >= self.max_seq_len:
            # Exceeded max length — fall back to eager
            _logger.warning("CUDA graph seq_pos %d >= max_seq_len %d, falling back",
                            state.cur_pos, self.max_seq_len)
            del self._graphs[bs]
            return self._eager_forward(input_ids, **kwargs)

        # Update static buffers IN-PLACE (same memory addresses as captured)
        state.static_input.copy_(input_ids)
        # Extend attention mask by 1 position
        state.static_mask[:, state.cur_pos - 1] = 1
        # Update position_ids to current position
        state.static_pos.fill_(state.cur_pos - 1)
        # Update cache_position
        state.static_cache_pos.fill_(state.cur_pos - 1)

        state.graph.replay()
        return state.static_logits.clone()

    def reset(self):
        """Release all captured graphs."""
        self._graphs.clear()
        self._call_counts.clear()

    @property
    def captured_sizes(self) -> list:
        return list(self._graphs.keys())
