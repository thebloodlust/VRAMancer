"""
Offloaded KV Cache Compressor — runs TurboQuant on a secondary GPU.

Moves compress/decompress work OFF the inference GPU to a dedicated
"worker" GPU (the VRAM lender, a WebGPU node, an NPU, etc.).  The
inference GPU only does model forward passes — zero compression overhead.

Architecture::

    GPU0 (inference)              GPU1 (worker/lender)
    ──────────────────           ──────────────────────
    model.forward()
      → new_k, new_v
      → PCIe transfer ─────────→ receive KV
                                  compress(KV) on GPU1 stream
                                  store compressed in GPU1 VRAM
      ← PCIe transfer ←───────── decompress() when needed
    attention(old + residual)

The key insight: compression is pipelined on a separate CUDA stream,
so it NEVER blocks the inference GPU's forward pass.

Usage::

    offloader = OffloadedCompressor(
        head_dim=128,
        worker_device="cuda:1",
        inference_device="cuda:0",
    )

    # In TurboQuantLayer.update():
    offloader.submit_compress(keys_out, values_out, layer_idx=0)
    old_k, old_v = offloader.get_decompressed(layer_idx=0, B=1, H=4)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.cuda
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from core.kv_quantizer import KVCacheCompressor
    _KV_QUANT = True
except ImportError:
    _KV_QUANT = False

from core.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class _LayerState:
    """Per-layer compressed state on the worker GPU."""
    compressed_keys: Optional[dict] = None
    compressed_values: Optional[dict] = None
    seq_len: int = 0
    # Event signaling compression is complete (for async pipeline)
    compress_event: Optional["torch.cuda.Event"] = None


class OffloadedCompressor:
    """Runs TurboQuant compress/decompress on a secondary (worker) GPU.

    The inference GPU sends raw KV tensors to the worker via PCIe.
    The worker compresses them asynchronously on its own CUDA stream.
    When the inference GPU needs old KV for attention, the worker
    decompresses and sends back.

    This removes ALL compression overhead from the inference GPU's
    critical path.  The cost is 2x PCIe transfers per spill/decompress
    cycle, but PCIe bandwidth (~25 GB/s) is much cheaper than the
    lost GPU compute time from synchronous compression.

    Parameters
    ----------
    head_dim : int
        Attention head dimension (e.g. 128 for Qwen2.5/Llama).
    worker_device : str
        CUDA device for compression work (e.g. "cuda:1").
    inference_device : str
        CUDA device running the model (e.g. "cuda:0").
    bits_per_angle : int
        TurboQuant bits per polar angle (default 3 → ~3.5 bits/dim).
    """

    def __init__(
        self,
        head_dim: int,
        worker_device: str = "cuda:1",
        inference_device: str = "cuda:0",
        bits_per_angle: int = 3,
    ):
        if not _TORCH or not _KV_QUANT:
            raise RuntimeError("OffloadedCompressor requires torch and core.kv_quantizer")

        self.worker_device = torch.device(worker_device)
        self.inference_device = torch.device(inference_device)
        self.head_dim = head_dim

        # Compressor lives on the WORKER GPU — all math happens there
        self._compressor = KVCacheCompressor(
            head_dim=head_dim,
            bits_per_angle=bits_per_angle,
        ).to(self.worker_device)

        # Dedicated CUDA stream on worker GPU for async compression
        self._worker_stream = torch.cuda.Stream(device=self.worker_device)

        # Per-layer compressed state (lives on worker GPU)
        self._layers: Dict[int, _LayerState] = {}

        # Lock for thread-safety (Flask multi-request scenarios)
        self._lock = threading.Lock()

        _logger.info(
            "OffloadedCompressor: inference=%s, worker=%s, "
            "head_dim=%d, %.1f bits/dim (%.1fx)",
            self.inference_device, self.worker_device,
            head_dim, self._compressor.bits_per_dim(),
            16.0 / self._compressor.bits_per_dim(),
        )

    def _get_layer(self, layer_idx: int) -> _LayerState:
        if layer_idx not in self._layers:
            self._layers[layer_idx] = _LayerState()
        return self._layers[layer_idx]

    def submit_compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Submit KV tensors for async compression on the worker GPU.

        Non-blocking: transfers to worker GPU and compresses on the
        worker's CUDA stream.  Call this from update() when the residual
        buffer overflows.

        Parameters
        ----------
        keys : Tensor [B, H, S, D] on inference_device
        values : Tensor [B, H, S, D] on inference_device
        layer_idx : int
        """
        B, H, S, D = keys.shape
        state = self._get_layer(layer_idx)

        with self._lock:
            # Transfer to worker GPU (non-blocking via PCIe)
            keys_worker = keys.to(self.worker_device, non_blocking=True)
            values_worker = values.to(self.worker_device, non_blocking=True)

            # Compress on worker stream (async, doesn't block inference GPU)
            with torch.cuda.stream(self._worker_stream):
                flat_k = keys_worker.reshape(-1, D).float()
                flat_v = values_worker.reshape(-1, D).float()
                state.compressed_keys = self._compressor.compress(flat_k)
                state.compressed_values = self._compressor.compress(flat_v)
                state.seq_len = S
                # Record event so we know when compression is done
                state.compress_event = torch.cuda.Event()
                state.compress_event.record(self._worker_stream)

    def get_decompressed(
        self,
        layer_idx: int,
        B: int,
        H: int,
        dtype: torch.dtype = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Decompress and transfer back to inference GPU.

        Synchronizes with the worker stream if compression is still
        in progress, then decompresses on the worker GPU and transfers
        the result to the inference GPU.

        Returns
        -------
        (keys, values) on inference_device, or None if no compressed data.
        """
        state = self._get_layer(layer_idx)
        if state.compressed_keys is None:
            return None

        with self._lock:
            # Wait for any pending compression to finish
            if state.compress_event is not None:
                state.compress_event.synchronize()

            # Decompress on worker GPU
            with torch.cuda.stream(self._worker_stream):
                flat_k = self._compressor.decompress(state.compressed_keys)
                flat_v = self._compressor.decompress(state.compressed_values)

                keys = flat_k.reshape(B, H, state.seq_len, -1)
                values = flat_v.reshape(B, H, state.seq_len, -1)

                if dtype is not None:
                    keys = keys.to(dtype)
                    values = values.to(dtype)

                # Transfer back to inference GPU (non-blocking)
                keys_inf = keys.to(self.inference_device, non_blocking=True)
                values_inf = values.to(self.inference_device, non_blocking=True)

                # Record event on worker stream
                transfer_event = torch.cuda.Event()
                transfer_event.record(self._worker_stream)

            # Must sync before inference GPU uses the tensors
            transfer_event.synchronize()

            return keys_inf, values_inf

    def has_compressed(self, layer_idx: int) -> bool:
        """Check if layer has compressed data."""
        state = self._get_layer(layer_idx)
        return state.compressed_keys is not None

    def get_compressed_seq_len(self, layer_idx: int) -> int:
        """Get number of compressed tokens for a layer."""
        state = self._get_layer(layer_idx)
        return state.seq_len

    def reset_layer(self, layer_idx: int) -> None:
        """Clear compressed state for a layer."""
        with self._lock:
            if layer_idx in self._layers:
                self._layers[layer_idx] = _LayerState()

    def reset(self) -> None:
        """Clear all compressed state."""
        with self._lock:
            self._layers.clear()

    @property
    def bits_per_dim(self) -> float:
        return self._compressor.bits_per_dim()
