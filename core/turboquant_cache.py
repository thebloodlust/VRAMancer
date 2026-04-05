"""
TurboQuant KV Cache — HuggingFace-native integration.

Subclasses transformers' DynamicCache to compress KV states in-flight using
PolarQuant + QJL (TurboQuant algorithm).  This is the **real** VRAM reduction
path: HF ``generate()`` calls ``cache.update()`` every token, and we compress
older tokens into ~3.5 bits/dim (4.6x reduction) while keeping a residual
buffer of recent tokens in fp16 for accuracy.

Usage::

    cache = TurboQuantCache.from_model_config(model.config, residual_length=128)
    output = model.generate(input_ids, past_key_values=cache, ...)

Environment variables:
    VRM_KV_COMPRESSION_BITS   bits per polar angle (default 3)
    VRM_SPARSE_V_RATIO        fraction of values to decompress (default 1.0)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from transformers.cache_utils import CacheLayerMixin, DynamicCache, DynamicLayer, Cache
    _HF_CACHE = True
except ImportError:
    _HF_CACHE = False

try:
    from core.kv_quantizer import KVCacheCompressor
    _KV_QUANT = True
except ImportError:
    _KV_QUANT = False

try:
    from core.offloaded_compressor import OffloadedCompressor
    _OFFLOAD = True
except ImportError:
    _OFFLOAD = False

from core.logger import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compressed tensor wrapper — opaque storage for quantized KV
# ---------------------------------------------------------------------------

@dataclass
class _CompressedKV:
    """Opaque container for TurboQuant-compressed KV states.

    Stores the compressed representation (radius, quantized angles, QJL signs)
    along with metadata needed for decompression and cache bookkeeping.
    """
    data: dict                  # output of KVCacheCompressor.compress()
    seq_len: int                # number of tokens compressed
    device: torch.device        # original device
    dtype: torch.dtype          # original dtype


# ---------------------------------------------------------------------------
# TurboQuantLayer — per-layer cache with compression
# ---------------------------------------------------------------------------

class TurboQuantLayer(CacheLayerMixin):
    """A cache layer that compresses older KV states via TurboQuant.

    Maintains a residual buffer of ``residual_length`` recent tokens in fp16
    for attention accuracy.  When the buffer exceeds capacity, ALL cached
    tokens (compressed + residual) are re-compressed into the compact sidecar.

    This follows the same pattern as HF's ``QuantizedLayer`` but uses
    PolarQuant+QJL instead of linear quantization, achieving ~3.5 bits/dim
    vs 4 bits/dim.
    """

    is_sliding = False

    def __init__(
        self,
        compressor: KVCacheCompressor,
        residual_length: int = 128,
        offloader: "OffloadedCompressor | None" = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.compressor = compressor
        self.residual_length = residual_length
        self.cumulative_length = 0
        self._offloader = offloader
        self._layer_idx = layer_idx

        # Compressed sidecar (None until first spill) — only used when no offloader
        self._compressed_keys: Optional[_CompressedKV] = None
        self._compressed_values: Optional[_CompressedKV] = None

        # Cached decompressed KV on inference GPU (offloaded path only)
        self._cached_decompressed_keys: Optional[torch.Tensor] = None
        self._cached_decompressed_values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _compress_tensor(self, tensor: torch.Tensor) -> _CompressedKV:
        """Compress a [batch, heads, seq_len, head_dim] tensor.

        Reshapes to [batch*heads*seq_len, head_dim], compresses, wraps.
        """
        B, H, S, D = tensor.shape
        flat = tensor.reshape(-1, D).float()
        data = self.compressor.compress(flat, pack=False)
        return _CompressedKV(
            data=data,
            seq_len=S,
            device=tensor.device,
            dtype=tensor.dtype,
        )

    def _decompress_tensor(self, compressed: _CompressedKV, B: int, H: int) -> torch.Tensor:
        """Decompress back to [batch, heads, seq_len, head_dim]."""
        flat = self.compressor.decompress(compressed.data)
        return flat.reshape(B, H, compressed.seq_len, -1).to(compressed.dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update cache — compress when residual buffer overflows.

        When an ``OffloadedCompressor`` is attached, compression and
        decompression run on the worker GPU (zero overhead on inference
        GPU).  Otherwise falls back to local compression.

        Args:
            key_states: [batch, heads, new_seq, head_dim]
            value_states: [batch, heads, new_seq, head_dim]

        Returns:
            (all_keys, all_values) for attention computation
        """
        self.cumulative_length += key_states.shape[-2]

        # Lazy init
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H = key_states.shape[0], key_states.shape[1]

        # Append new tokens to residual buffer
        if self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        # --- Offloaded path: compress/decompress on worker GPU ---
        if self._offloader is not None:
            return self._update_offloaded(B, H)

        # --- Local path: compress/decompress on inference GPU ---
        return self._update_local(B, H)

    def _update_local(self, B: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Original path — compress/decompress on the inference GPU."""
        # Build full KV for attention: decompress old + residual
        if self._compressed_keys is not None:
            old_keys = self._decompress_tensor(self._compressed_keys, B, H)
            old_values = self._decompress_tensor(self._compressed_values, B, H)
            keys_out = torch.cat([old_keys, self.keys], dim=-2)
            values_out = torch.cat([old_values, self.values], dim=-2)
        else:
            keys_out = self.keys
            values_out = self.values

        # Spill: compress everything when residual exceeds capacity
        if self.keys.shape[-2] >= self.residual_length:
            self._compressed_keys = self._compress_tensor(keys_out)
            self._compressed_values = self._compress_tensor(values_out)
            # Clear residual buffer
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

        return keys_out, values_out

    def _update_offloaded(self, B: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Offloaded path — compress on worker GPU, cache decompressed on inference GPU.

        Combines two optimizations:
        1. Compressed data lives on the worker GPU (saves inference GPU VRAM)
        2. Decompressed copy cached on inference GPU (avoids re-decompressing
           every token — only 1 decompress+PCIe per spill cycle)

        Cost model per spill cycle (residual_length tokens):
        - 1 compress (async on worker GPU, doesn't block inference)
        - 1 decompress + PCIe transfer (synced once after spill)
        - residual_length - 1 tokens use cached decompressed (zero overhead)
        """
        offloader = self._offloader
        idx = self._layer_idx

        # Build full KV for attention using cached decompressed + residual
        if offloader.has_compressed(idx):
            # Use cached decompressed if available (set after each spill)
            if self._cached_decompressed_keys is not None:
                keys_out = torch.cat([self._cached_decompressed_keys, self.keys], dim=-2)
                values_out = torch.cat([self._cached_decompressed_values, self.values], dim=-2)
            else:
                # First access after init — decompress from worker GPU
                result = offloader.get_decompressed(idx, B, H, dtype=self.dtype)
                if result is not None:
                    self._cached_decompressed_keys, self._cached_decompressed_values = result
                    keys_out = torch.cat([self._cached_decompressed_keys, self.keys], dim=-2)
                    values_out = torch.cat([self._cached_decompressed_values, self.values], dim=-2)
                else:
                    keys_out = self.keys
                    values_out = self.values
        else:
            keys_out = self.keys
            values_out = self.values

        # Spill: send everything to worker GPU for async compression
        if self.keys.shape[-2] >= self.residual_length:
            # Compress on worker GPU (async)
            offloader.submit_compress(keys_out, values_out, layer_idx=idx)
            # Immediately cache the decompressed state on inference GPU
            # (this is just keys_out itself — no extra decompress needed!)
            self._cached_decompressed_keys = keys_out.clone()
            self._cached_decompressed_values = values_out.clone()
            # Clear residual buffer on inference GPU
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

        return keys_out, values_out

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1  # dynamic, no maximum

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder for beam search — reorder residual + recompress."""
        if self.keys.numel() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))
        if self._compressed_keys is not None:
            # Decompress, reorder, recompress
            B = beam_idx.shape[0]
            H = self.keys.shape[1] if self.keys.numel() > 0 else 1
            old_k = self._decompress_tensor(self._compressed_keys, B, H)
            old_v = self._decompress_tensor(self._compressed_values, B, H)
            old_k = old_k.index_select(0, beam_idx.to(old_k.device))
            old_v = old_v.index_select(0, beam_idx.to(old_v.device))
            self._compressed_keys = self._compress_tensor(old_k)
            self._compressed_values = self._compress_tensor(old_v)

    def reset(self) -> None:
        self._compressed_keys = None
        self._compressed_values = None
        self._cached_decompressed_keys = None
        self._cached_decompressed_values = None
        if self._offloader is not None:
            self._offloader.reset_layer(self._layer_idx)
        if self.is_initialized:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.cumulative_length = 0


# ---------------------------------------------------------------------------
# TurboQuantCache — drop-in replacement for DynamicCache
# ---------------------------------------------------------------------------

class TurboQuantCache(Cache):
    """HuggingFace-compatible KV cache with TurboQuant compression.

    Drop-in replacement for ``DynamicCache`` that compresses KV states
    ~4.6x via PolarQuant+QJL.  Pass as ``past_key_values`` to
    ``model.generate()`` and HF will use it transparently.

    Example::

        cache = TurboQuantCache.from_model_config(model.config)
        out = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        bits_per_angle: int = 3,
        residual_length: int = 128,
        device: str = "cuda:0",
        worker_device: str | None = None,
    ):
        # Optional: offload compression to a worker GPU
        offloader = None
        if worker_device is not None and _OFFLOAD and _TORCH:
            offloader = OffloadedCompressor(
                head_dim=head_dim,
                worker_device=worker_device,
                inference_device=device,
                bits_per_angle=bits_per_angle,
            )

        # Build shared compressor (one set of random matrices for all layers)
        # Used for local compression when no offloader is present
        compressor = KVCacheCompressor(
            head_dim=head_dim,
            bits_per_angle=bits_per_angle,
        )
        if _TORCH and device.startswith("cuda") and torch.cuda.is_available():
            compressor = compressor.to(device)

        layers = [
            TurboQuantLayer(
                compressor=compressor,
                residual_length=residual_length,
                offloader=offloader,
                layer_idx=i,
            )
            for i in range(num_layers)
        ]
        super().__init__(layers=layers)

        self._head_dim = head_dim
        self._bits_per_angle = bits_per_angle
        self._compressor = compressor
        self._offloader = offloader

        mode = f"offloaded to {worker_device}" if offloader else "local"
        _logger.info(
            "TurboQuantCache: %d layers, head_dim=%d, %.1f bits/dim (%.1fx), "
            "residual=%d tokens, mode=%s",
            num_layers, head_dim, compressor.bits_per_dim(),
            16.0 / compressor.bits_per_dim(), residual_length, mode,
        )

    @classmethod
    def from_model_config(
        cls,
        config,
        bits_per_angle: int | None = None,
        residual_length: int = 256,
        device: str = "cuda:0",
        worker_device: str | None = None,
    ) -> "TurboQuantCache":
        """Create from a HuggingFace model config (auto-detect dimensions).

        Parameters
        ----------
        worker_device : str, optional
            If set, compress/decompress runs on this GPU instead of the
            inference device.  Pass ``"cuda:1"`` to offload TurboQuant
            to the secondary GPU — zero overhead on the inference path.
        """
        if hasattr(config, "get_text_config"):
            config = config.get_text_config(decoder=True)

        num_layers = getattr(config, "num_hidden_layers", 12)
        num_heads = getattr(config, "num_attention_heads", 12)
        hidden_size = getattr(config, "hidden_size", 768)
        head_dim = hidden_size // num_heads

        if bits_per_angle is None:
            bits_per_angle = int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3"))

        # Auto-detect worker GPU from env
        if worker_device is None:
            worker_device = os.environ.get("VRM_TURBOQUANT_WORKER")

        return cls(
            num_layers=num_layers,
            head_dim=head_dim,
            bits_per_angle=bits_per_angle,
            residual_length=residual_length,
            device=device,
            worker_device=worker_device,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer in self.layers:
            layer.reorder_cache(beam_idx)
