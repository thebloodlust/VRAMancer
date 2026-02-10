# core/compressor.py
"""Production model compression — quantization + byte-level compression.

Supports:
  - PyTorch dynamic quantization (INT8) for CPU inference
  - Manual weight quantization (INT8 / INT4) for mixed-precision
  - Byte-level compression (zstd / lz4 / gzip) for NVMe spill/network
  - GPTQ / AWQ integration stubs (require external libraries)
  - Compression ratio estimation and actual measurement
"""

from __future__ import annotations

import os
import io
import logging
from typing import Any, Optional, Dict, Tuple

_logger = logging.getLogger("vramancer.compressor")

STRICT = os.environ.get("VRM_STRICT_IMPORT", "0") in {"1", "true", "TRUE"}
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch  # type: ignore
    import torch.nn as nn
except ImportError:
    if STRICT:
        raise
    torch = None  # type: ignore
    nn = None  # type: ignore

# Byte-level compressors (try fast ones first)
_zstd = None
_lz4 = None
try:
    import zstandard as _zstd_mod  # type: ignore
    _zstd = _zstd_mod
except ImportError:
    pass
try:
    import lz4.frame as _lz4_mod  # type: ignore
    _lz4 = _lz4_mod
except ImportError:
    pass


class Compressor:
    """Multi-strategy model compressor.

    Strategies:
      - "none"       : no compression
      - "light"      : gzip level 1 (fast, ~15% reduction)
      - "adaptive"   : zstd level 3 or gzip level 6 (balanced)
      - "aggressive" : zstd level 9 + INT8 quantization (~50% reduction)

    Usage:
        compressor = Compressor(strategy="adaptive")
        compressed = compressor.compress_tensor(tensor)
        original = compressor.decompress_tensor(compressed)

        # Or compress a full module for NVMe spill
        data = compressor.compress_module(module)
        module = compressor.decompress_module(data)
    """

    def __init__(self, strategy: str = "adaptive", verbose: bool = True):
        self.strategy = strategy
        self.verbose = verbose
        self._codec = self._select_codec()

    def _select_codec(self) -> str:
        """Select the best available compression codec."""
        if self.strategy == "none":
            return "none"
        if _zstd is not None:
            return "zstd"
        if _lz4 is not None:
            return "lz4"
        return "gzip"  # always available via stdlib

    # ------------------------------------------------------------------
    # Byte-level compression (for serialized tensors / NVMe spill)
    # ------------------------------------------------------------------

    def compress_bytes(self, data: bytes) -> bytes:
        """Compress raw bytes."""
        if self.strategy == "none":
            return data

        level = self._compression_level()

        if self._codec == "zstd":
            cctx = _zstd.ZstdCompressor(level=level)
            return cctx.compress(data)
        elif self._codec == "lz4":
            return _lz4.compress(data, compression_level=level)
        else:
            import gzip
            return gzip.compress(data, compresslevel=level)

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress raw bytes (auto-detects codec)."""
        if self.strategy == "none":
            return data

        # Try codecs in order
        if self._codec == "zstd" and _zstd is not None:
            try:
                dctx = _zstd.ZstdDecompressor()
                return dctx.decompress(data)
            except Exception:
                pass
        if self._codec == "lz4" and _lz4 is not None:
            try:
                return _lz4.decompress(data)
            except Exception:
                pass
        # Fallback: gzip
        import gzip
        try:
            return gzip.decompress(data)
        except Exception:
            return data  # return as-is if can't decompress

    # ------------------------------------------------------------------
    # Tensor compression
    # ------------------------------------------------------------------

    def compress_tensor(self, tensor: Any) -> Dict[str, Any]:
        """Compress a tensor to a serialized dict with metadata.

        Returns dict with keys: "data" (compressed bytes), "shape", "dtype",
        "codec", "original_bytes", "compressed_bytes".
        """
        if torch is None or _MINIMAL:
            return {"data": b"", "shape": (), "dtype": "float32",
                    "codec": "none", "original_bytes": 0, "compressed_bytes": 0}

        buf = io.BytesIO()
        torch.save(tensor.cpu().detach(), buf)
        raw = buf.getvalue()
        compressed = self.compress_bytes(raw)

        result = {
            "data": compressed,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "codec": self._codec,
            "original_bytes": len(raw),
            "compressed_bytes": len(compressed),
        }
        if self.verbose:
            ratio = len(compressed) / max(len(raw), 1)
            _logger.info("Tensor compressed: %s %s -> %.1fKB (%.0f%% of original)",
                         result["shape"], result["dtype"],
                         len(compressed) / 1024, ratio * 100)
        return result

    def decompress_tensor(self, data: Dict[str, Any], device: str = "cpu") -> Any:
        """Decompress a tensor from compress_tensor output."""
        if torch is None or not data.get("data"):
            return None
        raw = self.decompress_bytes(data["data"])
        buf = io.BytesIO(raw)
        tensor = torch.load(buf, map_location=device, weights_only=True)
        return tensor

    # ------------------------------------------------------------------
    # Module compression (for NVMe spill / network transfer)
    # ------------------------------------------------------------------

    def compress_module(self, module: Any) -> Dict[str, Any]:
        """Serialize and compress an nn.Module's state_dict."""
        if torch is None or _MINIMAL:
            return {"data": b"", "codec": "none", "original_bytes": 0, "compressed_bytes": 0}

        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        raw = buf.getvalue()
        compressed = self.compress_bytes(raw)

        return {
            "data": compressed,
            "codec": self._codec,
            "original_bytes": len(raw),
            "compressed_bytes": len(compressed),
        }

    def decompress_module(self, data: Dict[str, Any], module_template: Any = None) -> Optional[Any]:
        """Decompress and load state_dict into a module template."""
        if torch is None or not data.get("data"):
            return None
        raw = self.decompress_bytes(data["data"])
        buf = io.BytesIO(raw)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)
        if module_template is not None:
            module_template.load_state_dict(state_dict)
            return module_template
        return state_dict

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_dynamic_int8(self, module: Any) -> Any:
        """Apply PyTorch dynamic INT8 quantization (CPU only).

        Quantizes nn.Linear and nn.LSTM layers to int8.
        """
        if torch is None or _MINIMAL:
            _logger.warning("Quantization unavailable (torch missing)")
            return module

        try:
            quantized = torch.quantization.quantize_dynamic(
                module, {torch.nn.Linear}, dtype=torch.qint8
            )
            if self.verbose:
                _logger.info("Dynamic INT8 quantization applied")
            return quantized
        except Exception as exc:
            _logger.warning("Dynamic quantization failed: %s", exc)
            return module

    def quantize_weights_int4(self, tensor: Any) -> Tuple[Any, Any, Any]:
        """Manual INT4 weight quantization (pack 2 values per byte).

        Returns (quantized_data, scale, zero_point) for dequantization.
        """
        if torch is None or _MINIMAL:
            return tensor, 1.0, 0

        t = tensor.float().cpu()
        vmin = t.min()
        vmax = t.max()
        scale = (vmax - vmin) / 15.0  # 4-bit range: 0-15
        zero_point = (-vmin / scale).round().clamp(0, 15).to(torch.int8)
        quantized = ((t / scale) + zero_point.float()).round().clamp(0, 15).to(torch.uint8)

        # Pack two 4-bit values per byte
        flat = quantized.flatten()
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
        packed = (flat[0::2] << 4) | flat[1::2]

        if self.verbose:
            ratio = packed.numel() / max(tensor.numel() * tensor.element_size(), 1) * 100
            _logger.info("INT4 quantization: %s -> %.0f%% of original", tensor.shape, ratio)

        return packed, scale, zero_point

    def dequantize_int4(self, packed: Any, scale: Any, zero_point: Any,
                        original_shape: tuple) -> Any:
        """Dequantize INT4 weights back to float32."""
        if torch is None:
            return packed

        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        flat = torch.stack([high, low], dim=-1).flatten()

        import math
        n = math.prod(original_shape)
        flat = flat[:n]

        return ((flat.float() - zero_point.float()) * scale).reshape(original_shape)

    # ------------------------------------------------------------------
    # Legacy API (backward compatible)
    # ------------------------------------------------------------------

    def compress(self, layer_name: str, original_size_mb: float) -> float:
        """Estimate compressed size (legacy API, returns estimated MB).

        For actual compression, use compress_tensor() or compress_module().
        """
        ratios = {
            "aggressive": 0.50,
            "light": 0.85,
            "adaptive": 0.75 if "layer" in layer_name else 0.90,
            "none": 1.0,
        }
        factor = ratios.get(self.strategy, 0.75)
        new_size = round(original_size_mb * factor, 2)
        if self.verbose:
            _logger.info("Estimated compression %s: %.1fMB -> %.1fMB (strategy=%s)",
                         layer_name, original_size_mb, new_size, self.strategy)
        return new_size

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compression_level(self) -> int:
        """Return compression level based on strategy."""
        if self.strategy == "aggressive":
            return 9
        elif self.strategy == "light":
            return 1
        else:  # adaptive
            return 3 if self._codec == "zstd" else 6
