"""Backward-compatibility shim — moved to core/kv_quantizer.py."""
from core.kv_quantizer import KVCacheCompressor, TurboQuantCompressor  # noqa: F401

__all__ = ["TurboQuantCompressor", "KVCacheCompressor"]
