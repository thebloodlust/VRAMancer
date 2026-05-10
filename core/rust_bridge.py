"""Graceful access layer for the optional ``vramancer_rust`` PyO3 extension.

Wraps every Rust call so that callers don't have to do their own
``try/except ImportError`` dances. When the crate is missing OR libcuda
is not loadable, helpers return ``None`` / ``False`` and the caller can
fall back to the pure-Python path.

Usage::

    from core.rust_bridge import rust, cuda_available, hmac_verify

    if rust is not None and cuda_available():
        rust.direct_vram_copy(...)
    else:
        # python fallback
        ...

Importing this module is **always safe** (no exceptions even if the .so
is missing or libcuda is absent).
"""
from __future__ import annotations

from typing import Any, Optional


def _safe_import() -> Optional[Any]:
    try:
        import vramancer_rust  # type: ignore
        return vramancer_rust
    except Exception:
        return None


# Cached module handle (None when the crate is missing).
rust: Optional[Any] = _safe_import()


def has_rust() -> bool:
    """True iff the ``vramancer_rust`` PyO3 module imports cleanly."""
    return rust is not None


def cuda_available() -> bool:
    """True iff the Rust crate is loaded AND libcuda.so.1 / nvcuda.dll is loadable.

    This is the authoritative check before calling any Rust CUDA function.
    Older crate versions without ``cuda_available()`` are assumed to have
    CUDA (best-effort).
    """
    if rust is None:
        return False
    fn = getattr(rust, "cuda_available", None)
    if fn is None:
        # Older crate built before this helper landed → assume present.
        return True
    try:
        return bool(fn())
    except Exception:
        return False


def hmac_verify(secret: bytes, payload: bytes, signature: bytes) -> Optional[bool]:
    """Constant-time HMAC-SHA256 verify via Rust. Returns None if Rust unavailable."""
    if rust is None:
        return None
    fn = getattr(rust, "verify_hmac_fast", None)
    if fn is None:
        return None
    try:
        return bool(fn(secret, payload, signature))
    except Exception:
        return None


__all__ = ["rust", "has_rust", "cuda_available", "hmac_verify"]
