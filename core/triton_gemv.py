"""VRAMancer Fused NF4 GEMV Kernel (Triton).

Numerically correct Triton implementation of BnB's ``gemv_4bit`` that fuses
NF4 dequantization with the GEMV in one pass.

**Current status (March 2026):** Correct but 0.6-0.9x vs BnB's hand-tuned
CUDA kernel (``kgemm_4bit_inference_naive``).  BnB's kernel uses vectorized
float4 reads + warp shuffles that Triton cannot easily match for this
bandwidth-bound operation.

The kernel is kept for:
- Reference implementation documenting the NF4 packed layout
- Potential future use when Triton gains better vectorized load support
- Fallback when BnB's native kernel is unavailable

BnB NF4 packing format (blocksize=64, compress_statistics=True):
- Weights: uint8 tensor, 2 elements per byte
- Hi nibble (bits 7-4) → even element (index 2i)
- Lo nibble (bits 3-0) → odd element (index 2i+1)
- Code table: 16 float32 values (NF4 quantization levels)
- Absmax: nested-quantized (uint8), must be dequantized first

Usage::

    from core.triton_gemv import triton_gemv_4bit, patch_bnb_gemv

    # Monkey-patch BnB globally
    patch_bnb_gemv()

    # Or call directly:
    out = triton_gemv_4bit(x, qweight, state=quant_state)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_logger = logging.getLogger("vramancer.triton_gemv")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------
_HAS_TRITON = False
_HAS_TORCH = False
_HAS_BNB = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

if not _MINIMAL:
    try:
        import triton
        import triton.language as tl
        _HAS_TRITON = True
    except ImportError:
        pass

    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_F
        _HAS_BNB = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# NF4 code table (hardcoded — same for all NF4 models, from BnB source)
# ---------------------------------------------------------------------------
NF4_CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _nf4_gemv_kernel(
        # Pointers
        x_ptr,           # [K] input vector (fp16)
        qw_ptr,          # [N*K//2] packed uint8 weights (row-major, 2 per byte)
        absmax_ptr,      # [num_blocks] fp32 absmax scales (pre-dequantized)
        code_ptr,        # [16] NF4 lookup table (fp32)
        out_ptr,         # [N] output vector (fp16)
        # Dimensions
        N,               # number of output features (rows)
        K,               # number of input features (columns)
        blocksize,       # quantization blocksize (64)
        # Tuning
        BLOCK_N: tl.constexpr,   # rows per program
        BLOCK_K: tl.constexpr,   # elements per iter (K dimension)
    ):
        """Fused NF4 dequant + GEMV, tiled over N.

        Grid: (cdiv(N, BLOCK_N),) — each program handles BLOCK_N rows.
        """
        pid = tl.program_id(0)
        row_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        row_mask = row_offs < N

        half_K = K // 2
        blocks_per_row = K // blocksize
        half_BLOCK: tl.constexpr = BLOCK_K // 2

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        num_iters = K // BLOCK_K

        for i in range(num_iters):
            byte_offs = tl.arange(0, half_BLOCK)  # [half_BLOCK]

            # 2D index: [BLOCK_N, half_BLOCK]
            byte_idx = row_offs[:, None] * half_K + (i * half_BLOCK + byte_offs[None, :])
            packed = tl.load(qw_ptr + byte_idx, mask=row_mask[:, None], other=0)

            # BnB packing: hi nibble → even element, lo nibble → odd element
            hi = (packed >> 4) & 0xF
            lo = packed & 0xF

            val_even = tl.load(code_ptr + hi)  # [BLOCK_N, half_BLOCK]
            val_odd = tl.load(code_ptr + lo)

            # Absmax scale indices (shared across rows for same K position)
            elem_even = i * BLOCK_K + 2 * byte_offs[None, :]  # [1, half_BLOCK]
            block_even = elem_even // blocksize
            block_odd = (elem_even + 1) // blocksize

            absmax_base = row_offs[:, None] * blocks_per_row  # [BLOCK_N, 1]
            scale_even = tl.load(absmax_ptr + absmax_base + block_even, mask=row_mask[:, None], other=0.0)
            scale_odd = tl.load(absmax_ptr + absmax_base + block_odd, mask=row_mask[:, None], other=0.0)

            w_even = val_even * scale_even  # [BLOCK_N, half_BLOCK]
            w_odd = val_odd * scale_odd

            x_even = tl.load(x_ptr + elem_even).to(tl.float32)  # [1, half_BLOCK]
            x_odd = tl.load(x_ptr + elem_even + 1).to(tl.float32)

            # Reduce K dimension for each row
            acc += tl.sum(w_even * x_even + w_odd * x_odd, axis=1)  # [BLOCK_N]

        tl.store(out_ptr + row_offs, acc.to(tl.float16), mask=row_mask)


def _dequant_absmax(state):
    """Pre-dequantize nested absmax (cheap, ~0.01ms for a 7B model layer)."""
    absmax = state.absmax
    if state.nested and hasattr(state, 'state2') and state.state2 is not None:
        absmax = bnb_F.dequantize_blockwise(absmax, state.state2) + state.offset
    return absmax.to(torch.float32)


def triton_gemv_4bit(
    x: torch.Tensor,
    qweight: torch.Tensor,
    state,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused NF4 dequant + GEMV using Triton.

    Args:
        x: Input vector [1, K] or [K] in fp16.
        qweight: Packed NF4 weights from BnB (uint8).
        state: BnB QuantState with absmax, code, blocksize, shape.
        out: Optional pre-allocated output tensor.

    Returns:
        Output tensor [1, N] or [N].
    """
    if not _HAS_TRITON:
        return bnb_F.gemv_4bit(x, qweight, out=out, state=state)

    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    N, K = state.shape  # original weight shape [N, K]

    # Pre-dequantize absmax (nested quantization)
    absmax = _dequant_absmax(state)

    # NF4 code table on device
    code = state.code.to(torch.float32).to(x.device)

    if out is None:
        out = torch.empty(x.shape[0], N, dtype=x.dtype, device=x.device)

    # Flatten x for the kernel
    x_flat = x.reshape(-1).contiguous()
    qw_flat = qweight.reshape(-1).contiguous()

    # Choose BLOCK_K: must divide K, power of 2, >= blocksize
    BLOCK_K = min(1024, K)
    while K % BLOCK_K != 0 and BLOCK_K > 64:
        BLOCK_K //= 2

    # Choose BLOCK_N: tile rows for occupancy
    BLOCK_N = 32
    if N >= 4096:
        BLOCK_N = 16  # smaller tiles for huge N to reduce register pressure

    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    _nf4_gemv_kernel[grid](
        x_flat, qw_flat, absmax, code, out.reshape(-1),
        N, K, state.blocksize,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# Monkey-patching utilities
# ---------------------------------------------------------------------------
_original_gemv = None


def _patched_gemv_4bit(A, B, out=None, transposed_A=False, transposed_B=False, state=None):
    """Drop-in replacement for bnb.functional.gemv_4bit."""
    if state is None:
        raise ValueError("state required")

    if not _HAS_TRITON or A.device.type != 'cuda':
        return _original_gemv(A, B, out=out, state=state)

    return triton_gemv_4bit(A, B, state=state, out=out)


def patch_bnb_gemv():
    """Monkey-patch BnB's gemv_4bit globally with our Triton kernel."""
    global _original_gemv
    if not _HAS_BNB or not _HAS_TRITON:
        _logger.warning("Cannot patch: Triton=%s BnB=%s", _HAS_TRITON, _HAS_BNB)
        return False

    _original_gemv = bnb_F.gemv_4bit
    bnb_F.gemv_4bit = _patched_gemv_4bit
    _logger.info("Patched bnb.functional.gemv_4bit with Triton fused kernel")
    return True


def unpatch_bnb_gemv():
    """Restore original BnB gemv_4bit."""
    global _original_gemv
    if _original_gemv is not None and _HAS_BNB:
        bnb_F.gemv_4bit = _original_gemv
        _original_gemv = None
        _logger.info("Restored original bnb.functional.gemv_4bit")


def patch_model_gemv(model):
    """Convenience: patch BnB globally so all Linear4bit in *model* use Triton."""
    return patch_bnb_gemv()
