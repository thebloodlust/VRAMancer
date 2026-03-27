"""
VRAMancer Triton GEMV for NVFP4 weights — M=1 decode bypass.

E2M1 FP4 has exactly 16 possible values → static LUT in L1 cache.
Bypasses cuBLASLt Tensor Core dispatch overhead at batch_size=1 (decode phase).

Weight packing (torchao convention):
  qdata[n, byte_idx]: low nibble (bits 0-3) = weight[n, 2*byte_idx]
                       high nibble (bits 4-7) = weight[n, 2*byte_idx+1]

Scale layout: this kernel expects UN-swizzled [N, K//16] float32 block scales
(not the cuBLAS blocked/swizzled format used by _scaled_mm).
The per_tensor_scale must be pre-multiplied into w_scale_row.
"""
from __future__ import annotations

try:
    import torch
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# E2M1 FP4: 16 values = {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {+, −}
# Nibble encoding: bit3 = sign, bits[2:0] = magnitude index
_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,      # nibble 0–7 (positive)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # nibble 8–15 (negative)
]

_LUT_CACHE: dict = {}


def _get_lut(device) -> "torch.Tensor":
    key = str(device)
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = torch.tensor(
            _E2M1_VALUES, dtype=torch.float32, device=device
        )
    return _LUT_CACHE[key]


if HAS_TRITON:
    @triton.jit
    def _nvfp4_gemv_kernel(
        x_ptr,             # [K] activation (any float → cast to fp32)
        w_qdata_ptr,       # [N, K//2] uint8 packed FP4
        w_scale_ptr,       # [N, K//16] float32 block scales (unswizzled)
        lut_ptr,           # [16] float32 E2M1 lookup table
        out_ptr,           # [N] float32 output
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wq_n: tl.constexpr,
        stride_ws_n: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, 16):
            byte_offs = tl.arange(0, 8)

            # Load 8 packed uint8 bytes → 16 FP4 weight values per row
            w_addr = (
                offs_n[:, None] * stride_wq_n
                + (k_start // 2)
                + byte_offs[None, :]
            )
            w_packed = tl.load(
                w_qdata_ptr + w_addr, mask=mask_n[:, None], other=0
            )

            # Unpack nibbles (torchao: low = even k, high = odd k)
            nib_lo = w_packed & 0x0F
            nib_hi = (w_packed >> 4) & 0x0F

            # LUT decode: nibble index → float32 FP4 value
            w_even = tl.load(lut_ptr + nib_lo)   # [BLOCK_N, 8]
            w_odd = tl.load(lut_ptr + nib_hi)    # [BLOCK_N, 8]

            # Load activation at even/odd K positions
            x_even = tl.load(
                x_ptr + k_start + byte_offs * 2,
                mask=(k_start + byte_offs * 2 < K), other=0.0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + k_start + byte_offs * 2 + 1,
                mask=(k_start + byte_offs * 2 + 1 < K), other=0.0,
            ).to(tl.float32)

            # Dot: 8 even pairs + 8 odd pairs = 16 elements
            dot = tl.sum(
                w_even * x_even[None, :] + w_odd * x_odd[None, :],
                axis=1,
            )

            # Block scale (1 per 16 elements, unswizzled row-major)
            scale = tl.load(
                w_scale_ptr + offs_n * stride_ws_n + (k_start // 16),
                mask=mask_n, other=0.0,
            )
            acc += dot * scale

        tl.store(out_ptr + offs_n, acc, mask=mask_n)


def nvfp4_gemv(
    x: "torch.Tensor",
    w_qdata: "torch.Tensor",
    w_scale_row: "torch.Tensor",
) -> "torch.Tensor":
    """
    GEMV for NVFP4: y = x @ W^T where W is FP4-quantized.

    Args:
        x: [1, K] or [K] activation tensor (bf16/fp16/fp32)
        w_qdata: [N, K//2] uint8 packed FP4 weights
        w_scale_row: [N, K//16] float32 unswizzled block scales
            (per_tensor_scale must be pre-multiplied in)

    Returns:
        [1, N] tensor in x.dtype
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton required for nvfp4_gemv")

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    M, K = x.shape
    N = w_qdata.shape[0]
    assert M == 1, f"GEMV requires M=1, got {M}"
    assert w_qdata.shape == (N, K // 2)
    assert w_scale_row.shape == (N, K // 16)

    lut = _get_lut(x.device)
    out = torch.empty(N, dtype=torch.float32, device=x.device)

    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)

    _nvfp4_gemv_kernel[grid](
        x.contiguous().view(-1),
        w_qdata,
        w_scale_row,
        lut,
        out,
        N=N, K=K,
        stride_wq_n=w_qdata.stride(0),
        stride_ws_n=w_scale_row.stride(0),
        BLOCK_N=BLOCK_N,
    )

    result = out.to(x.dtype).unsqueeze(0)
    return result.squeeze(0) if squeeze else result
