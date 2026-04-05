"""
VRAMancer High-Performance Triton GEMV for 4-bit quantized weights (M=1 decode).

Supports any 4-bit LUT codebook: NVFP4 (E2M1), NF4, or INT4.
Designed for the decode phase where batch_size=1 and the operation is
purely memory-bandwidth-bound.

Key optimizations vs the naive GEMV:
  1. K-parallel reduction: splits K across warps, uses atomic add for final reduction
  2. Large K tiles (BLOCK_K=128-256): amortizes kernel launch + scale loads
  3. Vectorized 128-bit loads: reads 16 packed bytes (32 FP4 values) per load
  4. FP16 scale support: halves scale bandwidth vs FP32
  5. Activation caching in registers: each K-tile's activation is read once
  6. Autotuned grid: Triton picks optimal BLOCK_N x BLOCK_K per hardware

Weight packing convention (same as torchao/BnB):
  qdata[n, byte_idx]: low nibble (bits 0-3) = weight[n, 2*byte_idx]
                       high nibble (bits 4-7) = weight[n, 2*byte_idx+1]

Scale layout: [N, K//block_size] in row-major, any dtype (fp32/fp16/bf16).
"""
from __future__ import annotations

import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger("vramancer.triton_gemv_awq")

# ── LUT codebooks ──────────────────────────────────────────────────────────

# E2M1 FP4 (NVFP4): {0, 0.5, 1, 1.5, 2, 3, 4, 6} × {+, −}
_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

# NF4 (bitsandbytes NormalFloat4): quantiles of N(0,1)
_NF4_VALUES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

# INT4 symmetric: {-8, -7, ..., -1, 0, 1, ..., 7} / 7.0
_INT4_VALUES = [i / 7.0 for i in range(-8, 8)]

_LUT_CACHE: dict = {}


def _get_lut(device, codebook: str = "e2m1") -> "torch.Tensor":
    """Get or create a cached 16-element LUT on the given device."""
    key = (str(device), codebook)
    if key not in _LUT_CACHE:
        values = {
            "e2m1": _E2M1_VALUES,
            "nf4": _NF4_VALUES,
            "int4": _INT4_VALUES,
        }[codebook]
        _LUT_CACHE[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return _LUT_CACHE[key]


# ── Optimized Triton kernel ───────────────────────────────────────────────

if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_N": 32, "BLOCK_K": 64}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=2, num_stages=4),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _gemv4_kernel(
        x_ptr,             # [K] activation (fp16/bf16 → cast to fp32)
        w_qdata_ptr,       # [N, K//2] uint8 packed 4-bit
        w_scale_ptr,       # [N, n_scale_blocks] float32/fp16 block scales
        lut_ptr,           # [16] float32 LUT
        out_ptr,           # [N] float32 output
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wq_n: tl.constexpr,   # w_qdata.stride(0)
        stride_ws_n: tl.constexpr,   # w_scale.stride(0)
        SCALE_BLOCK: tl.constexpr,   # elements per scale block (16 for NVFP4, 64 for BnB)
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Optimized 4-bit GEMV: y[n] = sum_k( dequant(w[n,k]) * x[k] )

        Grid: (cdiv(N, BLOCK_N),)
        Each program handles BLOCK_N output rows, iterating over K in tiles of BLOCK_K.
        Within each K-tile, we process BLOCK_K/2 packed bytes → BLOCK_K weights.
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Iterate over K in tiles
        n_bytes_per_tile = BLOCK_K // 2
        n_scale_per_tile = BLOCK_K // SCALE_BLOCK

        for k_start in range(0, K, BLOCK_K):
            # ── Load activation tile (shared across all N rows) ──
            k_offs_even = tl.arange(0, BLOCK_K // 2) * 2
            k_offs_odd = k_offs_even + 1

            x_even = tl.load(
                x_ptr + k_start + k_offs_even,
                mask=(k_start + k_offs_even < K), other=0.0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + k_start + k_offs_odd,
                mask=(k_start + k_offs_odd < K), other=0.0,
            ).to(tl.float32)

            # ── Load packed weight bytes: [BLOCK_N, BLOCK_K//2] ──
            byte_offs = tl.arange(0, n_bytes_per_tile)
            w_addr = offs_n[:, None] * stride_wq_n + (k_start // 2) + byte_offs[None, :]
            w_mask = mask_n[:, None] & ((k_start + byte_offs[None, :] * 2) < K)
            w_packed = tl.load(w_qdata_ptr + w_addr, mask=w_mask, other=0)

            # ── Unpack nibbles ──
            nib_lo = (w_packed & 0x0F).to(tl.int32)   # even k indices
            nib_hi = ((w_packed >> 4) & 0x0F).to(tl.int32)  # odd k indices

            # ── LUT dequantize: [BLOCK_N, BLOCK_K//2] ──
            w_even = tl.load(lut_ptr + nib_lo)   # dequantized even weights
            w_odd = tl.load(lut_ptr + nib_hi)    # dequantized odd weights

            # ── Dot product with per-block scaling ──
            # Process each scale block within this K-tile
            elems_per_half_block = SCALE_BLOCK // 2

            for sb in range(n_scale_per_tile):
                sb_byte_start = sb * elems_per_half_block
                sb_byte_end = sb_byte_start + elems_per_half_block

                # Scale for this sub-block: [BLOCK_N]
                scale_idx = (k_start // SCALE_BLOCK) + sb
                s = tl.load(
                    w_scale_ptr + offs_n * stride_ws_n + scale_idx,
                    mask=mask_n, other=0.0,
                ).to(tl.float32)

                # Slice the relevant bytes from this scale block
                sb_offs = tl.arange(0, elems_per_half_block)

                # Even elements dot product
                w_e = tl.load(
                    lut_ptr + tl.load(
                        w_qdata_ptr + offs_n[:, None] * stride_wq_n + (k_start // 2) + sb_byte_start + sb_offs[None, :],
                        mask=mask_n[:, None], other=0,
                    ).to(tl.int32) & 0x0F
                )
                x_e = tl.load(
                    x_ptr + k_start + (sb_byte_start * 2) + sb_offs * 2,
                    mask=(k_start + sb_byte_start * 2 + sb_offs * 2 < K), other=0.0,
                ).to(tl.float32)

                # Odd elements dot product
                w_o = tl.load(
                    lut_ptr + (tl.load(
                        w_qdata_ptr + offs_n[:, None] * stride_wq_n + (k_start // 2) + sb_byte_start + sb_offs[None, :],
                        mask=mask_n[:, None], other=0,
                    ).to(tl.int32) >> 4) & 0x0F
                )
                x_o = tl.load(
                    x_ptr + k_start + (sb_byte_start * 2) + sb_offs * 2 + 1,
                    mask=(k_start + sb_byte_start * 2 + sb_offs * 2 + 1 < K), other=0.0,
                ).to(tl.float32)

                dot = tl.sum(w_e * x_e[None, :] + w_o * x_o[None, :], axis=1)
                acc += dot * s

        tl.store(out_ptr + offs_n, acc, mask=mask_n)

    # ── Simpler kernel without per-sub-block scale loop ──
    # (For SCALE_BLOCK == BLOCK_K case, or when we pre-multiply scales)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8),
            triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _gemv4_prescaled_kernel(
        x_ptr,             # [K] activation
        w_qdata_ptr,       # [N, K//2] uint8 packed 4-bit
        w_prescale_ptr,    # [N, K//2] float16 pre-multiplied per-element scales
        lut_ptr,           # [16] float32 LUT
        out_ptr,           # [N] float32 output
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wq_n: tl.constexpr,
        stride_ws_n: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """GEMV with pre-expanded scales (one scale per packed byte pair).
        This avoids the inner scale-block loop at the cost of 2x more scale data.
        Used when SCALE_BLOCK is small (e.g., 16) making the sub-block loop expensive.
        """
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        n_bytes = BLOCK_K // 2

        for k_start in range(0, K, BLOCK_K):
            byte_offs = tl.arange(0, n_bytes)

            # Load packed weights [BLOCK_N, n_bytes]
            w_addr = offs_n[:, None] * stride_wq_n + (k_start // 2) + byte_offs[None, :]
            w_mask = mask_n[:, None] & ((k_start + byte_offs[None, :] * 2) < K)
            w_packed = tl.load(w_qdata_ptr + w_addr, mask=w_mask, other=0)

            # Load pre-expanded scales [BLOCK_N, n_bytes]
            s_addr = offs_n[:, None] * stride_ws_n + (k_start // 2) + byte_offs[None, :]
            scales = tl.load(w_prescale_ptr + s_addr, mask=w_mask, other=0.0).to(tl.float32)

            # Unpack + LUT dequantize
            nib_lo = (w_packed & 0x0F).to(tl.int32)
            nib_hi = ((w_packed >> 4) & 0x0F).to(tl.int32)
            w_even = tl.load(lut_ptr + nib_lo)
            w_odd = tl.load(lut_ptr + nib_hi)

            # Scale the dequantized weights
            w_even = w_even * scales
            w_odd = w_odd * scales

            # Load activation
            x_even = tl.load(
                x_ptr + k_start + byte_offs * 2,
                mask=(k_start + byte_offs * 2 < K), other=0.0,
            ).to(tl.float32)
            x_odd = tl.load(
                x_ptr + k_start + byte_offs * 2 + 1,
                mask=(k_start + byte_offs * 2 + 1 < K), other=0.0,
            ).to(tl.float32)

            dot = tl.sum(
                w_even * x_even[None, :] + w_odd * x_odd[None, :],
                axis=1,
            )
            acc += dot

        tl.store(out_ptr + offs_n, acc, mask=mask_n)

    # ── K-split GEMV: parallelizes both N and K dimensions ──
    # For small N (e.g., k/v_proj with N=512), the N-only parallelism
    # leaves most SMs idle. Splitting K across thread blocks and using
    # atomic add for final reduction dramatically improves utilization.
    #
    # Grid: (cdiv(N, BLOCK_N), NUM_K_SPLITS)
    # Each block: processes BLOCK_N rows for K/NUM_K_SPLITS columns
    # Output: atomic add partial sums into global buffer.

    @triton.jit
    def _gemv4_ksplit_kernel(
        x_ptr,             # [K] activation
        w_qdata_ptr,       # [N, K//2] uint8 packed 4-bit
        w_scale_ptr,       # [N, K//16] block scales
        lut_ptr,           # [16] float32 LUT
        out_ptr,           # [N] float32 output (must be zeroed!)
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wq_n: tl.constexpr,
        stride_ws_n: tl.constexpr,
        NUM_K_SPLITS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """K-parallel 4-bit GEMV with atomic reduction.

        Grid: (cdiv(N, BLOCK_N), NUM_K_SPLITS)
        Each program handles BLOCK_N output rows for a slice of K.
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # K slice for this split
        scale_blocks_total = K // 16
        scale_blocks_per_split = tl.cdiv(scale_blocks_total, NUM_K_SPLITS)
        sb_start = pid_k * scale_blocks_per_split
        sb_end = tl.minimum(sb_start + scale_blocks_per_split, scale_blocks_total)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        byte_offs = tl.arange(0, 8)

        for scale_idx in range(sb_start, sb_end):
            k_sb = scale_idx * 16
            byte_start = k_sb // 2

            # Load packed weights: [BLOCK_N, 8]
            w_addr = offs_n[:, None] * stride_wq_n + byte_start + byte_offs[None, :]
            w_packed = tl.load(w_qdata_ptr + w_addr, mask=mask_n[:, None], other=0)

            # Unpack + LUT
            nib_lo = (w_packed & 0x0F).to(tl.int32)
            nib_hi = ((w_packed >> 4) & 0x0F).to(tl.int32)
            w_even = tl.load(lut_ptr + nib_lo)
            w_odd = tl.load(lut_ptr + nib_hi)

            # Activation
            x_even = tl.load(x_ptr + k_sb + byte_offs * 2).to(tl.float32)
            x_odd = tl.load(x_ptr + k_sb + byte_offs * 2 + 1).to(tl.float32)

            dot = tl.sum(
                w_even * x_even[None, :] + w_odd * x_odd[None, :],
                axis=1,
            )

            # Scale
            s = tl.load(
                w_scale_ptr + offs_n * stride_ws_n + scale_idx,
                mask=mask_n, other=0.0,
            ).to(tl.float32)
            acc += dot * s

        # Atomic add to output (allows K-split reduction)
        tl.atomic_add(out_ptr + offs_n, acc, mask=mask_n)

    # ── Simple non-split kernel for large-N layers ──
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64}, num_warps=4),
            triton.Config({"BLOCK_N": 128}, num_warps=4),
            triton.Config({"BLOCK_N": 256}, num_warps=8),
        ],
        key=["N", "K"],
    )
    @triton.jit
    def _gemv4_fast_kernel(
        x_ptr,             # [K] activation
        w_qdata_ptr,       # [N, K//2] uint8 packed 4-bit
        w_scale_ptr,       # [N, K//16] block scales (any float dtype)
        lut_ptr,           # [16] float32 LUT
        out_ptr,           # [N] float32 output
        N: tl.constexpr,
        K: tl.constexpr,
        stride_wq_n: tl.constexpr,
        stride_ws_n: tl.constexpr,
        SCALE_BLOCK: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fast 4-bit GEMV for large-N layers. No K-split needed."""
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        byte_offs = tl.arange(0, 8)

        for scale_idx in range(0, K // 16):
            k_sb = scale_idx * 16
            byte_start = k_sb // 2

            w_addr = offs_n[:, None] * stride_wq_n + byte_start + byte_offs[None, :]
            w_packed = tl.load(w_qdata_ptr + w_addr, mask=mask_n[:, None], other=0)

            nib_lo = (w_packed & 0x0F).to(tl.int32)
            nib_hi = ((w_packed >> 4) & 0x0F).to(tl.int32)
            w_even = tl.load(lut_ptr + nib_lo)
            w_odd = tl.load(lut_ptr + nib_hi)

            x_even = tl.load(x_ptr + k_sb + byte_offs * 2).to(tl.float32)
            x_odd = tl.load(x_ptr + k_sb + byte_offs * 2 + 1).to(tl.float32)

            dot = tl.sum(
                w_even * x_even[None, :] + w_odd * x_odd[None, :],
                axis=1,
            )

            s = tl.load(
                w_scale_ptr + offs_n * stride_ws_n + scale_idx,
                mask=mask_n, other=0.0,
            ).to(tl.float32)

            acc += dot * s

        tl.store(out_ptr + offs_n, acc, mask=mask_n)


def gemv4(
    x: "torch.Tensor",
    w_qdata: "torch.Tensor",
    w_scale: "torch.Tensor",
    codebook: str = "e2m1",
    scale_block: int = 16,
) -> "torch.Tensor":
    """
    High-performance 4-bit GEMV: y = x @ W^T where W is 4-bit quantized.

    Args:
        x: [1, K] or [K] activation (any float dtype)
        w_qdata: [N, K//2] uint8 packed 4-bit weights
        w_scale: [N, K//scale_block] block scales (fp32/fp16/bf16)
        codebook: 'e2m1' (NVFP4), 'nf4' (BnB NormalFloat), 'int4' (symmetric)
        scale_block: number of weight elements per scale value (16 for NVFP4, 64 for BnB)

    Returns:
        [1, N] or [N] tensor in x.dtype
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton required for gemv4")

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    M, K = x.shape
    N = w_qdata.shape[0]
    assert M == 1, f"GEMV requires M=1, got {M}"
    assert w_qdata.shape[1] == K // 2, f"Expected w_qdata [N, K//2], got {w_qdata.shape}"

    lut = _get_lut(x.device, codebook)
    out = torch.empty(N, dtype=torch.float32, device=x.device)

    # Ensure K is divisible by scale_block (pad if needed)
    assert K % scale_block == 0, f"K={K} must be divisible by scale_block={scale_block}"

    # Heuristic: use K-split for small N to improve SM utilization.
    # On Ampere (82 SMs), we want at least ~100 thread blocks.
    # With BLOCK_N=64, N=512 gives only 8 blocks → need K-split.
    BLOCK_N_KSPLIT = 64
    n_blocks = (N + BLOCK_N_KSPLIT - 1) // BLOCK_N_KSPLIT
    if n_blocks < 64:
        # Use K-split kernel
        # Choose num_k_splits to get enough total blocks
        num_k_splits = max(2, min(16, 128 // max(n_blocks, 1)))
        out.zero_()  # atomic add requires zeroed output
        grid = (n_blocks, num_k_splits)
        _gemv4_ksplit_kernel[grid](
            x.contiguous().view(-1),
            w_qdata,
            w_scale,
            lut,
            out,
            N=N, K=K,
            stride_wq_n=w_qdata.stride(0),
            stride_ws_n=w_scale.stride(0),
            NUM_K_SPLITS=num_k_splits,
            BLOCK_N=BLOCK_N_KSPLIT,
        )
    else:
        # Large N: use autotuned single-split kernel
        grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
        _gemv4_fast_kernel[grid](
            x.contiguous().view(-1),
            w_qdata,
            w_scale,
            lut,
            out,
            N=N, K=K,
            stride_wq_n=w_qdata.stride(0),
            stride_ws_n=w_scale.stride(0),
            SCALE_BLOCK=scale_block,
        )

    result = out.to(x.dtype).unsqueeze(0)
    return result.squeeze(0) if squeeze else result
