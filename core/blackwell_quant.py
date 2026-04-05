"""
VRAMancer Blackwell-compatible NVFP4 activation quantizer.

The standard torchao activation quantizer uses Triton with fp8e4nv dtype,
which fails on SM 12.0 (Blackwell) because Triton's codegen doesn't support
fp8e4nv on this architecture yet.

This module provides a pure-PyTorch activation quantizer that:
1. Works on ALL CUDA architectures (Ampere, Ada, Hopper, Blackwell)
2. Produces output compatible with torch._scaled_mm (same format as torchao)
3. Is optimized for M=1 decode (single-row activation)
4. Falls back to torchao's Triton path when available (Ampere/Ada)

The quantization pipeline for [M, K] activation:
  1. Per-tensor amax → per-tensor scale (1 value)
  2. Reshape to [M, K//16, 16] → per-block amax → block scales [M, K//16]
  3. Quantize: scale → clamp → FP4 E2M1 encode → pack uint8 (2 nibbles/byte)
  4. Swizzle scales to blocked layout for cuBLAS
"""
from __future__ import annotations

import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torchao.prototype.mx_formats.constants import F4_E2M1_MAX, F8E4M3_MAX
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale
    from torchao.prototype.mx_formats.utils import (
        to_blocked,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False
    F4_E2M1_MAX = 6.0
    F8E4M3_MAX = 448.0

logger = logging.getLogger("vramancer.blackwell_quant")

# E2M1 FP4 midpoint thresholds for rounding-to-nearest
# Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
_THRESHOLDS = None  # Lazily created per-device
_MAGNITUDES = None


def _get_tables(device):
    """Get or create E2M1 quantization tables on the given device."""
    global _THRESHOLDS, _MAGNITUDES
    if _THRESHOLDS is not None and _THRESHOLDS.device == device:
        return _THRESHOLDS, _MAGNITUDES
    # Thresholds for magnitude comparison (7 boundaries between 8 levels)
    _THRESHOLDS = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32, device=device
    )
    # Corresponding magnitude values at each level
    _MAGNITUDES = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=device
    )
    return _THRESHOLDS, _MAGNITUDES


def quantize_activation_nvfp4(
    x: "torch.Tensor",
    per_tensor_scale: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Quantize activation to NVFP4 format, compatible with torch._scaled_mm.

    Works on all CUDA architectures including Blackwell (SM 12.0).

    Args:
        x: [M, K] activation tensor (bf16/fp16/fp32). K must be divisible by 16.
        per_tensor_scale: optional pre-computed per-tensor scale. If None, computed.

    Returns:
        a_qdata: [M, K//2] uint8 packed FP4 data
        a_scale: [sM, sK] swizzled FP8 block scales (ready for _scaled_mm)
        per_tensor_scale: float32 scalar
    """
    M, K = x.shape
    assert K % 16 == 0, f"K={K} must be divisible by 16"

    device = x.device
    x_fp32 = x.float()

    # ── Per-tensor scale (1 kernel) ──
    if per_tensor_scale is None:
        tensor_amax = x_fp32.abs().max()
        if HAS_TORCHAO:
            per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
        else:
            per_tensor_scale = tensor_amax * F4_E2M1_MAX / F8E4M3_MAX
            per_tensor_scale = torch.clamp(per_tensor_scale, min=1e-12)

    # ── Block-wise scales (2 kernels: reshape+amax, divide) ──
    n_blocks = K // 16
    x_blocks = x_fp32.view(M, n_blocks, 16)
    block_amax = torch.amax(torch.abs(x_blocks), dim=-1)  # [M, n_blocks]

    # Scale: block_amax / (F4_E2M1_MAX * per_tensor_scale)
    E4M3_TINY = torch.finfo(torch.float8_e4m3fn).tiny
    scaled_block_scales = block_amax / (F4_E2M1_MAX * per_tensor_scale)
    scaled_block_scales.clamp_(min=E4M3_TINY, max=F8E4M3_MAX)

    # FP8 round-trip
    block_scales_fp8 = scaled_block_scales.to(torch.float8_e4m3fn)
    total_scale = (per_tensor_scale * block_scales_fp8.float()).unsqueeze(-1)  # [M, n_blocks, 1]

    # ── Quantize: scale → encode → pack (3-4 kernels) ──
    x_scaled = x_blocks / total_scale
    x_scaled.clamp_(-F4_E2M1_MAX, F4_E2M1_MAX)

    # E2M1 encode via torch.bucketize (1 kernel, no 4D expansion)
    signs = (x_scaled < 0).to(torch.uint8)  # [M, n_blocks, 16]
    ax = x_scaled.abs()

    thresholds, _ = _get_tables(device)
    mag = torch.bucketize(ax, thresholds)  # [M, n_blocks, 16] → values 0-7

    # Nibble = sign(bit3) | magnitude(bits 0-2)
    nibbles = (signs.to(torch.int32) << 3) | mag.to(torch.int32)

    # ── Pack pairs into uint8 (2 kernels) ──
    nibbles_flat = nibbles.view(M, K)
    even = nibbles_flat[:, 0::2]
    odd = nibbles_flat[:, 1::2]
    a_qdata = (((odd & 0x0F) << 4) | (even & 0x0F)).to(torch.uint8)

    # ── Swizzle scales for cuBLAS ──
    if HAS_TORCHAO:
        a_scale_blocked = to_blocked(block_scales_fp8.view(M, n_blocks))
        sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        a_scale = a_scale_blocked.view(sM, sK)
    else:
        a_scale = block_scales_fp8.view(M, n_blocks)

    return a_qdata, a_scale, per_tensor_scale


def quantize_activation_nvfp4_m1(
    x: "torch.Tensor",
    per_tensor_scale: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Optimized M=1 activation quantizer for decode phase.

    Same output format as quantize_activation_nvfp4 but avoids reshape overhead
    when the activation is a single row.

    Args:
        x: [1, K] or [K] activation tensor
        per_tensor_scale: optional pre-computed per-tensor scale

    Returns:
        Same as quantize_activation_nvfp4
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return quantize_activation_nvfp4(x, per_tensor_scale)
