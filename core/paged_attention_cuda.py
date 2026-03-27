"""VRAMancer PagedAttention CUDA kernel — Python wrapper.

Loads the custom CUDA extension (csrc/paged_attention_kernel.cu) via JIT
compilation and exposes a high-level `paged_attention_decode()` API.

Falls back to a pure-PyTorch implementation when the CUDA kernel is
unavailable (compilation error, CPU-only, etc.).
"""
from __future__ import annotations

import math
import os
import logging
from typing import Any, Optional

_logger = logging.getLogger("vramancer.paged_attn_cuda")

# ---------------------------------------------------------------------------
# JIT-compile the CUDA extension
# ---------------------------------------------------------------------------

_cuda_module = None
_LOAD_ATTEMPTED = False


def _detect_and_set_cuda_arch():
    """Auto-detect GPU compute capabilities and set TORCH_CUDA_ARCH_LIST.

    This resolves mismatches between system nvcc version and PyTorch's
    default arch list (e.g., system nvcc 12.0 can't compile sm_120).
    """
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return  # user override

    import torch
    import subprocess

    # Get system nvcc max supported arch
    max_nvcc_arch = 86  # safe default
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                # e.g., "Cuda compilation tools, release 12.0, V12.0.140"
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                major, minor = parts.split(".")[:2]
                cuda_ver = int(major) * 10 + int(minor)
                # CUDA 12.0 → max sm_90, 12.4+ → sm_100, 12.8+ → sm_120
                if cuda_ver >= 128:
                    max_nvcc_arch = 120
                elif cuda_ver >= 124:
                    max_nvcc_arch = 100
                elif cuda_ver >= 118:
                    max_nvcc_arch = 90
                else:
                    max_nvcc_arch = 86
                break
    except Exception:
        pass

    # Collect actual GPU architectures
    archs = set()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = cap[0] * 10 + cap[1]
        archs.add(min(arch, max_nvcc_arch))

    if archs:
        arch_str = ";".join(f"{a // 10}.{a % 10}" for a in sorted(archs))
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str
        _logger.info("Auto-detected CUDA arch list: %s", arch_str)


def _load_cuda_module():
    """Lazy JIT compilation of the CUDA PagedAttention kernel."""
    global _cuda_module, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _cuda_module
    _LOAD_ATTEMPTED = True

    if os.environ.get("VRM_MINIMAL_TEST"):
        return None

    try:
        import torch
        if not torch.cuda.is_available():
            return None

        from torch.utils.cpp_extension import load

        csrc_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "csrc",
        )
        kernel_path = os.path.join(csrc_dir, "paged_attention_kernel.cu")
        if not os.path.exists(kernel_path):
            _logger.warning("CUDA kernel source not found: %s", kernel_path)
            return None

        # Detect supported architectures from current GPU(s),
        # clamped to what the system nvcc can compile.
        _detect_and_set_cuda_arch()

        _cuda_module = load(
            name="vramancer_paged_attn",
            sources=[kernel_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        _logger.info("PagedAttention CUDA kernel compiled successfully")
        return _cuda_module

    except Exception as e:
        _logger.warning("CUDA kernel compilation failed: %s", e)
        return None


def has_cuda_paged_attention() -> bool:
    """Check if the compiled CUDA PagedAttention kernel is available."""
    return _load_cuda_module() is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def paged_attention_decode(
    query: Any,           # [B, H, D] fp32
    kv_pool: Any,         # [P, L, 2, KH, PS, D] fp16 or fp32
    page_table: Any,      # [B, max_pages_per_seq] int32
    context_lens: Any,    # [B] int32
    layer_idx: int,
    scale: Optional[float] = None,
) -> Any:
    """Compute single-token attention directly from paged KV cache.

    This avoids materializing contiguous KV tensors (to_hf_cache).

    Args:
        query: Query tensor [batch, num_heads, head_dim] in fp32.
        kv_pool: Paged KV cache pool [max_pages, num_layers, 2, num_kv_heads, page_size, head_dim].
        page_table: Page indices for each batch element [batch, max_pages_per_seq] as int32.
        context_lens: Context length (num cached tokens) per batch element [batch] as int32.
        layer_idx: Which transformer layer to read KV from.
        scale: Attention scale factor. Defaults to 1/sqrt(head_dim).

    Returns:
        Attention output [batch, num_heads, head_dim] in fp32.
    """
    import torch

    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Ensure correct dtypes
    if page_table.dtype != torch.int32:
        page_table = page_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    if query.dtype != torch.float32:
        query = query.float()

    # Try CUDA kernel
    mod = _load_cuda_module()
    if mod is not None:
        try:
            return mod.paged_attention_decode(
                query.contiguous(),
                kv_pool.contiguous(),
                page_table.contiguous(),
                context_lens.contiguous(),
                layer_idx,
                scale,
            )
        except Exception as e:
            _logger.warning("CUDA kernel failed, falling back to PyTorch: %s", e)

    # Fallback: pure PyTorch implementation
    return _pytorch_paged_attention_decode(
        query, kv_pool, page_table, context_lens, layer_idx, scale
    )


def _pytorch_paged_attention_decode(
    query, kv_pool, page_table, context_lens, layer_idx, scale
):
    """Pure PyTorch fallback for paged attention decode.

    Less efficient (materializes KV per request) but always works.
    """
    import torch

    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = kv_pool.shape[3]
    page_size = kv_pool.shape[4]
    kv_head_ratio = num_heads // num_kv_heads

    outputs = []
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        if ctx_len <= 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=query.device))
            continue

        num_pages = (ctx_len + page_size - 1) // page_size
        page_ids = page_table[b, :num_pages].long()

        # Gather K, V from pool: [num_pages, num_kv_heads, page_size, head_dim]
        k_pages = kv_pool[page_ids, layer_idx, 0].float()  # [P, KH, PS, D]
        v_pages = kv_pool[page_ids, layer_idx, 1].float()

        # Reshape to [KH, total_tokens, D] then trim
        k_flat = k_pages.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len, :]
        v_flat = v_pages.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len, :]

        # Expand for GQA: [KH, S, D] -> [H, S, D]
        if kv_head_ratio > 1:
            k_flat = k_flat.repeat_interleave(kv_head_ratio, dim=0)
            v_flat = v_flat.repeat_interleave(kv_head_ratio, dim=0)

        # Attention: Q [H, 1, D] @ K^T [H, D, S] -> [H, 1, S]
        q = query[b].unsqueeze(1)  # [H, 1, D]
        scores = torch.bmm(q, k_flat.transpose(1, 2)) * scale  # [H, 1, S]
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights, v_flat).squeeze(1)  # [H, D]
        outputs.append(out)

    return torch.stack(outputs)


# ---------------------------------------------------------------------------
# Q4 quantized KV — fused dequantization variant
# ---------------------------------------------------------------------------

def paged_attention_decode_q4(
    query: Any,            # [B, H, D] fp32
    kv_pool_q4: Any,       # [P, L, 2, KH, PS, D/2] uint8 packed nibbles
    kv_scales: Any,        # [P, L, 2, KH, PS, num_groups] fp16
    kv_zeros: Any,         # [P, L, 2, KH, PS, num_groups] fp16
    page_table: Any,       # [B, max_pages_per_seq] int32
    context_lens: Any,     # [B] int32
    layer_idx: int,
    scale: Optional[float] = None,
) -> Any:
    """PagedAttention decode with Q4-quantized KV cache.

    Dequantizes 4-bit packed KV values inline during attention computation,
    avoiding the overhead of a separate decompress-then-attend pass.
    Falls back to dequantize + standard fp32 kernel if CUDA Q4 kernel unavailable.
    """
    import torch

    head_dim = query.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if page_table.dtype != torch.int32:
        page_table = page_table.to(torch.int32)
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    if query.dtype != torch.float32:
        query = query.float()

    # Try fused CUDA Q4 kernel
    mod = _load_cuda_module()
    if mod is not None and hasattr(mod, 'paged_attention_decode_q4'):
        try:
            return mod.paged_attention_decode_q4(
                query.contiguous(),
                kv_pool_q4.contiguous(),
                kv_scales.contiguous(),
                kv_zeros.contiguous(),
                page_table.contiguous(),
                context_lens.contiguous(),
                layer_idx,
                scale,
            )
        except Exception as e:
            _logger.warning("CUDA Q4 kernel failed, falling back: %s", e)

    # Fallback: dequantize Q4 → fp32, then use standard kernel
    return _pytorch_paged_attention_decode_q4(
        query, kv_pool_q4, kv_scales, kv_zeros,
        page_table, context_lens, layer_idx, scale,
    )


def _pytorch_paged_attention_decode_q4(
    query, kv_pool_q4, kv_scales, kv_zeros,
    page_table, context_lens, layer_idx, scale,
):
    """Pure PyTorch fallback: dequantize Q4 then compute attention."""
    import torch

    batch_size, num_heads, head_dim = query.shape
    num_kv_heads = kv_pool_q4.shape[3]
    page_size = kv_pool_q4.shape[4]
    kv_head_ratio = num_heads // num_kv_heads

    def _dequant_page(packed, scales, zeros):
        """Dequantize a packed uint8 page [KH, PS, D/2] → [KH, PS, D] fp32."""
        lo = (packed & 0x0F).float()
        hi = ((packed >> 4) & 0x0F).float()
        # Interleave: [KH, PS, D/2] → [KH, PS, D]
        vals = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
        # Expand scales/zeros from [KH, PS, G] to [KH, PS, D]
        group_size = 32
        sc = scales.float().repeat_interleave(group_size, dim=-1)[..., :vals.shape[-1]]
        zp = zeros.float().repeat_interleave(group_size, dim=-1)[..., :vals.shape[-1]]
        return (vals - zp) * sc

    outputs = []
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        if ctx_len <= 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=query.device))
            continue

        num_pages = (ctx_len + page_size - 1) // page_size
        page_ids = page_table[b, :num_pages].long()

        k_q4 = kv_pool_q4[page_ids, layer_idx, 0]
        v_q4 = kv_pool_q4[page_ids, layer_idx, 1]
        k_sc = kv_scales[page_ids, layer_idx, 0]
        v_sc = kv_scales[page_ids, layer_idx, 1]
        k_zp = kv_zeros[page_ids, layer_idx, 0]
        v_zp = kv_zeros[page_ids, layer_idx, 1]

        k_pages = _dequant_page(k_q4, k_sc, k_zp)
        v_pages = _dequant_page(v_q4, v_sc, v_zp)

        k_flat = k_pages.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len, :]
        v_flat = v_pages.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :ctx_len, :]

        if kv_head_ratio > 1:
            k_flat = k_flat.repeat_interleave(kv_head_ratio, dim=0)
            v_flat = v_flat.repeat_interleave(kv_head_ratio, dim=0)

        q = query[b].unsqueeze(1)
        scores = torch.bmm(q, k_flat.transpose(1, 2)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights, v_flat).squeeze(1)
        outputs.append(out)

    return torch.stack(outputs)


__all__ = ["paged_attention_decode", "paged_attention_decode_q4", "has_cuda_paged_attention"]
