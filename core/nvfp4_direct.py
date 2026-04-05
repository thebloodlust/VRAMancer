"""
VRAMancer Direct FP4 Linear — Bypasses torchao tensor subclass dispatch overhead.

Instead of using NVFP4Tensor (which intercepts every aten op via __torch_dispatch__),
this module:
1. Pre-quantizes weights to FP4 + stores swizzled scales as plain buffers (one-time)
2. Quantizes activations via Triton kernel (fused, no Python loop)
3. Calls torch._scaled_mm directly (no tensor subclass dispatch)

The torchao dispatch chain for each Linear:
  F.linear → __torch_dispatch__(aten.linear) → nvfp4_linear() →
  NVFP4Tensor.to_nvfp4(activation) → _addmm_nvfp4_dispatch() → _scaled_mm

Our bypass:
  forward() → triton_quantize_nvfp4(activation) → _scaled_mm
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except (ImportError, AttributeError):
    HAS_TORCH = False
    # Provide stubs so the class can be defined even with mock torch
    import types as _t
    torch = _t.SimpleNamespace(Tensor=None)  # type: ignore[assignment]
    nn = _t.SimpleNamespace(Module=object)  # type: ignore[assignment]

try:
    from torchao.prototype.mx_formats.constants import F4_E2M1_MAX, F8E4M3_MAX
    from torchao.prototype.mx_formats.kernels import (
        triton_quantize_nvfp4,
        f32_to_f4_unpacked,
        pack_uint4,
    )
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        nvfp4_quantize,
        per_tensor_amax_to_scale,
    )
    from torchao.prototype.mx_formats.utils import (
        to_blocked,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

logger = logging.getLogger("vramancer.nvfp4_direct")

# Device compute capability cache — avoids repeated cudaDeviceGetAttribute calls
_CC_CACHE: dict = {}


def _get_device_cc(device) -> tuple:
    """Get (major, minor) compute capability for a CUDA device, cached."""
    key = str(device)
    if key not in _CC_CACHE:
        _CC_CACHE[key] = torch.cuda.get_device_capability(device) if HAS_TORCH else (0, 0)
    return _CC_CACHE[key]


def _from_blocked(blocked_flat: "torch.Tensor", rows: int, cols: int) -> "torch.Tensor":
    """Inverse of torchao's to_blocked() — unswizzle scales to [rows, cols]."""
    import math as _math
    data = blocked_flat.float().flatten()
    n_row_blocks = _math.ceil(rows / 128)
    n_col_blocks = _math.ceil(cols / 4)
    total_blocks = n_row_blocks * n_col_blocks

    data = data.view(total_blocks, 32, 16)
    data = data.view(total_blocks, 32, 4, 4)
    data = data.transpose(1, 2)                                       # → [B, 4, 32, 4]
    data = data.reshape(n_row_blocks, n_col_blocks, 128, 4)
    data = data.permute(0, 2, 1, 3)                                   # → [R, 128, C, 4]
    data = data.reshape(n_row_blocks * 128, n_col_blocks * 4)
    return data[:rows, :cols].contiguous()


class DirectFP4Linear(nn.Module):
    """
    Direct FP4 Linear that bypasses torchao's __torch_dispatch__ overhead.

    Key insight from torchao source (nvfp4_tensor.py):
    - aten.t.default creates non-contiguous views of qdata and scale
    - _addmm_nvfp4_dispatch passes w.qdata.t().view(fp4x2) (non-contiguous!) to _scaled_mm
    - cuBLAS handles the transpose via column-major layout flags
    - Weight scales stay in original [N, K//16] swizzled orientation

    So we: store original weight data + scales, create .t() view at init, done.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias_data: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Populated by from_nvfp4_tensor() or from_linear()
        self.register_buffer('w_qdata', None)          # [N, K//2] uint8 packed FP4
        self.register_buffer('w_scale', None)           # swizzled scales for [N, K//16]
        self.register_buffer('w_per_tensor_scale', None)  # scalar float32
        self.register_buffer('w_scale_row', None)       # [N, K//16] float32 unswizzled (for GEMV)

        if bias_data is not None:
            self.register_buffer('bias', bias_data)
        else:
            self.bias = None

        self._orig_dtype = torch.bfloat16
        self._use_triton = True

        # Cached views (set in _init_views)
        self._w_fp4_t = None    # w_qdata.t().view(float4_e2m1fn_x2) — non-contiguous
        self._w_scale_fp8 = None  # w_scale.view(float8_e4m3fn),

    def _init_views(self):
        """Create cached non-contiguous views for the weight. No data copy."""
        if self.w_qdata is not None:
            # Non-contiguous transposed view — exactly what torchao passes to _scaled_mm
            self._w_fp4_t = self.w_qdata.t().view(torch.float4_e2m1fn_x2)
            self._w_scale_fp8 = self.w_scale.view(torch.float8_e4m3fn)

    def _apply(self, fn, recurse=True):
        """Override to invalidate cached views when .to(device)/.cuda()/etc. is called."""
        self._w_fp4_t = None
        self._w_scale_fp8 = None
        return super()._apply(fn, recurse=recurse)

    @classmethod
    def from_nvfp4_tensor(cls, linear: nn.Linear, nvfp4_weight) -> "DirectFP4Linear":
        """Extract raw FP4 data from an existing NVFP4Tensor."""
        in_features = nvfp4_weight.shape[1]
        out_features = nvfp4_weight.shape[0]
        bias_data = getattr(linear, 'bias', None)

        mod = cls(in_features, out_features, bias_data=bias_data)
        mod._orig_dtype = nvfp4_weight.orig_dtype
        mod._use_triton = getattr(nvfp4_weight, 'use_triton_kernel', True)

        # Detach from NVFP4Tensor — no clone needed, just take ownership
        # The original NVFP4Tensor will be GC'd after replacement
        mod.w_qdata = nvfp4_weight.qdata.detach()

        # Scales: must be in original [N, K//16] swizzled layout
        if nvfp4_weight.is_swizzled_scales:
            mod.w_scale = nvfp4_weight.scale.detach()
        else:
            N, K = out_features, in_features
            w_scale = nvfp4_weight.scale.view(N, K // 16)
            mod.w_scale = to_blocked(w_scale)

        if nvfp4_weight.per_tensor_scale is not None:
            mod.w_per_tensor_scale = nvfp4_weight.per_tensor_scale.detach()

        # Compute unswizzled row-major scales for GEMV bypass
        # Store as BF16 to halve DRAM bandwidth (FP32 scales = 33% of total traffic)
        try:
            N, K = out_features, in_features
            if nvfp4_weight.is_swizzled_scales:
                scale_flat = nvfp4_weight.scale.detach().flatten()
                unswizzled = _from_blocked(scale_flat, N, K // 16)
            else:
                unswizzled = nvfp4_weight.scale.reshape(N, K // 16).float()
            if nvfp4_weight.per_tensor_scale is not None:
                unswizzled = unswizzled * nvfp4_weight.per_tensor_scale.float()
            mod.w_scale_row = unswizzled.to(torch.bfloat16)
        except Exception:
            pass  # w_scale_row stays None, GEMV bypass disabled

        mod._init_views()
        return mod

    @classmethod
    def from_linear(cls, linear: nn.Linear,
                    device: Optional[torch.device] = None) -> "DirectFP4Linear":
        """Quantize a regular nn.Linear to FP4 and store as plain buffers."""
        weight = linear.weight.data.to(torch.bfloat16)
        if device:
            weight = weight.to(device)

        out_features, in_features = weight.shape
        bias_data = linear.bias.data if linear.bias is not None else None

        mod = cls(in_features, out_features, bias_data=bias_data)
        mod._orig_dtype = torch.bfloat16

        # Per-tensor scale (2-level scaling)
        tensor_amax = torch.max(torch.abs(weight))
        mod.w_per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

        # Quantize weights to FP4
        blockwise_scales, data_lp = nvfp4_quantize(
            weight, block_size=16, per_tensor_scale=mod.w_per_tensor_scale
        )
        mod.w_qdata = data_lp
        if device:
            mod.w_qdata = mod.w_qdata.to(device)

        # Swizzle scales for cuBLAS
        scale_view = blockwise_scales.view(out_features, in_features // 16)
        mod.w_scale = to_blocked(scale_view)
        if device:
            mod.w_scale = mod.w_scale.to(device)

        # Unswizzled scales for GEMV bypass (BF16 to halve DRAM bandwidth)
        row_scale = blockwise_scales.view(out_features, in_features // 16).float()
        if mod.w_per_tensor_scale is not None:
            row_scale = row_scale * mod.w_per_tensor_scale.float()
        mod.w_scale_row = row_scale.to(torch.bfloat16)
        if device:
            mod.w_scale_row = mod.w_scale_row.to(device)

        mod._init_views()
        return mod

    def _quantize_activation_triton(self, x_2d: torch.Tensor):
        """Fused activation quantization — tries fastest path available.

        Priority: Triton fused > CUDA fused > torchao Triton > Python fallback.
        On Blackwell (SM 12.0), Triton fails on fp8e4nv so CUDA fused is used.
        """
        M, K = x_2d.shape

        # --- Fast path 1: VRAMancer Triton single-kernel quantizer ---
        try:
            from core.triton_fused_nvfp4_quant import fused_nvfp4_quantize
            a_qdata, a_scale_raw, act_per_tensor_scale = fused_nvfp4_quantize(x_2d)

            # Convert scales to FP8 and swizzle for cuBLAS
            a_scale_fp8 = a_scale_raw.to(torch.float8_e4m3fn)
            scale_view = a_scale_fp8.view(M, K // 16)
            a_scale = to_blocked(scale_view)

            sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
            a_scale = a_scale.view(sM, sK)

            return a_qdata, a_scale, act_per_tensor_scale
        except Exception:
            pass  # Fall through

        # --- Fast path 2: VRAMancer fused CUDA kernel (works on Blackwell) ---
        try:
            from core.fused_act_quant import fused_quantize_activation_nvfp4
            a_qdata, a_scale, act_per_tensor_scale = fused_quantize_activation_nvfp4(x_2d)
            return a_qdata, a_scale, act_per_tensor_scale
        except Exception:
            pass  # Fall through

        # --- Standard path: torchao's triton_quantize_nvfp4 ---
        # Per-tensor scale
        tensor_amax = torch.max(torch.abs(x_2d))
        act_per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

        # Fused Triton kernel: quantize + pack + swizzle scales in one pass
        a_scale, a_qdata = triton_quantize_nvfp4(x_2d, act_per_tensor_scale)

        # Reshape scales to canonical swizzled dimensions
        sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        a_scale = a_scale.view(sM, sK)

        return a_qdata, a_scale, act_per_tensor_scale

    def _quantize_activation_python(self, x_2d: torch.Tensor):
        """Fallback Python activation quantization (when Triton unavailable)."""
        M, K = x_2d.shape

        tensor_amax = torch.max(torch.abs(x_2d))
        act_per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)

        # Block-wise quantization
        x_fp32 = x_2d.float().reshape(M, -1, 16)
        max_abs = torch.amax(torch.abs(x_fp32), dim=-1)
        block_scale = max_abs / F4_E2M1_MAX

        scaled_block_scales = block_scale / act_per_tensor_scale
        E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
        scaled_block_scales_fp8 = torch.clamp(
            scaled_block_scales, min=E4M3_EPS, max=F8E4M3_MAX
        ).to(torch.float8_e4m3fn)

        total_scale = act_per_tensor_scale * scaled_block_scales_fp8.to(torch.float32)
        x_scaled = x_fp32 / total_scale.unsqueeze(-1)
        x_scaled = torch.clamp(x_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)

        data_lp = f32_to_f4_unpacked(x_scaled.view(M, K))
        a_qdata = pack_uint4(data_lp)

        scale_view = scaled_block_scales_fp8.view(M, K // 16)
        a_scale = to_blocked(scale_view)

        return a_qdata, a_scale, act_per_tensor_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Direct FP4 forward — bypasses torchao __torch_dispatch__.

        Priority chain:
          0. Blackwell (CC>=10): Fused W4A16 GEMM for ALL batch sizes
             → On-the-fly FP4→BF16 dequant + BF16 tensor cores in one kernel.
             → Bypasses the broken _scaled_mm CUTLASS FP4 path entirely.
             → Novel approach: no other framework has this.
          1. M=1: Triton GEMV (W4A16 LUT kernel, non-Blackwell fallback)
          2. M>1: W4A4 via torch._scaled_mm (non-Blackwell)
          3. Emergency fallback: W4A16 split-K GEMM
        """
        # --- Path 0: Fused W4A16 GEMM on Blackwell (ALL batch sizes) ---
        # Uses BF16 tensor cores (mature) with on-the-fly FP4 weight dequant.
        # Eliminates: activation quantization, _scaled_mm, CUTLASS FP4 dependency.
        if self.w_scale_row is not None and x.is_cuda:
            cc = _get_device_cc(x.device)
            if cc[0] >= 10:  # Blackwell SM 12.0+
                orig_shape = x.shape
                x_2d = x.reshape(-1, self.in_features)
                M = x_2d.shape[0]

                # Priority A: CUDA C kernel (hand-optimized, fastest)
                try:
                    if M == 1:
                        from core.fp4_cuda import fp4_gemv_cuda
                        result = fp4_gemv_cuda(
                            x_2d.squeeze(0), self.w_qdata, self.w_scale_row
                        ).unsqueeze(0)
                    else:
                        from core.fp4_cuda import fp4_dequant_gemm
                        result = fp4_dequant_gemm(
                            x_2d, self.w_qdata, self.w_scale_row
                        )
                    result = result.to(self._orig_dtype)
                    if self.bias is not None:
                        result = result + self.bias.to(self._orig_dtype)
                    return result.reshape(*orig_shape[:-1], self.out_features)
                except Exception:
                    pass

                # Priority B: Triton fused GEMM (fallback if CUDA JIT fails)
                try:
                    from core.triton_fp4_gemm import fp4_fused_gemm
                    result = fp4_fused_gemm(x_2d, self.w_qdata, self.w_scale_row)
                    result = result.to(self._orig_dtype)
                    if self.bias is not None:
                        result = result + self.bias.to(self._orig_dtype)
                    return result.reshape(*orig_shape[:-1], self.out_features)
                except Exception:
                    pass  # Fall through to legacy paths

        # --- Path 1: M=1 GEMV (W4A16, non-Blackwell fallback) ---
        if self.w_scale_row is not None:
            x_flat = x.reshape(-1, self.in_features) if x.dim() != 2 else x
            if x_flat.shape[0] == 1:
                try:
                    from core.triton_gemv_nvfp4 import nvfp4_gemv
                    result = nvfp4_gemv(x_flat, self.w_qdata, self.w_scale_row)
                    if self.bias is not None:
                        result = result + self.bias.to(result.dtype)
                    if x.dim() != 2:
                        result = result.reshape(*x.shape[:-1], self.out_features)
                    return result
                except Exception:
                    pass  # Fall through

        # --- Path 2: W4A4 via torch._scaled_mm (M>1 prefill) ---
        if self._w_fp4_t is None:
            self._init_views()

        if self._w_fp4_t is not None:
            orig_shape = x.shape
            x_2d = x.reshape(-1, self.in_features)

            # Quantize activation (Triton > Python fallback)
            if self._use_triton:
                a_qdata, a_scale, act_pts = self._quantize_activation_triton(x_2d)
            else:
                a_qdata, a_scale, act_pts = self._quantize_activation_python(x_2d)

            # cuBLAS FP4 matmul
            result = torch._scaled_mm(
                a_qdata.view(torch.float4_e2m1fn_x2),
                self._w_fp4_t,
                a_scale.view(torch.float8_e4m3fn),
                self._w_scale_fp8,
                bias=None,
                out_dtype=self._orig_dtype,
            )

            # Per-tensor scale product
            if self.w_per_tensor_scale is not None:
                scale_product = act_pts * self.w_per_tensor_scale
                result = result * scale_product.to(self._orig_dtype)

            if self.bias is not None:
                result = result + self.bias.to(self._orig_dtype)

            return result.reshape(*orig_shape[:-1], self.out_features)

        # --- Path 3 (emergency): W4A16 GEMM if _scaled_mm unavailable ---
        if self.w_scale_row is not None:
            try:
                from core.triton_fp4_gemm import fp4_dequant_gemm
                orig_shape = x.shape
                x_2d = x.reshape(-1, self.in_features)
                result = fp4_dequant_gemm(x_2d, self.w_qdata, self.w_scale_row)
                result = result.to(self._orig_dtype)
                if self.bias is not None:
                    result = result + self.bias.to(self._orig_dtype)
                return result.reshape(*orig_shape[:-1], self.out_features)
            except Exception:
                pass

        raise RuntimeError("No viable FP4 computation path available")


# ── Dynamo safety ─────────────────────────────────────────────
# Mark DirectFP4Linear.forward as opaque to torch._dynamo so that
# torch.compile does not trace into the complex kernel dispatch
# logic (try/except, dynamic imports, CUDA API calls).  Without
# this, Dynamo tracing hangs on DirectFP4Linear's custom ops.
# CUDA Graph capture is unaffected — it records GPU ops directly.
if HAS_TORCH:
    try:
        import torch._dynamo
        DirectFP4Linear.forward = torch._dynamo.disable(
            DirectFP4Linear.forward
        )
    except (AttributeError, TypeError, ImportError):
        pass  # torch._dynamo not available


def replace_with_direct_fp4(model: "nn.Module", verbose: bool = True) -> int:
    """
    Replace NVFP4Tensor-quantized Linear layers with DirectFP4Linear.

    Walks the module tree, finds layers with NVFP4Tensor weights,
    replaces them with DirectFP4Linear storing plain buffers.
    """
    if not HAS_TORCHAO:
        logger.warning("torchao not available, cannot replace layers")
        return 0

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not isinstance(module.weight, NVFP4Tensor):
            continue

        direct = DirectFP4Linear.from_nvfp4_tensor(module, module.weight)

        # Replace in parent
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], direct)

        if verbose:
            logger.info(f"Replaced {name}: [{module.out_features}, {module.in_features}]")
        replaced += 1

    if verbose:
        logger.info(f"Total replaced: {replaced}")

    # Free stale NVFP4Tensor references
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return replaced


def compile_for_decode(
    model: "nn.Module",
    max_cache_len: int = 2048,
    device: "Optional[torch.device]" = None,
) -> "tuple[callable, object]":
    """
    Prepare a DirectFP4 model for compiled decode (11x faster on Blackwell).

    Uses StaticCache + torch.compile(mode='reduce-overhead') to eliminate
    Python dispatch overhead by capturing the entire decode step as a CUDA Graph.

    Returns:
        compiled_fn: callable(token_ids, cache, cache_position) -> logits
        static_cache: StaticCache to pass to generate() or use manually

    Usage::

        model = load_and_quantize(...)
        replace_with_direct_fp4(model)
        model = model.to(device)

        compiled_decode, cache = compile_for_decode(model, device=device)

        # Use with transformers generate():
        output = model.generate(
            **inputs,
            past_key_values=cache,
            cache_implementation="static",
            compile_prefill=False,
        )
    """
    from transformers import StaticCache

    if device is None:
        device = next(model.parameters()).device

    config = model.config
    static_cache = StaticCache(
        config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.set_float32_matmul_precision("high")

    def _decode_step(model, input_ids, cache, cache_position):
        return model(
            input_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

    compiled_fn = torch.compile(
        _decode_step, mode="reduce-overhead", fullgraph=False
    )

    logger.info(
        "Compiled decode ready (StaticCache max_len=%d, device=%s)",
        max_cache_len, device,
    )
    return compiled_fn, static_cache
