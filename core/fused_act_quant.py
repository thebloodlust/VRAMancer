"""
VRAMancer fused CUDA activation quantizer for NVFP4.

Single kernel: reads BF16/FP16 activation, outputs packed FP4 + FP8 scales.
Works on ALL CUDA architectures. Eliminates the 15+ kernel launch overhead
of the PyTorch-ops version (543 us → target <30 us).
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger("vramancer.fused_act_quant")

_EXTENSION = None
_EXTENSION_LOAD_ATTEMPTED = False


def _load_extension():
    """JIT compile the CUDA extension."""
    global _EXTENSION, _EXTENSION_LOAD_ATTEMPTED
    if _EXTENSION_LOAD_ATTEMPTED:
        return _EXTENSION
    _EXTENSION_LOAD_ATTEMPTED = True

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError:
        return None

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr float F4_E2M1_MAX = 6.0f;
constexpr float F8E4M3_MAX = 448.0f;
constexpr float F8E4M3_TINY = 1.175494e-38f;
constexpr int QUANT_BLOCK = 16;

__device__ __forceinline__ int quantize_e2m1(float val) {
    int sign = (val < 0.0f) ? 8 : 0;
    float ax = fabsf(val);
    int mag = 0;
    mag += (ax >= 0.25f);
    mag += (ax >= 0.75f);
    mag += (ax >= 1.25f);
    mag += (ax >= 1.75f);
    mag += (ax >= 2.5f);
    mag += (ax >= 3.5f);
    mag += (ax >= 5.0f);
    return sign | mag;
}

__device__ __forceinline__ float read_val(const void* ptr, int idx, bool is_bf16) {
    if (is_bf16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    } else {
        return __half2float(reinterpret_cast<const half*>(ptr)[idx]);
    }
}

// Single-kernel fused quantizer for M=1.
// Layout:
//   shared[0..n_blocks-1] = block amaxes (float)
//   shared[n_blocks]      = global max amax (int bits for atomicMax)
__global__ void fused_quant_kernel(
    const void* __restrict__ x_ptr,
    uint8_t* __restrict__ out_qdata,     // [K/2]
    float* __restrict__ out_block_scales, // [n_blocks]
    float* __restrict__ out_pts,         // [1] per_tensor_scale
    int K, int n_blocks, bool is_bf16
) {
    extern __shared__ char smem_raw[];
    float* block_amax = reinterpret_cast<float*>(smem_raw);
    int* global_max_int = reinterpret_cast<int*>(&block_amax[n_blocks]);

    int tid = threadIdx.x;

    // Init shared
    if (tid == 0) *global_max_int = 0;
    __syncthreads();

    // Phase 1: compute block amaxes
    for (int blk = tid; blk < n_blocks; blk += blockDim.x) {
        int base = blk * QUANT_BLOCK;
        float bmax = 0.0f;
        for (int i = 0; i < QUANT_BLOCK; i++) {
            bmax = fmaxf(bmax, fabsf(read_val(x_ptr, base + i, is_bf16)));
        }
        block_amax[blk] = bmax;
        // Float atomicMax via int reinterpret (works for non-negative floats)
        atomicMax(global_max_int, __float_as_int(bmax));
    }
    __syncthreads();

    // Compute per-tensor scale (torchao convention: amax / F8E4M3_MAX / F4_E2M1_MAX)
    float tensor_amax = __int_as_float(*global_max_int);
    float pts = tensor_amax / F8E4M3_MAX / F4_E2M1_MAX;
    pts = fmaxf(pts, 1e-12f);

    if (tid == 0) out_pts[0] = pts;

    // Phase 2: quantize and pack
    for (int blk = tid; blk < n_blocks; blk += blockDim.x) {
        int base = blk * QUANT_BLOCK;
        float bmax = block_amax[blk];

        // Block scale = bmax / F4_E2M1_MAX (range of values this block covers)
        float raw_block_scale = bmax / F4_E2M1_MAX;

        // Scaled block scale = raw / pts (for FP8 storage)
        float sbs = raw_block_scale / pts;
        sbs = fminf(fmaxf(sbs, F8E4M3_TINY), F8E4M3_MAX);
        out_block_scales[blk] = sbs;

        // Total scale = pts * sbs ≈ raw_block_scale
        float total_scale = pts * sbs;
        float inv_scale = (total_scale > 1e-30f) ? (1.0f / total_scale) : 0.0f;

        // Quantize 16 values → 8 packed bytes
        for (int pair = 0; pair < QUANT_BLOCK / 2; pair++) {
            float v0 = read_val(x_ptr, base + pair * 2, is_bf16) * inv_scale;
            float v1 = read_val(x_ptr, base + pair * 2 + 1, is_bf16) * inv_scale;
            v0 = fminf(fmaxf(v0, -F4_E2M1_MAX), F4_E2M1_MAX);
            v1 = fminf(fmaxf(v1, -F4_E2M1_MAX), F4_E2M1_MAX);
            int n0 = quantize_e2m1(v0);
            int n1 = quantize_e2m1(v1);
            out_qdata[base / 2 + pair] = (uint8_t)((n1 << 4) | (n0 & 0x0F));
        }
    }
}

std::vector<torch::Tensor> fused_activation_quant_nvfp4(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2, "Expected [M, K]");
    TORCH_CHECK(x.size(0) == 1, "Only M=1 supported");
    int K = x.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");

    bool is_bf16 = (x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(is_bf16 || x.scalar_type() == torch::kFloat16, "bf16 or fp16 required");

    // Guard: set CUDA device to match tensor's device
    const at::cuda::CUDAGuard device_guard(x.device());

    int n_blocks = K / 16;
    auto dev = x.device();
    auto qdata = torch::empty({1, K / 2}, torch::dtype(torch::kUInt8).device(dev));
    auto scales = torch::empty({n_blocks}, torch::dtype(torch::kFloat32).device(dev));
    auto pts = torch::empty({1}, torch::dtype(torch::kFloat32).device(dev));

    int nthreads = ((std::min(n_blocks, 1024) + 31) / 32) * 32;
    nthreads = std::max(nthreads, 32);
    int smem = (n_blocks + 1) * sizeof(float);

    fused_quant_kernel<<<1, nthreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr(), qdata.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), pts.data_ptr<float>(),
        K, n_blocks, is_bf16
    );
    return {qdata, scales, pts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_activation_quant_nvfp4", &fused_activation_quant_nvfp4,
          "Fused NVFP4 activation quantization (CUDA)");
}
"""

    cpp_src = ""  # All in CUDA file

    # Write to temp file
    src_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "fused_act_quant")
    os.makedirs(src_dir, exist_ok=True)
    cuda_path = os.path.join(src_dir, "fused_act_quant.cu")
    with open(cuda_path, 'w') as f:
        f.write(cuda_src)

    try:
        # Find ninja
        try:
            from ninja import BIN_DIR
            os.environ["PATH"] = f"{BIN_DIR}:{os.environ.get('PATH', '')}"
        except ImportError:
            pass

        _EXTENSION = load(
            name="fused_act_quant",
            sources=[cuda_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        logger.info("Fused activation quantizer CUDA extension loaded")
    except Exception as e:
        logger.warning("Failed to compile fused activation quantizer: %s", e)
        _EXTENSION = None

    return _EXTENSION


def fused_quantize_activation_nvfp4(
    x: "torch.Tensor",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Fused single-kernel NVFP4 activation quantization.

    Returns:
        a_qdata: [1, K//2] uint8 packed FP4
        a_scale: [sM, sK] swizzled FP8 scales (ready for _scaled_mm)
        per_tensor_scale: float32 scalar
    """
    import torch

    ext = _load_extension()
    if ext is None:
        # Fallback to PyTorch implementation
        from core.blackwell_quant import quantize_activation_nvfp4
        return quantize_activation_nvfp4(x)

    if x.dim() == 1:
        x = x.unsqueeze(0)

    M, K = x.shape

    # Call fused CUDA kernel
    qdata, raw_scales, per_tensor_scale_tensor = ext.fused_activation_quant_nvfp4(x)

    per_tensor_scale = per_tensor_scale_tensor  # [1] tensor

    # Convert raw FP32 scales to FP8 and swizzle for cuBLAS
    try:
        from torchao.prototype.mx_formats.utils import (
            hp_data_dims_to_swizzled_scale_dims_nvfp4,
        )
        block_scales_fp8 = raw_scales.to(torch.float8_e4m3fn)
        sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
        scatter_idx = _get_scatter_indices(K, block_scales_fp8.device)
        out_flat = torch.zeros(sM * sK, dtype=torch.float8_e4m3fn, device=block_scales_fp8.device)
        # Scatter FP8 scales into swizzled positions (1 kernel vs ~10 for to_blocked)
        out_flat.view(torch.uint8).scatter_(0, scatter_idx, block_scales_fp8.view(torch.uint8))
        a_scale = out_flat.view(sM, sK)
    except ImportError:
        a_scale = raw_scales.to(torch.float8_e4m3fn).view(M, K // 16)

    return qdata, a_scale, per_tensor_scale


# Cache for pre-computed scatter indices per (K, device)
_SCATTER_CACHE: dict = {}


def _get_scatter_indices(K: int, device: "torch.device") -> "torch.Tensor":
    """Pre-compute and cache the to_blocked scatter index map for a given K."""
    import torch
    cache_key = (K, device)
    if cache_key in _SCATTER_CACHE:
        return _SCATTER_CACHE[cache_key]

    from torchao.prototype.mx_formats.utils import (
        to_blocked,
        hp_data_dims_to_swizzled_scale_dims_nvfp4,
    )

    n_scales = K // 16
    sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(1, K)

    # Use sentinel values to discover the mapping
    sentinel = torch.arange(1, n_scales + 1, dtype=torch.float32).view(1, n_scales)
    blocked = to_blocked(sentinel).flatten()

    scatter_idx = torch.zeros(n_scales, dtype=torch.long)
    for i in range(n_scales):
        positions = (blocked == float(i + 1)).nonzero(as_tuple=False).flatten()
        scatter_idx[i] = positions[0].item()

    scatter_idx = scatter_idx.to(device)
    _SCATTER_CACHE[cache_key] = scatter_idx
    return scatter_idx
