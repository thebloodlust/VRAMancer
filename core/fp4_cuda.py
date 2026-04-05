"""
VRAMancer FP4 CUDA Kernels — JIT-compiled hand-optimized GEMV + dequant.

Provides two operations:
  1. fp4_gemv_cuda(x, w_qdata, w_scale_row) → GEMV for M=1 decode
  2. fp4_dequant_gemm(x, w_qdata, w_scale_row) → dequant + cuBLAS GEMM for M>1

The GEMV kernel is THE solution for Blackwell FP4 inference:
  - Replaces torch._scaled_mm (broken CUTLASS FP4 kernels)
  - Replaces Triton GEMV (5x overhead vs cuBLAS)
  - Hand-written CUDA with vectorized loads + warp shuffle reduction
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("vramancer.fp4_cuda")

_MODULE = None
_BUILD_FAILED = False


def _get_module():
    """JIT-compile the CUDA kernel on first use, cache for subsequent calls."""
    global _MODULE, _BUILD_FAILED
    if _MODULE is not None:
        return _MODULE
    if _BUILD_FAILED:
        return None

    try:
        import torch
        from torch.utils.cpp_extension import load

        cuda_src = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "csrc", "fp4_gemv.cu"
        )
        if not os.path.exists(cuda_src):
            logger.warning(f"CUDA source not found: {cuda_src}")
            _BUILD_FAILED = True
            return None

        # Ensure ninja is on PATH
        try:
            from ninja import BIN_DIR
            os.environ["PATH"] = BIN_DIR + ":" + os.environ.get("PATH", "")
        except ImportError:
            pass

        # Detect GPU compute capability for native compilation
        import torch
        import subprocess
        import tempfile
        # Find highest CC across all GPUs (e.g. RTX 3090=86 + RTX 5070 Ti=120)
        gpu_cc = 86
        for i in range(torch.cuda.device_count()):
            maj, minor = torch.cuda.get_device_capability(i)
            gpu_cc = max(gpu_cc, maj * 10 + minor)

        # Probe nvcc to find highest supported arch (nvcc may be older than GPU)
        nvcc_path = "nvcc"
        best_cc = 86  # safe fallback
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w") as tmp:
            tmp.write("__global__ void k(){}\n")
            tmp.flush()
            for probe_cc in [gpu_cc, 90, 89, 86]:
                try:
                    r = subprocess.run(
                        [nvcc_path, f"-arch=compute_{probe_cc}", "-c",
                         tmp.name, "-o", "/dev/null"],
                        capture_output=True, timeout=15,
                    )
                    if r.returncode == 0:
                        best_cc = probe_cc
                        break
                except Exception:
                    continue

        arch_flags = [
            f"--generate-code=arch=compute_{best_cc},code=compute_{best_cc}",
        ]
        # Add native SASS if nvcc can target it directly
        if best_cc == gpu_cc:
            arch_flags.append(
                f"--generate-code=arch=compute_{best_cc},code=sm_{best_cc}"
            )

        logger.info(
            f"JIT-compiling FP4 CUDA kernels "
            f"(max GPU CC {gpu_cc}, nvcc target compute_{best_cc}) ..."
        )
        _MODULE = load(
            name="vramancer_fp4_gemv",
            sources=[cuda_src],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ] + arch_flags,
            verbose=False,
        )
        logger.info("FP4 CUDA kernels compiled successfully")
        return _MODULE

    except Exception as e:
        logger.warning(f"FP4 CUDA kernel compilation failed: {e}")
        _BUILD_FAILED = True
        return None


def fp4_gemv_cuda(
    x: "torch.Tensor",
    w_qdata: "torch.Tensor",
    w_scale_row: "torch.Tensor",
) -> "torch.Tensor":
    """
    Hand-optimized CUDA GEMV for FP4 weights (M=1 decode).

    - Warp-level K-reduction with __shfl_xor_sync
    - Vectorized 4-byte weight loads
    - E2M1 LUT in constant memory (100% L1 hit)
    - Zero intermediate tensors

    Args:
        x: [K] or [1, K] bfloat16 activation
        w_qdata: [N, K//2] uint8 packed FP4 weights
        w_scale_row: [N, K//16] float32 block scales (per_tensor premultiplied)

    Returns:
        [N] or [1, N] bfloat16 output
    """
    mod = _get_module()
    if mod is None:
        raise RuntimeError("FP4 CUDA kernels not available")
    return mod.fp4_gemv(x, w_qdata, w_scale_row)


def fp4_dequant_gemm(
    x: "torch.Tensor",
    w_qdata: "torch.Tensor",
    w_scale_row: "torch.Tensor",
) -> "torch.Tensor":
    """
    FP4 W4A16 GEMM for M>1 (prefill): CUDA dequant + cuBLAS BF16 GEMM.

    Two-step approach:
      1. CUDA kernel: FP4→BF16 dequant (all elements, one kernel launch)
      2. torch.mm: BF16 GEMM via cuBLAS (maximally optimized)

    Uses a single reusable buffer for the dequantized weights.

    Args:
        x: [M, K] or [batch..., K] activation (any float dtype)
        w_qdata: [N, K//2] uint8 packed FP4 weights
        w_scale_row: [N, K//16] float32 block scales

    Returns:
        [M, N] or [batch..., N] bfloat16 output
    """
    import torch

    mod = _get_module()
    if mod is None:
        raise RuntimeError("FP4 CUDA kernels not available")

    orig_shape = x.shape
    K = x.shape[-1]
    x_2d = x.reshape(-1, K).to(torch.bfloat16)
    N = w_qdata.shape[0]

    # Step 1: CUDA dequant FP4 → BF16 (single kernel, no Python overhead)
    w_bf16 = mod.fp4_dequant(w_qdata, w_scale_row)  # [N, K] bf16

    # Step 2: cuBLAS GEMM (maximally optimized for BF16)
    out = torch.mm(x_2d, w_bf16.T)  # [M, N]

    del w_bf16  # free immediately
    return out.reshape(*orig_shape[:-1], N)
