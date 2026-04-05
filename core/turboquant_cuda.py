"""VRAMancer TurboQuant CUDA kernel — Python wrapper.

Loads the fused CUDA extension (csrc/turboquant_kernel.cu) via JIT
compilation. Provides compress/decompress that run the entire
PolarQuant + QJL pipeline in ONE kernel launch (vs ~80 separate
PyTorch kernel launches in the Python path).

Falls back to TritonKVCompressOps (PyTorch GPU ops) or pure-Python
CPU path when the kernel is unavailable.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

_logger = logging.getLogger("vramancer.turboquant_cuda")

_cuda_module = None
_LOAD_ATTEMPTED = False


def _detect_and_set_cuda_arch():
    """Auto-detect GPU compute capabilities and set TORCH_CUDA_ARCH_LIST."""
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return

    import torch
    import subprocess

    max_nvcc_arch = 86
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                parts = line.split("release")[-1].strip().split(",")[0].strip()
                major, minor = parts.split(".")[:2]
                cuda_ver = int(major) * 10 + int(minor)
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

    archs = set()
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        arch = cap[0] * 10 + cap[1]
        archs.add(min(arch, max_nvcc_arch))

    if archs:
        sorted_archs = sorted(archs)
        parts = [f"{a // 10}.{a % 10}" for a in sorted_archs]
        # Add +PTX to highest arch for forward compatibility with newer GPUs
        # (e.g. SM 9.0 PTX runs on SM 12.0 via driver JIT)
        parts[-1] = parts[-1] + "+PTX"
        arch_str = ";".join(parts)
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_str
        _logger.info("TurboQuant CUDA arch list: %s", arch_str)


def _load_cuda_module():
    """Lazy JIT compilation of the TurboQuant CUDA kernel."""
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
        kernel_path = os.path.join(csrc_dir, "turboquant_kernel.cu")
        if not os.path.exists(kernel_path):
            _logger.warning("TurboQuant kernel source not found: %s", kernel_path)
            return None

        _detect_and_set_cuda_arch()

        _cuda_module = load(
            name="vramancer_turboquant",
            sources=[kernel_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        _logger.info("TurboQuant CUDA kernel compiled successfully")
        return _cuda_module

    except Exception as e:
        _logger.warning("TurboQuant CUDA kernel compilation failed: %s", e)
        return None


def has_cuda_turboquant() -> bool:
    """Check if the compiled TurboQuant CUDA kernel is available."""
    return _load_cuda_module() is not None


class CUDATurboQuantOps:
    """Drop-in replacement for TritonKVCompressOps using fused CUDA kernels.

    One kernel launch for compress (vs ~80 in Python path).
    One kernel launch for decompress (vs ~14 in Python path).
    """

    @staticmethod
    def compress_gpu(
        kv,            # [N, head_dim] or [..., head_dim]
        head_dim: int,
        padded_dim: int,
        n_levels: int,
        bits_per_angle: int,
        hadamard_signs,  # [padded_dim] float
        jl_matrix,       # [qjl_dim, padded_dim] float
    ) -> dict:
        import torch

        mod = _load_cuda_module()
        if mod is None:
            raise RuntimeError("TurboQuant CUDA kernel not available")

        orig_shape = kv.shape
        flat = kv.reshape(-1, head_dim).contiguous().float()

        # Ensure signs/jl on same device, float32
        device = flat.device
        signs_f = hadamard_signs.to(device=device, dtype=torch.float32).contiguous()
        jl_f = jl_matrix.to(device=device, dtype=torch.float32).contiguous()
        qjl_dim = jl_f.shape[0]

        # Set current CUDA device to match the tensor device — required because
        # the PTX JIT cache is per-device and the kernel .so is compiled once.
        with torch.cuda.device(device):
            radius, angles, qjl_signs, qjl_norms = mod.compress(
                flat, signs_f, jl_f,
                padded_dim, n_levels, bits_per_angle, qjl_dim,
            )

        # Unpack angles into per-level list (for API compatibility)
        angles_list = []
        offset = 0
        d = padded_dim
        for level in range(n_levels):
            half_d = d // 2
            level_angles = angles[:, offset:offset + half_d]
            angles_list.append(level_angles)
            offset += half_d
            d = half_d

        return {
            "radius": radius,          # [N] fp16
            "angles": angles_list,     # list of [N, d/2^(l+1)] uint8
            "qjl_signs": qjl_signs,    # [N, qjl_dim] uint8
            "qjl_norms": qjl_norms.unsqueeze(-1),  # [N, 1] fp16
            "shape": orig_shape,
        }

    @staticmethod
    def decompress_gpu(
        compressed: dict,
        head_dim: int,
        padded_dim: int,
        n_levels: int,
        bits_per_angle: int,
        hadamard_signs,
    ) -> "torch.Tensor":
        import torch

        mod = _load_cuda_module()
        if mod is None:
            raise RuntimeError("TurboQuant CUDA kernel not available")

        radius = compressed["radius"]  # [N] fp16
        device = radius.device
        signs_f = hadamard_signs.to(device=device, dtype=torch.float32).contiguous()

        # Re-pack angles into flat [N, padded_dim - 1]
        angles_list = compressed["angles"]
        angles_flat = torch.cat(angles_list, dim=-1).contiguous()

        with torch.cuda.device(device):
            out = mod.decompress(
                radius.contiguous(), angles_flat, signs_f,
                head_dim, padded_dim, n_levels, bits_per_angle,
            )

        return out.reshape(compressed["shape"])

    # Attention score: decompress keys then dot product
    # (could be fused later, but the decompress is the bottleneck)
    @staticmethod
    def attention_score_gpu(
        q,
        compressed_k: dict,
        head_dim: int,
        padded_dim: int,
        n_levels: int,
        bits_per_angle: int,
        hadamard_signs,
        jl_matrix,
        qjl_dim: int,
    ) -> "torch.Tensor":
        import torch
        import math

        # Decompress keys via CUDA kernel
        k_reconstructed = CUDATurboQuantOps.decompress_gpu(
            compressed_k, head_dim, padded_dim, n_levels,
            bits_per_angle, hadamard_signs,
        )
        k_flat = k_reconstructed.reshape(-1, head_dim)

        # Rotate query (Hadamard) — use PyTorch ops for query (small tensor)
        flat_q = q.reshape(-1, head_dim)
        if padded_dim > head_dim:
            flat_q = torch.nn.functional.pad(flat_q, (0, padded_dim - head_dim))

        # Hadamard on query — small, Python overhead negligible
        from core.triton_kv_quant import TritonKVCompressOps
        q_rot = TritonKVCompressOps.hadamard_transform(
            flat_q, hadamard_signs, forward=True
        )

        # Reconstruct keys in rotated space for polar dot product
        # Actually we need keys in original space for q @ k^T
        scores = flat_q @ k_flat.t()

        # QJL correction
        device = q.device
        jl_f = jl_matrix.to(device=device, dtype=torch.float32)
        q_proj = flat_q @ jl_f.t()
        signs_float = compressed_k["qjl_signs"].float() * 2 - 1
        raw = q_proj @ signs_float.t()
        scale = math.sqrt(math.pi / 2) / qjl_dim
        correction = raw * scale * compressed_k["qjl_norms"].float().t()

        return scores + correction
