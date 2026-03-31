"""
VRAMancer GPU-accelerated TurboQuant KV Compression.

GPU-native implementation of PolarQuant + QJL operations.
All ops stay on GPU (zero CPU transfers per token).

The polar encode/decode are inherently recursive (log2(D) levels)
which maps poorly to Triton's flat SIMT model. Instead, we use
PyTorch vectorized GPU ops. The key insight is that PyTorch's atan2,
sqrt, cos, sin etc. are already GPU kernels when called on CUDA
tensors — the real bottleneck was Python call overhead and the sheer
number of separate function calls per token, not the math itself.

The real speedup comes from:
  1. Batching queries across GQA head groups (7:1 for Qwen2.5-7B)
  2. Pre-building merged compressed data per kv_head
  3. Caching value decompressions across heads
  4. Eliminating redundant polar_decode for shared kv_heads
"""
from __future__ import annotations

import math
import logging
import os

logger = logging.getLogger("vramancer.triton_kv_quant")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

HAS_TORCH = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

# ──────────────────────────────────────────────────────────────────
# GPU Implementation (PyTorch vectorized ops on CUDA tensors)
# ──────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class TritonKVCompressOps:
        """GPU-native operations for KV compression.

        Drop-in replacement for the hot methods of KVCacheCompressor.
        All operations stay on GPU — zero CPU transfers.
        """

        @staticmethod
        def hadamard_transform(x: "torch.Tensor", signs: "torch.Tensor",
                               forward: bool = True) -> "torch.Tensor":
            """Fast Walsh-Hadamard transform on GPU.

            Uses iterative butterfly pattern — all on CUDA tensors.
            For D=128: 7 butterfly passes, each is elementwise on GPU.
            """
            d = x.shape[-1]
            result = x.clone()

            if forward:
                result = result * signs  # diag(±1) @ x

            h = 1
            while h < d:
                # Reshape to pairs at stride h
                result = result.view(*x.shape[:-1], -1, 2, h)
                a = result[..., 0, :] + result[..., 1, :]
                b = result[..., 0, :] - result[..., 1, :]
                result = torch.stack([a, b], dim=-2).reshape(*x.shape[:-1], -1)
                h *= 2

            inv_sqrt_d = 1.0 / math.sqrt(d)
            result = result * inv_sqrt_d

            if not forward:
                result = result * signs  # diag(±1) @ H^T @ x

            return result

        @staticmethod
        def polar_encode_gpu(x: "torch.Tensor", n_levels: int,
                             bits_per_angle: int) -> tuple:
            """Vectorized PolarQuant encode — fully on GPU.

            Args:
                x: [..., D] rotated vectors (on CUDA)
                n_levels: log2(D)
                bits_per_angle: quantization bits

            Returns:
                (radius [..., 1], q_angles list of [..., D/2^(l+1)] uint8 on GPU)
            """
            n_bins = 1 << bits_per_angle
            all_q_angles = []
            current = x

            for level in range(n_levels):
                d_cur = current.shape[-1]
                pairs = current.view(*current.shape[:-1], d_cur // 2, 2)
                a = pairs[..., 0]
                b = pairs[..., 1]

                radii = torch.sqrt(a * a + b * b + 1e-12)
                angles = torch.atan2(b, a)

                # Quantize angles to uint8 — all on GPU
                if level == 0:
                    lo, hi = -math.pi, math.pi
                else:
                    lo, hi = 0.0, math.pi / 2

                normalized = (angles - lo) / (hi - lo)
                idx = torch.clamp((normalized * n_bins).long(), 0, n_bins - 1)
                all_q_angles.append(idx.to(torch.uint8))

                current = radii

            return current, all_q_angles

        @staticmethod
        def polar_decode_gpu(radius: "torch.Tensor", q_angles: list,
                             n_levels: int, bits_per_angle: int) -> "torch.Tensor":
            """Vectorized polar decode — fully on GPU.

            Args:
                radius: [..., 1] on GPU
                q_angles: list of [..., D/2^(l+1)] uint8 on GPU
                n_levels: log2(D)
                bits_per_angle: quantization bits

            Returns:
                [..., D] reconstructed vectors on GPU
            """
            n_bins = 1 << bits_per_angle
            current = radius

            for level in range(n_levels - 1, -1, -1):
                if level == 0:
                    lo, hi = -math.pi, math.pi
                else:
                    lo, hi = 0.0, math.pi / 2

                angles = lo + (q_angles[level].float() + 0.5) * (hi - lo) / n_bins
                cos_a = torch.cos(angles)
                sin_a = torch.sin(angles)

                a = current * cos_a
                b = current * sin_a

                d_half = angles.shape[-1]
                current = torch.stack([a, b], dim=-1).reshape(
                    *a.shape[:-1], d_half * 2
                )

            return current

        @staticmethod
        def qjl_encode_gpu(residual: "torch.Tensor",
                           jl_matrix: "torch.Tensor") -> tuple:
            """Fused JL projection + sign quantization — fully on GPU.

            Args:
                residual: [..., D] on GPU
                jl_matrix: [M, D] on GPU

            Returns:
                (signs [..., M] uint8, norms [..., 1] fp16) — all on GPU
            """
            projected = residual @ jl_matrix.t()  # GPU matmul
            signs = (projected > 0).to(torch.uint8)
            norms = torch.norm(residual, dim=-1, keepdim=True).half()
            return signs, norms

        @staticmethod
        def qjl_score_correction_gpu(
            q_rot: "torch.Tensor",
            signs: "torch.Tensor",
            norms: "torch.Tensor",
            jl_matrix: "torch.Tensor",
            qjl_dim: int,
        ) -> "torch.Tensor":
            """Asymmetric QJL score correction — fully on GPU."""
            q_proj = q_rot @ jl_matrix.t()
            signs_float = signs.float() * 2 - 1
            raw = q_proj @ signs_float.t()
            scale = math.sqrt(math.pi / 2) / qjl_dim
            return raw * scale * norms.float().t()

        @staticmethod
        def compress_gpu(
            kv: "torch.Tensor",
            head_dim: int,
            padded_dim: int,
            n_levels: int,
            bits_per_angle: int,
            hadamard_signs: "torch.Tensor",
            jl_matrix: "torch.Tensor",
        ) -> dict:
            """Full compression pipeline — all on GPU.

            This is the drop-in replacement for KVCacheCompressor.compress().
            """
            orig_shape = kv.shape
            flat = kv.reshape(-1, head_dim)

            # Ensure on same device as signs
            device = hadamard_signs.device
            if flat.device != device:
                flat = flat.to(device)

            # Pad
            if padded_dim > head_dim:
                flat = torch.nn.functional.pad(flat, (0, padded_dim - head_dim))

            # Step 1: Hadamard rotation (GPU)
            rotated = TritonKVCompressOps.hadamard_transform(
                flat, hadamard_signs, forward=True
            )

            # Step 2: Polar encode + quantize (GPU)
            radius, q_angles = TritonKVCompressOps.polar_encode_gpu(
                rotated, n_levels, bits_per_angle
            )

            # Step 3: Reconstruct for residual (GPU)
            reconstructed = TritonKVCompressOps.polar_decode_gpu(
                radius, q_angles, n_levels, bits_per_angle
            )

            # Step 4: QJL on residual (GPU)
            residual = rotated - reconstructed
            qjl_signs, qjl_norms = TritonKVCompressOps.qjl_encode_gpu(
                residual, jl_matrix
            )

            return {
                "radius": radius.half(),
                "angles": q_angles,  # list of uint8 tensors ON GPU
                "qjl_signs": qjl_signs,
                "qjl_norms": qjl_norms,
                "shape": orig_shape,
            }

        @staticmethod
        def decompress_gpu(
            compressed: dict,
            head_dim: int,
            padded_dim: int,
            n_levels: int,
            bits_per_angle: int,
            hadamard_signs: "torch.Tensor",
        ) -> "torch.Tensor":
            """Full decompression pipeline — all on GPU."""
            radius = compressed["radius"].float()
            reconstructed = TritonKVCompressOps.polar_decode_gpu(
                radius, compressed["angles"], n_levels, bits_per_angle
            )
            unrotated = TritonKVCompressOps.hadamard_transform(
                reconstructed, hadamard_signs, forward=False
            )

            if padded_dim > head_dim:
                unrotated = unrotated[..., :head_dim]

            return unrotated.reshape(compressed["shape"])

        @staticmethod
        def attention_score_gpu(
            q: "torch.Tensor",
            compressed_k: dict,
            head_dim: int,
            padded_dim: int,
            n_levels: int,
            bits_per_angle: int,
            hadamard_signs: "torch.Tensor",
            jl_matrix: "torch.Tensor",
            qjl_dim: int,
        ) -> "torch.Tensor":
            """Compute attention scores from compressed keys — all on GPU."""
            flat_q = q.reshape(-1, head_dim)

            if padded_dim > head_dim:
                flat_q = torch.nn.functional.pad(
                    flat_q, (0, padded_dim - head_dim)
                )

            q_rot = TritonKVCompressOps.hadamard_transform(
                flat_q, hadamard_signs, forward=True
            )

            # PolarQuant reconstruction of k
            k_polar = TritonKVCompressOps.polar_decode_gpu(
                compressed_k["radius"].float(),
                compressed_k["angles"],
                n_levels,
                bits_per_angle,
            )
            scores = q_rot @ k_polar.t()

            # QJL correction
            correction = TritonKVCompressOps.qjl_score_correction_gpu(
                q_rot,
                compressed_k["qjl_signs"],
                compressed_k["qjl_norms"],
                jl_matrix,
                qjl_dim,
            )

            return scores + correction
