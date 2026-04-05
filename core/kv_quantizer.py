"""
VRAMancer KV Cache Quantizer — PolarQuant + QJL compression.

Implements the TurboQuant algorithm (arxiv:2504.19874, ICLR 2026)
with PolarQuant (arxiv:2502.02617) and QJL (Kacham et al., AAAI 2025).

Two-stage compression:
  Stage 1 — PolarQuant: random rotation -> recursive Cartesian-to-Polar conversion
            -> quantize concentrated angles. Handles most of the signal.
  Stage 2 — QJL 1-bit: project residual error via JL matrix -> sign-bit quantization.
            Asymmetric estimator computes attention scores directly (no reconstruction).

Sparse V optimization (March 2026):
  After computing attention weights from compressed keys, only ~10% of tokens
  carry meaningful weight. Sparse V skips value decompression for the other ~90%,
  yielding massive speedup with near-zero quality loss.

Typical budget: 3 bits/dim PolarQuant + 0.5 bits/dim QJL ~ 3.5 bits/dim total.
Achieves 4x+ KV memory reduction with near-zero accuracy loss.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

_logger = logging.getLogger("vramancer.kv_quantizer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class _Stub:
        Module = object
    nn = _Stub()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bit-packing utilities — store N-bit values in packed uint8 tensors
# ---------------------------------------------------------------------------

def _pack_bits(signs):
    """Pack a [..., M] uint8 tensor of 0/1 into [..., ceil(M/8)] bytes (MSB-first)."""
    *batch, M = signs.shape
    pad = (8 - M % 8) % 8
    if pad:
        signs = F.pad(signs, (0, pad))
    packed_M = signs.shape[-1] // 8
    signs = signs.reshape(*batch, packed_M, 8)
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                           dtype=torch.uint8, device=signs.device)
    return (signs * weights).sum(dim=-1).to(torch.uint8)


def _unpack_bits(packed, M):
    """Unpack [..., ceil(M/8)] packed bytes back to [..., M] uint8 of 0/1."""
    *batch, packed_M = packed.shape
    bits = packed.unsqueeze(-1).expand(*batch, packed_M, 8)
    shifts = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0],
                          dtype=torch.uint8, device=packed.device)
    unpacked = ((bits >> shifts) & 1).reshape(*batch, packed_M * 8)
    return unpacked[..., :M]


def _pack_angles(angles_list, bits_per_angle):
    """Pack list of 2D quantized angle tensors [N, cols_i] into [N, packed_bytes].

    Concatenates all angle values per row, expands to individual bits,
    then packs 8 bits per byte.  Fully vectorized (no Python row loops).
    """
    if not angles_list:
        return torch.tensor([], dtype=torch.uint8)
    flat = torch.cat(angles_list, dim=-1)  # [N, total_cols]
    N, total_cols = flat.shape
    # Expand each uint8 value to bits_per_angle bits: [N, total_cols, bpa]
    shifts = torch.arange(bits_per_angle - 1, -1, -1,
                          device=flat.device, dtype=torch.uint8)
    bits = ((flat.unsqueeze(-1) >> shifts) & 1).reshape(N, -1)  # [N, total_cols*bpa]
    return _pack_bits(bits)  # [N, ceil(total_cols*bpa/8)]


def _unpack_angles(packed, N, level_dims, bits_per_angle):
    """Unpack [N, packed_bytes] back to list of [N, cols_i] uint8 tensors.

    level_dims: list of int -- column count per level (e.g. [64,32,16,8,4,2,1]).
    """
    total_cols = sum(level_dims)
    total_bits = total_cols * bits_per_angle
    bits = _unpack_bits(packed, total_bits)  # [N, total_bits]
    # Reconstruct values: [N, total_cols, bpa] -> sum weighted bits -> [N, total_cols]
    shifts = torch.arange(bits_per_angle - 1, -1, -1,
                          device=packed.device, dtype=torch.uint8)
    bits_3d = bits.reshape(N, total_cols, bits_per_angle)
    flat = (bits_3d * (1 << shifts)).sum(dim=-1).to(torch.uint8)  # [N, total_cols]
    # Split into per-level tensors
    result = []
    offset = 0
    for d in level_dims:
        result.append(flat[:, offset:offset + d])
        offset += d
    return result


# ---------------------------------------------------------------------------
# GPU-accelerated ops — priority: fused CUDA kernel > PyTorch GPU ops > CPU
# ---------------------------------------------------------------------------
_GPU_OPS = None
_GPU_OPS_RESOLVED = False
_PYTORCH_GPU_OPS = None

class BatchedCompressedRef(dict):
    """Lazy-slicing view into a batched compress result.

    Acts like a normal compressed dict but defers tensor slicing until
    first access.  Creating 56 of these costs ~0.008ms vs ~1.6ms for
    eager slicing (560 PyTorch view objects).
    """
    __slots__ = ("_backing", "_off", "_n", "_shape", "_resolved")

    def __init__(self, backing: dict, offset: int, n_rows: int, shape: tuple):
        super().__init__()
        self._backing = backing
        self._off = offset
        self._n = n_rows
        self._shape = shape
        self._resolved = False

    def _resolve(self):
        if self._resolved:
            return
        o, n = self._off, self._n
        b = self._backing
        super().__setitem__("radius", b["radius"][o:o + n])
        super().__setitem__("angles", [a[o:o + n] for a in b["angles"]])
        super().__setitem__("qjl_signs", b["qjl_signs"][o:o + n])
        super().__setitem__("qjl_norms", b["qjl_norms"][o:o + n])
        super().__setitem__("shape", self._shape)
        self._backing = None  # allow GC of backing if all refs resolved
        self._resolved = True

    def __getitem__(self, key):
        if not self._resolved:
            self._resolve()
        return super().__getitem__(key)

    def __contains__(self, key):
        if not self._resolved:
            # Fast-path: we know the keys without resolving
            return key in ("radius", "angles", "qjl_signs", "qjl_norms", "shape")
        return super().__contains__(key)

    def get(self, key, default=None):
        if not self._resolved:
            if key in ("radius", "angles", "qjl_signs", "qjl_norms", "shape"):
                self._resolve()
                return super().__getitem__(key)
            return default
        return super().get(key, default)

    def keys(self):
        if not self._resolved:
            self._resolve()
        return super().keys()

    def values(self):
        if not self._resolved:
            self._resolve()
        return super().values()

    def items(self):
        if not self._resolved:
            self._resolve()
        return super().items()

try:
    from core.triton_kv_quant import TritonKVCompressOps, HAS_TORCH as _TKV_TORCH
    if _TKV_TORCH:
        _PYTORCH_GPU_OPS = TritonKVCompressOps
except ImportError:
    pass


def _resolve_gpu_ops():
    """Lazy resolution: try CUDA fused kernel first, fall back to PyTorch GPU ops."""
    global _GPU_OPS, _GPU_OPS_RESOLVED
    if _GPU_OPS_RESOLVED:
        return _GPU_OPS
    _GPU_OPS_RESOLVED = True
    try:
        from core.turboquant_cuda import CUDATurboQuantOps, has_cuda_turboquant
        if has_cuda_turboquant():
            _GPU_OPS = CUDATurboQuantOps
            _logger.info("TurboQuant: using fused CUDA kernel (1 launch per compress)")
            return _GPU_OPS
    except ImportError:
        pass
    if _PYTORCH_GPU_OPS is not None:
        _GPU_OPS = _PYTORCH_GPU_OPS
        _logger.info("TurboQuant: using PyTorch GPU ops (~80 launches per compress)")
    return _GPU_OPS


class KVCacheCompressor(nn.Module):
    """
    KV cache compressor using PolarQuant + QJL (TurboQuant algorithm).

    Args:
        head_dim: dimension of each attention head (e.g. 128)
        bits_per_angle: quantization bits for polar angles (default 3)
        qjl_dim: JL projection dimension (default head_dim // 2)
    """

    def __init__(self, head_dim: int, bits_per_angle: int = 3,
                 qjl_dim: Optional[int] = None, force_cpu: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.bits_per_angle = bits_per_angle
        self._force_cpu = force_cpu

        # Pad to next power of 2 for recursive polar subdivision
        self._padded_dim = 1 << math.ceil(math.log2(max(head_dim, 2)))
        self._n_levels = int(math.log2(self._padded_dim))

        self.qjl_dim = qjl_dim or self._padded_dim // 2

        # Random rotation via randomized Hadamard: H @ diag(+/-1)
        signs = torch.randint(0, 2, (self._padded_dim,)).float() * 2 - 1
        self.register_buffer("_hadamard_signs", signs)

        # JL projection matrix (Gaussian, scaled)
        P = torch.randn(self.qjl_dim, self._padded_dim) / math.sqrt(self.qjl_dim)
        self.register_buffer("jl_matrix", P)

    # -- Fast Walsh-Hadamard transform --

    def _hadamard(self, x):
        d = x.shape[-1]
        h = 1
        result = x.clone()
        while h < d:
            result = result.view(*result.shape[:-1], -1, 2, h)
            a = result[..., 0, :] + result[..., 1, :]
            b = result[..., 0, :] - result[..., 1, :]
            result = torch.stack([a, b], dim=-2).reshape(
                *x.shape[:-1], -1
            )
            h *= 2
        return result / math.sqrt(d)

    def _rotate(self, x):
        return self._hadamard(x * self._hadamard_signs)

    def _unrotate(self, x):
        return self._hadamard(x) * self._hadamard_signs

    # -- PolarQuant encode / decode --

    def _polar_encode(self, x):
        all_angles = []
        current = x

        for _ in range(self._n_levels):
            d_cur = current.shape[-1]
            pairs = current.view(*current.shape[:-1], d_cur // 2, 2)
            a, b = pairs[..., 0], pairs[..., 1]

            radii = torch.sqrt(a * a + b * b + 1e-12)
            angles = torch.atan2(b, a)

            all_angles.append(angles)
            current = radii

        return current, all_angles

    def _quantize_angle(self, angle, level):
        n_bins = 1 << self.bits_per_angle
        if level == 0:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = 0.0, math.pi / 2
        normalized = (angle - lo) / (hi - lo)
        idx = torch.clamp((normalized * n_bins).long(), 0, n_bins - 1)
        return idx.to(torch.uint8)

    def _dequantize_angle(self, idx, level):
        n_bins = 1 << self.bits_per_angle
        if level == 0:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = 0.0, math.pi / 2
        return lo + (idx.float() + 0.5) * (hi - lo) / n_bins

    def _polar_decode(self, radius, q_angles):
        current = radius
        for level in range(self._n_levels - 1, -1, -1):
            angles = self._dequantize_angle(q_angles[level], level)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)
            a = current * cos_a
            b = current * sin_a
            d_half = angles.shape[-1]
            current = torch.stack([a, b], dim=-1).reshape(
                *a.shape[:-1], d_half * 2
            )
        return current

    # -- QJL encode --

    def _qjl_encode(self, residual):
        projected = residual @ self.jl_matrix.t()
        signs = (projected > 0).to(torch.uint8)
        norms = torch.norm(residual, dim=-1, keepdim=True).half()
        return signs, norms

    def _qjl_score_correction(self, q, signs, norms):
        q_proj = q @ self.jl_matrix.t()
        signs_float = signs.float() * 2 - 1
        raw = q_proj @ signs_float.t()
        scale = math.sqrt(math.pi / 2) / self.qjl_dim
        return raw * scale * norms.float().t()

    # -- Packing --

    def _use_gpu_ops(self, tensor):
        ops = _resolve_gpu_ops()
        return (ops is not None
                and tensor.is_cuda
                and not self._force_cpu)

    def _get_gpu_ops(self):
        return _resolve_gpu_ops()

    def _pack_compressed(self, d):
        """Pack compressed dict for compact storage.

        Packs QJL signs (1-bit per elem -> 8 per byte) and 2D angles
        (bits_per_angle per value -> packed contiguous bytes).
        """
        N = d["qjl_signs"].shape[0]
        qjl_dim = d["qjl_signs"].shape[-1]
        level_dims = [a.shape[-1] for a in d["angles"]]
        return {
            "radius": d["radius"],
            "angles_packed": _pack_angles(d["angles"], self.bits_per_angle),
            "qjl_signs_packed": _pack_bits(d["qjl_signs"]),
            "qjl_norms": d["qjl_norms"],
            "shape": d["shape"],
            "_N": N,
            "_qjl_dim": qjl_dim,
            "_level_dims": level_dims,
        }

    def _unpack_compressed(self, d):
        """Unpack a packed compressed dict back to working format."""
        if "angles" in d:
            return d
        N = d["_N"]
        return {
            "radius": d["radius"],
            "angles": _unpack_angles(
                d["angles_packed"], N, d["_level_dims"],
                self.bits_per_angle,
            ),
            "qjl_signs": _unpack_bits(d["qjl_signs_packed"], d["_qjl_dim"]),
            "qjl_norms": d["qjl_norms"],
            "shape": d["shape"],
        }

    # -- Public API --

    def compress(self, kv, pack: bool = True):
        """Compress KV cache vectors.

        Args:
            kv: [seq_len, head_dim] (or [..., head_dim])

        Returns:
            dict with packed keys: radius, angles_packed, qjl_signs_packed, etc.
        """
        if self._use_gpu_ops(kv):
            result = self._get_gpu_ops().compress_gpu(
                kv, self.head_dim, self._padded_dim, self._n_levels,
                self.bits_per_angle, self._hadamard_signs, self.jl_matrix,
            )
            return self._pack_compressed(result) if pack else result

        orig_shape = kv.shape
        flat = kv.reshape(-1, self.head_dim)

        if self._padded_dim > self.head_dim:
            flat = F.pad(flat, (0, self._padded_dim - self.head_dim))

        rotated = self._rotate(flat)

        radius, angles_list = self._polar_encode(rotated)
        q_angles = [
            self._quantize_angle(a, level)
            for level, a in enumerate(angles_list)
        ]

        reconstructed = self._polar_decode(radius, q_angles)

        residual = rotated - reconstructed
        qjl_signs, qjl_norms = self._qjl_encode(residual)

        raw = {
            "radius": radius.half(),
            "angles": q_angles,
            "qjl_signs": qjl_signs,
            "qjl_norms": qjl_norms,
            "shape": orig_shape,
        }
        return self._pack_compressed(raw) if pack else raw

    def compress_batch(self, kv_list: list, pack: bool = False) -> list:
        """Compress multiple KV tensors in a single kernel launch.

        Instead of N separate compress() calls (each launching a CUDA kernel),
        concatenate all vectors and launch once. For Qwen2.5-7B this turns
        56 kernel launches into 1, reducing overhead from 3.2ms to 0.06ms.

        By default returns raw GPU dicts (``pack=False``): angles stay as uint8
        tensor lists on GPU, QJL signs stay as uint8 tensors — no Python
        bit-packing. This cuts per-token overhead from 20ms to <0.5ms.

        Raw dicts are directly usable by ``decompress()`` and
        ``attention_score()`` (``_unpack_compressed`` is a no-op for them).
        Call with ``pack=True`` only for offload / serialization.

        Args:
            kv_list: list of [*, head_dim] tensors to compress
            pack: if True, bit-pack angles/signs for compact storage

        Returns:
            list of compressed dicts
        """
        if not kv_list:
            return []
        if len(kv_list) == 1:
            return [self.compress(kv_list[0])]

        # Check if GPU ops available and all tensors on CUDA
        first = kv_list[0]
        if not (self._use_gpu_ops(first) and all(t.is_cuda for t in kv_list)):
            return [self.compress(kv) for kv in kv_list]

        # Record shapes and flatten
        shapes = [kv.shape for kv in kv_list]
        rows_per = [kv.reshape(-1, self.head_dim).shape[0] for kv in kv_list]
        flat_all = torch.cat([kv.reshape(-1, self.head_dim) for kv in kv_list], dim=0)

        # Single kernel launch
        result = self._get_gpu_ops().compress_gpu(
            flat_all, self.head_dim, self._padded_dim, self._n_levels,
            self.bits_per_angle, self._hadamard_signs, self.jl_matrix,
        )

        # Split results back — use lazy refs when not packing (0.008ms vs 1.6ms)
        out = []
        offset = 0
        for shape, n_rows in zip(shapes, rows_per):
            if pack:
                sliced = {
                    "radius": result["radius"][offset:offset + n_rows],
                    "angles": [a[offset:offset + n_rows] for a in result["angles"]],
                    "qjl_signs": result["qjl_signs"][offset:offset + n_rows],
                    "qjl_norms": result["qjl_norms"][offset:offset + n_rows],
                    "shape": shape,
                }
                out.append(self._pack_compressed(sliced))
            else:
                out.append(BatchedCompressedRef(result, offset, n_rows, shape))
            offset += n_rows

        return out

    def decompress(self, compressed):
        """Reconstruct approximate KV vectors (for values or debugging).
        Does NOT use QJL (QJL is for attention scores, not reconstruction).
        """
        compressed = self._unpack_compressed(compressed)
        radius_tensor = compressed["radius"]
        if self._use_gpu_ops(radius_tensor):
            return self._get_gpu_ops().decompress_gpu(
                compressed, self.head_dim, self._padded_dim,
                self._n_levels, self.bits_per_angle, self._hadamard_signs,
            )

        radius = radius_tensor.float()
        reconstructed = self._polar_decode(radius, compressed["angles"])
        unrotated = self._unrotate(reconstructed)

        if self._padded_dim > self.head_dim:
            unrotated = unrotated[..., :self.head_dim]

        return unrotated.reshape(compressed["shape"])

    def attention_score(self, q, compressed_k):
        """Compute attention scores using compressed keys (no full reconstruction).

        q: [n_queries, head_dim]
        compressed_k: dict from compress()
        Returns: [n_queries, seq_len]
        """
        compressed_k = self._unpack_compressed(compressed_k)
        if self._use_gpu_ops(q):
            return self._get_gpu_ops().attention_score_gpu(
                q, compressed_k, self.head_dim, self._padded_dim,
                self._n_levels, self.bits_per_angle,
                self._hadamard_signs, self.jl_matrix, self.qjl_dim,
            )

        flat_q = q.reshape(-1, self.head_dim)

        if self._padded_dim > self.head_dim:
            flat_q = F.pad(flat_q, (0, self._padded_dim - self.head_dim))
        q_rot = self._rotate(flat_q)

        k_polar = self._polar_decode(
            compressed_k["radius"].float(),
            compressed_k["angles"],
        )
        scores = q_rot @ k_polar.t()

        correction = self._qjl_score_correction(
            q_rot,
            compressed_k["qjl_signs"],
            compressed_k["qjl_norms"],
        )

        return scores + correction

    def sparse_v_attend(self, q, compressed_k, compressed_v_list,
                        scale=None, sparse_v_ratio=0.1):
        """Sparse V attention: scores from compressed keys, selective value decompression."""
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores = self.attention_score(q, compressed_k) * scale
        weights = torch.softmax(scores, dim=-1)

        seq_len = weights.shape[-1]
        k = max(1, int(math.ceil(sparse_v_ratio * seq_len)))

        if k >= seq_len:
            v_vecs = [self.decompress(cv) for cv in compressed_v_list]
            v_mat = torch.cat(v_vecs, dim=0)
            return weights @ v_mat

        topk_weights, topk_indices = weights.topk(k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        v_vecs = []
        for idx in topk_indices.squeeze(0):
            v_vecs.append(self.decompress(compressed_v_list[idx.item()]))
        v_mat = torch.cat(v_vecs, dim=0)

        return topk_weights @ v_mat

    def bits_per_dim(self):
        """Compute actual compression ratio in bits per dimension."""
        d = self._padded_dim
        n_angles = d - 1
        angle_bits = n_angles * self.bits_per_angle
        radius_bits = 16  # fp16
        qjl_bits = self.qjl_dim + 16  # 1-bit signs + fp16 norm
        total = angle_bits + radius_bits + qjl_bits
        return total / self.head_dim


# Backward-compat alias
TurboQuantCompressor = KVCacheCompressor
