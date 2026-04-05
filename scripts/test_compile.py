#!/usr/bin/env python3
"""Test torch.compile speedup on TurboQuant core operations."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch, math, time
torch.set_float32_matmul_precision('high')

device = 'cuda:0'
D = 128
n_levels = 7
n_bins = 8

def compress_core(flat, signs, jl_matrix):
    """Core compress: Hadamard + polar_encode + polar_decode + QJL."""
    # Hadamard rotation
    result = flat * signs
    h = 1
    shape = flat.shape
    while h < D:
        result = result.view(*shape[:-1], -1, 2, h)
        a = result[..., 0, :] + result[..., 1, :]
        b = result[..., 0, :] - result[..., 1, :]
        result = torch.stack([a, b], dim=-2).reshape(*shape[:-1], -1)
        h *= 2
    rotated = result * (1.0 / math.sqrt(D))

    # Polar encode + quantize
    current = rotated
    all_angles = []
    for level in range(n_levels):
        d_cur = current.shape[-1]
        pairs = current.view(*current.shape[:-1], d_cur // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        radii = torch.sqrt(a * a + b * b + 1e-12)
        angles = torch.atan2(b, a)
        if level == 0:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = 0.0, math.pi / 2
        idx = torch.clamp(((angles - lo) / (hi - lo) * n_bins).long(), 0, n_bins - 1).to(torch.uint8)
        all_angles.append(idx)
        current = radii

    # Polar decode for residual
    recon = current
    for level in range(n_levels - 1, -1, -1):
        if level == 0:
            lo, hi = -math.pi, math.pi
        else:
            lo, hi = 0.0, math.pi / 2
        ang = lo + (all_angles[level].float() + 0.5) * (hi - lo) / n_bins
        a2 = recon * torch.cos(ang)
        b2 = recon * torch.sin(ang)
        recon = torch.stack([a2, b2], dim=-1).reshape(*a2.shape[:-1], a2.shape[-1] * 2)

    # QJL
    residual = rotated - recon
    projected = residual @ jl_matrix.t()
    qjl_signs = (projected > 0).to(torch.uint8)
    qjl_norms = torch.norm(residual, dim=-1, keepdim=True).half()

    return current, all_angles, qjl_signs, qjl_norms

def bench(fn, warmup=10, iters=200, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label}: {ms:.3f} ms")
    return ms

# Test inputs
signs = (torch.randint(0, 2, (D,), device=device).float() * 2 - 1)
jl_matrix = torch.randn(64, D, device=device) / math.sqrt(64)
flat = torch.randn(4, D, device=device)  # n_kv_heads=4

print("=== RAW (no torch.compile) ===")
t_raw = bench(lambda: compress_core(flat, signs, jl_matrix), label="compress_core")

print("\n=== COMPILING (torch.compile mode=default) ===")
compiled_default = torch.compile(compress_core, mode="default", fullgraph=False)
# Long warmup for compilation
print("  warmup (compiling)...")
for _ in range(3):
    compiled_default(flat, signs, jl_matrix)
torch.cuda.synchronize()
print("  compilation done")
t_compiled = bench(lambda: compiled_default(flat, signs, jl_matrix), label="compile-default")
print(f"  speedup vs raw: {t_raw/t_compiled:.1f}x")

print("\n=== COMPILING (torch.compile mode=reduce-overhead) ===")
compiled_ro = torch.compile(compress_core, mode="reduce-overhead", fullgraph=False)
print("  warmup (compiling + CUDA graph capture)...")
for _ in range(5):
    compiled_ro(flat, signs, jl_matrix)
torch.cuda.synchronize()
print("  compilation done")
t_ro = bench(lambda: compiled_ro(flat, signs, jl_matrix), label="compile-reduce-overhead")
print(f"  speedup vs raw: {t_raw/t_ro:.1f}x")

print(f"\n=== SUMMARY ===")
print(f"Raw:                 {t_raw:.3f} ms")
print(f"compile(default):    {t_compiled:.3f} ms ({t_raw/t_compiled:.1f}x)")
print(f"compile(reduce-ovh): {t_ro:.3f} ms ({t_raw/t_ro:.1f}x)")
