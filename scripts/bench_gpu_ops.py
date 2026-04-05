#!/usr/bin/env python3
"""Micro-benchmark: GPU TurboQuant ops vs CPU ops.

Measures compress(), decompress(), attention_score() for both paths.
This is the critical benchmark — the CPU path caused 65x-175x slowdown.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import time
import torch
from core.kv_quantizer import KVCacheCompressor


def bench(fn, warmup=5, iters=100, label=""):
    """Time a function with warmup and CUDA sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000  # ms
    print(f"  {label}: {elapsed:.3f} ms/call")
    return elapsed


def main():
    device = "cuda:0"
    head_dim = 128

    for n_tokens in [1, 32, 256]:
        print(f"\n{'='*60}")
        print(f"HEAD_DIM={head_dim}, N_TOKENS={n_tokens}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        comp_cpu = KVCacheCompressor(head_dim=head_dim, bits_per_angle=3, force_cpu=True).to(device)

        torch.manual_seed(42)
        comp_gpu = KVCacheCompressor(head_dim=head_dim, bits_per_angle=3, force_cpu=False).to(device)
        comp_gpu._hadamard_signs = comp_cpu._hadamard_signs.clone()
        comp_gpu.jl_matrix = comp_cpu.jl_matrix.clone()

        kv = torch.randn(n_tokens, head_dim, device=device)
        q = torch.randn(1, head_dim, device=device)

        iters = 200 if n_tokens <= 32 else 50

        # --- compress ---
        print("\ncompress():")
        t_cpu = bench(lambda: comp_cpu.compress(kv), iters=iters, label="CPU path")
        t_gpu = bench(lambda: comp_gpu.compress(kv), iters=iters, label="GPU path")
        print(f"  speedup: {t_cpu/t_gpu:.1f}x")

        # Pre-compress for decompress/attention_score benchmarks
        c_cpu = comp_cpu.compress(kv)
        c_gpu = comp_gpu.compress(kv)

        # --- decompress ---
        print("\ndecompress():")
        t_cpu = bench(lambda: comp_cpu.decompress(c_cpu), iters=iters, label="CPU path")
        t_gpu = bench(lambda: comp_gpu.decompress(c_gpu), iters=iters, label="GPU path")
        print(f"  speedup: {t_cpu/t_gpu:.1f}x")

        # --- attention_score ---
        print("\nattention_score():")
        t_cpu = bench(lambda: comp_cpu.attention_score(q, c_cpu), iters=iters, label="CPU path")
        t_gpu = bench(lambda: comp_gpu.attention_score(q, c_gpu), iters=iters, label="GPU path")
        print(f"  speedup: {t_cpu/t_gpu:.1f}x")


if __name__ == "__main__":
    main()
