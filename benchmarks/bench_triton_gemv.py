#!/usr/bin/env python3
"""Test and benchmark the Triton NF4 GEMV kernel vs BnB native.

Usage:
    python benchmarks/bench_triton_gemv.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import bitsandbytes as bnb
import bitsandbytes.functional as bnb_F
from core.triton_gemv import triton_gemv_4bit, _dequant_absmax

print("=" * 70)
print(" Triton NF4 GEMV — Correctness & Performance")
print("=" * 70)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print()

# Test sizes matching real model dimensions (Qwen2.5-7B)
SIZES = [
    (3584, 3584, "Qwen 7B hidden"),      # q_proj, k_proj, v_proj
    (3584, 18944, "Qwen 7B gate/up"),    # gate_proj, up_proj
    (18944, 3584, "Qwen 7B down"),       # down_proj
    (3584, 152064, "Qwen 7B lm_head"),   # lm_head (huge)
    (4096, 4096, "Generic 4K"),
]

WARMUP = 10
ITERS = 100

for K, N, desc in SIZES:
    print(f"\n--- {desc}: [{K}] x [{N}, {K}] → [{N}] ---")

    # Create and quantize weight
    w = torch.randn(N, K, dtype=torch.float16, device='cuda:0')
    qw, qs = bnb_F.quantize_4bit(w, quant_type='nf4', blocksize=64, compress_statistics=True)

    x = torch.randn(1, K, dtype=torch.float16, device='cuda:0')

    # === Correctness test ===
    # BnB reference
    out_bnb = bnb_F.gemv_4bit(x, qw.t(), out=None, state=qs)

    # Triton
    out_triton = triton_gemv_4bit(x, qw, state=qs)

    # Dequantize + matmul reference (most accurate)
    absmax = _dequant_absmax(qs)
    code = qs.code.to(torch.float32)
    w_deq = bnb_F.dequantize_4bit(qw, qs)
    out_ref = (x.float() @ w_deq.t().float()).half()

    diff_bnb = (out_bnb - out_ref).abs().max().item()
    diff_triton = (out_triton - out_ref).abs().max().item()
    diff_vs = (out_triton - out_bnb).abs().max().item()

    print(f"  Correctness: BnB vs ref={diff_bnb:.6f}  Triton vs ref={diff_triton:.6f}  Triton vs BnB={diff_vs:.6f}")

    if diff_triton > 1.0:
        print(f"  *** WARNING: Large error! ***")
        # Debug: check first 10 elements
        print(f"  BnB[:10]:    {out_bnb[0, :10].tolist()}")
        print(f"  Triton[:10]: {out_triton[0, :10].tolist()}")
        print(f"  Ref[:10]:    {out_ref[0, :10].tolist()}")
        continue

    # === Performance test ===
    # Warmup
    for _ in range(WARMUP):
        bnb_F.gemv_4bit(x, qw.t(), out=None, state=qs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        bnb_F.gemv_4bit(x, qw.t(), out=None, state=qs)
    torch.cuda.synchronize()
    bnb_us = (time.perf_counter() - t0) / ITERS * 1e6

    # Triton warmup (first call compiles)
    for _ in range(WARMUP):
        triton_gemv_4bit(x, qw, state=qs)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        triton_gemv_4bit(x, qw, state=qs)
    torch.cuda.synchronize()
    triton_us = (time.perf_counter() - t0) / ITERS * 1e6

    speedup = bnb_us / triton_us
    tag = "FASTER" if speedup > 1.0 else "SLOWER"
    print(f"  BnB:    {bnb_us:8.1f} µs")
    print(f"  Triton: {triton_us:8.1f} µs  ({speedup:.2f}x {tag})")

    del w, qw, qs, x, out_bnb, out_triton, out_ref, w_deq
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print(" DONE")
print("=" * 70)
