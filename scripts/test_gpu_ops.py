#!/usr/bin/env python3
"""Test GPU vs CPU correctness for TurboQuant KV compression ops."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from core.kv_quantizer import KVCacheCompressor

def main():
    torch.manual_seed(42)
    device = "cuda:0"
    head_dim = 128

    # Create two compressors with SAME random state
    torch.manual_seed(42)
    comp_cpu = KVCacheCompressor(head_dim=head_dim, bits_per_angle=3, force_cpu=True).to(device)

    torch.manual_seed(42)
    comp_gpu = KVCacheCompressor(head_dim=head_dim, bits_per_angle=3, force_cpu=False).to(device)

    # Copy buffers so they match exactly
    comp_gpu._hadamard_signs = comp_cpu._hadamard_signs.clone()
    comp_gpu.jl_matrix = comp_cpu.jl_matrix.clone()

    # Test data
    kv = torch.randn(32, head_dim, device=device)
    q = torch.randn(4, head_dim, device=device)

    # Compress with both
    c_cpu = comp_cpu.compress(kv)
    c_gpu = comp_gpu.compress(kv)

    print("=== COMPRESS ===")
    print(f"radius match: {torch.allclose(c_cpu['radius'], c_gpu['radius'])}")
    for i, (a_c, a_g) in enumerate(zip(c_cpu["angles"], c_gpu["angles"])):
        match = (a_c == a_g).all().item()
        print(f"angles[{i}] match: {match}")
    print(f"qjl_signs match: {(c_cpu['qjl_signs'] == c_gpu['qjl_signs']).all().item()}")
    print(f"qjl_norms match: {torch.allclose(c_cpu['qjl_norms'], c_gpu['qjl_norms'])}")

    # Decompress
    d_cpu = comp_cpu.decompress(c_cpu)
    d_gpu = comp_gpu.decompress(c_gpu)
    cosine = torch.nn.functional.cosine_similarity(d_cpu.flatten(), d_gpu.flatten(), dim=0)
    print(f"\n=== DECOMPRESS ===")
    print(f"cosine similarity: {cosine.item():.6f}")
    print(f"max abs diff: {(d_cpu - d_gpu).abs().max().item():.8f}")

    # Attention scores
    s_cpu = comp_cpu.attention_score(q, c_cpu)
    s_gpu = comp_gpu.attention_score(q, c_gpu)
    score_cos = torch.nn.functional.cosine_similarity(s_cpu.flatten(), s_gpu.flatten(), dim=0)
    print(f"\n=== ATTENTION SCORE ===")
    print(f"cosine similarity: {score_cos.item():.6f}")
    print(f"max abs diff: {(s_cpu - s_gpu).abs().max().item():.8f}")

    all_pass = True
    # Verify exact match for discrete outputs
    for i, (a_c, a_g) in enumerate(zip(c_cpu["angles"], c_gpu["angles"])):
        if not (a_c == a_g).all():
            all_pass = False
    if not (c_cpu["qjl_signs"] == c_gpu["qjl_signs"]).all():
        all_pass = False
    if cosine.item() < 0.999:
        all_pass = False
    if score_cos.item() < 0.999:
        all_pass = False

    print(f"\n{'PASS' if all_pass else 'FAIL'} — GPU ops {'match' if all_pass else 'DIFFER from'} CPU ops")

if __name__ == "__main__":
    main()
