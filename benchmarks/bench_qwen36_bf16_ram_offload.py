#!/usr/bin/env python3
"""Qwen3.6-35B-A3B BF16 across 2 GPUs + system RAM offload — the honest proof.

This is the REAL "extend VRAM into system RAM" path (UVM-style oversubscription
via accelerate's CPU offload). ReBAR only accelerates the PCIe transit; it does
NOT add capacity. We give accelerate an explicit budget:

    cuda:0 (RTX 3090)    -> 22 GiB
    cuda:1 (RTX 5070 Ti) -> 13 GiB
    cpu  (system RAM)    -> 90 GiB   (you have 154 GiB free)

72 GB BF16 weights fit easily in 35 GiB VRAM + 90 GiB RAM. Cold layers live in
RAM and stream over PCIe on demand. This runs a model that OOMs on either GPU
alone AND on both GPUs combined without offload.

Usage:
    python benchmarks/bench_qwen36_bf16_ram_offload.py
"""
import os
import sys
import time
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL = os.environ.get("VRM_MODEL", "/home/jeremie/models/Qwen3.6-35B-A3B")
MAX_TOKENS = int(os.environ.get("VRM_MAX_TOKENS", "48"))
GPU0_GIB = os.environ.get("VRM_GPU0_GIB", "22")
GPU1_GIB = os.environ.get("VRM_GPU1_GIB", "13")
CPU_GIB = os.environ.get("VRM_CPU_GIB", "90")

PROMPTS = [
    "Write a Python function that implements binary search:",
    "Explain quantum entanglement in simple terms:",
    "What is the time complexity of merge sort and why?",
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 72)
    print(" Qwen3.6-35B-A3B BF16 — 2 GPU + system RAM offload (honest 'extend VRAM')")
    print(f" Model: {MODEL}")
    print(f" Budget: cuda:0={GPU0_GIB}GiB  cuda:1={GPU1_GIB}GiB  cpu={CPU_GIB}GiB")
    print("=" * 72)

    max_memory = {0: f"{GPU0_GIB}GiB", 1: f"{GPU1_GIB}GiB", "cpu": f"{CPU_GIB}GiB"}

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"\n  Loaded in {load_time:.1f}s")

    # Report device placement summary
    dev_counts = {}
    for _, p in model.named_parameters():
        d = str(p.device)
        dev_counts[d] = dev_counts.get(d, 0) + p.numel()
    total = sum(dev_counts.values())
    print("  Parameter placement:")
    for d, n in sorted(dev_counts.items()):
        print(f"    {d}: {n/1e9:.2f}B params ({100*n/total:.1f}%)")

    for i in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(i)
        print(f"    GPU {i}: {(tot-free)/1024**3:.1f}GB used / {tot/1024**3:.1f}GB")

    # Warmup
    print("\n  Warmup...")
    inp = tok("Hello", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model.generate(**inp, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    print(f"  Benchmarking ({len(PROMPTS)} prompts x {MAX_TOKENS} tokens)...")
    total_gen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for p in PROMPTS:
            inp = tok(p, return_tensors="pt").to("cuda:0")
            out = model.generate(**inp, max_new_tokens=MAX_TOKENS,
                                 do_sample=False, use_cache=True)
            total_gen += out.shape[1] - inp["input_ids"].shape[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tok_s = total_gen / elapsed

    print(f"\n  {total_gen} tokens in {elapsed:.2f}s => {tok_s:.2f} tok/s")

    result = {
        "model": MODEL,
        "dtype": "bf16",
        "placement": {d: round(n/1e9, 2) for d, n in dev_counts.items()},
        "load_time_s": round(load_time, 1),
        "tok_s": round(tok_s, 2),
        "tokens": total_gen,
        "elapsed_s": round(elapsed, 2),
    }
    print(f"\nRESULT_JSON: {json.dumps(result)}")
    return result


if __name__ == "__main__":
    main()
