#!/usr/bin/env python3
"""Qwen3.6-35B-A3B NVFP4 on a single Blackwell GPU (RTX 5070 Ti) with CPU offload.

The HONEST realization of "full NVFP4 on a 16 GB card":
  - 22 GB NVFP4 weights don't fit in 16 GB VRAM.
  - FP4 tensor-core compute ONLY works on Blackwell (SM 10+), so only the
    5070 Ti can run it — the 3090 (Ampere) has no FP4 kernels.
  - ReBAR/VRAM-lending CANNOT help: FP4 matmul needs weights RESIDENT in the
    compute GPU's own VRAM. Remote/lent memory = PCIe streaming = no win.
  - The real lever is the MoE structure (256 experts, 8 active/token):
    attention + hot experts stay resident in NVFP4 on the 5070 Ti (real
    Blackwell compute); cold experts spill to system RAM via cpu_offload_gb
    and stream in on demand.

Usage:
    CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
        python benchmarks/bench_qwen36_nvfp4_offload.py
"""
import os
import sys
import time
import json

MODEL = os.environ.get("VRM_MODEL", "/home/jeremie/models/Qwen3.6-35B-A3B-NVFP4")
CPU_OFFLOAD_GB = float(os.environ.get("VRM_CPU_OFFLOAD_GB", "16"))
MAX_MODEL_LEN = int(os.environ.get("VRM_MAX_MODEL_LEN", "2048"))
GPU_UTIL = float(os.environ.get("VRM_GPU_UTIL", "0.95"))
MAX_TOKENS = int(os.environ.get("VRM_MAX_TOKENS", "64"))
ENFORCE_EAGER = os.environ.get("VRM_ENFORCE_EAGER", "1") == "1"

PROMPTS = [
    "Write a Python function that implements binary search:",
    "Explain quantum entanglement in simple terms:",
    "Refactor this loop to be more efficient:\nfor i in range(len(a)):\n    b.append(a[i]*2)",
    "What is the time complexity of merge sort and why?",
    "Write a SQL query to find the second highest salary:",
]


def main():
    from vllm import LLM, SamplingParams

    print("=" * 72)
    print(" Qwen3.6-35B-A3B NVFP4 — single Blackwell GPU + CPU offload")
    print(f" Model: {MODEL}")
    print(f" cpu_offload_gb={CPU_OFFLOAD_GB}  max_model_len={MAX_MODEL_LEN}"
          f"  gpu_util={GPU_UTIL}  enforce_eager={ENFORCE_EAGER}")
    print("=" * 72)

    load_start = time.perf_counter()
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="auto",                      # modelopt NVFP4 auto-detected
        gpu_memory_utilization=GPU_UTIL,
        cpu_offload_gb=CPU_OFFLOAD_GB,     # cold experts -> system RAM
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=ENFORCE_EAGER,       # skip CUDA graphs -> saves VRAM
    )
    load_time = time.perf_counter() - load_start
    print(f"\n  Loaded in {load_time:.1f}s")

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # Warmup
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8))

    start = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tok_s = total_tokens / elapsed

    print(f"\n  {len(PROMPTS)} prompts, {total_tokens} tokens in {elapsed:.2f}s")
    print(f"  => {tok_s:.1f} tok/s (aggregate)")
    print(f"\n  Sample output:\n  {outs[0].outputs[0].text[:200]!r}")

    result = {
        "model": MODEL,
        "quant": "NVFP4 (modelopt MIXED_PRECISION)",
        "gpu": "RTX 5070 Ti (Blackwell SM12.0)",
        "cpu_offload_gb": CPU_OFFLOAD_GB,
        "load_time_s": round(load_time, 1),
        "tok_s": round(tok_s, 1),
        "tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
    }
    print(f"\nRESULT_JSON: {json.dumps(result)}")
    return result


if __name__ == "__main__":
    main()
