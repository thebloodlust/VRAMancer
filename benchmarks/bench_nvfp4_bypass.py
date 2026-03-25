#!/usr/bin/env python3
"""
Benchmark: DirectFP4Linear bypass vs torchao NVFP4Tensor baseline.

Tests on a real Qwen2.5-7B model to measure the actual speedup
from bypassing torchao's __torch_dispatch__ overhead.

Usage:
    python3 benchmarks/bench_nvfp4_bypass.py [--model MODEL] [--max-tokens N]
"""

import argparse
import gc
import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torch.nn as nn


def check_blackwell():
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i)
        if cc[0] >= 10:
            return i
    print("ERROR: No Blackwell GPU (CC >= 10.0) found")
    sys.exit(1)


def get_gpu_mem(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1024**3


def nvfp4_filter_fn(module, fqn):
    """Exclude lm_head and embedding layers from FP4 quantization."""
    if not isinstance(module, nn.Linear):
        return False
    skip = ("lm_head", "embed_tokens", "wte", "wpe")
    return not any(s in fqn for s in skip)


def bench_generation(model, tokenizer, prompt, max_tokens, label, device):
    """Benchmark autoregressive generation speed."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize(device)

    # Benchmark
    speeds = []
    for run in range(3):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        n_tok = out.shape[1] - input_len
        speed = n_tok / dt
        speeds.append(speed)
        print(f"  [{label}] Run {run+1}: {n_tok} tok, {dt:.2f}s, {speed:.1f} tok/s")

    avg = sum(speeds) / len(speeds)
    best = max(speeds)
    vram = get_gpu_mem(device)
    print(f"  [{label}] AVG: {avg:.1f} tok/s  BEST: {best:.1f} tok/s  VRAM: {vram:.2f} GB")
    return avg, best, vram


def load_model_nvfp4_torchao(model_name, device):
    """Load model with standard torchao NVFP4 (baseline)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao import quantize_
    from torchao.prototype.mx_formats import (
        NVFP4DynamicActivationNVFP4WeightConfig,
    )

    print(f"\n{'='*60}")
    print(f"Loading {model_name} — torchao NVFP4 baseline")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Set CUDA device context so is_sm_at_least_100() sees Blackwell GPU
    torch.cuda.set_device(device)
    config = NVFP4DynamicActivationNVFP4WeightConfig()
    quantize_(model, config, filter_fn=nvfp4_filter_fn)
    model = model.to(device)
    torch.cuda.synchronize(device)
    load_time = time.time() - t0
    vram = get_gpu_mem(device)
    print(f"Load time: {load_time:.1f}s, VRAM: {vram:.2f} GB")

    return model, tokenizer


def load_model_nvfp4_direct(model_name, device):
    """Load model with DirectFP4CachedLinear bypass."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao import quantize_
    from torchao.prototype.mx_formats import (
        NVFP4DynamicActivationNVFP4WeightConfig,
    )

    # Add core/ to path for import
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.nvfp4_direct import replace_with_direct_fp4

    print(f"\n{'='*60}")
    print(f"Loading {model_name} — DirectFP4 bypass")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Set CUDA device context so is_sm_at_least_100() sees Blackwell GPU
    torch.cuda.set_device(device)
    # Step 1: Quantize with torchao (to get FP4 weights)
    config = NVFP4DynamicActivationNVFP4WeightConfig()
    quantize_(model, config, filter_fn=nvfp4_filter_fn)

    # Step 2: Replace NVFP4Tensor layers with DirectFP4CachedLinear
    import logging
    logging.basicConfig(level=logging.INFO)
    n_replaced = replace_with_direct_fp4(model, verbose=True)
    print(f"Replaced {n_replaced} layers with DirectFP4CachedLinear")

    # Step 3: Move to GPU
    model = model.to(device)
    torch.cuda.synchronize(device)
    load_time = time.time() - t0
    vram = get_gpu_mem(device)
    print(f"Load time: {load_time:.1f}s, VRAM: {vram:.2f} GB")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt", default="Explain the concept of GPU memory pooling in distributed systems.")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    args = parser.parse_args()

    gpu_idx = check_blackwell()
    device = torch.device(f"cuda:{gpu_idx}")
    gpu_name = torch.cuda.get_device_name(gpu_idx)
    cc = torch.cuda.get_device_capability(gpu_idx)
    print(f"Blackwell GPU: {gpu_name} (CC {cc[0]}.{cc[1]}) at cuda:{gpu_idx}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")

    results = {}

    # === Baseline: torchao NVFP4 ===
    if not args.skip_baseline:
        model, tokenizer = load_model_nvfp4_torchao(args.model, device)
        avg, best, vram = bench_generation(model, tokenizer, args.prompt,
                                            args.max_tokens, "torchao NVFP4", device)
        results["torchao"] = {"avg": avg, "best": best, "vram": vram}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # === DirectFP4 bypass ===
    if not args.skip_direct:
        model, tokenizer = load_model_nvfp4_direct(args.model, device)
        avg, best, vram = bench_generation(model, tokenizer, args.prompt,
                                            args.max_tokens, "DirectFP4", device)
        results["direct"] = {"avg": avg, "best": best, "vram": vram}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # === Summary ===
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:20s}: AVG {r['avg']:.1f} tok/s  BEST {r['best']:.1f} tok/s  VRAM {r['vram']:.2f} GB")

    if "torchao" in results and "direct" in results:
        speedup = results["direct"]["avg"] / results["torchao"]["avg"]
        print(f"\n  DirectFP4 speedup: {speedup:.2f}x over torchao baseline")


if __name__ == "__main__":
    main()
