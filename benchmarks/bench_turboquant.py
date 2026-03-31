#!/usr/bin/env python3
"""VRAMancer TurboQuant + Sparse V benchmark.

Measures the impact of KV cache compression (TurboQuant) and Sparse V
selective decompression on inference performance.

Test matrix:
  1. BF16 baseline (no compression)
  2. TurboQuant KV compression (3 bits/angle, ~4.6x reduction)
  3. TurboQuant + Sparse V 30% (decompress top 30% of values)
  4. TurboQuant + Sparse V 10% (decompress top 10% of values)
  5. NVFP4 + TurboQuant + Sparse V 10% (full stack, Blackwell only)

Reports: tok/s, VRAM peak, KV cache overhead, context capacity.

Usage:
    # Single-GPU (RTX 3090)
    python benchmarks/bench_turboquant.py --model gpt2

    # Blackwell GPU (NVFP4 + TurboQuant)
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_turboquant.py \\
        --model Qwen/Qwen2.5-7B-Instruct --include-nvfp4

    # Multi-GPU heterogeneous
    python benchmarks/bench_turboquant.py --model gpt2 --num-gpus 2

    # Long context stress test
    python benchmarks/bench_turboquant.py --model gpt2 --context-lengths 128,512,2048,8192
"""
from __future__ import annotations

import os
import sys
import time
import gc
import json
import argparse
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Remove test mode flags
os.environ.pop("VRM_MINIMAL_TEST", None)
os.environ.pop("VRM_TEST_MODE", None)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required for benchmarks")
    sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────

def gpu_info(device_id: int = 0) -> str:
    if not torch.cuda.is_available():
        return "No CUDA"
    name = torch.cuda.get_device_name(device_id)
    cc = torch.cuda.get_device_capability(device_id)
    free, total = torch.cuda.mem_get_info(device_id)
    return f"{name} CC={cc[0]}.{cc[1]}, {free/1e9:.1f}/{total/1e9:.1f} GB"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram_used_gb(device_id: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device_id) / 1e9


# ── Benchmark configurations ──────────────────────────────────────

CONFIGS = {
    "BF16 baseline": {
        "VRM_KV_COMPRESSION": "",
        "VRM_SPARSE_V_RATIO": "1.0",
        "VRM_QUANTIZATION": "",
    },
    "TurboQuant (3-bit)": {
        "VRM_KV_COMPRESSION": "turboquant",
        "VRM_KV_COMPRESSION_BITS": "3",
        "VRM_SPARSE_V_RATIO": "1.0",
        "VRM_QUANTIZATION": "",
    },
    "TurboQuant + Sparse V 30%": {
        "VRM_KV_COMPRESSION": "turboquant",
        "VRM_KV_COMPRESSION_BITS": "3",
        "VRM_SPARSE_V_RATIO": "0.3",
        "VRM_QUANTIZATION": "",
    },
    "TurboQuant + Sparse V 10%": {
        "VRM_KV_COMPRESSION": "turboquant",
        "VRM_KV_COMPRESSION_BITS": "3",
        "VRM_SPARSE_V_RATIO": "0.1",
        "VRM_QUANTIZATION": "",
    },
}

NVFP4_CONFIGS = {
    "NVFP4 baseline": {
        "VRM_KV_COMPRESSION": "",
        "VRM_SPARSE_V_RATIO": "1.0",
        "VRM_QUANTIZATION": "nvfp4",
    },
    "NVFP4 + TurboQuant + Sparse V 10%": {
        "VRM_KV_COMPRESSION": "turboquant",
        "VRM_KV_COMPRESSION_BITS": "3",
        "VRM_SPARSE_V_RATIO": "0.1",
        "VRM_QUANTIZATION": "nvfp4",
    },
}


PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land",
    "Python is a programming language that",
    "The best way to learn programming is",
    "In the year 2050 the world will",
]


def generate_long_prompt(target_tokens: int) -> str:
    """Generate a prompt that's approximately target_tokens long."""
    base = "The quick brown fox jumps over the lazy dog. "
    # ~10 tokens per sentence
    repeats = max(1, target_tokens // 10)
    return base * repeats


# ── Single-config benchmark ──────────────────────────────────────

def bench_config(
    model_name: str,
    config_name: str,
    env_vars: Dict[str, str],
    prompts: List[str],
    max_tokens: int,
    num_gpus: int,
) -> Dict[str, Any]:
    """Run a single benchmark configuration via VRAMancer pipeline."""

    # Set env vars
    for k, v in env_vars.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)

    clear_gpu()

    from core.inference_pipeline import InferencePipeline, reset_pipeline
    reset_pipeline()

    result = {
        "config": config_name,
        "model": model_name,
        "num_gpus": num_gpus,
        "max_tokens": max_tokens,
    }

    try:
        t_load = time.perf_counter()
        pipeline = InferencePipeline()
        pipeline.load(model_name, num_gpus=num_gpus)
        load_time = time.perf_counter() - t_load
        result["load_time_s"] = round(load_time, 2)

        # Check if TurboQuant is active
        kv_active = False
        if hasattr(pipeline, '_paged_kv') and pipeline._paged_kv:
            kv_active = pipeline._paged_kv.kv_compression_active
        result["kv_compression_active"] = kv_active

        # Warmup
        warmup_prompt = "Hello world"
        pipeline.generate(warmup_prompt, max_new_tokens=8)
        torch.cuda.synchronize()
        clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
        total_tokens = 0
        total_time = 0.0
        per_prompt = []

        # Get tokenizer for accurate token counting
        _tokenizer = getattr(pipeline.backend, "tokenizer", None)

        for prompt in prompts:
            t0 = time.perf_counter()
            result_text = pipeline.generate(prompt, max_new_tokens=max_tokens)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            # Count actual generated tokens (generate() returns only new text, no prompt)
            if _tokenizer is not None:
                tokens = len(_tokenizer.encode(result_text))
                tokens = max(tokens, 1)
            else:
                tokens = len(result_text) // 4  # rough estimate
            tps = tokens / elapsed if elapsed > 0 else 0
            per_prompt.append({"tokens": tokens, "time_s": round(elapsed, 3), "tok_s": round(tps, 1)})
            total_tokens += tokens
            total_time += elapsed

        avg_tps = total_tokens / total_time if total_time > 0 else 0
        peak_vram = vram_used_gb(0)

        # Multi-GPU VRAM
        if num_gpus > 1 and torch.cuda.device_count() > 1:
            for i in range(1, min(num_gpus, torch.cuda.device_count())):
                peak_vram += torch.cuda.max_memory_allocated(i) / 1e9

        result.update({
            "total_tokens": total_tokens,
            "total_time_s": round(total_time, 3),
            "avg_tok_s": round(avg_tps, 1),
            "peak_vram_gb": round(peak_vram, 2),
            "per_prompt": per_prompt,
            "status": "OK",
        })

        # Cleanup
        reset_pipeline()
        del pipeline
        clear_gpu()

    except Exception as e:
        result["status"] = f"FAILED: {e}"
        # Cleanup on failure
        try:
            from core.inference_pipeline import reset_pipeline
            reset_pipeline()
        except Exception:
            pass
        clear_gpu()

    return result


# ── Context length scaling benchmark ─────────────────────────────

def bench_context_scaling(
    model_name: str,
    context_lengths: List[int],
    num_gpus: int,
) -> List[Dict[str, Any]]:
    """Measure VRAM usage and tok/s at various context lengths with/without TurboQuant."""
    results = []

    for ctx_len in context_lengths:
        prompt = generate_long_prompt(ctx_len)

        for config_name, env_vars in [
            ("BF16", {"VRM_KV_COMPRESSION": "", "VRM_SPARSE_V_RATIO": "1.0", "VRM_QUANTIZATION": ""}),
            ("TurboQuant+SparseV10%", {"VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3",
                                        "VRM_SPARSE_V_RATIO": "0.1", "VRM_QUANTIZATION": ""}),
        ]:
            result = bench_config(
                model_name, f"{config_name} ctx={ctx_len}",
                env_vars, [prompt], max_tokens=32, num_gpus=num_gpus,
            )
            result["context_length"] = ctx_len
            results.append(result)

    return results


# ── Standalone KV compressor microbenchmark ──────────────────────

def bench_kv_compressor():
    """Microbenchmark: compress/decompress throughput and compression ratio."""
    print("\n" + "=" * 60)
    print("  KV Compressor Microbenchmark")
    print("=" * 60)

    from core.kv_quantizer import KVCacheCompressor

    head_dims = [64, 128]
    seq_lengths = [128, 512, 2048]

    for head_dim in head_dims:
        torch.manual_seed(42)
        comp = KVCacheCompressor(head_dim=head_dim, bits_per_angle=3)
        bpd = comp.bits_per_dim()
        ratio = 16.0 / bpd

        print(f"\n  head_dim={head_dim}, {bpd:.1f} bits/dim ({ratio:.1f}x compression)")
        print(f"  {'seq_len':>8s}  {'compress':>10s}  {'decompress':>10s}  {'score':>10s}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

        for seq_len in seq_lengths:
            keys = torch.randn(seq_len, head_dim)
            q = torch.randn(1, head_dim)

            # Compress
            t0 = time.perf_counter()
            ck = comp.compress(keys)
            t_compress = (time.perf_counter() - t0) * 1000

            # Decompress
            t0 = time.perf_counter()
            _ = comp.decompress(ck)
            t_decompress = (time.perf_counter() - t0) * 1000

            # Attention score (asymmetric, no decompression)
            t0 = time.perf_counter()
            _ = comp.attention_score(q, ck)
            t_score = (time.perf_counter() - t0) * 1000

            print(f"  {seq_len:>8d}  {t_compress:>8.2f}ms  {t_decompress:>8.2f}ms  "
                  f"{t_score:>8.2f}ms")


# ── Main ─────────────────────────────────────────────────────────

def print_results_table(results: List[Dict[str, Any]]):
    """Print a formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Configuration':<40s}  {'tok/s':>8s}  {'VRAM GB':>8s}  {'Status':>10s}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*10}")

    baseline_tps = None
    for r in results:
        name = r["config"]
        if r["status"] == "OK":
            tps = r["avg_tok_s"]
            vram = r["peak_vram_gb"]
            if baseline_tps is None:
                baseline_tps = tps
                delta = ""
            else:
                pct = ((tps - baseline_tps) / baseline_tps * 100) if baseline_tps else 0
                delta = f" ({pct:+.1f}%)"
            print(f"  {name:<40s}  {tps:>7.1f}  {vram:>7.2f}  {'OK' + delta:>10s}")
        else:
            print(f"  {name:<40s}  {'—':>8s}  {'—':>8s}  {'FAILED':>10s}")


def main():
    parser = argparse.ArgumentParser(description="VRAMancer TurboQuant + Sparse V benchmark")
    parser.add_argument("--model", default="gpt2", help="Model name/path")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per prompt")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts")
    parser.add_argument("--include-nvfp4", action="store_true", help="Include NVFP4 configs (Blackwell)")
    parser.add_argument("--nvfp4-only", action="store_true", help="Only run NVFP4 configs (skip BF16)")
    parser.add_argument("--context-lengths", type=str, default="",
                        help="Comma-separated context lengths for scaling test (e.g. 128,512,2048)")
    parser.add_argument("--micro-only", action="store_true", help="Only run KV compressor microbenchmark")
    parser.add_argument("--output", type=str, default="", help="Save JSON results to file")
    args = parser.parse_args()

    print("VRAMancer TurboQuant + Sparse V Benchmark")
    print("=" * 60)
    n_gpus = torch.cuda.device_count()
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    for i in range(min(n_gpus, args.num_gpus)):
        print(f"GPU {i}: {gpu_info(i)}")
    print()

    # ── KV compressor microbenchmark ──
    bench_kv_compressor()

    if args.micro_only:
        return

    # ── Pipeline benchmarks ──
    prompts = PROMPTS[:args.num_prompts]

    if args.nvfp4_only:
        configs = dict(NVFP4_CONFIGS)
    else:
        configs = dict(CONFIGS)

    if args.include_nvfp4 and not args.nvfp4_only:
        cc = torch.cuda.get_device_capability(0)
        if cc[0] >= 10:
            configs.update(NVFP4_CONFIGS)
            print(f"\nNVFP4 enabled (CC={cc[0]}.{cc[1]})")
        else:
            print(f"\nNVFP4 skipped (CC={cc[0]}.{cc[1]} < 10.0, need Blackwell)")

    results = []
    for config_name, env_vars in configs.items():
        print(f"\n>>> {config_name}")
        r = bench_config(args.model, config_name, env_vars, prompts,
                         args.max_tokens, args.num_gpus)
        results.append(r)
        if r["status"] == "OK":
            print(f"  {r['avg_tok_s']:.1f} tok/s, {r['peak_vram_gb']:.2f} GB VRAM, "
                  f"{r['total_tokens']} tokens in {r['total_time_s']:.1f}s")
        else:
            print(f"  {r['status']}")

    print_results_table(results)

    # ── Context scaling ──
    if args.context_lengths:
        ctx_lens = [int(x) for x in args.context_lengths.split(",")]
        print(f"\n\n{'=' * 60}")
        print(f"  CONTEXT SCALING TEST: {ctx_lens}")
        print(f"{'=' * 60}")
        ctx_results = bench_context_scaling(args.model, ctx_lens, args.num_gpus)
        print_results_table(ctx_results)
        results.extend(ctx_results)

    # ── Save results ──
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
