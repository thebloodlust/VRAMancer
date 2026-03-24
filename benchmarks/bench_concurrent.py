#!/usr/bin/env python3
"""VRAMancer Concurrent Throughput Benchmark.

Measures tokens/second under concurrent load using the ContinuousBatcher.
This is the key metric where VRAMancer was weakest vs vLLM.

Tests:
  1. Sequential baseline (1 request at a time)
  2. Concurrent batches (N requests simultaneously)
  3. Sustained load (stream of requests over time)

Usage:
    # Quick smoke test (GPT-2)
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_concurrent.py --model gpt2

    # Realistic workload (1.1B model, 20 concurrent)
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_concurrent.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --concurrent 20

    # Stress test
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_concurrent.py \\
        --model gpt2 --concurrent 50 --total 200 --max-tokens 64
"""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import Future, as_completed

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land",
    "Python is a programming language that",
    "The best way to learn programming is",
    "In the year 2050 the world will",
    "Machine learning models can be trained to",
    "The largest language model ever built has",
    "Quantum computing will change the world by",
    "Space exploration in the 21st century has",
    "The ocean covers most of the planet and",
]


def _get_prompt(idx: int) -> str:
    return PROMPTS[idx % len(PROMPTS)]


def bench_sequential(pipeline, num_requests: int, max_tokens: int) -> dict:
    """Baseline: one request at a time, no batching."""
    import torch

    total_tokens = 0
    start = time.perf_counter()
    for i in range(num_requests):
        prompt = _get_prompt(i)
        result = pipeline.generate(prompt, max_new_tokens=max_tokens)
        total_tokens += max_tokens
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "mode": "sequential",
        "requests": num_requests,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(total_tokens / elapsed, 1),
        "req_s": round(num_requests / elapsed, 2),
    }


def bench_concurrent_batch(
    batcher, num_requests: int, max_tokens: int, timeout: float = 300.0
) -> dict:
    """Concurrent: submit all requests at once, measure total throughput."""
    import torch

    futures = []
    start = time.perf_counter()

    for i in range(num_requests):
        prompt = _get_prompt(i)
        fut = batcher.submit(prompt, max_new_tokens=max_tokens)
        futures.append(fut)

    # Collect all results
    completed = 0
    errors = 0
    for fut in futures:
        try:
            fut.result(timeout=timeout)
            completed += 1
        except Exception as e:
            errors += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = completed * max_tokens

    return {
        "mode": "concurrent_batch",
        "requests": num_requests,
        "completed": completed,
        "errors": errors,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(total_tokens / elapsed, 1) if elapsed > 0 else 0,
        "req_s": round(completed / elapsed, 2) if elapsed > 0 else 0,
    }


def bench_sustained_load(
    batcher, total_requests: int, concurrent: int,
    max_tokens: int, timeout: float = 300.0
) -> dict:
    """Sustained: maintain N in-flight requests, submit new ones as old complete."""
    import torch

    completed = 0
    errors = 0
    submitted = 0
    active: list = []

    start = time.perf_counter()

    # Initial burst
    for i in range(min(concurrent, total_requests)):
        prompt = _get_prompt(i)
        fut = batcher.submit(prompt, max_new_tokens=max_tokens)
        active.append(fut)
        submitted += 1

    # Sustain
    while completed + errors < total_requests:
        # Check for completions
        new_active = []
        for fut in active:
            if fut.done():
                try:
                    fut.result(timeout=0.001)
                    completed += 1
                except Exception:
                    errors += 1
                # Submit replacement
                if submitted < total_requests:
                    prompt = _get_prompt(submitted)
                    new_fut = batcher.submit(prompt, max_new_tokens=max_tokens)
                    new_active.append(new_fut)
                    submitted += 1
            else:
                new_active.append(fut)
        active = new_active

        if active:
            time.sleep(0.001)

        # Safety timeout
        if time.perf_counter() - start > timeout:
            errors += len(active)
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = completed * max_tokens

    return {
        "mode": "sustained_load",
        "total_requests": total_requests,
        "concurrent": concurrent,
        "completed": completed,
        "errors": errors,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(total_tokens / elapsed, 1) if elapsed > 0 else 0,
        "req_s": round(completed / elapsed, 2) if elapsed > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="VRAMancer concurrent throughput benchmark")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrent", type=int, default=10,
                        help="Number of concurrent requests")
    parser.add_argument("--total", type=int, default=50,
                        help="Total requests for sustained load test")
    parser.add_argument("--sequential", type=int, default=5,
                        help="Number of sequential baseline requests")
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from core.inference_pipeline import InferencePipeline
    from core.continuous_batcher import ContinuousBatcher

    short = args.model.split("/")[-1]

    print(f"\n{'='*70}")
    print(f"VRAMancer Concurrent Throughput — {short}")
    print(f"Max tokens: {args.max_tokens} | Concurrent: {args.concurrent} | Total: {args.total}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")

    # Load model via pipeline
    print(f"\nLoading {args.model}...")
    pipe = InferencePipeline(
        backend_name="huggingface",
        enable_metrics=False,
        enable_discovery=False,
        verbose=False,
    )
    pipe.load(args.model, num_gpus=1)

    backend = pipe.backend
    results = {"model": short, "max_tokens": args.max_tokens}

    # 1. Sequential baseline
    if not args.skip_sequential:
        print(f"\n[1/3] Sequential baseline ({args.sequential} requests)...")
        r = bench_sequential(pipe, args.sequential, args.max_tokens)
        results["sequential"] = r
        print(f"  => {r['tok_s']} tok/s | {r['req_s']} req/s")

    # 2. Setup batcher
    print(f"\nInitializing ContinuousBatcher (max_batch={args.max_batch_size})...")
    batcher = ContinuousBatcher(
        model=backend.model,
        tokenizer=backend.tokenizer,
        max_batch_size=args.max_batch_size,
        device="auto",
    )
    batcher.start()

    # 3. Concurrent batch
    print(f"\n[2/3] Concurrent batch ({args.concurrent} requests at once)...")
    r = bench_concurrent_batch(batcher, args.concurrent, args.max_tokens)
    results["concurrent_batch"] = r
    print(f"  => {r['tok_s']} tok/s | {r['req_s']} req/s | "
          f"{r['completed']}/{r['requests']} completed")

    # 4. Sustained load
    print(f"\n[3/3] Sustained load ({args.total} total, {args.concurrent} concurrent)...")
    r = bench_sustained_load(batcher, args.total, args.concurrent, args.max_tokens)
    results["sustained_load"] = r
    print(f"  => {r['tok_s']} tok/s | {r['req_s']} req/s | "
          f"{r['completed']}/{r['total_requests']} completed")

    batcher.stop()

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"{'Mode':<20} {'Tok/s':>8} {'Req/s':>8} {'Completed':>10} {'Time(s)':>8}")
    print("-" * 60)
    for key in ["sequential", "concurrent_batch", "sustained_load"]:
        if key in results:
            r = results[key]
            comp = r.get("completed", r.get("requests", "-"))
            print(f"{r['mode']:<20} {r['tok_s']:>8.1f} {r['req_s']:>8.2f} "
                  f"{comp!s:>10} {r['elapsed_s']:>8.1f}")

    # Speedup calculation
    seq_toks = results.get("sequential", {}).get("tok_s", 0)
    if seq_toks > 0:
        for key in ["concurrent_batch", "sustained_load"]:
            if key in results:
                speedup = results[key]["tok_s"] / seq_toks
                print(f"\n{key} vs sequential: {speedup:.2f}x throughput")

    batcher_stats = batcher.stats()
    print(f"\nBatcher stats: {json.dumps(batcher_stats, indent=2, default=str)}")

    print(f"\n{json.dumps(results, indent=2)}")

    pipe.shutdown()
    return results


if __name__ == "__main__":
    main()
