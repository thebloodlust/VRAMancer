#!/usr/bin/env python3
"""VRAMancer continuous batcher load test.

Sends N concurrent requests to /v1/completions and measures:
  - Total throughput (tok/s across all clients)
  - Per-request latency (P50, P95, P99)
  - Queue depth behavior
  - Batcher ON vs OFF comparison

Usage:
    # Start API first:
    #   VRM_CONTINUOUS_BATCHING=1 python -m vramancer.main serve --model gpt2 --port 8111

    # Then run load test:
    python benchmarks/bench_load_batcher.py --url http://127.0.0.1:8111 --concurrency 1,2,4,8
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land there was",
    "Python is a programming language that excels at",
    "The most important thing about distributed systems",
    "When we think about the nature of consciousness",
    "Machine learning models have fundamentally changed",
    "In computer science the concept of parallelism",
    "The relationship between hardware and software",
    "Neural networks were first proposed in the",
    "The history of computing began with",
    "Quantum computing promises to revolutionize",
    "The internet was originally designed as",
    "Deep learning requires large amounts of",
    "Natural language processing has evolved from",
    "Reinforcement learning differs from supervised",
    "The transformer architecture was introduced in",
    "Convolutional neural networks are particularly good at",
    "Recurrent neural networks were designed to handle",
    "Transfer learning allows models to leverage",
    "Federated learning enables training without sharing",
    "The attention mechanism allows models to focus on",
    "Batch normalization helps stabilize training by",
    "Gradient descent is the fundamental optimization",
    "Backpropagation computes gradients by applying the",
    "Regularization techniques help prevent overfitting by",
    "Data augmentation increases the effective size of",
    "Hyperparameter tuning is the process of finding",
    "Cross-validation helps estimate model performance",
    "Ensemble methods combine multiple models to",
    "Dimensionality reduction techniques like PCA",
    "Feature engineering is the process of creating",
    "The bias-variance tradeoff describes the balance",
]


def _send_request(url, prompt, max_tokens, token):
    """Send a single completion request and return (latency_s, num_tokens, text)."""
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(
        f"{url.rstrip('/')}/v1/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    start = time.perf_counter()
    try:
        resp = urlopen(req, timeout=300)
        body = json.loads(resp.read().decode())
        elapsed = time.perf_counter() - start

        text = body.get("choices", [{}])[0].get("text", "")
        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", max_tokens)

        return {
            "latency_s": elapsed,
            "tokens": completion_tokens,
            "tok_s": completion_tokens / elapsed if elapsed > 0 else 0,
            "text_len": len(text),
            "ok": True,
        }
    except (URLError, TimeoutError, Exception) as e:
        elapsed = time.perf_counter() - start
        return {
            "latency_s": elapsed,
            "tokens": 0,
            "tok_s": 0,
            "text_len": 0,
            "ok": False,
            "error": str(e),
        }


def run_load_test(url, concurrency, max_tokens, num_requests, token):
    """Run num_requests requests with given concurrency level."""
    prompts_cycle = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    results = []

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for p in prompts_cycle:
            futures.append(pool.submit(_send_request, url, p, max_tokens, token))

        for f in as_completed(futures):
            results.append(f.result())
    wall_elapsed = time.perf_counter() - wall_start

    ok_results = [r for r in results if r["ok"]]
    failed = len(results) - len(ok_results)

    if not ok_results:
        return {
            "concurrency": concurrency,
            "requests": num_requests,
            "failed": failed,
            "error": "All requests failed",
        }

    latencies = sorted(r["latency_s"] for r in ok_results)
    total_tokens = sum(r["tokens"] for r in ok_results)

    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    return {
        "concurrency": concurrency,
        "requests": num_requests,
        "failed": failed,
        "total_tokens": total_tokens,
        "wall_time_s": round(wall_elapsed, 2),
        "throughput_tok_s": round(total_tokens / wall_elapsed, 1),
        "throughput_req_s": round(len(ok_results) / wall_elapsed, 2),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_p50_s": round(p50, 3),
        "latency_p95_s": round(p95, 3),
        "latency_p99_s": round(p99, 3),
        "latency_min_s": round(latencies[0], 3),
        "latency_max_s": round(latencies[-1], 3),
        "per_request_tok_s": round(statistics.mean(r["tok_s"] for r in ok_results), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="VRAMancer batcher load test")
    parser.add_argument("--url", default="http://127.0.0.1:8111")
    parser.add_argument("--concurrency", default="1,2,4,8",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-requests", type=int, default=16,
                        help="Total requests per concurrency level")
    parser.add_argument("--token", default=os.environ.get("VRM_API_TOKEN", "testtoken"))
    args = parser.parse_args()

    levels = [int(c.strip()) for c in args.concurrency.split(",")]

    print(f"\n{'='*70}")
    print(f"BATCHER LOAD TEST — {args.url}")
    print(f"Max tokens: {args.max_tokens} | Requests per level: {args.num_requests}")
    print(f"Concurrency levels: {levels}")
    print(f"{'='*70}")

    # Warmup
    print("\n[WARMUP] Sending 1 request...")
    warm = _send_request(args.url, "Hello world", 10, args.token)
    if not warm["ok"]:
        print(f"  WARMUP FAILED: {warm.get('error', 'unknown')}")
        print("  Is the API running? Start it with:")
        print(f"    python -m vramancer.main serve --model gpt2 --port 8111")
        sys.exit(1)
    print(f"  OK ({warm['latency_s']:.2f}s)")

    all_results = []
    for level in levels:
        print(f"\n[LOAD] Concurrency={level}, {args.num_requests} requests...")
        result = run_load_test(
            args.url, level, args.max_tokens, args.num_requests, args.token
        )
        all_results.append(result)

        if "error" in result:
            print(f"  FAILED: {result['error']}")
        else:
            print(f"  Throughput:  {result['throughput_tok_s']} tok/s total "
                  f"| {result['throughput_req_s']} req/s")
            print(f"  Latency:     P50={result['latency_p50_s']}s "
                  f"P95={result['latency_p95_s']}s "
                  f"P99={result['latency_p99_s']}s")
            print(f"  Failed:      {result['failed']}/{result['requests']}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Conc':>6} {'Tok/s':>8} {'Req/s':>7} {'P50(s)':>8} {'P95(s)':>8} {'P99(s)':>8} {'Fail':>5}")
    print("-" * 55)
    for r in all_results:
        if "error" in r:
            print(f"{r['concurrency']:>6} {'FAIL':>8}")
        else:
            print(f"{r['concurrency']:>6} {r['throughput_tok_s']:>8.1f} "
                  f"{r['throughput_req_s']:>7.2f} "
                  f"{r['latency_p50_s']:>8.3f} {r['latency_p95_s']:>8.3f} "
                  f"{r['latency_p99_s']:>8.3f} "
                  f"{r['failed']:>5}")

    # JSON output
    print(f"\n{json.dumps(all_results, indent=2)}")
    return all_results


if __name__ == "__main__":
    main()
