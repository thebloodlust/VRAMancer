#!/usr/bin/env python3
"""Concurrent benchmark for the continuous batcher (manual run only)."""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _request(url, prompt, max_tokens, token):
    import urllib.request
    body = json.dumps({
        "model": "vramancer",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-API-Token"] = token
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="Explain quantum entanglement in one sentence.")
    p.add_argument("--token", default=os.environ.get("VRM_API_TOKEN", ""))
    p.add_argument("--out", default="bench_batcher_concurrent.json")
    args = p.parse_args()

    if os.environ.get("VRM_MINIMAL_TEST") == "1":
        print("Skipped under VRM_MINIMAL_TEST=1")
        return

    latencies = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_request, args.url, args.prompt, args.max_tokens, args.token)
                   for _ in range(args.total)]
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as exc:
                print(f"  request failed: {exc}")
    elapsed = time.perf_counter() - t_start

    latencies.sort()
    result = {
        "concurrency": args.concurrency,
        "total_requests": args.total,
        "successful": len(latencies),
        "wall_seconds": round(elapsed, 3),
        "throughput_rps": round(len(latencies) / elapsed, 2) if elapsed else 0,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p99": round(latencies[int(len(latencies) * 0.99)], 3) if latencies else None,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
