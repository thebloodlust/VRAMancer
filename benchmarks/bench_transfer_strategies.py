#!/usr/bin/env python3
"""Benchmark: GPU-to-GPU transfer strategies (T2.3 — ReBAR validation).

Compares the transport strategies TransferManager can fall back through:
  - Strategy 1   : direct CUDA P2P (cudaMemcpyPeer), if can_device_access_peer
  - Strategy 1.7 : ReBAR full-window (BAR-optimal chunks, experimental.cross_vendor_bridge.ReBarTransport)
  - Strategy 2   : CPU-pipelined double-buffer (experimental.cross_vendor_bridge.PipelinedTransport, default chunk)
  - Strategy 4   : plain CPU-staged (.cpu().to(device), no pinning tricks)

Runs in "degraded mode" if ReBAR / cross-vendor hardware isn't present:
strategies that aren't applicable are skipped and reported as such, the
script still completes and prints whatever strategies ARE measurable on
the current machine.

Usage:
    source .venv/bin/activate
    VRM_EXPERIMENTAL=1 python benchmarks/bench_transfer_strategies.py
"""
import os
import sys
import time
import json

os.environ.setdefault("VRM_EXPERIMENTAL", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experimental.cross_vendor_bridge import ReBarTransport, PipelinedTransport

SRC_GPU = 1
DST_GPU = 0
SIZES_MB = [1, 4, 16, 64, 256, 1024]
N_ITERS = 5


def make_tensor(size_mb: int) -> torch.Tensor:
    numel = (size_mb * 1024 * 1024) // 2  # bf16 = 2 bytes
    return torch.randn(numel, dtype=torch.bfloat16, device=f"cuda:{SRC_GPU}")


def bench_cpu_staged(tensor: torch.Tensor) -> float:
    torch.cuda.synchronize(SRC_GPU)
    t0 = time.perf_counter()
    out = tensor.to("cpu").to(f"cuda:{DST_GPU}")
    torch.cuda.synchronize(DST_GPU)
    del out
    return time.perf_counter() - t0


def bench_transport(transport, tensor: torch.Tensor) -> float:
    torch.cuda.synchronize(SRC_GPU)
    t0 = time.perf_counter()
    out, _ = transport.transfer(SRC_GPU, DST_GPU, tensor)
    torch.cuda.synchronize(DST_GPU)
    del out
    return time.perf_counter() - t0


def gbps(size_mb: int, seconds: float) -> float:
    return (size_mb * 1024 * 1024 * 8) / (seconds * 1e9)


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Needs >=2 CUDA GPUs — skipping (degraded mode, no GPU).")
        return

    can_p2p = torch.cuda.can_device_access_peer(SRC_GPU, DST_GPU)
    rebar = ReBarTransport()
    pipeline = PipelinedTransport()

    print("=" * 70)
    print(f"  Transfer strategies: GPU {SRC_GPU} -> GPU {DST_GPU}")
    print(f"  can_device_access_peer: {can_p2p}")
    print(f"  ReBAR available (BAR>4GB): {rebar.available}  full_window={rebar.full_window}")
    if rebar.available:
        print(f"  ReBAR info: {rebar.info()}")
    print("=" * 70)

    results = []
    for size_mb in SIZES_MB:
        row = {"size_mb": size_mb}

        # Strategy 4: plain CPU-staged
        times = []
        for _ in range(N_ITERS):
            t = make_tensor(size_mb)
            times.append(bench_cpu_staged(t))
            del t
        avg = sum(times) / len(times)
        row["cpu_staged_ms"] = avg * 1000
        row["cpu_staged_gbps"] = gbps(size_mb, avg)

        # Strategy 2: PipelinedTransport (default chunk, no ReBAR sizing)
        times = []
        for _ in range(N_ITERS):
            t = make_tensor(size_mb)
            times.append(bench_transport(pipeline, t))
            del t
        avg = sum(times) / len(times)
        row["pipelined_ms"] = avg * 1000
        row["pipelined_gbps"] = gbps(size_mb, avg)

        # Strategy 1.7: ReBAR full-window (only if BAR > 4GB on either GPU)
        if rebar.available:
            times = []
            for _ in range(N_ITERS):
                t = make_tensor(size_mb)
                times.append(bench_transport(rebar, t))
                del t
            avg = sum(times) / len(times)
            row["rebar_ms"] = avg * 1000
            row["rebar_gbps"] = gbps(size_mb, avg)
        else:
            row["rebar_ms"] = None
            row["rebar_gbps"] = None

        results.append(row)
        print(
            f"{size_mb:6d} MB  |  CPU-staged: {row['cpu_staged_gbps']:6.1f} Gbps  |  "
            f"Pipelined: {row['pipelined_gbps']:6.1f} Gbps  |  "
            f"ReBAR: {row['rebar_gbps']:.1f} Gbps" if row["rebar_gbps"] else
            f"{size_mb:6d} MB  |  CPU-staged: {row['cpu_staged_gbps']:6.1f} Gbps  |  "
            f"Pipelined: {row['pipelined_gbps']:6.1f} Gbps  |  ReBAR: n/a (BAR <= 4GB)"
        )

    out_path = os.path.join(os.path.dirname(__file__), "results", "bench_transfer_strategies.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "can_device_access_peer": can_p2p,
            "rebar_available": rebar.available,
            "rebar_full_window": rebar.full_window,
            "rebar_info": rebar.info(),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
