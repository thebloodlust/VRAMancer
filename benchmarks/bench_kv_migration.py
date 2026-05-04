#!/usr/bin/env python3
"""Benchmark: KV cache migration latency — P2P vs CPU-staged.

Simulates VRAMLendingPool preemption: migrating KV pages from GPU 1 → GPU 0
when GPU 1 reclaims its VRAM. Uses real page sizes from Qwen2.5-14B KV layout.

Page size formula (bfloat16):
  2 (K+V) × num_layers × num_kv_heads × tokens_per_page × head_dim × 2 bytes

Qwen2.5-14B: 2 × 48 × 8 × 16 × 128 × 2 = 3,145,728 bytes ≈ 3.0 MB/page

Usage:
    source .venv/bin/activate
    python benchmarks/bench_kv_migration.py
"""
import os, sys, time, json, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.transfer_manager import TransferManager

# ── KV page geometry (Qwen2.5-14B) ──────────────────────────────────────────
NUM_LAYERS       = 48
NUM_KV_HEADS     = 8
TOKENS_PER_PAGE  = 16
HEAD_DIM         = 128
DTYPE            = torch.bfloat16
BYTES_PER_ELT    = 2

PAGE_BYTES = 2 * NUM_LAYERS * NUM_KV_HEADS * TOKENS_PER_PAGE * HEAD_DIM * BYTES_PER_ELT
PAGE_MB    = PAGE_BYTES / 1024**2
PAGE_NUMEL = PAGE_BYTES // BYTES_PER_ELT

# Reclaim scenarios: (label, num_pages)
SCENARIOS = [
    ("10 pages  (idle GPU, small reclaim)", 10),
    ("50 pages  (moderate context)",        50),
    ("100 pages (4K-token request)",        100),
    ("300 pages (12K-token request)",       300),
    ("500 pages (20K-token request)",       500),
]
N_ITERS = 8          # repeats per scenario
SRC_GPU  = 1         # GPU 1 (5070 Ti) → lending GPU
DST_GPU  = 0         # GPU 0 (3090)    → borrower reclaims here


def alloc_kv_pages(n: int, gpu: int) -> torch.Tensor:
    """Allocate n KV pages as a contiguous tensor on given GPU."""
    return torch.randn(n * PAGE_NUMEL, dtype=DTYPE, device=f"cuda:{gpu}")


def _measure(fn, n_pages: int) -> dict:
    """Run fn() N_ITERS times and return timing stats."""
    total_bytes = n_pages * PAGE_BYTES
    times = []
    for _ in range(N_ITERS):
        torch.cuda.synchronize(SRC_GPU)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(DST_GPU)
        times.append(time.perf_counter() - t0)
    avg_ms  = sum(times) / len(times) * 1000
    p50_ms  = sorted(times)[len(times) // 2] * 1000
    p95_ms  = sorted(times)[int(len(times) * 0.95)] * 1000
    bw_gbps = (total_bytes / 1e9) / (avg_ms / 1000)
    return dict(avg_ms=avg_ms, p50_ms=p50_ms, p95_ms=p95_ms,
                bw_gbps=bw_gbps, total_mb=total_bytes / 1024**2)


def run_rust_p2p(n_pages: int, tm: TransferManager) -> dict:
    """Rust cuMemcpyPeerAsync (true P2P DMA via Proxmox PCIe P2P)."""
    pages = alloc_kv_pages(n_pages, SRC_GPU)
    # Warmup
    for _ in range(2):
        tm.send_activation(SRC_GPU, DST_GPU, pages)
    torch.cuda.synchronize(DST_GPU)
    result = _measure(lambda: tm.send_activation(SRC_GPU, DST_GPU, pages), n_pages)
    del pages; torch.cuda.empty_cache()
    return result


def run_cpu_staged(n_pages: int) -> dict:
    """Pure Python CPU-staged: GPU→pinned CPU→GPU (no Rust, no P2P)."""
    pages = alloc_kv_pages(n_pages, SRC_GPU)
    pinned = torch.empty(n_pages * PAGE_NUMEL, dtype=DTYPE,
                         pin_memory=True)
    # Warmup
    for _ in range(2):
        pinned.copy_(pages, non_blocking=False)
        dst = pinned.to(f"cuda:{DST_GPU}", non_blocking=False)
        torch.cuda.synchronize(DST_GPU)

    result = _measure(
        lambda: (
            pinned.copy_(pages, non_blocking=False),
            pinned.to(f"cuda:{DST_GPU}", non_blocking=False),
        ),
        n_pages,
    )
    del pages, pinned, dst
    torch.cuda.empty_cache()
    return result


def run_raw_to(n_pages: int) -> dict:
    """PyTorch .to() — what accelerate uses during inference layer handoff."""
    pages = alloc_kv_pages(n_pages, SRC_GPU)
    # Warmup
    for _ in range(2):
        _ = pages.to(f"cuda:{DST_GPU}", non_blocking=False)
    torch.cuda.synchronize(DST_GPU)
    result = _measure(
        lambda: pages.to(f"cuda:{DST_GPU}", non_blocking=False),
        n_pages,
    )
    del pages; torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print("  VRAMancer — KV Migration Latency: 3 methods compared")
    print(f"  Page: {PAGE_MB:.2f} MB  ({NUM_LAYERS}L × {NUM_KV_HEADS}kv × "
          f"{TOKENS_PER_PAGE}tok × {HEAD_DIM}dim × bf16)")
    print(f"  Direction: GPU {SRC_GPU} ({torch.cuda.get_device_name(SRC_GPU)}) "
          f"→ GPU {DST_GPU} ({torch.cuda.get_device_name(DST_GPU)})")
    print(f"  Iters: {N_ITERS}  |  can_device_access_peer: "
          f"{torch.cuda.can_device_access_peer(SRC_GPU, DST_GPU)}")
    print("=" * 70)

    tm = TransferManager(verbose=False)

    methods = [
        ("Rust cuMemcpyPeer (P2P DMA)",  "p2p_rust"),
        ("Python CPU-staged (pinned)",   "cpu_staged"),
        ("PyTorch .to() (accelerate)",   "raw_to"),
    ]

    all_res = {tag: {} for _, tag in methods}

    for label, tag in methods:
        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")
        print(f"  {'Scenario':42s}  {'avg':>7s}  {'p95':>7s}  {'BW':>9s}")
        print(f"  {'─'*42}  {'─'*7}  {'─'*7}  {'─'*9}")
        for slabel, n_pages in SCENARIOS:
            if tag == "p2p_rust":
                r = run_rust_p2p(n_pages, tm)
            elif tag == "cpu_staged":
                r = run_cpu_staged(n_pages)
            else:
                r = run_raw_to(n_pages)
            all_res[tag][f"{n_pages}pages"] = r
            print(f"  {slabel:42s}  {r['avg_ms']:>5.1f}ms  "
                  f"{r['p95_ms']:>5.1f}ms  {r['bw_gbps']:>7.2f} GB/s")

    # Delta table
    print(f"\n{'═'*70}")
    print("  SPEEDUP vs CPU-staged (Python pinned)")
    print(f"{'═'*70}")
    print(f"  {'Scenario':28s}  {'CPU-staged':>10s}  {'Rust P2P':>9s}  "
          f"{'PyTorch .to()':>13s}  {'P2P gain':>9s}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*9}  {'─'*13}  {'─'*9}")

    for slabel, n_pages in SCENARIOS:
        tag_key = f"{n_pages}pages"
        cpu = all_res["cpu_staged"][tag_key]
        p2p = all_res["p2p_rust"][tag_key]
        raw = all_res["raw_to"][tag_key]
        gain = (cpu["avg_ms"] - p2p["avg_ms"]) / cpu["avg_ms"] * 100
        short = slabel.split("(")[0].strip()
        print(f"  {short:28s}  {cpu['avg_ms']:>8.1f}ms  "
              f"{p2p['avg_ms']:>7.1f}ms  {raw['avg_ms']:>11.1f}ms  "
              f"  {gain:>+5.1f}%")

    # Save
    out = {
        "page_mb": round(PAGE_MB, 3),
        "geometry": {"layers": NUM_LAYERS, "kv_heads": NUM_KV_HEADS,
                     "tokens_per_page": TOKENS_PER_PAGE, "head_dim": HEAD_DIM},
        "gpu_src": torch.cuda.get_device_name(SRC_GPU),
        "gpu_dst": torch.cuda.get_device_name(DST_GPU),
        **all_res,
    }
    with open("bench_kv_migration.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: bench_kv_migration.json")


if __name__ == "__main__":
    main()
