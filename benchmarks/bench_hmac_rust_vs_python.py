"""Benchmark HMAC-SHA256 verify: Python stdlib vs Rust PyO3 (vramancer_rust).

Measures throughput on realistic AITP packet sizes (256 B headers up to
64 KB tensor frames). Outputs a simple table; non-zero exit code if the
Rust path is unavailable so CI can skip cleanly.

Usage::

    python benchmarks/bench_hmac_rust_vs_python.py [--iterations 100000]

Honest claim policy: if the speedup measured on this hardware is below
3×, log a warning. The README claims "100× faster" which is unrealistic
for HMAC (Python's hashlib is C-backed) — we measure and tell the truth.
"""
from __future__ import annotations

import argparse
import hmac
import os
import statistics
import sys
import time
from hashlib import sha256

import secrets

try:
    from core.rust_bridge import rust, has_rust  # type: ignore
except Exception:
    rust = None
    def has_rust() -> bool:  # type: ignore[misc]
        return False


def _bench_python(secret: bytes, payload: bytes, sig: bytes, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        expected = hmac.new(secret, payload, sha256).digest()
        hmac.compare_digest(expected, sig)
    return time.perf_counter() - t0


def _bench_rust(secret: bytes, payload: bytes, sig: bytes, n: int) -> float:
    fn = rust.verify_hmac_fast  # type: ignore[union-attr]
    t0 = time.perf_counter()
    for _ in range(n):
        fn(secret, payload, sig)
    return time.perf_counter() - t0


def _gen_signed(secret: bytes, payload: bytes) -> bytes:
    return hmac.new(secret, payload, sha256).digest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=50_000,
                    help="iterations per payload size (default: 50k)")
    ap.add_argument("--sizes", default="256,1024,4096,16384,65536",
                    help="payload sizes (bytes) CSV")
    args = ap.parse_args()

    if not has_rust():
        print("[skip] vramancer_rust not installed in this venv. "
              "Build with: cd rust_core && cargo build --release --features cuda "
              "&& maturin develop --features cuda")
        return 1

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    secret = secrets.token_bytes(32)

    print(f"\nHMAC-SHA256 verify benchmark — {args.iterations:,} iters per size")
    print(f"  Python: hmac + hashlib.sha256 (C-backed)")
    print(f"  Rust:   vramancer_rust.verify_hmac_fast (sha2 crate)\n")
    print(f"  {'size':>8}  {'py µs/op':>10}  {'rust µs/op':>11}  {'speedup':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}")

    speedups: list[float] = []
    for sz in sizes:
        payload = os.urandom(sz)
        sig = _gen_signed(secret, payload)
        # Warmup
        _bench_python(secret, payload, sig, 1000)
        _bench_rust(secret, payload, sig, 1000)
        py_t = _bench_python(secret, payload, sig, args.iterations)
        rs_t = _bench_rust(secret, payload, sig, args.iterations)
        py_us = py_t * 1e6 / args.iterations
        rs_us = rs_t * 1e6 / args.iterations
        speedup = py_t / rs_t if rs_t > 0 else float("inf")
        speedups.append(speedup)
        print(f"  {sz:>8}  {py_us:>10.2f}  {rs_us:>11.2f}  {speedup:>7.2f}x")

    median = statistics.median(speedups)
    print(f"\n  Median speedup: {median:.2f}x")
    if median < 3.0:
        print(f"  [WARNING] Sub-3x speedup measured. The 'Rust HMAC 100x faster'")
        print(f"  claim in docs is unrealistic — Python's hashlib is C-backed.")
        print(f"  Real win comes from GIL-free batch verify (verify_hmac_batch).")
    elif median < 10.0:
        print(f"  [OK] Modest speedup, GIL-release matters more for batch paths.")
    else:
        print(f"  [GOOD] Significant speedup, keep using Rust path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
