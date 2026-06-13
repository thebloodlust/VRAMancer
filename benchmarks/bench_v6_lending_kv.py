"""V6.D Phase 3+4 lending KV benchmark.

Compares decode latency with and without VRM_KV_LEND / VRM_KV_LEND_ATTENTION on a
model that overflows GPU0's KV pool and would normally evict to CPU DRAM.

The VRAMancer lending pool lends spare VRAM from GPU1 (RTX 3090) to GPU0
(RTX 5070 Ti) for KV cache overflow, avoiding slow PCIe DRAM reads.

Run manually on 2-GPU box (RTX 5070 Ti + RTX 3090):

  # Baseline (KV overflow goes to CPU DRAM)
  python benchmarks/bench_v6_lending_kv.py

  # Phase 3+4 (KV overflow goes to GPU1 lending pool via P2P)
  VRM_KV_LEND=1 VRM_KV_LEND_ATTENTION=1 python benchmarks/bench_v6_lending_kv.py

  # Compare both automatically
  python benchmarks/bench_v6_lending_kv.py --compare

Expected outcome when lending is active:
  - tok/s should be >= baseline (no regression)
  - GPU1 VRAM usage should increase (pages borrowed from lender)
  - GPU0 VRAM usage should stay lower (overflow offloaded to GPU1, not CPU)

Metrics captured per run:
  - tok/s (generation throughput)
  - Peak VRAM per GPU (via nvidia-smi / pynvml)
  - Whether lending leases were created (pool_stats.active_leases > 0)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_CFG = {
    "model": os.getenv("VRM_BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "max_new": 256,
    "num_gpus": int(os.getenv("VRM_BENCH_GPUS", "2")),
}
PROMPT = ("Once upon a time in a land far away, " * 50).strip()  # ~400 tokens

OUT_JSON = Path("benchmarks/results/bench_v6_lending_kv.json")


def _vram_per_gpu() -> dict:
    """Read VRAM usage via pynvml (insensitive to CUDA_VISIBLE_DEVICES)."""
    out: dict = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[f"gpu{i}"] = {
                "used_mb": m.used // (1024 * 1024),
                "total_mb": m.total // (1024 * 1024),
            }
    except Exception:
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                out[f"gpu{i}"] = {
                    "used_mb": (total - free) // (1024 * 1024),
                    "total_mb": total // (1024 * 1024),
                }
        except Exception:
            pass
    return out


def run_single(label: str, kv_lend: bool, kv_lend_attention: bool) -> dict:
    """Run one benchmark variant and return metrics."""
    print(f"\n{'='*60}")
    print(f"[bench_v6_lending_kv] Variant: {label}")
    print(f"  VRM_KV_LEND={'1' if kv_lend else '0'}")
    print(f"  VRM_KV_LEND_ATTENTION={'1' if kv_lend_attention else '0'}")
    print(f"  Model: {_CFG['model']}  MAX_NEW: {_CFG['max_new']} tokens")

    if kv_lend:
        os.environ["VRM_KV_LEND"] = "1"
    else:
        os.environ.pop("VRM_KV_LEND", None)

    if kv_lend_attention:
        os.environ["VRM_KV_LEND_ATTENTION"] = "1"
    else:
        os.environ.pop("VRM_KV_LEND_ATTENTION", None)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("VRM_FORCE_MULTI_GPU", "1")
    os.environ.pop("VRM_MINIMAL_TEST", None)

    result: dict = {
        "label": label,
        "kv_lend": kv_lend,
        "kv_lend_attention": kv_lend_attention,
        "model": _CFG["model"],
        "max_new": _CFG["max_new"],
        "num_gpus": _CFG["num_gpus"],
    }

    try:
        from core.inference_pipeline import get_pipeline, reset_pipeline

        reset_pipeline()
        pipe = get_pipeline(
            backend_name="huggingface",
            enable_metrics=False,
            enable_discovery=False,
        )

        vram_pre = _vram_per_gpu()
        t_load = time.perf_counter()
        pipe.load(_CFG["model"], num_gpus=_CFG["num_gpus"])
        result["load_time_s"] = round(time.perf_counter() - t_load, 2)
        vram_post_load = _vram_per_gpu()

        # Lending pool info
        pool = getattr(pipe, "lending_pool", None)
        result["pool_active"] = pool is not None
        if pool is not None:
            try:
                result["pool_stats_pre_gen"] = pool.stats()
            except Exception:
                pass

        # Warmup
        _ = pipe.generate(PROMPT[:200], max_new_tokens=16)

        # Benchmark
        vram_pre_gen = _vram_per_gpu()
        t0 = time.perf_counter()
        _ = pipe.generate(PROMPT, max_new_tokens=_CFG["max_new"])
        elapsed = time.perf_counter() - t0
        vram_post_gen = _vram_per_gpu()

        result["tok_s"] = round(_CFG["max_new"] / elapsed, 2) if elapsed > 0 else 0.0
        result["elapsed_s"] = round(elapsed, 2)
        result["vram_pre_load"] = vram_pre
        result["vram_post_load"] = vram_post_load
        result["vram_post_gen"] = vram_post_gen
        result["vram_delta_gen_mb"] = {
            k: vram_post_gen[k]["used_mb"] - vram_pre_gen[k]["used_mb"]
            for k in vram_post_gen if k in vram_pre_gen
        }

        if pool is not None:
            try:
                result["pool_stats_post_gen"] = pool.stats()
            except Exception:
                pass

        result["status"] = "ok"
        print(f"  => {result['tok_s']} tok/s  load={result['load_time_s']}s")
        print(f"  => VRAM delta: {result['vram_delta_gen_mb']}")

        pipe.shutdown()
        reset_pipeline()

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:400]
        print(f"  => ERROR: {e}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V6.D lending KV benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Run both baseline and lending variants sequentially")
    parser.add_argument("--model", default=_CFG["model"], help="HF model id")
    parser.add_argument("--max-new", type=int, default=_CFG["max_new"])
    args = parser.parse_args()

    _CFG["model"] = args.model
    _CFG["max_new"] = args.max_new

    kv_lend = os.getenv("VRM_KV_LEND") == "1"
    kv_lend_attn = os.getenv("VRM_KV_LEND_ATTENTION") == "1"

    if args.compare:
        runs = [
            run_single("baseline", kv_lend=False, kv_lend_attention=False),
            run_single("lending_kv", kv_lend=True, kv_lend_attention=True),
        ]
        print("\n" + "="*60)
        print("COMPARISON")
        for r in runs:
            status = r.get("status", "?")
            tps = r.get("tok_s", "—")
            pool = r.get("pool_active", "—")
            print(f"  {r['label']:15s}: {tps} tok/s  pool_active={pool}  status={status}")
        if len(runs) == 2 and all(r["status"] == "ok" for r in runs):
            base_tps = runs[0]["tok_s"]
            lend_tps = runs[1]["tok_s"]
            delta = lend_tps - base_tps
            pct = (delta / base_tps * 100) if base_tps else 0
            print(f"  Delta: {delta:+.2f} tok/s ({pct:+.1f} %)")
    else:
        label = "lending_kv" if (kv_lend or kv_lend_attn) else "baseline"
        runs = [run_single(label, kv_lend=kv_lend, kv_lend_attention=kv_lend_attn)]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "bench": "bench_v6_lending_kv",
        "phase": "V6.D Phase 3+4",
        "runs": runs,
    }, indent=2))
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()
