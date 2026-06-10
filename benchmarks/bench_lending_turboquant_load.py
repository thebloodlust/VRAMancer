#!/usr/bin/env python3
"""V6 — VRAM Lending Pool under sustained KV-cache pressure, with/without
TurboQuant KV compression (HF backend, 2-GPU).

Companion to ``bench_lending_hetero_real.py``. That bench proved the lending
pool prevents an OOM at model-load time, but in a short run no lease was
actually triggered (``total_leases_created == 0``). This bench instead drives
many longer generations back-to-back to grow the KV cache and check whether:

  - a lease is actually created/reclaimed during sustained inference
    (``pool._stats['total_leases_created']``), and
  - enabling ``VRM_KV_COMPRESSION=turboquant`` (PolarQuant+QJL KV cache,
    ~4.6x smaller) changes lease behaviour or throughput.

Each variant runs in a fresh subprocess (torch CUDA state + lending pool
singleton must not leak across runs).

Usage:
    python benchmarks/bench_lending_turboquant_load.py
    python benchmarks/bench_lending_turboquant_load.py --model Qwen/Qwen2.5-14B-Instruct \\
        --max-new 96 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a Python function that implements binary search:",
    "The future of renewable energy depends on",
    "In distributed computing, consistency and availability",
    "To optimize GPU memory usage in deep learning,",
]

OUT_JSON = Path("benchmarks/results/bench_lending_turboquant_load.json")
OUT_MD = Path("benchmarks/results/bench_lending_turboquant_load.md")


def _gpu_topology() -> list[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({"index": i, "name": name, "total_mb": mem.total // (1024 * 1024)})
        return out
    except Exception:
        return []


def _run_variant(
    label: str,
    kv_compression: str,
    model: str,
    num_gpus: int,
    max_new: int,
    repeats: int,
    warmup: int,
    timeout: int,
) -> dict:
    print(f"\n=== Variant: {label} (VRM_KV_COMPRESSION={kv_compression!r}) ===")

    script = textwrap.dedent(f"""
        import os, sys, json, time, gc
        os.environ['VRM_VRAM_LENDING'] = '1'
        os.environ['VRM_FORCE_MULTI_GPU'] = '1'
        kv_comp = {kv_compression!r}
        if kv_comp:
            os.environ['VRM_KV_COMPRESSION'] = kv_comp
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ.pop('VRM_MINIMAL_TEST', None)
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        import torch
        try:
            import pynvml
            pynvml.nvmlInit()
        except Exception:
            pynvml = None

        def _vram_per_gpu():
            out = {{}}
            if pynvml is not None:
                try:
                    for i in range(pynvml.nvmlDeviceGetCount()):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        out[f'gpu{{i}}'] = {{
                            'used_mb': m.used // (1024 * 1024),
                            'total_mb': m.total // (1024 * 1024),
                        }}
                    return out
                except Exception:
                    pass
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                out[f'gpu{{i}}'] = {{
                    'used_mb': (total - free) // (1024 * 1024),
                    'total_mb': total // (1024 * 1024),
                }}
            return out

        PROMPTS = {PROMPTS!r}
        MAX_NEW = {max_new}
        REPEATS = {repeats}
        WARMUP = {warmup}
        MODEL = {model!r}
        NUM_GPUS = {num_gpus}

        result = {{
            'label': {label!r},
            'kv_compression': kv_comp or 'none',
            'vram_pre_load': _vram_per_gpu(),
        }}

        try:
            from core.inference_pipeline import InferencePipeline, reset_pipeline
            reset_pipeline()

            pipe = InferencePipeline(
                backend_name='huggingface',
                enable_metrics=False,
                enable_discovery=False,
                verbose=False,
            )
            t_load_start = time.perf_counter()
            pipe.load(MODEL, num_gpus=NUM_GPUS)
            result['load_time_s'] = round(time.perf_counter() - t_load_start, 2)

            pool = getattr(pipe, 'lending_pool', None)
            result['pool_active'] = pool is not None

            if pool is not None and hasattr(pool, '_budgets'):
                result['pool_registered_gpus'] = [
                    {{
                        'gpu_id': b.gpu_id,
                        'device_name': getattr(b, 'device_name', ''),
                        'lendable_bytes': int(getattr(b, 'lendable_bytes', 0)),
                    }}
                    for b in pool._budgets.values()
                ]
                result['stats_pre'] = dict(pool._stats)

            result['turboquant_active'] = pipe._turboquant_cache_factory is not None
            result['vram_post_load'] = _vram_per_gpu()

            # Warmup
            for _ in range(min(WARMUP, len(PROMPTS))):
                pipe.generate(PROMPTS[0], max_new_tokens=8)
            torch.cuda.synchronize()

            # Sustained generation: cycle through prompts REPEATS times
            total_gen = 0
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                for p in PROMPTS:
                    _ = pipe.generate(p, max_new_tokens=MAX_NEW)
                    total_gen += MAX_NEW
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            result['tok_s'] = round(total_gen / elapsed, 2) if elapsed > 0 else 0.0
            result['total_tokens'] = total_gen
            result['elapsed_s'] = round(elapsed, 2)
            result['vram_post_gen'] = _vram_per_gpu()

            if pool is not None:
                result['stats_post'] = dict(pool._stats)
                if hasattr(pool, '_budgets'):
                    result['pool_lendable_post'] = [
                        {{'gpu_id': b.gpu_id, 'lendable_bytes': int(getattr(b, 'lendable_bytes', 0))}}
                        for b in pool._budgets.values()
                    ]

            result['ok'] = True

        except Exception as e:
            import traceback
            result['ok'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()

        print("RESULT_JSON:" + json.dumps(result))
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )

    out_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            out_line = line[len("RESULT_JSON:"):]
    if out_line is None:
        return {
            "label": label, "ok": False,
            "error": "no RESULT_JSON in subprocess output",
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
        }
    return json.loads(out_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=2,
                    help="how many times to cycle through PROMPTS")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print("[V6] VRAM Lending Pool under load — TurboQuant KV-cache A/B")
    print(f"Model: {args.model}, num_gpus={args.num_gpus}, "
          f"max_new={args.max_new}, repeats={args.repeats}")

    topo = _gpu_topology()
    results = {"gpu_topology": topo, "model": args.model, "variants": []}

    for label, kv_comp in [("baseline", ""), ("turboquant", "turboquant")]:
        r = _run_variant(
            label, kv_comp, args.model, args.num_gpus,
            args.max_new, args.repeats, args.warmup, args.timeout,
        )
        results["variants"].append(r)
        print(json.dumps(r, indent=2)[:2000])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown summary
    lines = [
        "# VRAM Lending Pool under load — TurboQuant KV-cache A/B",
        "",
        f"Model: `{args.model}`, GPUs: {args.num_gpus}, "
        f"max_new={args.max_new}, repeats={args.repeats} "
        f"({args.repeats * len(PROMPTS)} generations per variant)",
        "",
        "| Variant | ok | load_s | tok/s | leases_created | leases_reclaimed | "
        "bytes_reclaimed | turboquant_active |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results["variants"]:
        if not r.get("ok"):
            lines.append(f"| {r['label']} | FAIL | - | - | - | - | - | - |")
            continue
        sp = r.get("stats_post", {})
        lines.append(
            f"| {r['label']} | ok | {r.get('load_time_s')} | {r.get('tok_s')} | "
            f"{sp.get('total_leases_created', 0)} | {sp.get('total_leases_reclaimed', 0)} | "
            f"{sp.get('total_bytes_reclaimed', 0)} | {r.get('turboquant_active')} |"
        )
    lines.append("")
    for r in results["variants"]:
        lines.append(f"## {r['label']}")
        lines.append("```json")
        lines.append(json.dumps(r, indent=2))
        lines.append("```")
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
