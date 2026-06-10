#!/usr/bin/env python3
"""V6 — VRAM Lending Pool under artificial VRAM pressure (HF backend, 2-GPU).

``bench_lending_turboquant_load.py`` showed that with both GPUs mostly free,
a 14B model + 800 generated tokens never needs to borrow VRAM
(``total_leases_created == 0``). Loading a bigger model is not a viable way
to force this path: Qwen3.6-35B-A3B currently runs via llama.cpp, and the
lending pool is gated off for that backend entirely
(``inference_pipeline.py:327`` — ``backend_type not in ('vllm', 'llamacpp')``).

Instead, this bench pre-allocates a "filler" CUDA tensor (dummy data, content
irrelevant) on one GPU *before* loading the model, to artificially shrink its
free VRAM. This simulates "another tenant already using this GPU" without
needing a different model or real concurrent load.

Usage:
    python benchmarks/bench_lending_vram_pressure.py --filler-gpu 1 --filler-mb 9000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a Python function that implements binary search:",
    "The future of renewable energy depends on",
    "In distributed computing, consistency and availability",
    "To optimize GPU memory usage in deep learning,",
]

OUT_JSON = Path("benchmarks/results/bench_lending_vram_pressure.json")
OUT_MD = Path("benchmarks/results/bench_lending_vram_pressure.md")


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
    lending_enabled: bool,
    filler_gpu: int,
    filler_mb: int,
    model: str,
    num_gpus: int,
    max_new: int,
    repeats: int,
    warmup: int,
    timeout: int,
) -> dict:
    print(f"\n=== Variant: {label} (LENDING={'1' if lending_enabled else '0'}, "
          f"filler={filler_mb}MB on gpu{filler_gpu}) ===")

    script = textwrap.dedent(f"""
        import os, sys, json, time, gc
        os.environ['VRM_VRAM_LENDING'] = {'"1"' if lending_enabled else '"0"'}
        os.environ['VRM_FORCE_MULTI_GPU'] = '1'
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
        FILLER_GPU = {filler_gpu}
        FILLER_MB = {filler_mb}

        result = {{
            'label': {label!r},
            'lending_enabled': {lending_enabled!r},
            'filler_gpu': FILLER_GPU,
            'filler_mb': FILLER_MB,
            'vram_pre_filler': _vram_per_gpu(),
        }}

        # --- Dummy "filler" tensor: occupy VRAM on one GPU before model load ---
        # Content is irrelevant (uninitialised CUDA memory) — only the byte
        # footprint matters, simulating another tenant's allocation.
        filler = None
        if FILLER_MB > 0:
            n_floats = (FILLER_MB * 1024 * 1024) // 4
            filler = torch.empty(n_floats, dtype=torch.float32, device=f'cuda:{{FILLER_GPU}}')
            torch.cuda.synchronize()
        result['vram_post_filler'] = _vram_per_gpu()

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

            result['vram_post_load'] = _vram_per_gpu()

            for _ in range(min(WARMUP, len(PROMPTS))):
                pipe.generate(PROMPTS[0], max_new_tokens=8)
            torch.cuda.synchronize()

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

        del filler
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
    ap.add_argument("--filler-gpu", type=int, default=1,
                    help="GPU index to pre-fill with dummy data (default: 1, the 16GB 5070 Ti)")
    ap.add_argument("--filler-mb", type=int, default=9000,
                    help="MB of dummy CUDA memory to allocate before model load")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print("[V6] VRAM Lending Pool under artificial VRAM pressure")
    print(f"Model: {args.model}, num_gpus={args.num_gpus}, "
          f"filler={args.filler_mb}MB on gpu{args.filler_gpu}")

    topo = _gpu_topology()
    results = {"gpu_topology": topo, "model": args.model, "variants": []}

    for label, lending in [("LENDING_ON", True), ("LENDING_OFF", False)]:
        r = _run_variant(
            label, lending, args.filler_gpu, args.filler_mb,
            args.model, args.num_gpus, args.max_new, args.repeats,
            args.warmup, args.timeout,
        )
        results["variants"].append(r)
        print(json.dumps(r, indent=2)[:2000])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    lines = [
        "# VRAM Lending Pool under artificial VRAM pressure",
        "",
        f"Model: `{args.model}`, GPUs: {args.num_gpus}, "
        f"filler={args.filler_mb}MB pre-allocated on gpu{args.filler_gpu} "
        f"before model load.",
        "",
        "| Variant | ok | load_s | tok/s | leases_created | leases_reclaimed | "
        "bytes_reclaimed | error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results["variants"]:
        if not r.get("ok"):
            lines.append(f"| {r['label']} | FAIL | - | - | - | - | - | "
                          f"{r.get('error', '')[:80]} |")
            continue
        sp = r.get("stats_post", {})
        lines.append(
            f"| {r['label']} | ok | {r.get('load_time_s')} | {r.get('tok_s')} | "
            f"{sp.get('total_leases_created', 0)} | {sp.get('total_leases_reclaimed', 0)} | "
            f"{sp.get('total_bytes_reclaimed', 0)} | |"
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
