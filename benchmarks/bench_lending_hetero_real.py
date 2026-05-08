#!/usr/bin/env python3
"""V5 P13bis — VRAM Lending Pool real-load bench (HF backend, 2-GPU).

Companion to ``bench_deepseek_engram.py``. The DeepSeek bench shows that the
lending pool *can* reserve VRAM on the 3090 and validates a P2P data plane,
but vLLM's spawned worker doesn't actually use the lending buffer for its
weights (cpu_offload_gb keeps them in pinned DRAM accessed via UVA).

This bench runs a model through VRAMancer's HuggingFace backend instead.
The HF path is the one ``InferencePipeline._init_lending_pool()`` is wired
to (see ``inference_pipeline.py:327`` — the lending pool init is gated on
``num_gpus > 1`` and ``backend_type not in ('vllm', 'llamacpp')``). With
Qwen2.5-14B BF16 on a 3090 (24 GB) + 5070 Ti (16 GB), the lending pool is
exercised end-to-end during inference, and ReBAR + Rust P2P provide the
data plane (see TransferManager strategies 1.5 and 1.7).

A/B comparison:
  - ``VRM_VRAM_LENDING=1`` (default): pool active, cooperative GPU memory.
  - ``VRM_VRAM_LENDING=0``: pool disabled, baseline behaviour.

Each variant runs in a fresh subprocess so torch CUDA state never leaks.
Outputs ``benchmarks/results/bench_lending_hetero_real.{json,md}``.

Usage:
    python benchmarks/bench_lending_hetero_real.py
    python benchmarks/bench_lending_hetero_real.py --model Qwen/Qwen2.5-14B-Instruct
    python benchmarks/bench_lending_hetero_real.py --max-new 32 --warmup 5
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
DEFAULT_MAX_NEW = 64
DEFAULT_WARMUP = 10

OUT_JSON = Path("benchmarks/results/bench_lending_hetero_real.json")
OUT_MD = Path("benchmarks/results/bench_lending_hetero_real.md")


def _gpu_topology() -> list[dict]:
    """Read GPU topology via NVML so it works regardless of CUDA_VISIBLE_DEVICES."""
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
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            out.append({
                "index": i,
                "name": name,
                "sm": f"{major}.{minor}",
                "total_mb": mem.total // (1024 * 1024),
                "free_mb": mem.free // (1024 * 1024),
                "used_mb": mem.used // (1024 * 1024),
            })
        return out
    except Exception:
        return []


def _run_variant(
    label: str,
    lending_enabled: bool,
    model: str,
    num_gpus: int,
    max_new: int,
    warmup: int,
    timeout: int,
) -> dict:
    """Run one A/B variant in an isolated subprocess.

    Subprocess isolation is mandatory: torch.cuda caches device state at first
    init, and the lending pool registers a singleton inside ``core.vram_lending``.
    Running both variants in the same Python process would leak the singleton
    across A and B.
    """
    print(f"\n=== Variant: {label} (VRM_VRAM_LENDING={'1' if lending_enabled else '0'}) ===")

    script = textwrap.dedent(f"""
        import os, sys, json, time, gc
        os.environ['VRM_VRAM_LENDING'] = {'"1"' if lending_enabled else '"0"'}
        # InferencePipeline.load() auto-reduces num_gpus to 1 when the model
        # fits on a single GPU (cross-GPU transfer overhead). For a lending
        # pool A/B bench we *want* the multi-GPU path even for small models —
        # the pool only initialises when num_gpus > 1 (see inference_pipeline:327).
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
        WARMUP = {warmup}
        MODEL = {model!r}
        NUM_GPUS = {num_gpus}

        result = {{
            'label': {label!r},
            'lending_enabled_env': {lending_enabled!r},
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

            # Whether the lending pool actually attached (the gate at
            # inference_pipeline.py:327 may decide not to init it for several
            # reasons — single GPU, vLLM/llamacpp, or VRM_VRAM_LENDING=0).
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

            result['vram_post_load'] = _vram_per_gpu()

            # Warmup — discard timing
            for _ in range(min(WARMUP, len(PROMPTS))):
                pipe.generate(PROMPTS[0], max_new_tokens=8)
            torch.cuda.synchronize()

            # Benchmark
            total_gen = 0
            t0 = time.perf_counter()
            for p in PROMPTS:
                _ = pipe.generate(p, max_new_tokens=MAX_NEW)
                total_gen += MAX_NEW
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            result['tok_s'] = round(total_gen / elapsed, 2) if elapsed > 0 else 0.0
            result['total_tokens'] = total_gen
            result['elapsed_s'] = round(elapsed, 2)
            result['vram_post_bench'] = _vram_per_gpu()

            # Per-GPU peak usage delta vs pre-load — proxy for "did lending move data"
            result['vram_delta_mb'] = {{
                k: result['vram_post_bench'][k]['used_mb'] - result['vram_pre_load'][k]['used_mb']
                for k in result['vram_post_bench']
                if k in result['vram_pre_load']
            }}

            # Pool stats (lease counts, evictions, etc.) when available
            if pool is not None and hasattr(pool, 'stats'):
                try:
                    result['pool_stats'] = pool.stats()
                except Exception as e:
                    result['pool_stats_error'] = str(e)[:200]

            try:
                pipe.shutdown()
            except Exception:
                pass
            del pipe
            reset_pipeline()
            torch.cuda.empty_cache()
            gc.collect()
            result['status'] = 'ok'
        except torch.cuda.OutOfMemoryError as e:
            result['status'] = 'OOM'
            result['error'] = str(e)[:300]
            result['vram_post_oom'] = _vram_per_gpu()
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)[:300]

        print('===VRM_RESULT_BEGIN===')
        print(json.dumps(result, indent=2))
        print('===VRM_RESULT_END===')
    """)

    t0 = time.perf_counter()
    try:
        cp = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "label": label, "status": "timeout",
            "elapsed_subprocess_s": round(time.perf_counter() - t0, 1),
        }

    sub_elapsed = round(time.perf_counter() - t0, 1)

    stdout = cp.stdout
    if "===VRM_RESULT_BEGIN===" in stdout and "===VRM_RESULT_END===" in stdout:
        block = stdout.split("===VRM_RESULT_BEGIN===", 1)[1].split("===VRM_RESULT_END===", 1)[0].strip()
        try:
            parsed = json.loads(block)
            parsed["elapsed_subprocess_s"] = sub_elapsed
            parsed["returncode"] = cp.returncode
            return parsed
        except json.JSONDecodeError as e:
            return {
                "label": label, "status": "parse_error",
                "error": f"{e}",
                "stdout_tail": stdout[-1500:],
                "stderr_tail": cp.stderr[-1500:],
                "elapsed_subprocess_s": sub_elapsed,
            }

    return {
        "label": label, "status": "no_marker",
        "stdout_tail": stdout[-1500:],
        "stderr_tail": cp.stderr[-1500:],
        "returncode": cp.returncode,
        "elapsed_subprocess_s": sub_elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VRAM lending A/B bench")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model id (default {DEFAULT_MODEL})")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=DEFAULT_MAX_NEW)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--timeout", type=int, default=900,
                        help="Per-variant subprocess timeout (s)")
    parser.add_argument("--lending-only", action="store_true",
                        help="Run only the LENDING_ON variant (skip baseline)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the LENDING_OFF variant")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to output filenames (e.g. '_llama3_8b')")
    args = parser.parse_args()

    out_json = OUT_JSON.with_name(OUT_JSON.stem + args.out_suffix + OUT_JSON.suffix)
    out_md = OUT_MD.with_name(OUT_MD.stem + args.out_suffix + OUT_MD.suffix)

    print("=" * 70)
    print("[V5 P13bis] VRAM Lending Pool — real-load A/B (HF backend)")
    print("=" * 70)
    print(f"  Model       : {args.model}")
    print(f"  num_gpus    : {args.num_gpus}")
    print(f"  prompts     : {len(PROMPTS)} × {args.max_new} new tokens")
    print(f"  warmup      : {args.warmup} iterations")

    topo = _gpu_topology()
    if topo:
        print(f"  Topology    :")
        for g in topo:
            print(f"    GPU{g['index']}: {g['name']} (SM {g['sm']}, "
                  f"{g['total_mb']} MB, {g['free_mb']} MB free)")
    print()

    runs: list[dict] = []
    if not args.lending_only:
        runs.append(_run_variant(
            "LENDING_OFF", lending_enabled=False, model=args.model,
            num_gpus=args.num_gpus, max_new=args.max_new, warmup=args.warmup,
            timeout=args.timeout,
        ))
    if not args.baseline_only:
        runs.append(_run_variant(
            "LENDING_ON", lending_enabled=True, model=args.model,
            num_gpus=args.num_gpus, max_new=args.max_new, warmup=args.warmup,
            timeout=args.timeout,
        ))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "bench": "bench_lending_hetero_real",
        "phase": "V5 P13bis (companion to bench_deepseek_engram)",
        "model": args.model,
        "num_gpus": args.num_gpus,
        "max_new": args.max_new,
        "warmup": args.warmup,
        "topology": topo,
        "runs": runs,
    }, indent=2))

    md = ["# VRAM Lending Pool — A/B (V5 P13bis)", "",
          f"> Model: `{args.model}`  num_gpus: {args.num_gpus}  "
          f"prompts: {len(PROMPTS)} × {args.max_new} new tokens", ""]
    if topo:
        md.append("## GPU topology")
        md.append("")
        md.append("| Index | Name | SM | Total VRAM | Free at start |")
        md.append("|-------|------|----|------------|---------------|")
        for g in topo:
            md.append(f"| {g['index']} | {g['name']} | {g['sm']} | "
                      f"{g['total_mb']} MB | {g['free_mb']} MB |")
        md.append("")

    md += ["## Results", "",
           "| Variant | Status | Pool active | Load (s) | tok/s | VRAM Δ (MB per GPU) |",
           "|---------|--------|-------------|----------|-------|---------------------|"]
    for r in runs:
        status = r.get("status", "?")
        pool_active = r.get("pool_active", "—")
        load_s = r.get("load_time_s", "—")
        tok_s = r.get("tok_s", "—")
        deltas = r.get("vram_delta_mb", {})
        delta_str = ", ".join(f"{k}: {v}" for k, v in deltas.items()) or "—"
        md.append(f"| {r.get('label', '?')} | {status} | {pool_active} | "
                  f"{load_s} | {tok_s} | {delta_str} |")
    md.append("")

    for r in runs:
        regs = r.get("pool_registered_gpus")
        if regs:
            md.append(f"### Lending pool registry — {r.get('label')}")
            md.append("")
            md.append("| GPU id | Name | Lendable VRAM (GB) |")
            md.append("|--------|------|---------------------|")
            for g in regs:
                lendable_gb = g.get("lendable_bytes", 0) / (1024 ** 3)
                md.append(f"| {g.get('gpu_id', '?')} | "
                          f"{g.get('device_name', '?')} | "
                          f"{lendable_gb:.2f} |")
            md.append("")
        if r.get("status") == "error" and r.get("error"):
            md.append(f"### Error — {r.get('label')}")
            md.append("")
            md.append("```")
            md.append(r["error"][:400])
            md.append("```")
            md.append("")

    if all(r.get("status") == "ok" for r in runs) and len(runs) == 2:
        off = next(r for r in runs if r["label"] == "LENDING_OFF")
        on = next(r for r in runs if r["label"] == "LENDING_ON")
        if off.get("tok_s") and on.get("tok_s"):
            delta = on["tok_s"] - off["tok_s"]
            pct = (delta / off["tok_s"]) * 100 if off["tok_s"] else 0
            md += ["## Lending impact", "",
                   f"- LENDING_OFF: **{off['tok_s']} tok/s**",
                   f"- LENDING_ON : **{on['tok_s']} tok/s** "
                   f"({delta:+.2f} tok/s, {pct:+.1f} %)",
                   ""]
    elif any(r.get("status") == "OOM" for r in runs):
        oom = [r["label"] for r in runs if r.get("status") == "OOM"]
        md += ["## Lending impact", "",
               f"- OOM in variant(s): {', '.join(oom)} — when LENDING_OFF OOMs "
               "and LENDING_ON succeeds, that **is** the proof: cooperative "
               "pooling allows running a model that cannot fit otherwise.",
               ""]

    md += ["## What the table is showing", "",
           "- ``Pool active`` reflects whether ``InferencePipeline.lending_pool`` "
           "was actually instantiated (gated by ``num_gpus > 1`` and "
           "``backend_type not in ('vllm', 'llamacpp')``).",
           "- ``VRAM Δ`` is the post-bench minus pre-load delta per physical GPU. "
           "When lending is active, the pool reserves a buffer on the lender — "
           "VRAM Δ on the lender increases without the model being pinned there.",
           "- The data plane underneath the lending pool uses ``TransferManager`` "
           "(Rust P2P bypass for ≥ 512 KB activations, ReBAR-pipelined for "
           "larger transfers when BAR1 ≥ VRAM — both are active on this host, "
           "see ``docs/reports/REBAR_PROXMOX_BENCHMARK.md``).",
           "",
           "*Generated by VRAMancer bench_lending_hetero_real.py*", ""]
    out_md.write_text("\n".join(md))

    print(f"\nResults written to {out_json}")
    print(f"Summary written to {out_md}")
    return 0 if all(r.get("status") in ("ok", "OOM") for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
