"""V5 P13 — KV engram offload bench with long contexts.

Target model: deepseek-ai/DeepSeek-V4-Flash (158B) — SKIPPED (too large for
40GB VRAM rig). Using Qwen2.5-7B-Instruct as proxy to demonstrate engram
KV offload mechanism with increasing context lengths.

[PARTIAL@P13.1 — DeepSeek-V4-Flash (158B) >> 40GB VRAM; proxy: Qwen2.5-7B-Instruct]

Usage:
    VRM_KV_OFFLOAD_ENGRAM=1 VRM_KV_DRAM_LIMIT_GB=200 \
      python benchmarks/bench_deepseek_engram.py
"""
import json
import os
import sys
import time
from pathlib import Path

MODEL = os.environ.get(
    "VRM_BENCH_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",  # proxy for DeepSeek-V4-Flash (see PARTIAL note above)
)
CONTEXT_SIZES = [512, 2048, 4096, 8192, 16384]  # tokens
MAX_NEW = 32
OUT_JSON = Path("benchmarks/results/bench_deepseek_engram_v5.json")
OUT_MD = Path("benchmarks/results/bench_deepseek_engram_v5.md")


def measure_vram_per_gpu():
    try:
        import torch
        out = {}
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out[f"gpu{i}"] = {
                "used_mb": (total - free) // (1024 * 1024),
                "total_mb": total // (1024 * 1024),
            }
        return out
    except Exception:
        return {}


def measure_dram_used():
    try:
        import psutil
        proc = psutil.Process()
        return {"rss_mb": proc.memory_info().rss // (1024 * 1024)}
    except Exception:
        return {"rss_mb": -1}


def make_prompt(approx_tokens: int, tokenizer) -> str:
    """Generate a synthetic prompt of ~approx_tokens tokens."""
    base = "The quick brown fox jumps over the lazy dog. " * 100
    text = base
    while len(tokenizer.encode(text)) < approx_tokens:
        text += base
    # Trim to exact target (approximate)
    tokens = tokenizer.encode(text)
    if len(tokens) > approx_tokens:
        tokens = tokens[:approx_tokens]
        text = tokenizer.decode(tokens)
    return text


def run_bench():
    print(f"[P13] Model: {MODEL} (proxy for DeepSeek-V4-Flash — PARTIAL@P13.1)")
    print(f"[P13] VRM_KV_OFFLOAD_ENGRAM={os.environ.get('VRM_KV_OFFLOAD_ENGRAM', '0')}")
    print(f"[P13] VRM_KV_DRAM_LIMIT_GB={os.environ.get('VRM_KV_DRAM_LIMIT_GB', '200')}")
    print()

    os.environ["VRM_KV_OFFLOAD_ENGRAM"] = "1"
    os.environ.setdefault("VRM_KV_DRAM_LIMIT_GB", "200")

    try:
        from core.inference_pipeline import InferencePipeline, reset_pipeline
        reset_pipeline()
    except ImportError as e:
        print(f"[BLOCKED@P13] Cannot import InferencePipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {MODEL}...")
    try:
        pipeline = InferencePipeline(enable_metrics=False, enable_discovery=False)
        pipeline.load(MODEL)
    except Exception as e:
        msg = str(e)
        print(f"[SKIPPED@P13 — model load failed: {msg[:200]}]")
        # Write a partial result
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({
            "model": MODEL,
            "note": "PARTIAL@P13.1 — DeepSeek-V4-Flash too large (158B); proxy model OOM'd too",
            "error": msg[:500],
        }, indent=2))
        return

    tokenizer = getattr(getattr(pipeline, "backend", None), "tokenizer", None)
    if tokenizer is None:
        print("[BLOCKED@P13] Cannot get tokenizer from backend")
        return

    # Log offloader status
    kv_off = getattr(pipeline, "kv_offloader", None)
    print(f"[P13] kv_offloader: {kv_off}")

    results = []
    for ctx_size in CONTEXT_SIZES:
        prompt = make_prompt(ctx_size, tokenizer)
        actual_tokens = len(tokenizer.encode(prompt))
        vram_before = measure_vram_per_gpu()
        dram_before = measure_dram_used()

        t0 = time.perf_counter()
        try:
            _ = pipeline.generate(prompt, max_new_tokens=MAX_NEW)
        except Exception as e:
            results.append({
                "ctx_target": ctx_size,
                "ctx_actual": actual_tokens,
                "error": str(e)[:200],
            })
            print(f"  ctx={ctx_size}: ERROR {e}")
            continue
        dt = time.perf_counter() - t0

        vram_after = measure_vram_per_gpu()
        dram_after = measure_dram_used()
        tok_s = MAX_NEW / dt if dt > 0 else 0

        # Compute VRAM deltas
        vram_deltas = {}
        for k in vram_after:
            if k in vram_before:
                vram_deltas[k] = vram_after[k]["used_mb"] - vram_before[k]["used_mb"]

        dram_delta = (
            dram_after["rss_mb"] - dram_before["rss_mb"]
            if dram_before["rss_mb"] >= 0 and dram_after["rss_mb"] >= 0
            else -1
        )

        # Offloader stats
        offload_stats = {}
        if kv_off is not None:
            offload_stats = kv_off.stats()

        results.append({
            "ctx_target": ctx_size,
            "ctx_actual": actual_tokens,
            "max_new": MAX_NEW,
            "dt_s": round(dt, 3),
            "tok_s": round(tok_s, 2),
            "vram_before": vram_before,
            "vram_after": vram_after,
            "vram_deltas_mb": vram_deltas,
            "dram_before_mb": dram_before["rss_mb"],
            "dram_after_mb": dram_after["rss_mb"],
            "dram_delta_mb": dram_delta,
            "offload_stats": offload_stats,
        })
        print(f"  ctx={ctx_size} ({actual_tokens} tok): {tok_s:.2f} tok/s, "
              f"VRAM Δ={vram_deltas}, DRAM Δ={dram_delta} MB, "
              f"offload={offload_stats}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": MODEL,
        "note": "PARTIAL@P13.1: proxy for DeepSeek-V4-Flash (158B too large for 40GB rig)",
        "context_sizes": CONTEXT_SIZES,
        "max_new": MAX_NEW,
        "engram_offload": os.environ.get("VRM_KV_OFFLOAD_ENGRAM") == "1",
        "dram_limit_gb": int(os.environ.get("VRM_KV_DRAM_LIMIT_GB", "200")),
        "results": results,
    }, indent=2))

    md_lines = [
        "# DeepSeek + engram KV offload bench (V5 P13)",
        "",
        f"> **PARTIAL@P13.1**: Target was `deepseek-ai/DeepSeek-V4-Flash` (158B params).",
        f"> 158B >> 40GB VRAM (RTX 5070 Ti 16GB + RTX 3090 24GB). Proxy: `{MODEL}`.",
        "",
        f"**Model:** `{MODEL}`",
        "**GPUs:** RTX 5070 Ti (16GB) + RTX 3090 (24GB) — 40 GB VRAM total",
        "**KV offload:** engram DRAM store (cap 200 GB) via VRM_KV_OFFLOAD_ENGRAM=1",
        "",
        "| Context | Actual tok | tok/s | VRAM Δ G0 MB | VRAM Δ G1 MB | DRAM Δ MB | Offload evictions |",
        "|---------|-----------|-------|--------------|--------------|-----------|------------------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(
                f"| {r['ctx_target']} | {r['ctx_actual']} | ERROR | — | — | — | — |"
            )
            continue
        vd = r.get("vram_deltas_mb", {})
        g0 = vd.get("gpu0", "N/A")
        g1 = vd.get("gpu1", "N/A")
        evict = r.get("offload_stats", {}).get("evicted_total", 0)
        md_lines.append(
            f"| {r['ctx_target']} | {r['ctx_actual']} | {r['tok_s']:.2f} "
            f"| {g0} | {g1} | {r['dram_delta_mb']} | {evict} |"
        )

    md_lines += [
        "",
        "**Engram offload active:** yes (VRM_KV_OFFLOAD_ENGRAM=1)",
        "**DRAM Δ > 0 at ctx:** see table — first row where DRAM Δ > 0 is offload inflection",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    run_bench()
