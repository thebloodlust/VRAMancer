#!/usr/bin/env python3
"""Bench Mistral Medium 3.5 128B GGUF (UD-IQ2_XXS) sur RTX 3090 + RTX 5070 Ti.

Mesure :
- VRAM usage par GPU
- Tokens/sec en génération
- Latence prefill (prompt processing)
- KV cache RAM @ contextes 32K, 128K, 256K (Q4_0 type_k/v)

Usage:
    python benchmarks/bench_mistral_medium_gguf.py
    python benchmarks/bench_mistral_medium_gguf.py --ctx 32768,131072
"""
from __future__ import annotations

import os
import sys
import time
import json
import gc
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--unsloth--Mistral-Medium-3.5-128B-GGUF/Mistral-Medium-3.5-128B-UD-IQ2_XXS.gguf"
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUT = RESULTS_DIR / "bench_mistral_medium_gguf_v5.json"

PROMPT = (
    "Write a Python function that computes the n-th Fibonacci number using "
    "memoization. Then explain its time complexity in detail."
)
N_PREDICT = 128

# Tensor split : proportions relatives (pas des MiB absolus en llama-cpp-python 0.3.x)
# CUDA0 = RTX 3090 (24576 MiB) → 0.60, CUDA1 = RTX 5070 Ti (15841 MiB) → 0.40
# NOTE: llama.cpp assigne GPU0 en premier, donc RTX 3090=0 (24GB), 5070 Ti=1 (16GB)
TENSOR_SPLIT = [0.60, 0.40]


def get_vram_gb() -> list[float]:
    """VRAM used per GPU via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
        return [int(x.strip()) / 1024 for x in out.strip().split("\n")]
    except Exception:
        return []


def get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except Exception:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    avail_kb = int(line.split()[1])
                    return (16 * 1024 * 1024 - avail_kb) / 1e6  # rough
        return 0.0


def run_bench(n_ctx: int) -> dict:
    from llama_cpp import Llama

    print(f"\n{'='*60}\n  n_ctx = {n_ctx:,}\n{'='*60}")

    ram_before = get_ram_gb()
    vram_before = get_vram_gb()

    t_load_start = time.time()
    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,          # tout en VRAM
            tensor_split=TENSOR_SPLIT,  # [0.60, 0.40] = proportions relatives
            n_ctx=n_ctx,
            type_k=2,                 # Q4_0 KV cache
            type_v=2,
            offload_kqv=False,        # KV en CPU RAM, libère VRAM pour grands contextes
            flash_attn=True,
            verbose=False,
            n_threads=8,
            n_threads_batch=8,
        )
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return {"n_ctx": n_ctx, "error": f"load: {e}", "load_time_s": time.time() - t_load_start}

    load_time = time.time() - t_load_start
    ram_after_load = get_ram_gb()
    vram_after_load = get_vram_gb()
    print(f"  Load: {load_time:.1f}s  |  RAM delta: +{ram_after_load - ram_before:.1f} GB  |  VRAM: {vram_after_load}")

    # Warmup (1 token)
    try:
        _ = llm("Hi", max_tokens=1, echo=False)
    except Exception:
        pass

    # Bench
    t_gen_start = time.time()
    try:
        output = llm(PROMPT, max_tokens=N_PREDICT, echo=False, temperature=0.0)
    except Exception as e:
        print(f"  GENERATE FAILED: {e}")
        del llm; gc.collect()
        return {"n_ctx": n_ctx, "load_time_s": round(load_time, 2),
                "error": f"generate: {e}"}

    elapsed = time.time() - t_gen_start
    ram_peak = get_ram_gb()
    vram_peak = get_vram_gb()

    n_generated = output["usage"]["completion_tokens"]
    n_prompt = output["usage"]["prompt_tokens"]
    tok_s = round(n_generated / elapsed, 2) if elapsed > 0 else 0
    text = output["choices"][0]["text"]

    print(f"  Prefill: {n_prompt} tokens | Generated: {n_generated} tokens")
    print(f"  Speed: {tok_s} tok/s | Gen time: {elapsed:.1f}s")
    print(f"  RAM peak: +{ram_peak - ram_before:.1f} GB | VRAM peak: {vram_peak}")
    print(f"  Output preview: {text[:120]!r}")

    result = {
        "n_ctx": n_ctx,
        "load_time_s": round(load_time, 2),
        "gen_time_s": round(elapsed, 2),
        "n_prompt_tokens": n_prompt,
        "n_generated_tokens": n_generated,
        "tokens_per_sec": tok_s,
        "ram_delta_load_gb": round(ram_after_load - ram_before, 2),
        "ram_peak_delta_gb": round(ram_peak - ram_before, 2),
        "vram_before_gb": vram_before,
        "vram_after_load_gb": vram_after_load,
        "vram_peak_gb": vram_peak,
        "output_preview": text[:300],
    }

    del llm
    gc.collect()
    time.sleep(2)  # laisser le GPU se vider

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", default="32768,131072,262144",
                        help="Context sizes comma-separated")
    args = parser.parse_args()

    ctx_list = [int(x.strip()) for x in args.ctx.split(",")]

    if not MODEL_PATH.exists():
        print(f"ERROR: modèle introuvable : {MODEL_PATH}")
        sys.exit(1)

    print(f"Modèle : {MODEL_PATH.name}  ({MODEL_PATH.stat().st_size / 1e9:.1f} GB)")
    print(f"Contextes à tester : {ctx_list}")

    results = []
    for ctx in ctx_list:
        res = run_bench(ctx)
        results.append(res)
        with open(OUT, "w") as f:
            json.dump({
                "model": "Mistral-Medium-3.5-128B-UD-IQ2_XXS",
                "tensor_split": TENSOR_SPLIT,
                "type_k": "Q4_0",
                "type_v": "Q4_0",
                "offload_kv": False,
                "n_predict": N_PREDICT,
                "results": results,
            }, f, indent=2)
        print(f"\n  → sauvegardé dans {OUT}")

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    for r in results:
        if "error" in r:
            print(f"  ctx={r['n_ctx']:>7,} : ERREUR — {r['error']}")
        else:
            print(f"  ctx={r['n_ctx']:>7,} : {r['tokens_per_sec']:5.1f} tok/s | "
                  f"RAM +{r['ram_peak_delta_gb']:.1f} GB | load {r['load_time_s']:.0f}s")


if __name__ == "__main__":
    main()
