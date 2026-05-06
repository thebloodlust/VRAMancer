#!/usr/bin/env python3
"""Bench Devstral-Small-22B NF4 + TurboQuant KV compression via VRAMancer.

Mesure l'impact de TurboQuant (PolarQuant + QJL, ~3.5 bits/dim, ~4.6× KV reduction)
sur le débit de génération et l'utilisation VRAM d'un modèle code 22B NF4.

Configs testées :
  A) Baseline  : NF4, KV BF16, VRM_KV_COMPRESSION non défini
  B) TurboQuant: NF4, TurboQuant 3-bit, VRM_KV_COMPRESSION=turboquant
  C) TQ+SparseV: NF4, TurboQuant + SparseV 10%, VRM_SPARSE_V_RATIO=0.1

Contextes : 8K, 32K tokens

Hardware : RTX 5070 Ti (CUDA0, 16GB) — single GPU (BnB NF4 multi-GPU upstream bug)
           RTX 3090 utilisé si NF4 ne tient pas dans 16GB

Note TurboQuant : applicable uniquement au pipeline HF (non llama.cpp). Chaque
token génère des appels compress() / decompress() via core/kv_quantizer.py (GPU).
"""
from __future__ import annotations

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "unsloth/Devstral-Small-2505-bnb-4bit"
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUT = RESULTS_DIR / "bench_devstral_turboquant_v5.json"

CODING_PROMPT = (
    "<s>[INST] Write a complete Python implementation of a LRU Cache using "
    "OrderedDict with get() and put() methods. Include docstrings and type hints. "
    "Then implement a decorator version. [/INST]"
)
N_PREDICT = 256


def get_vram_gb() -> list[float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode()
        return [round(int(x.strip()) / 1024, 2) for x in out.strip().split("\n")]
    except Exception:
        return []


def get_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().used / 1e9
    except Exception:
        return 0.0


def run_config(config_name: str, n_ctx: int, env_overrides: dict[str, str]) -> dict[str, Any]:
    """Run a single bench config in subprocess to ensure clean VRAM state."""
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}  |  ctx: {n_ctx:,}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env.update(env_overrides)

    script = f"""
import os, sys, time, gc, json
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

import torch
import subprocess

def get_vram():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5).decode()
        return [round(int(x.strip()) / 1024, 2) for x in out.strip().split("\\n")]
    except: return []

MODEL_ID = "{MODEL_ID}"
PROMPT = {CODING_PROMPT!r}
N_PREDICT = {N_PREDICT}
N_CTX = {n_ctx}

try:
    from core.inference_pipeline import InferencePipeline, reset_pipeline
    reset_pipeline()

    vram_before = get_vram()
    t_load = time.time()
    # Force HuggingFace backend — TurboQuant KV compression only applies to HF pipeline
    pipe = InferencePipeline(backend_name="huggingface")
    pipe.load(MODEL_ID, num_gpus=1)
    load_time = time.time() - t_load
    vram_after_load = get_vram()
    print(f"LOAD_OK load_time={{load_time:.1f}}s vram_after={{vram_after_load}}", flush=True)

    # Warmup
    try:
        _ = pipe.generate("Hello", max_new_tokens=4, do_sample=False)
    except Exception as e:
        print(f"WARMUP_FAIL {{e}}", flush=True)

    vram_after_warmup = get_vram()

    # Bench
    t0 = time.time()
    output = pipe.generate(PROMPT, max_new_tokens=N_PREDICT, do_sample=False)
    elapsed = time.time() - t0

    vram_peak = get_vram()

    # Count tokens from output
    from core.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(MODEL_ID)
    n_out_tokens = len(tokenizer.encode(output)) if output else 0
    tok_s = round(n_out_tokens / elapsed, 2) if elapsed > 0 else 0

    print(f"RESULT tok_s={{tok_s}} n_tokens={{n_out_tokens}} elapsed={{elapsed:.1f}}s vram_peak={{vram_peak}}", flush=True)
    print(f"OUTPUT_PREVIEW {{output[:200]!r}}", flush=True)

    reset_pipeline()
    gc.collect()
    torch.cuda.empty_cache()

except Exception as e:
    import traceback
    print(f"ERROR {{type(e).__name__}}: {{e}}", flush=True)
    traceback.print_exc()
"""

    t_start = time.time()
    vram_before = get_vram_gb()

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    result_data: dict = {}

    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        print(f"  {line}")
        lines.append(line)

        if line.startswith("LOAD_OK"):
            parts = line.split()
            for p in parts[1:]:
                if p.startswith("load_time="):
                    result_data["load_time_s"] = float(p.split("=")[1].rstrip("s"))
        elif line.startswith("RESULT"):
            parts = line.split()
            for p in parts[1:]:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                if k == "tok_s":
                    result_data["tokens_per_sec"] = float(v)
                elif k == "n_tokens":
                    result_data["n_generated_tokens"] = int(v)
                elif k == "elapsed":
                    result_data["gen_time_s"] = float(v.rstrip("s"))
        elif line.startswith("ERROR"):
            result_data["error"] = line

    proc.wait()
    total_time = time.time() - t_start
    vram_after = get_vram_gb()

    result_data.update({
        "config": config_name,
        "n_ctx": n_ctx,
        "vram_before_gb": vram_before,
        "vram_after_gb": vram_after,
        "total_wall_time_s": round(total_time, 1),
        "env_overrides": {k: v for k, v in env_overrides.items() if k.startswith("VRM_")},
    })

    if "tokens_per_sec" in result_data:
        print(f"\n  → {result_data['tokens_per_sec']} tok/s ({result_data.get('n_generated_tokens', '?')} tokens in {result_data.get('gen_time_s', '?')}s)")
    elif "error" in result_data:
        print(f"\n  → FAILED: {result_data['error']}")

    return result_data


def main() -> None:
    base_env = {
        "VRM_MINIMAL_TEST": "",
        "VRM_BACKEND_ALLOW_STUB": "",
        # BitsAndBytes requires libnvJitLink.so.13 from nvidia/cu13 package
        "LD_LIBRARY_PATH": (
            "/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:"
            + os.environ.get("LD_LIBRARY_PATH", "")
        ),
    }

    configs = [
        # (name, ctx, env_overrides)
        ("A_baseline_8k",    8192,  {**base_env}),
        ("A_baseline_32k",   32768, {**base_env}),
        ("B_turboquant_8k",  8192,  {**base_env, "VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3"}),
        ("B_turboquant_32k", 32768, {**base_env, "VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3"}),
        ("C_tq_sparsev_8k",  8192,  {**base_env, "VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "0.1"}),
        ("C_tq_sparsev_32k", 32768, {**base_env, "VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "0.1"}),
    ]

    print(f"Modèle  : {MODEL_ID} (NF4 BnB, ~14GB)")
    print(f"Backend : VRAMancer InferencePipeline (num_gpus=1, RTX 5070 Ti 16GB)")
    print(f"Configs : {len(configs)} (baseline + TurboQuant + TQ+SparseV @ 8K/32K)")
    print(f"Output  : {OUT}")

    all_results: list[dict] = []

    for (name, ctx, env) in configs:
        res = run_config(name, ctx, env)
        all_results.append(res)

        # Sauvegarde intermédiaire
        with open(OUT, "w") as f:
            json.dump({
                "model": MODEL_ID,
                "hardware": "RTX 5070 Ti (CUDA0, 16GB) — single GPU NF4 BnB",
                "n_predict": N_PREDICT,
                "results": all_results,
            }, f, indent=2)
        print(f"\n  → sauvegardé dans {OUT}")

        # Pause entre configs pour laisser VRAM se stabiliser
        import time as _t
        _t.sleep(5)

    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    for r in all_results:
        cfg = r.get("config", "?")
        ctx = r.get("n_ctx", 0)
        if "tokens_per_sec" in r:
            print(f"  {cfg:30s} (ctx={ctx:>6,}) : {r['tokens_per_sec']:6.1f} tok/s")
        else:
            print(f"  {cfg:30s} (ctx={ctx:>6,}) : ERREUR — {r.get('error', '?')}")


if __name__ == "__main__":
    main()
