"""V5 P13 — DeepSeek-V4-Flash dual-GPU + engram KV offload bench.

Backend: vLLM 0.20.1 (supporte DeepseekV4ForCausalLM nativement)
Model: deepseek-ai/DeepSeek-V4-Flash (159 GB safetensors, FP8/FP4 MoE)
  - 158B MoE parameters, 256 experts, 6 active/tok
  - RTX 3090 (24 GB) + RTX 5070 Ti (16 GB) = 40 GB VRAM
  - tensor_parallel_size=2 → modèle splitté sur 2 GPUs
  - cpu_offload_gb=120 → ~120 GB de poids offloadés en DRAM (185 GB dispo)
  - KV cache croît avec la longueur de contexte → engram DRAM effect

Demonstrates:
  1. Modèle 159 GB > VRAM totale (40 GB) tourne via cpu_offload (DRAM 185 GB)
  2. DRAM usage croît avec la taille du contexte (KV cache en RAM)
  3. Dual-GPU TP=2 split proportionnel VRAM

Usage:
    VRM_KV_OFFLOAD_ENGRAM=1 VRM_KV_DRAM_LIMIT_GB=185 \\
      python benchmarks/bench_deepseek_engram.py
"""
import json
import os
import sys
import time
from pathlib import Path

HF_MODEL_ID = os.environ.get("VRM_BENCH_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
# cpu_offload_gb: modèle 159 GB, VRAM 40 GB → offloader ~120 GB en DRAM
CPU_OFFLOAD_GB = float(os.environ.get("VRM_BENCH_CPU_OFFLOAD_GB", "120"))
# max_model_len: limité par mémoire disponible sur TP=2
MAX_MODEL_LEN = int(os.environ.get("VRM_BENCH_MAX_MODEL_LEN", "4096"))
CONTEXT_SIZES = [512, 1024, 2048, 4096]
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


def make_prompt_str(approx_chars: int) -> str:
    """Generate a synthetic prompt of approximately approx_chars characters."""
    base = "The quick brown fox jumps over the lazy dog. " * 50
    text = base
    while len(text) < approx_chars:
        text += base
    return text[:approx_chars]


def run_bench():
    print(f"[P13] DeepSeek-V4-Flash — REAL MODEL (158B MoE, 159 GB safetensors)")
    print(f"[P13] Backend: vLLM 0.20.1 (DeepseekV4ForCausalLM natif)")
    print(f"[P13] Model: {HF_MODEL_ID}")
    print(f"[P13] tensor_parallel_size=2, cpu_offload_gb={CPU_OFFLOAD_GB}")
    print(f"[P13] max_model_len={MAX_MODEL_LEN}")
    print(f"[P13] VRM_KV_OFFLOAD_ENGRAM={os.environ.get('VRM_KV_OFFLOAD_ENGRAM', '0')}")
    print()

    os.environ["VRM_KV_OFFLOAD_ENGRAM"] = "1"
    os.environ.setdefault("VRM_KV_DRAM_LIMIT_GB", "185")
    # vLLM avec TP=2 sur GPUs hétérogènes requiert PCI_BUS_ID order
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    try:
        from core.inference_pipeline import InferencePipeline, reset_pipeline
        reset_pipeline()
    except ImportError as e:
        print(f"[BLOCKED@P13] Cannot import InferencePipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {HF_MODEL_ID} via vLLM (TP=2, cpu_offload_gb={CPU_OFFLOAD_GB})...")
    print("  Note: premier lancement = téléchargement ~159 GB depuis HF Hub")
    try:
        # vLLM backend: supporte DeepseekV4ForCausalLM nativement.
        # TP=2 split le modèle sur RTX 3090 (24 GB) + RTX 5070 Ti (16 GB).
        # cpu_offload_gb=120 offload ~120 GB de poids vers DRAM (185 GB disponibles)
        # → seuls ~40 GB de poids restent en VRAM → modèle 159 GB tourne!
        # KV cache pour les tokens en DRAM → engram DRAM effect.
        pipeline = InferencePipeline(backend_name="vllm", enable_metrics=False, enable_discovery=False)
        pipeline.load(
            HF_MODEL_ID,
            num_gpus=2,
            tensor_parallel_size=2,    # forcer TP=2 même si VRAMancer détecte single-GPU
            kv_cache_dtype="fp8",      # DeepseekV4 exige fp8 kv-cache (MLA attention)
            cpu_offload_gb=CPU_OFFLOAD_GB,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.92,
            enforce_eager=True,        # évite cuda graph OOM sur grand modèle
        )
    except Exception as e:
        msg = str(e)
        print(f"[FAILED@P13 — model load failed: {msg[:400]}]")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({
            "model": HF_MODEL_ID,
            "backend": "vllm",
            "note": "FAILED — vLLM load error",
            "error": msg[:800],
        }, indent=2))
        return

    # Prompt scale: taille en chars (approx ~1 char ≈ 0.25 tok pour EN)
    char_sizes = {512: 2048, 1024: 4096, 2048: 8192, 4096: 16384}

    vram_init = measure_vram_per_gpu()
    dram_init = measure_dram_used()
    print(f"\n[P13] Modèle chargé. VRAM: {vram_init}, DRAM RSS: {dram_init['rss_mb']} MB")

    results = []
    for ctx_size in CONTEXT_SIZES:
        prompt = make_prompt_str(char_sizes[ctx_size])
        vram_before = measure_vram_per_gpu()
        dram_before = measure_dram_used()

        t0 = time.perf_counter()
        try:
            _ = pipeline.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.0)
        except Exception as e:
            results.append({
                "ctx_target": ctx_size,
                "error": str(e)[:300],
            })
            print(f"  ctx≈{ctx_size} tok: ERROR {e}")
            continue
        dt = time.perf_counter() - t0

        vram_after = measure_vram_per_gpu()
        dram_after = measure_dram_used()
        tok_s = MAX_NEW / dt if dt > 0 else 0

        vram_deltas = {}
        for k in vram_after:
            if k in vram_before:
                vram_deltas[k] = vram_after[k]["used_mb"] - vram_before[k]["used_mb"]

        dram_delta = (
            dram_after["rss_mb"] - dram_before["rss_mb"]
            if dram_before["rss_mb"] >= 0 and dram_after["rss_mb"] >= 0
            else -1
        )

        results.append({
            "ctx_target": ctx_size,
            "prompt_chars": len(prompt),
            "max_new": MAX_NEW,
            "dt_s": round(dt, 3),
            "tok_s": round(tok_s, 2),
            "vram_before": vram_before,
            "vram_after": vram_after,
            "vram_deltas_mb": vram_deltas,
            "dram_before_mb": dram_before["rss_mb"],
            "dram_after_mb": dram_after["rss_mb"],
            "dram_delta_mb": dram_delta,
        })
        print(f"  ctx≈{ctx_size} tok ({len(prompt)} chars): {tok_s:.2f} tok/s, "
              f"VRAM Δ={vram_deltas}, DRAM Δ={dram_delta} MB")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": HF_MODEL_ID,
        "backend": "vllm",
        "note": (
            "DeepSeek-V4-Flash 158B MoE, 159 GB safetensors. "
            "vLLM TP=2 (RTX 3090 24GB + RTX 5070 Ti 16GB = 40GB VRAM). "
            f"cpu_offload_gb={CPU_OFFLOAD_GB} → ~120 GB poids en DRAM (185 GB dispo). "
            "KV cache en DRAM → engram DRAM effect."
        ),
        "tensor_parallel_size": 2,
        "cpu_offload_gb": CPU_OFFLOAD_GB,
        "max_model_len": MAX_MODEL_LEN,
        "context_sizes": CONTEXT_SIZES,
        "max_new": MAX_NEW,
        "engram_offload": True,
        "dram_limit_gb": int(os.environ.get("VRM_KV_DRAM_LIMIT_GB", "185")),
        "vram_init": vram_init,
        "dram_init_mb": dram_init["rss_mb"],
        "results": results,
    }, indent=2))

    md_lines = [
        "# DeepSeek-V4-Flash + engram DRAM offload bench (V5 P13)",
        "",
        "> **REAL DeepSeek-V4-Flash** — 158B MoE params, 159 GB safetensors (FP8/FP4).",
        "> vLLM TP=2 (RTX 3090 24 GB + RTX 5070 Ti 16 GB = 40 GB VRAM).",
        f"> cpu_offload_gb={CPU_OFFLOAD_GB} → ~120 GB de poids en DRAM (185 GB dispo).",
        "> KV cache dans DRAM pour les grandes séquences → engram DRAM effect.",
        "",
        f"**Model:** `{HF_MODEL_ID}`  **Backend:** vLLM 0.20.1",
        "**GPUs:** RTX 3090 (24 GB) + RTX 5070 Ti (16 GB) — 40 GB VRAM total",
        f"**cpu_offload_gb:** {CPU_OFFLOAD_GB}  **max_model_len:** {MAX_MODEL_LEN}",
        "",
        "| Context tok | tok/s | VRAM Δ G0 MB | VRAM Δ G1 MB | DRAM Δ MB |",
        "|-------------|-------|--------------|--------------|-----------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r['ctx_target']} | ERROR | — | — | — |")
            continue
        vd = r.get("vram_deltas_mb", {})
        g0 = vd.get("gpu0", "N/A")
        g1 = vd.get("gpu1", "N/A")
        md_lines.append(
            f"| {r['ctx_target']} | {r['tok_s']:.2f} | {g0} | {g1} | {r['dram_delta_mb']} |"
        )

    md_lines += [
        "",
        "**Engram offload:** VRM_KV_OFFLOAD_ENGRAM=1  **DRAM Δ > 0** = KV spill en DRAM visible",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    run_bench()
