#!/usr/bin/env python3
"""Benchmark DeepSeek-V4-Flash IQ2_XS-XL GGUF sur 2 GPUs hétérogènes.

Hardware cible :
  - CUDA 0 : RTX 3090 (Ampere SM8.6, 24 GB)
  - CUDA 1 : RTX 5070 Ti (Blackwell SM12.0, 16 GB)
  Total VRAM : 40 GB / Modèle : 87 GB → ~47 GB en RAM CPU

Modes testés :
  1. 2-GPU tensor_split auto      — le showcase VRAMancer (40 GB VRAM + 47 GB CPU RAM)
  2. Single GPU CUDA 1 (5070 Ti)  — référence, quelques layers GPU
  3. Single GPU CUDA 0 (3090)     — référence

STATUT (2026-05-23) :
  ⚠ Le GGUF teamblobfish utilise l'architecture "deepseek4" — une arch CUSTOM
  non présente dans llama.cpp officiel (ni 0.3.22 ni 0.3.23).

  Différences structurelles vs deepseek2 (llama.cpp) :
    - Pas de kv_lora_rank (GQA standard au lieu de MLA)
    - Hyper connections (residual stream adaptatif)
    - Attention sinks (fenêtre glissante avec sink tokens)
    - Output LoRA (projection sortie décomposée en AB)
    - Pas encore upstreamé dans llama.cpp officiel

  Solution provisoire si disponible : construire llama.cpp depuis le fork
  teamblobfish avec support deepseek4 natif.

  Alternative fonctionnelle dès maintenant : vLLM avec le modèle FP8 officiel
  (deepseek-ai/DeepSeek-V4-Flash, 160 GB, nécessite 4×GPU A100/H100).

Usage :
    python benchmarks/bench_deepseek_flash_gguf.py
    python benchmarks/bench_deepseek_flash_gguf.py --model ~/models/DeepSeek-V4-Flash-IQ2_XS
    python benchmarks/bench_deepseek_flash_gguf.py --only-dual  # skip single-GPU tests
"""
import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("VRM_MINIMAL_TEST", None)

MODEL_DIR = os.path.expanduser("~/models/DeepSeek-V4-Flash-IQ2_XS/IQ2_XS-XL")
SHARD1 = os.path.join(MODEL_DIR, "DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00002.gguf")

# llama.cpp charge automatiquement les shards si on lui passe le shard 1
MODEL_PATH = SHARD1

PROMPTS = [
    "Explain why mixture-of-experts models are efficient:",
    "Write a Python async HTTP server with aiohttp:",
    "The key advantage of VRAMancer over vLLM is",
    "Quantum computing will revolutionize",
]

MAX_TOKENS = 128
N_CTX = 4096


def get_vram_info():
    import torch
    info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info.append({
            "id": i,
            "name": p.name,
            "total_gb": round(total / 1024**3, 1),
            "free_gb": round(free / 1024**3, 1),
        })
    return info


def print_vram(label=""):
    import torch
    if label:
        print(f"  [{label}]")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        print(f"    GPU {i}: {used/1e9:.2f} GB used / {total/1e9:.2f} GB total")


def run_bench(llm, label: str, n_warmup: int = 1) -> dict:
    """Exécute le benchmark sur les PROMPTS, retourne stats."""
    import torch

    print(f"\n  {'─'*50}")
    print(f"  Benchmark : {label}")
    print(f"  {'─'*50}")

    latencies = []
    tokens_total = 0

    for i, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        out = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,  # greedy pour reproductibilité
            echo=False,
        )
        dt = time.perf_counter() - t0

        n_tok = out["usage"]["completion_tokens"] if "usage" in out else MAX_TOKENS
        tok_s = n_tok / dt
        tokens_total += n_tok

        if i >= n_warmup:
            latencies.append(dt)
            print(f"    [{i+1}] {n_tok} tokens en {dt:.2f}s → {tok_s:.1f} tok/s")
        else:
            print(f"    [warmup] {n_tok} tokens en {dt:.2f}s → {tok_s:.1f} tok/s (ignoré)")

    avg_toks = tokens_total / sum(1 + len(p.split()) for p in PROMPTS[n_warmup:]) if latencies else 0
    avg_dt = sum(latencies) / len(latencies) if latencies else 0
    avg_tokps = (MAX_TOKENS * len(latencies)) / sum(latencies) if latencies else 0

    print(f"\n  → Moyenne : {avg_tokps:.1f} tok/s sur {len(latencies)} runs")
    return {
        "label": label,
        "avg_tok_s": round(avg_tokps, 1),
        "avg_latency_s": round(avg_dt, 2),
        "n_runs": len(latencies),
    }


def load_llamacpp(n_gpu_layers: int, gpu_ids: list, tensor_split=None, verbose=False):
    """Charge le modèle llama-cpp-python directement."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python non installé.")
        print("  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Modèle introuvable : {MODEL_PATH}")
        print(f"  Assurez-vous que le téléchargement est terminé.")
        sys.exit(1)

    kwargs = dict(
        model_path=MODEL_PATH,
        n_gpu_layers=n_gpu_layers,
        n_ctx=N_CTX,
        flash_attn=True,
        verbose=verbose,
        split_mode=1,  # LLAMA_SPLIT_MODE_LAYER
    )

    if tensor_split is not None:
        kwargs["tensor_split"] = tensor_split

    # Sélection des GPUs via CUDA_VISIBLE_DEVICES
    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print(f"\n  Chargement : n_gpu_layers={n_gpu_layers}, GPUs={gpu_ids}, tensor_split={tensor_split}")
    t0 = time.time()
    llm = Llama(**kwargs)
    load_time = time.time() - t0
    print(f"  Chargé en {load_time:.1f}s")

    if old_cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    return llm


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V4-Flash GGUF 2-GPU")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin vers le fichier .gguf")
    parser.add_argument("--only-dual", action="store_true", help="Skip single-GPU tests")
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.model != MODEL_PATH:
        globals()["MODEL_PATH"] = args.model
    globals()["N_CTX"] = args.n_ctx
    globals()["MAX_TOKENS"] = args.max_tokens

    import torch
    print("=" * 60)
    print("DeepSeek-V4-Flash IQ2_XS-XL — Benchmark VRAMancer 2-GPU")
    print("=" * 60)
    print(f"\nPyTorch : {torch.__version__}")
    gpus = get_vram_info()
    for g in gpus:
        print(f"  GPU {g['id']}: {g['name']} — {g['free_gb']:.1f}/{g['total_gb']:.1f} GB free")
    print(f"\nModèle  : {MODEL_PATH}")
    print(f"Context : {N_CTX} tokens, max_tokens={MAX_TOKENS}")

    results = []

    # ── TEST 1 : 2-GPU hétérogène (showcase) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1 : 2-GPU hétérogène (5070 Ti + 3090)")
    print("  tensor_split=[0.4, 0.6]  (proportionnel à la VRAM libre)")
    print("=" * 60)
    # 5070 Ti = CUDA 0 (16 GB) → 40% ; 3090 = CUDA 1 (24 GB) → 60%
    # n_gpu_layers=-1 : max layers GPU, le reste CPU RAM
    llm_dual = load_llamacpp(
        n_gpu_layers=-1,
        gpu_ids=[0, 1],
        tensor_split=[0.4, 0.6],
        verbose=args.verbose,
    )
    print_vram("après chargement dual-GPU")
    res = run_bench(llm_dual, "2-GPU (5070Ti 40% + 3090 60%)")
    results.append(res)
    del llm_dual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.only_dual:
        # ── TEST 2 : Single GPU — 5070 Ti seulement ──────────────────────────
        print("\n" + "=" * 60)
        print("TEST 2 : Single GPU — RTX 5070 Ti (CUDA 0, 16 GB)")
        print("  La majorité des layers sera en CPU RAM")
        print("=" * 60)
        # 16 GB / ~87 GB = ~18% du modèle en VRAM
        # n_gpu_layers estimé : ~20 layers sur ~60 total
        llm_5070 = load_llamacpp(
            n_gpu_layers=20,
            gpu_ids=[0],
            tensor_split=None,
            verbose=args.verbose,
        )
        print_vram("après chargement 5070 Ti seul")
        res = run_bench(llm_5070, "Single GPU — RTX 5070 Ti (20 layers GPU)")
        results.append(res)
        del llm_5070
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── TEST 3 : Single GPU — RTX 3090 seulement ─────────────────────────
        print("\n" + "=" * 60)
        print("TEST 3 : Single GPU — RTX 3090 (CUDA 1, 24 GB)")
        print("=" * 60)
        # 24 GB / ~87 GB = ~27% du modèle en VRAM
        llm_3090 = load_llamacpp(
            n_gpu_layers=30,
            gpu_ids=[1],
            tensor_split=None,
            verbose=args.verbose,
        )
        print_vram("après chargement 3090 seul")
        res = run_bench(llm_3090, "Single GPU — RTX 3090 (30 layers GPU)")
        results.append(res)
        del llm_3090
        gc.collect()

    # ── Résumé ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ — DeepSeek-V4-Flash IQ2_XS-XL")
    print("=" * 60)
    print(f"  {'Configuration':<40} {'tok/s':>8}")
    print(f"  {'-'*48}")
    for r in results:
        print(f"  {r['label']:<40} {r['avg_tok_s']:>7.1f}")

    if len(results) >= 2:
        dual = results[0]["avg_tok_s"]
        single_best = max(r["avg_tok_s"] for r in results[1:])
        speedup = dual / single_best if single_best > 0 else 0
        print(f"\n  Speedup 2-GPU vs best single : {speedup:.2f}x")

    print()


if __name__ == "__main__":
    main()
