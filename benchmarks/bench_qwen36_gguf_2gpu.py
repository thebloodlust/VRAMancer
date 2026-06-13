#!/usr/bin/env python3
"""Benchmark Qwen3.6-35B-A3B Q4_K_M GGUF sur 2 GPUs hétérogènes (full-GPU).

Le chemin "réel" recommandé : GGUF Q4_K_M via llama.cpp. Contrairement au
checkpoint NVFP4 MIXED_PRECISION (qui charge les experts MoE en BF16 → OOM)
ou au BF16 offload RAM (0.28 tok/s, PCIe-bound), le GGUF Q4_K_M quantifie
réellement les experts MoE (~20 GB) et tient ENTIÈREMENT dans la VRAM combinée
des deux GPUs → vitesse pleine.

Hardware cible :
  - RTX 3090   (Ampere   SM8.6, 24 GB)
  - RTX 5070 Ti (Blackwell SM12.0, 16 GB)
  Total VRAM : ~40 GB / Modèle Q4_K_M : ~20 GB → tout en VRAM + KV cache.

Architecture : qwen35moe (hybride DeltaNet/SSM + MoE) — supportée nativement
par la llama.cpp embarquée dans llama-cpp-python 0.3.22 (vérifié : symboles
llm_build_qwen35moe, llm_build_qwen3next, llm_build_delta_net_base présents).

PRÉREQUIS : llama-cpp-python compilé avec CUDA (sm_86 + sm_120) :
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86;120" \
        pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python \
        "llama-cpp-python==0.3.22"

Usage :
    python benchmarks/bench_qwen36_gguf_2gpu.py
    python benchmarks/bench_qwen36_gguf_2gpu.py --only-dual
    python benchmarks/bench_qwen36_gguf_2gpu.py --max-tokens 256 --n-ctx 8192
"""
import argparse
import gc
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("VRM_MINIMAL_TEST", None)

MODEL_DIR = os.path.expanduser("~/models/Qwen3.6-35B-A3B-GGUF")

PROMPTS = [
    "Explain why mixture-of-experts models are efficient:",
    "Write a Python async HTTP server with aiohttp:",
    "The key advantage of VRAMancer over vLLM is",
    "Describe the architecture of a transformer in three sentences:",
]

MAX_TOKENS = 128
N_CTX = 4096


def find_gguf(model_dir: str) -> str:
    """Trouve le .gguf Q4_K_M (ou le shard 00001) dans le dossier."""
    if os.path.isfile(model_dir) and model_dir.endswith(".gguf"):
        return model_dir
    candidates = sorted(glob.glob(os.path.join(model_dir, "**", "*.gguf"), recursive=True))
    if not candidates:
        print(f"ERROR: aucun .gguf trouvé dans {model_dir}")
        sys.exit(1)
    # Préférer Q4_K_M, et le premier shard si sharded
    q4 = [c for c in candidates if "Q4_K_M" in c or "q4_k_m" in c.lower()]
    pool = q4 if q4 else candidates
    shard1 = [c for c in pool if "00001-of-" in c]
    return shard1[0] if shard1 else pool[0]


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
            "free_bytes": free,
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


def compute_tensor_split(gpus):
    """tensor_split proportionnel à la VRAM libre de chaque GPU."""
    total_free = sum(g["free_bytes"] for g in gpus)
    if total_free <= 0:
        return None
    return [round(g["free_bytes"] / total_free, 3) for g in gpus]


def load_llamacpp(model_path, n_gpu_layers, gpu_ids, tensor_split=None,
                  split_mode=1, n_ctx=N_CTX, verbose=False):
    try:
        from llama_cpp import Llama, llama_supports_gpu_offload
    except ImportError:
        print("ERROR: llama-cpp-python non installé.")
        sys.exit(1)

    if not llama_supports_gpu_offload():
        print("ERROR: llama-cpp-python compilé SANS CUDA (CPU-only).")
        print('  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86;120" \\')
        print('    pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python==0.3.22"')
        sys.exit(1)

    kwargs = dict(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        flash_attn=True,
        verbose=verbose,
        split_mode=split_mode,
    )
    if tensor_split is not None:
        kwargs["tensor_split"] = tensor_split

    old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print(f"\n  Chargement : n_gpu_layers={n_gpu_layers}, GPUs={gpu_ids}, "
          f"split_mode={split_mode}, tensor_split={tensor_split}")
    t0 = time.time()
    llm = Llama(**kwargs)
    load_time = time.time() - t0
    print(f"  Chargé en {load_time:.1f}s")

    if old_cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    return llm


def run_bench(llm, label: str, n_warmup: int = 1) -> dict:
    print(f"\n  {'-'*50}")
    print(f"  Benchmark : {label}")
    print(f"  {'-'*50}")

    latencies = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.0, echo=False)
        dt = time.perf_counter() - t0
        n_tok = out["usage"]["completion_tokens"] if "usage" in out else MAX_TOKENS
        tok_s = n_tok / dt if dt > 0 else 0
        if i >= n_warmup:
            latencies.append((n_tok, dt))
            print(f"    [{i+1}] {n_tok} tokens en {dt:.2f}s -> {tok_s:.1f} tok/s")
        else:
            print(f"    [warmup] {n_tok} tokens en {dt:.2f}s -> {tok_s:.1f} tok/s (ignoré)")

    tot_tok = sum(n for n, _ in latencies)
    tot_dt = sum(d for _, d in latencies)
    avg_tokps = tot_tok / tot_dt if tot_dt > 0 else 0
    print(f"\n  -> Moyenne : {avg_tokps:.1f} tok/s sur {len(latencies)} runs")
    return {"label": label, "avg_tok_s": round(avg_tokps, 1), "n_runs": len(latencies)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.6-35B-A3B Q4_K_M GGUF 2-GPU")
    parser.add_argument("--model", default=None, help="Chemin .gguf ou dossier")
    parser.add_argument("--only-dual", action="store_true", help="Skip single-GPU tests")
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    globals()["N_CTX"] = args.n_ctx
    globals()["MAX_TOKENS"] = args.max_tokens

    model_path = find_gguf(args.model or MODEL_DIR)

    import torch
    print("=" * 60)
    print("Qwen3.6-35B-A3B Q4_K_M GGUF — Benchmark VRAMancer 2-GPU")
    print("=" * 60)
    print(f"\nPyTorch : {torch.__version__}")
    gpus = get_vram_info()
    for g in gpus:
        print(f"  GPU {g['id']}: {g['name']} — {g['free_gb']:.1f}/{g['total_gb']:.1f} GB free")
    print(f"\nModèle  : {model_path}")
    print(f"Taille  : {os.path.getsize(model_path)/1e9:.1f} GB (shard 1)")
    print(f"Context : {N_CTX} tokens, max_tokens={MAX_TOKENS}")

    results = []

    # ── TEST 1 : 2-GPU full offload (showcase) ───────────────────────────────
    ts = compute_tensor_split(gpus)
    print("\n" + "=" * 60)
    print("TEST 1 : 2-GPU full offload (n_gpu_layers=-1)")
    print(f"  tensor_split={ts} (proportionnel VRAM libre)")
    print("=" * 60)
    llm_dual = load_llamacpp(
        model_path, n_gpu_layers=-1, gpu_ids=None,
        tensor_split=ts, split_mode=1, n_ctx=N_CTX, verbose=args.verbose,
    )
    print_vram("après chargement dual-GPU")
    results.append(run_bench(llm_dual, "2-GPU full offload"))
    del llm_dual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.only_dual:
        # ── TEST 2 : Single GPU 0 ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"TEST 2 : Single GPU 0 — {gpus[0]['name']}")
        print("=" * 60)
        llm0 = load_llamacpp(
            model_path, n_gpu_layers=-1, gpu_ids=[0],
            tensor_split=None, split_mode=0, n_ctx=N_CTX, verbose=args.verbose,
        )
        print_vram("après chargement GPU 0 seul")
        results.append(run_bench(llm0, f"Single GPU 0 ({gpus[0]['name']})"))
        del llm0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Résumé ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ — Qwen3.6-35B-A3B Q4_K_M GGUF")
    print("=" * 60)
    print(f"  {'Configuration':<40} {'tok/s':>8}")
    print(f"  {'-'*48}")
    for r in results:
        print(f"  {r['label']:<40} {r['avg_tok_s']:>7.1f}")
    if len(results) >= 2:
        dual = results[0]["avg_tok_s"]
        single_best = max(r["avg_tok_s"] for r in results[1:])
        if single_best > 0:
            print(f"\n  Speedup 2-GPU vs best single : {dual/single_best:.2f}x")
    print()


if __name__ == "__main__":
    main()
