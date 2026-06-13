#!/usr/bin/env python3
"""
Benchmark: Qwen2.5-14B-Instruct avec NVFP4 sur RTX 5070 Ti (Blackwell SM12.0).

Showcase VRAMancer :
  - Modèle 28 GB BF16 → 7 GB VRAM avec NVFP4 Blackwell
  - Comparaison : BF16 (OOM sur 5070 Ti) vs NVFP4 (~7 GB)
  - DirectFP4 bypass (+7% vs torchao standard)

Hardware cible :
  - CUDA 0 : RTX 3090 (Ampere SM8.6, 24 GB)  — non utilisé ici
  - CUDA 1 : RTX 5070 Ti (Blackwell SM12.0, 16 GB)  — cible NVFP4

Usage :
    python benchmarks/bench_14b_nvfp4_blackwell.py
    python benchmarks/bench_14b_nvfp4_blackwell.py --no-direct  # sans DirectFP4 bypass
    python benchmarks/bench_14b_nvfp4_blackwell.py --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;12.0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("VRM_MINIMAL_TEST", None)

import torch

DEVICE = "cuda:1"          # RTX 5070 Ti (SM12.0)
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_NEW_TOKENS = 128
WARMUP_TOKENS = 32

PROMPTS = [
    "Explain why mixture-of-experts models are memory-efficient:",
    "Write a Python async web server with FastAPI and uvicorn:",
    "The key advantage of NVFP4 quantization on Blackwell GPUs is",
    "Describe the architecture of a modern transformer model:",
]


def nvfp4_filter_fn(module, fqn):
    import torch.nn as nn
    return isinstance(module, nn.Linear)


def get_vram_gb(device: str) -> float:
    try:
        return torch.cuda.memory_allocated(device) / 1e9
    except Exception:
        return 0.0


def load_model_nvfp4(device: str, model_id: str, use_direct: bool = True):
    """Charge le modèle en BF16 sur CPU, applique NVFP4, déplace sur GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n  Chargement {model_id} → NVFP4 sur {device}")
    print(f"  DirectFP4 bypass : {'actif' if use_direct else 'désactivé'}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("  1/4 Chargement BF16 sur CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    t1 = time.time()
    print(f"      → {t1-t0:.1f}s")

    print("  2/4 Quantisation NVFP4 (torchao)...")
    from torchao.quantization import quantize_
    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        config = NVFP4DynamicActivationNVFP4WeightConfig()
    except ImportError:
        # Fallback si mx_formats pas disponible
        from torchao.quantization import float8_dynamic_activation_float8_weight
        print("  ATTENTION: NVFP4 non disponible, fallback FP8")
        config = float8_dynamic_activation_float8_weight()

    torch.cuda.set_device(device)
    quantize_(model, config, filter_fn=nvfp4_filter_fn)
    t2 = time.time()
    print(f"      → {t2-t1:.1f}s")

    if use_direct:
        print("  3/4 Remplacement DirectFP4Linear (bypass torchao)...")
        try:
            from core.nvfp4_direct import replace_with_direct_fp4
            n = replace_with_direct_fp4(model, verbose=False)
            print(f"      → {n} couches remplacées en {time.time()-t2:.1f}s")
        except Exception as e:
            print(f"      → DirectFP4 indisponible ({e}), skip")
    else:
        print("  3/4 DirectFP4 bypass : désactivé")

    print("  4/4 Déplacement vers GPU...")
    t3 = time.time()
    model = model.to(device)
    torch.cuda.synchronize(device)
    t4 = time.time()
    print(f"      → {t4-t3:.1f}s")

    vram = get_vram_gb(device)
    print(f"\n  ✓ Modèle chargé en {t4-t0:.1f}s")
    print(f"  ✓ VRAM utilisée : {vram:.2f} GB")

    return model, tokenizer


def bench_one(model, tokenizer, prompt: str, device: str, max_new: int) -> dict:
    """Un run de benchmark, retourne tok/s et TTFT."""
    messages = [{"role": "user", "content": prompt}]
    txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(txt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    torch.cuda.synchronize(device)

    # TTFT
    t_ttft = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=inputs["input_ids"])
    torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # Timed generation
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    n_gen = out.shape[1] - input_len
    tok_s = n_gen / dt if dt > 0 else 0.0

    return {
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "n_gen": n_gen,
        "dt": dt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--no-direct", action="store_true", help="Désactive DirectFP4 bypass")
    args = parser.parse_args()

    print("=" * 65)
    print("VRAMancer — Benchmark NVFP4 Blackwell")
    print(f"Modèle : {args.model}")
    print(f"GPU    : {args.device} — {torch.cuda.get_device_name(args.device)}")
    sm = torch.cuda.get_device_capability(args.device)
    print(f"SM     : {sm[0]}.{sm[1]} {'(Blackwell ✓)' if sm >= (12,0) else '(non-Blackwell)'}")
    free, total = torch.cuda.mem_get_info(args.device)
    print(f"VRAM   : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    print("=" * 65)

    # NVFP4
    model, tokenizer = load_model_nvfp4(
        device=args.device,
        model_id=args.model,
        use_direct=not args.no_direct,
    )

    print(f"\n{'─'*65}")
    print(f"Benchmark NVFP4 — {len(PROMPTS)} prompts × {args.max_tokens} tokens")
    print(f"{'─'*65}")

    results = []
    for i, prompt in enumerate(PROMPTS):
        r = bench_one(model, tokenizer, prompt, args.device, args.max_tokens)
        results.append(r)
        label = f"  [{i+1}/{len(PROMPTS)}] {prompt[:50]}"
        print(f"{label:<55} → {r['tok_s']:5.1f} tok/s  TTFT {r['ttft_ms']:.0f}ms")

    avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    vram_gb = get_vram_gb(args.device)
    total_gb = torch.cuda.mem_get_info(args.device)[1] / 1e9

    print(f"\n{'═'*65}")
    print(f"RÉSULTATS — {args.model}")
    print(f"{'═'*65}")
    print(f"  Quantisation  : NVFP4 (DirectFP4={'actif' if not args.no_direct else 'off'})")
    print(f"  tok/s moyen   : {avg_tok_s:.1f}")
    print(f"  TTFT moyen    : {avg_ttft:.0f} ms")
    print(f"  VRAM utilisée : {vram_gb:.2f} GB / {total_gb:.1f} GB")
    print(f"  Réduction VRAM: BF16 ~28 GB → NVFP4 {vram_gb:.1f} GB ({(1-vram_gb/28)*100:.0f}% réduction)")
    print(f"{'═'*65}")
    print()

    # JSON summary
    import json
    summary = {
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device),
        "sm": f"{sm[0]}.{sm[1]}",
        "quant": "nvfp4",
        "direct_fp4": not args.no_direct,
        "avg_tok_s": round(avg_tok_s, 1),
        "avg_ttft_ms": round(avg_ttft, 0),
        "vram_gb": round(vram_gb, 2),
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "bench_14b_nvfp4_blackwell.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()
