#!/usr/bin/env python3
"""Benchmark: AWQ vs BnB NF4 vs fp16 — all with TurboEngine torch.compile.

Tests Qwen2.5-7B-Instruct across quantization methods on RTX 3090.
AWQ loaded via autoawq native API (bypasses transformers 5.3 gptqmodel requirement).
"""
import gc
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_OFFLINE"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.turbo_engine import TurboEngine

PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
BENCH_ROUNDS = 5
# Auto-detect largest GPU
_best_gpu = 0
_best_mem = 0
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_memory
    if mem > _best_mem:
        _best_mem = mem
        _best_gpu = i
GPU = f"cuda:{_best_gpu}"
GPU_IDX = _best_gpu


def measure_hf(model, tokenizer, device, label):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    for _ in range(2):
        model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
    times = []
    for _ in range(BENCH_ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    n_gen = out.shape[1] - inputs["input_ids"].shape[1]
    avg = sum(times) / len(times)
    tps = n_gen / avg
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  {label}: {tps:.1f} tok/s ({n_gen} tok, {avg:.3f}s)")
    print(f'    "{text[:80]}..."')
    return tps


def measure_turbo(model, tokenizer, device, label):
    engine = TurboEngine(model, tokenizer, device=device, compile=True)
    print(f"  Compiling + warmup...")
    engine.warmup(prompt=PROMPT, n_tokens=MAX_NEW, rounds=3)
    times = []
    for _ in range(BENCH_ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = engine.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    n_gen = len(tokenizer.encode(text))
    tps = n_gen / avg
    print(f"  {label}: {tps:.1f} tok/s ({n_gen} tok, {avg:.3f}s)")
    print(f'    "{text[:80]}..."')
    del engine
    gc.collect(); torch.cuda.empty_cache()
    return tps


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def vram_gb():
    return torch.cuda.memory_allocated(GPU_IDX) / 1e9


results = {}
gpu_name = torch.cuda.get_device_name(GPU_IDX)
print(f"GPU: {gpu_name} ({GPU})")
print(f"VRAM total: {torch.cuda.get_device_properties(GPU_IDX).total_memory / 1e9:.1f} GB")
print()

# ═══════════════════════════════════════════════════════════════
# 1. AWQ via autoawq native loader
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[1/3] AWQ — Qwen2.5-7B-Instruct-AWQ (autoawq native loader)")
print("=" * 65)
try:
    cleanup()
    from awq import AutoAWQForCausalLM
    model_path = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    awq_model = AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=True,
    )
    model = awq_model.model.to(GPU)
    vram = vram_gb()
    print(f"  VRAM: {vram:.1f} GB")

    tps_hf = measure_hf(model, tokenizer, GPU, "AWQ HF generate()")
    results["AWQ HF"] = tps_hf

    tps_turbo = measure_turbo(model, tokenizer, GPU, "AWQ TurboCompiled")
    results["AWQ Turbo"] = tps_turbo
    results["AWQ VRAM"] = vram

    del model, awq_model, tokenizer
    cleanup()
except Exception as e:
    print(f"  AWQ FAILED: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# 2. BnB NF4
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("[2/3] BnB NF4 — Qwen2.5-7B-Instruct")
print("=" * 65)
try:
    cleanup()
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=GPU,
    )
    vram = vram_gb()
    print(f"  VRAM: {vram:.1f} GB")

    tps_hf = measure_hf(model, tokenizer, GPU, "NF4 HF generate()")
    results["NF4 HF"] = tps_hf

    tps_turbo = measure_turbo(model, tokenizer, GPU, "NF4 TurboCompiled")
    results["NF4 Turbo"] = tps_turbo
    results["NF4 VRAM"] = vram

    del model, tokenizer
    cleanup()
except Exception as e:
    print(f"  NF4 FAILED: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# 3. fp16 full precision
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("[3/3] fp16 — Qwen2.5-7B-Instruct")
print("=" * 65)
try:
    cleanup()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map=GPU,
    )
    vram = vram_gb()
    print(f"  VRAM: {vram:.1f} GB")

    tps_hf = measure_hf(model, tokenizer, GPU, "fp16 HF generate()")
    results["fp16 HF"] = tps_hf

    tps_turbo = measure_turbo(model, tokenizer, GPU, "fp16 TurboCompiled")
    results["fp16 Turbo"] = tps_turbo
    results["fp16 VRAM"] = vram

    del model, tokenizer
    cleanup()
except Exception as e:
    print(f"  fp16 FAILED: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("SUMMARY — Qwen2.5-7B-Instruct on", gpu_name)
print("=" * 65)
print(f"{'Config':<25s} {'tok/s':>8s} {'VRAM':>8s} {'vs HF':>8s}")
print("-" * 55)
for label in ["AWQ HF", "AWQ Turbo", "NF4 HF", "NF4 Turbo", "fp16 HF", "fp16 Turbo"]:
    if label in results:
        base_key = label.split()[0] + " HF"
        base = results.get(base_key, results[label])
        speedup = results[label] / base if base > 0 else 0
        vram_key = label.split()[0] + " VRAM"
        vram = results.get(vram_key, 0)
        print(f"  {label:<23s} {results[label]:>7.1f} {vram:>6.1f}GB {speedup:>6.2f}x")
print()
