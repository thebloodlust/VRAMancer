#!/usr/bin/env python3
"""Benchmark: RTX 5070 Ti (Blackwell SM_120) — NF4 and fp16 with TurboEngine.

Qwen2.5-7B-Instruct on cuda:1 (RTX 5070 Ti, 16GB).
fp16 is tight (~15.2GB) — may OOM. NF4 should fit easily (~5GB).
"""
import gc
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.turbo_engine import TurboEngine

PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
BENCH_ROUNDS = 5

# Force RTX 5070 Ti — find it by name
GPU_IDX = None
for i in range(torch.cuda.device_count()):
    if "5070" in torch.cuda.get_device_name(i):
        GPU_IDX = i
        break
if GPU_IDX is None:
    print("ERROR: RTX 5070 Ti not found!")
    sys.exit(1)
GPU = f"cuda:{GPU_IDX}"


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
    torch.cuda.reset_peak_memory_stats(GPU_IDX)


def vram_gb():
    return torch.cuda.memory_allocated(GPU_IDX) / 1e9


results = {}
gpu_name = torch.cuda.get_device_name(GPU_IDX)
total_vram = torch.cuda.get_device_properties(GPU_IDX).total_memory / 1e9
print(f"GPU: {gpu_name} ({GPU}, {total_vram:.1f} GB)")
print(f"Arch: SM_{torch.cuda.get_device_capability(GPU_IDX)[0]}{torch.cuda.get_device_capability(GPU_IDX)[1]}")
print()

# ═══════════════════════════════════════════════════════════════
# 1. BnB NF4
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[1/2] BnB NF4 — Qwen2.5-7B-Instruct on", gpu_name)
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
# 2. fp16 full precision
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("[2/2] fp16 — Qwen2.5-7B-Instruct on", gpu_name)
print("  (WARNING: ~15.2 GB needed, 16 GB available — may OOM)")
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
print(f"SUMMARY — Qwen2.5-7B-Instruct on {gpu_name}")
print("=" * 65)
print(f"{'Config':<25s} {'tok/s':>8s} {'VRAM':>8s} {'vs HF':>8s}")
print("-" * 55)
for label in ["NF4 HF", "NF4 Turbo", "fp16 HF", "fp16 Turbo"]:
    if label in results:
        base_key = label.split()[0] + " HF"
        base = results.get(base_key, results[label])
        speedup = results[label] / base if base > 0 else 0
        vram_key = label.split()[0] + " VRAM"
        vram = results.get(vram_key, 0)
        print(f"  {label:<23s} {results[label]:>7.1f} {vram:>6.1f}GB {speedup:>6.2f}x")

# Cross-GPU comparison (hardcoded RTX 3090 results from previous runs)
print()
print("Cross-GPU comparison (RTX 3090 from previous run):")
print("-" * 55)
ref = {"NF4 HF": 20.5, "NF4 Turbo": 29.4, "fp16 HF": 36.5, "fp16 Turbo": 49.1}
for label in ["NF4 HF", "NF4 Turbo", "fp16 HF", "fp16 Turbo"]:
    if label in results:
        ratio = results[label] / ref[label]
        print(f"  {label:<23s}  5070Ti:{results[label]:>6.1f}  3090:{ref[label]:>6.1f}  ratio:{ratio:>5.2f}x")
print()
