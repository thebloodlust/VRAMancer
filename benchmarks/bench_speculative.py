#!/usr/bin/env python3
"""Benchmark: Speculative Decoding with TurboEngine.

Uses Qwen2.5-0.5B-Instruct as draft model + Qwen2.5-7B-Instruct NF4 as main.
Compares: HF generate, TurboEngine, SpeculativeTurboEngine.
"""
import gc
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.turbo_engine import TurboEngine, SpeculativeTurboEngine

PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
BENCH_ROUNDS = 5
GAMMA = 5  # draft tokens per round

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
gpu_name = torch.cuda.get_device_name(GPU_IDX)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(GPU_IDX)


def vram_gb():
    return torch.cuda.memory_allocated(GPU_IDX) / 1e9


print(f"GPU: {gpu_name} ({GPU}, {torch.cuda.get_device_properties(GPU_IDX).total_memory / 1e9:.1f} GB)")
print(f"Draft model: Qwen/Qwen2.5-0.5B-Instruct (fp16)")
print(f"Main model: Qwen/Qwen2.5-7B-Instruct (NF4)")
print(f"Gamma: {GAMMA}")
print()

# ═══════════════════════════════════════════════════════════════
# Load models
# ═══════════════════════════════════════════════════════════════
print("Loading draft model (0.5B fp16)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
draft_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map=GPU,
)
draft_vram = vram_gb()
print(f"  Draft VRAM: {draft_vram:.1f} GB")

print("Loading main model (7B NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
main_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map=GPU,
)
total_vram = vram_gb()
print(f"  Total VRAM (draft + main): {total_vram:.1f} GB")
print()

# ═══════════════════════════════════════════════════════════════
# 1. Baseline: NF4 HF generate()
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[1/3] NF4 HF generate() baseline")
print("=" * 65)
inputs = tokenizer(PROMPT, return_tensors="pt").to(GPU)
for _ in range(2):
    main_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()

times = []
for _ in range(BENCH_ROUNDS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = main_model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

n_gen = out.shape[1] - inputs["input_ids"].shape[1]
avg_hf = sum(times) / len(times)
tps_hf = n_gen / avg_hf
text_hf = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  NF4 HF: {tps_hf:.1f} tok/s ({n_gen} tok, {avg_hf:.3f}s)")
print(f'    "{text_hf[:80]}..."')
print()

# ═══════════════════════════════════════════════════════════════
# 2. TurboEngine (no speculation)
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("[2/3] TurboEngine (compiled, no speculation)")
print("=" * 65)
turbo = TurboEngine(main_model, tokenizer, device=GPU, compile=True)
print("  Compiling + warmup...")
turbo.warmup(prompt=PROMPT, n_tokens=MAX_NEW, rounds=3)

times = []
for _ in range(BENCH_ROUNDS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = turbo.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

avg_turbo = sum(times) / len(times)
n_turbo = len(tokenizer.encode(text))
tps_turbo = n_turbo / avg_turbo
print(f"  TurboEngine: {tps_turbo:.1f} tok/s ({n_turbo} tok, {avg_turbo:.3f}s)")
print(f'    "{text[:80]}..."')
del turbo
gc.collect(); torch.cuda.empty_cache()
print()

# ═══════════════════════════════════════════════════════════════
# 3. SpeculativeTurboEngine (draft=0.5B, main=7B NF4)
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print(f"[3/3] SpeculativeTurboEngine (gamma={GAMMA})")
print("=" * 65)
spec = SpeculativeTurboEngine(
    main_model=main_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    device=GPU,
    gamma=GAMMA,
    compile_main=True,   # Will detect already-compiled and skip
    compile_draft=True,
)
print("  Compiling + warmup (draft + main)...")
spec.warmup(prompt=PROMPT, n_tokens=MAX_NEW, rounds=3)

times = []
for _ in range(BENCH_ROUNDS):
    # Reset stats per round
    spec.total_drafted = 0
    spec.total_accepted = 0
    spec.total_rounds = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = spec.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

avg_spec = sum(times) / len(times)
n_spec = len(tokenizer.encode(text))
tps_spec = n_spec / avg_spec
stats = spec.stats
print(f"  Speculative: {tps_spec:.1f} tok/s ({n_spec} tok, {avg_spec:.3f}s)")
print(f"  Acceptance: {stats['acceptance_rate']*100:.1f}% ({stats['accepted']}/{stats['drafted']})")
print(f"  Effective tok/round: {stats['effective_tokens_per_round']:.1f}")
print(f'    "{text[:80]}..."')
print()

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print(f"SUMMARY — Speculative Decoding on {gpu_name}")
print("=" * 65)
print(f"  NF4 HF generate():    {tps_hf:>7.1f} tok/s  (1.00x)")
print(f"  TurboEngine:          {tps_turbo:>7.1f} tok/s  ({tps_turbo/tps_hf:.2f}x)")
print(f"  Speculative (γ={GAMMA}):   {tps_spec:>7.1f} tok/s  ({tps_spec/tps_hf:.2f}x)")
print(f"  Spec vs Turbo:        {tps_spec/tps_turbo:.2f}x")
print(f"  Acceptance rate:      {stats['acceptance_rate']*100:.1f}%")
print(f"  VRAM total:           {total_vram:.1f} GB")
print()
