#!/usr/bin/env python3
"""Benchmark: Fixed Speculative Decoding — quick test without torch.compile.

Measures the overhead of speculative decoding to determine if it can
beat TurboEngine. No torch.compile to get fast iteration.
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
BENCH_ROUNDS = 3

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

print(f"GPU: {gpu_name} ({GPU})")
print(f"NO torch.compile — measuring raw speculative decode overhead")
print()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load draft model (fp16)
print("Loading draft: Qwen2.5-0.5B-Instruct fp16...")
draft_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16,
    device_map={"": GPU}, trust_remote_code=True,
)
draft_model.eval()
print(f"  Draft VRAM: {torch.cuda.memory_allocated(GPU_IDX)/1e9:.1f} GB")

# Load main model (NF4)
print("Loading main: Qwen2.5-7B-Instruct NF4...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
main_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config,
    device_map={"": GPU}, trust_remote_code=True,
)
main_model.eval()
total_vram = torch.cuda.memory_allocated(GPU_IDX) / 1e9
print(f"  Total VRAM: {total_vram:.1f} GB")
print()


def bench(name, gen_fn, rounds=BENCH_ROUNDS):
    """Benchmark a generation function."""
    # warmup
    gen_fn()
    torch.cuda.synchronize()
    times = []
    text = ""
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = gen_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    n_tok = len(tokenizer.encode(text))
    tps = n_tok / avg
    return tps, n_tok, avg, text


# 1. TurboEngine reference (no compile) — pure autoregressive
print("=" * 65)
print("[1] TurboEngine (no compile) — reference")
print("=" * 65)
turbo = TurboEngine(main_model, tokenizer, device=GPU, compile=False)
tps_turbo, n_turbo, avg_turbo, text_turbo = bench(
    "Turbo", lambda: turbo.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
)
print(f"  TurboEngine: {tps_turbo:.1f} tok/s ({n_turbo} tok, {avg_turbo:.3f}s)")
print(f'    "{text_turbo[:80]}..."')
print()

# 2. Speculative gamma=3
print("=" * 65)
print("[2] SpeculativeTurboEngine gamma=3 (no compile)")
print("=" * 65)
spec3 = SpeculativeTurboEngine(
    main_model=main_model, draft_model=draft_model,
    tokenizer=tokenizer, device=GPU, gamma=3,
    compile_main=False, compile_draft=False,
)
tps_s3, n_s3, avg_s3, text_s3 = bench(
    "Spec3", lambda: (
        setattr(spec3, 'total_drafted', 0) or
        setattr(spec3, 'total_accepted', 0) or
        setattr(spec3, 'total_rounds', 0) or
        spec3.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    )
)
s3 = spec3.stats
print(f"  Speculative γ=3: {tps_s3:.1f} tok/s ({n_s3} tok, {avg_s3:.3f}s)")
print(f"  Acceptance: {s3['acceptance_rate']*100:.1f}% ({s3['accepted']}/{s3['drafted']})")
print(f'    "{text_s3[:80]}..."')
print()

# 3. Speculative gamma=5
print("=" * 65)
print("[3] SpeculativeTurboEngine gamma=5 (no compile)")
print("=" * 65)
spec5 = SpeculativeTurboEngine(
    main_model=main_model, draft_model=draft_model,
    tokenizer=tokenizer, device=GPU, gamma=5,
    compile_main=False, compile_draft=False,
)
tps_s5, n_s5, avg_s5, text_s5 = bench(
    "Spec5", lambda: (
        setattr(spec5, 'total_drafted', 0) or
        setattr(spec5, 'total_accepted', 0) or
        setattr(spec5, 'total_rounds', 0) or
        spec5.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    )
)
s5 = spec5.stats
print(f"  Speculative γ=5: {tps_s5:.1f} tok/s ({n_s5} tok, {avg_s5:.3f}s)")
print(f"  Acceptance: {s5['acceptance_rate']*100:.1f}% ({s5['accepted']}/{s5['drafted']})")
print(f'    "{text_s5[:80]}..."')
print()

# 4. Speculative gamma=8
print("=" * 65)
print("[4] SpeculativeTurboEngine gamma=8 (no compile)")
print("=" * 65)
spec8 = SpeculativeTurboEngine(
    main_model=main_model, draft_model=draft_model,
    tokenizer=tokenizer, device=GPU, gamma=8,
    compile_main=False, compile_draft=False,
)
tps_s8, n_s8, avg_s8, text_s8 = bench(
    "Spec8", lambda: (
        setattr(spec8, 'total_drafted', 0) or
        setattr(spec8, 'total_accepted', 0) or
        setattr(spec8, 'total_rounds', 0) or
        spec8.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    )
)
s8 = spec8.stats
print(f"  Speculative γ=8: {tps_s8:.1f} tok/s ({n_s8} tok, {avg_s8:.3f}s)")
print(f"  Acceptance: {s8['acceptance_rate']*100:.1f}% ({s8['accepted']}/{s8['drafted']})")
print(f'    "{text_s8[:80]}..."')
print()

# Summary
print("=" * 65)
print(f"SUMMARY — Speculative Decoding on {gpu_name}")
print("=" * 65)
print(f"  TurboEngine (ref):    {tps_turbo:>7.1f} tok/s  (1.00x)")
print(f"  Speculative γ=3:      {tps_s3:>7.1f} tok/s  ({tps_s3/tps_turbo:.2f}x)  accept={s3['acceptance_rate']*100:.0f}%")
print(f"  Speculative γ=5:      {tps_s5:>7.1f} tok/s  ({tps_s5/tps_turbo:.2f}x)  accept={s5['acceptance_rate']*100:.0f}%")
print(f"  Speculative γ=8:      {tps_s8:>7.1f} tok/s  ({tps_s8/tps_turbo:.2f}x)  accept={s8['acceptance_rate']*100:.0f}%")
print(f"  VRAM total: {total_vram:.1f} GB")
print()
