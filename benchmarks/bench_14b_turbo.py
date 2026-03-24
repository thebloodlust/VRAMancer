#!/usr/bin/env python3
"""Benchmark: 14B 2-GPU TurboEngine — VRAMancer's core value proposition.

Qwen2.5-14B-Instruct (28GB bf16) doesn't fit on any single GPU.
VRAMancer splits it across RTX 3090 (24GB) + RTX 5070 Ti (16GB).

Compares:
1. HF generate() with accelerate device_map
2. MultiGPUTurboEngine (custom decode loop, no compile)

torch.compile is NOT used for multi-GPU because accelerate's dispatch
hooks (AlignDevicesHook) need to intercept each layer forward for
cross-device tensor transfers.
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
from core.turbo_engine import create_turbo_engine

MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
BENCH_ROUNDS = 3
PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function that implements merge sort with comments.",
    "What are the key differences between TCP and UDP protocols?",
]

# GPU info
print("=" * 70)
print(" VRAMancer 14B 2-GPU TurboEngine Benchmark")
print("=" * 70)
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {name} — {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
print()

# ── Load model across 2 GPUs ──
print(f"Loading {MODEL} across 2 GPUs (bf16, accelerate device_map)...")
t0_load = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
load_time = time.perf_counter() - t0_load

# Print device map
device_map = getattr(model, "hf_device_map", {})
device_counts = {}
for name_layer, dev in device_map.items():
    dev_str = str(dev)
    device_counts[dev_str] = device_counts.get(dev_str, 0) + 1
print(f"  Loaded in {load_time:.1f}s")
print(f"  Device distribution: {dict(device_counts)}")
for i in range(torch.cuda.device_count()):
    used = torch.cuda.memory_allocated(i) / 1e9
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {used:.1f} GB / {total:.1f} GB")
print()

# ── 1. HF generate() baseline ──
print("=" * 70)
print("[1/3] HF model.generate() — baseline")
print("=" * 70)

# warmup
with torch.no_grad():
    inp = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    model.generate(**inp, max_new_tokens=10, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

times_hf = []
text_hf = ""
for prompt in PROMPTS:
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = out.shape[1] - inp["input_ids"].shape[1]
    times_hf.append((n_gen, elapsed))
    if not text_hf:
        text_hf = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

total_tok_hf = sum(t[0] for t in times_hf)
total_time_hf = sum(t[1] for t in times_hf)
tps_hf = total_tok_hf / total_time_hf
print(f"  HF generate: {tps_hf:.1f} tok/s ({total_tok_hf} tok, {total_time_hf:.2f}s)")
print(f'    "{text_hf[:100]}..."')
print()

# ── 2. MultiGPUTurboEngine ──
print("=" * 70)
print("[2/3] MultiGPUTurboEngine (custom decode loop)")
print("=" * 70)

engine = create_turbo_engine(model, tokenizer, max_seq_len=4096)
print(f"  Engine type: {type(engine).__name__}")

# warmup
engine.generate(PROMPT, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

times_turbo = []
text_turbo = ""
for prompt in PROMPTS:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = engine.generate(prompt, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n_gen = len(tokenizer.encode(text))
    times_turbo.append((n_gen, elapsed))
    if not text_turbo:
        text_turbo = text

total_tok_turbo = sum(t[0] for t in times_turbo)
total_time_turbo = sum(t[1] for t in times_turbo)
tps_turbo = total_tok_turbo / total_time_turbo
print(f"  MultiGPUTurbo: {tps_turbo:.1f} tok/s ({total_tok_turbo} tok, {total_time_turbo:.2f}s)")
print(f'    "{text_turbo[:100]}..."')
print()

# ── 3. Also try NF4 single-GPU if possible (on RTX 3090) ──
print("=" * 70)
print("[3/3] NF4 single GPU (RTX 3090) — if it fits")
print("=" * 70)

# Free 14B model
del model, engine
gc.collect()
torch.cuda.empty_cache()

# Find the 3090
gpu_3090 = None
for i in range(torch.cuda.device_count()):
    if "3090" in torch.cuda.get_device_name(i):
        gpu_3090 = i
        break

tps_nf4 = 0
text_nf4 = ""
if gpu_3090 is not None:
    try:
        print(f"  Loading 14B NF4 on GPU {gpu_3090} (RTX 3090)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_nf4 = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=bnb_config,
            device_map={"": f"cuda:{gpu_3090}"},
            trust_remote_code=True,
        )
        model_nf4.eval()
        used = torch.cuda.memory_allocated(gpu_3090) / 1e9
        print(f"  NF4 VRAM: {used:.1f} GB")

        # warmup
        with torch.no_grad():
            inp = tokenizer(PROMPT, return_tensors="pt").to(f"cuda:{gpu_3090}")
            model_nf4.generate(**inp, max_new_tokens=10, do_sample=False, use_cache=True)
            torch.cuda.synchronize()

        times_nf4 = []
        for prompt in PROMPTS:
            inp = tokenizer(prompt, return_tensors="pt").to(f"cuda:{gpu_3090}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model_nf4.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_gen = out.shape[1] - inp["input_ids"].shape[1]
            times_nf4.append((n_gen, elapsed))
            if not text_nf4:
                text_nf4 = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

        total_tok_nf4 = sum(t[0] for t in times_nf4)
        total_time_nf4 = sum(t[1] for t in times_nf4)
        tps_nf4 = total_tok_nf4 / total_time_nf4
        print(f"  NF4 single-GPU: {tps_nf4:.1f} tok/s ({total_tok_nf4} tok, {total_time_nf4:.2f}s)")
        print(f'    "{text_nf4[:100]}..."')

        del model_nf4
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  NF4 single-GPU failed: {e}")
else:
    print("  No RTX 3090 found, skipping NF4 test")
print()

# ── Summary ──
print("=" * 70)
print(f" SUMMARY — {MODEL.split('/')[-1]} (14B)")
print("=" * 70)
print(f"  BF16 2-GPU HF generate:        {tps_hf:>7.1f} tok/s  (1.00x)")
print(f"  BF16 2-GPU MultiGPUTurbo:      {tps_turbo:>7.1f} tok/s  ({tps_turbo/tps_hf:.2f}x)")
if tps_nf4 > 0:
    print(f"  NF4 single-GPU (3090):         {tps_nf4:>7.1f} tok/s  ({tps_nf4/tps_hf:.2f}x)")
print()
print("  KEY: BF16 14B (28GB) DOESN'T FIT on any single GPU →")
print("       VRAMancer enables inference by splitting across heterogeneous GPUs")
if tps_nf4 > tps_hf:
    print(f"  NF4 on single GPU is {tps_nf4/tps_hf:.1f}x faster than BF16 2-GPU")
    print(f"  (avoids inter-GPU transfer overhead)")
print()
