#!/usr/bin/env python3
"""Quick AWQ-only benchmark using autoawq native loader."""
import gc, os, sys, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "0"

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.turbo_engine import TurboEngine

PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
ROUNDS = 5

# Auto-detect best GPU
best_gpu, best_mem = 0, 0
for i in range(torch.cuda.device_count()):
    m = torch.cuda.get_device_properties(i).total_memory
    if m > best_mem:
        best_mem = m
        best_gpu = i
GPU = f"cuda:{best_gpu}"
print(f"GPU: {torch.cuda.get_device_name(best_gpu)} ({GPU}, {best_mem/1e9:.1f} GB)")

# Load AWQ via autoawq
from awq import AutoAWQForCausalLM
model_id = "Qwen/Qwen2.5-7B-Instruct-AWQ"
print(f"Loading {model_id} via autoawq (fuse_layers=True)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
awq_model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True)
model = awq_model.model
# autoawq loads to cuda:0 by default
dev = str(next(model.parameters()).device)
print(f"  Model on: {dev}")
print(f"  VRAM: {torch.cuda.memory_allocated(int(dev.split(':')[-1])) / 1e9:.1f} GB")

# Use model's actual device
GPU = dev

# HF generate()
inputs = tokenizer(PROMPT, return_tensors="pt").to(GPU)
for _ in range(2):
    model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
times = []
for _ in range(ROUNDS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
avg = sum(times) / len(times)
n_gen = out.shape[1] - inputs["input_ids"].shape[1]
tps_hf = n_gen / avg
text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  AWQ HF generate(): {tps_hf:.1f} tok/s ({n_gen} tok, {avg:.3f}s)")
print(f'    "{text[:80]}..."')

# TurboEngine
print("  Compiling + warmup...")
engine = TurboEngine(model, tokenizer, device=GPU, compile=True)
engine.warmup(prompt=PROMPT, n_tokens=MAX_NEW, rounds=3)
times = []
for _ in range(ROUNDS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = engine.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
avg = sum(times) / len(times)
n_gen = len(tokenizer.encode(text))
tps_turbo = n_gen / avg
print(f"  AWQ TurboCompiled: {tps_turbo:.1f} tok/s ({n_gen} tok, {avg:.3f}s)")
print(f'    "{text[:80]}..."')

print()
print(f"SUMMARY — AWQ Qwen2.5-7B-Instruct on {torch.cuda.get_device_name(int(GPU.split(':')[-1]))}")
print(f"  AWQ HF:    {tps_hf:.1f} tok/s")
print(f"  AWQ Turbo: {tps_turbo:.1f} tok/s ({tps_turbo/tps_hf:.2f}x)")
