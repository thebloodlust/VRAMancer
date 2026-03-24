#!/usr/bin/env python3
"""Attempt to squeeze max tok/s from fp16 on RTX 3090.

Tests:
1. TurboEngine compile="default" (baseline: 49.1 tok/s)
2. Static KV cache with pre-allocated buffer
3. CUDA graphs for decode step
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Explain the theory of general relativity in simple terms."
MAX_NEW = 128
WARMUP = 5

print("=" * 70)
print(" VRAMancer — Maximum Throughput Experiment (fp16)")
print("=" * 70)
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Model: {MODEL}")
print()

# Load fp16 model
print("Loading fp16 model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16,
    device_map={"": "cuda:0"}, trust_remote_code=True,
)
model.eval()
vram = torch.cuda.memory_allocated(0) / 1e9
print(f"  VRAM: {vram:.1f} GB", flush=True)

# ── Helper: benchmark decode ──
def bench_decoded(desc, forward_fn, n_tokens=MAX_NEW, warmup=WARMUP):
    """Time n_tokens decode steps using forward_fn."""
    inp = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    
    # Prefill
    with torch.no_grad():
        out = model(inp["input_ids"], use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    
    # Warmup decode
    for _ in range(warmup):
        with torch.no_grad():
            out = forward_fn(next_tok, past_kv)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    
    # Reset for benchmark
    inp = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model(inp["input_ids"], use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1:].argmax(dim=-1)
    
    tokens = [next_tok.item()]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(n_tokens):
            out = forward_fn(next_tok, past_kv)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            tokens.append(next_tok.item())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    
    tps = n_tokens / elapsed
    text = tokenizer.decode(tokens[:50], skip_special_tokens=True)
    print(f"  {desc}: {tps:.1f} tok/s ({n_tokens} tok, {elapsed:.2f}s)", flush=True)
    print(f'    "{text[:80]}..."', flush=True)
    return tps

# ── 1. Baseline: raw model forward ──
print("\n[1/4] Baseline (uncompiled model.forward)", flush=True)
tps_base = bench_decoded("Baseline", lambda tok, kv: model(tok, past_key_values=kv, use_cache=True))

# ── 2. TurboEngine default compile (existing approach) ──
print("\n[2/4] TurboEngine (compile=default)", flush=True)
from core.turbo_engine import TurboEngine
engine = TurboEngine(model, tokenizer, device="cuda:0", max_seq_len=4096, compile=True)

# Warmup compile
engine.generate(PROMPT, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

t0 = time.perf_counter()
text = engine.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
n_gen = len(tokenizer.encode(text))
tps_turbo = n_gen / elapsed
print(f"  TurboEngine: {tps_turbo:.1f} tok/s ({n_gen} tok, {elapsed:.2f}s)", flush=True)
print(f'    "{text[:80]}..."', flush=True)

# ── 3. CUDA stream with non-blocking transfers ──
print("\n[3/4] CUDA stream optimization", flush=True)
# Create a dedicated compute stream
compute_stream = torch.cuda.Stream()

def forward_stream(tok, kv):
    with torch.cuda.stream(compute_stream):
        out = model(tok, past_key_values=kv, use_cache=True)
    compute_stream.synchronize()
    return out

tps_stream = bench_decoded("Stream", forward_stream)

# ── 4. torch.compile with reduce-overhead (CUDA graphs) ──
print("\n[4/4] torch.compile reduce-overhead (CUDA graphs)", flush=True)
try:
    # Replace inner model with reduce-overhead compiled version
    original_inner = model.model
    model.model = torch.compile(original_inner, mode="reduce-overhead", fullgraph=False)
    
    # Warmup: 3 full forward passes to trigger CUDA graph capture
    inp = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    print("  Warmup (CUDA graph capture)...", flush=True)
    for i in range(3):
        with torch.no_grad():
            out = model(inp["input_ids"], use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:].argmax(dim=-1)
        with torch.no_grad():
            for _ in range(5):
                out = model(tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                tok = out.logits[:, -1:].argmax(dim=-1)
        del out, past, tok
    torch.cuda.synchronize()
    print("  reduce-overhead compile OK, running benchmark...", flush=True)
    
    def forward_reduce(tok, kv):
        return model(tok, past_key_values=kv, use_cache=True)
    
    tps_cudagraph = bench_decoded("ReduceOverhead", forward_reduce)
    
    # Restore original
    model.model = original_inner
except Exception as e:
    print(f"  reduce-overhead FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()
    tps_cudagraph = 0
    try:
        model.model = original_inner
    except:
        pass

# ── Summary ──
print("\n" + "=" * 70)
print(" SUMMARY — fp16 Qwen2.5-7B on RTX 3090")
print("=" * 70)
print(f"  Baseline (uncompiled):     {tps_base:>6.1f} tok/s (1.00x)")
print(f"  TurboEngine (compiled):    {tps_turbo:>6.1f} tok/s ({tps_turbo/tps_base:.2f}x)")
print(f"  Stream optimization:       {tps_stream:>6.1f} tok/s ({tps_stream/tps_base:.2f}x)")
if tps_cudagraph > 0:
    print(f"  reduce-overhead:           {tps_cudagraph:>6.1f} tok/s ({tps_cudagraph/tps_base:.2f}x)")
print()
