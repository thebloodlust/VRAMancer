#!/usr/bin/env python3
"""Ultimate benchmark: compile(reduce-overhead) + StaticCache on fp16 7B.

This is the triple-win combo:
1. Inductor kernel fusion (torch.compile)
2. CUDA Graph capture (reduce-overhead) 
3. Static memory layout (StaticCache)

Previously crashed with DynamicCache. StaticCache is designed for this.
"""
import torch, time, os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_float32_matmul_precision('high')

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
MAX_NEW = 128
MAX_CACHE = 512
PROMPT = "Explain quantum computing in simple terms:"

def reset_cache(sc):
    for layer in sc.layers:
        if hasattr(layer, 'keys') and layer.is_initialized:
            layer.keys.zero_()
            layer.values.zero_()

print(f"=== Ultimate Benchmark: {MODEL} fp16 ===")
print(f"    compile modes × StaticCache\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map=DEV, torch_dtype=torch.float16,
)
model.eval()
cfg = model.config
ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB\n")

g_inp = torch.zeros(1, 1, dtype=torch.long, device=DEV)
g_pos = torch.zeros(1, dtype=torch.long, device=DEV)

def run_decode(fwd_fn, n=MAX_NEW, label=""):
    """Run decode loop with a given forward function + StaticCache."""
    sc = StaticCache(config=cfg, max_cache_len=MAX_CACHE)
    cp = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = fwd_fn(input_ids=ids, past_key_values=sc,
                    cache_position=cp, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        t = logits.argmax(dim=-1)
    
    gen = [t.item()]
    for i in range(n - 1):
        g_inp[0, 0] = t.item()
        g_pos.fill_(plen + i)
        with torch.no_grad():
            out = fwd_fn(input_ids=g_inp, past_key_values=sc,
                        cache_position=g_pos, use_cache=True)
            logits = model.lm_head(out.last_hidden_state)
            t = logits.argmax(dim=-1)
        gen.append(t.item())
        if tokenizer.eos_token_id and t.item() == tokenizer.eos_token_id:
            break
    return gen

def benchmark(fwd_fn, label, warmup=3, runs=5, n=MAX_NEW):
    # Warmup
    for _ in range(warmup):
        run_decode(fwd_fn, n=5)
        torch.cuda.synchronize()
    
    ts = []
    for _ in range(runs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen = run_decode(fwd_fn, n=n)
        torch.cuda.synchronize()
        ts.append(len(gen) / (time.perf_counter() - t0))
    
    avg = sum(ts) / len(ts)
    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"  {label}: {avg:.1f} tok/s")
    print(f'    "{text[:70]}..."')
    return avg

results = {}

# ── 1. Eager + StaticCache ──
print("[1] Eager + StaticCache...")
r1 = benchmark(model.model.forward, "eager+StaticCache")
results["eager+StaticCache"] = r1

# ── 2. compile(default) + StaticCache ──
print("\n[2] compile(default) + StaticCache...")
compiled_default = torch.compile(model.model.forward, mode="default", fullgraph=False)
print("  Compiling JIT warmup...")
r2 = benchmark(compiled_default, "compile(default)+StaticCache", warmup=5)
results["compile(default)+StaticCache"] = r2

# ── 3. compile(reduce-overhead) + StaticCache ──
print("\n[3] compile(reduce-overhead) + StaticCache...")
try:
    # Need a fresh compile — can't recompile over existing
    # torch._dynamo.reset()
    compiled_ro = torch.compile(model.model.forward, mode="reduce-overhead", fullgraph=False)
    print("  Compiling JIT warmup (CUDA Graph capture)...")
    r3 = benchmark(compiled_ro, "compile(reduce-overhead)+StaticCache", warmup=5)
    results["compile(reduce-overhead)+StaticCache"] = r3
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    r3 = 0

# ── 4. Manual CUDA Graph + StaticCache ──
print("\n[4] Manual CUDA Graph + StaticCache...")
try:
    sc_graph = StaticCache(config=cfg, max_cache_len=MAX_CACHE)
    
    # Prefill
    cp = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=sc_graph,
                         cache_position=cp, use_cache=True)
        first = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
    
    # Warmup decode
    g_inp.copy_(first.view(1, 1))
    for wi in range(3):
        g_pos.fill_(plen + wi)
        with torch.no_grad():
            out = model.model(input_ids=g_inp, past_key_values=sc_graph,
                             cache_position=g_pos, use_cache=True)
            g_inp.copy_(model.lm_head(out.last_hidden_state).argmax(dim=-1).view(1, 1))
    torch.cuda.synchronize()
    
    # Reset & re-prefill
    reset_cache(sc_graph)
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=sc_graph,
                         cache_position=cp, use_cache=True)
        first = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
    g_inp.copy_(first.view(1, 1))
    g_pos.fill_(plen)
    with torch.no_grad():
        out = model.model(input_ids=g_inp, past_key_values=sc_graph,
                         cache_position=g_pos, use_cache=True)
        model.lm_head(out.last_hidden_state)
    g_pos.fill_(plen + 1)
    g_inp.copy_(model.lm_head(out.last_hidden_state).argmax(dim=-1).view(1, 1))
    torch.cuda.synchronize()
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.model(input_ids=g_inp, past_key_values=sc_graph,
                         cache_position=g_pos, use_cache=True)
        g_out = model.lm_head(out.last_hidden_state)
    torch.cuda.synchronize()
    print("  CUDA Graph captured")
    
    def run_graph(n=MAX_NEW):
        reset_cache(sc_graph)
        cp = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = model.model(input_ids=ids, past_key_values=sc_graph,
                             cache_position=cp, use_cache=True)
            tok = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
        gen = [tok.item()]
        g_inp.copy_(tok.view(1, 1))
        for i in range(n - 1):
            g_pos.fill_(plen + i)
            graph.replay()
            tok_val = g_out[:, -1:, :].argmax(dim=-1).item()
            gen.append(tok_val)
            g_inp[0, 0] = tok_val
            if tokenizer.eos_token_id and tok_val == tokenizer.eos_token_id:
                break
        return gen
    
    run_graph(5); torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen = run_graph(MAX_NEW)
        torch.cuda.synchronize()
        ts.append(len(gen) / (time.perf_counter() - t0))
    r4 = sum(ts) / len(ts)
    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"  CUDA Graph: {r4:.1f} tok/s")
    print(f'    "{text[:70]}..."')
    results["CUDAGraph+StaticCache"] = r4
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    r4 = 0

# ── Summary ──
print(f"\n{'='*60}")
print(f"Qwen2.5-7B-Instruct fp16 on RTX 3090:")
for k, v in results.items():
    print(f"  {k:40s} {v:6.1f} tok/s")
print(f"\nPrevious results:")
print(f"  {'HF generate (fp16) baseline':40s}  37.2 tok/s")
print(f"  {'TurboEngine compile(default)+DynCache':40s}  49.0 tok/s")
print(f"  {'BnB NF4 compile(default)+StaticCache':40s}  30.9 tok/s")
print(f"  {'Ollama target':40s} 60-100 tok/s")
