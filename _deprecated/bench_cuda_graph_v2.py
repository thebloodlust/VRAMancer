#!/usr/bin/env python3
"""CUDA Graph decode engine — the ultimate weapon.

Approach: StaticCache + manual CUDA Graph capture.
Eliminates 73% Python overhead (35ms/token on NF4 7B).
Target: 48ms → ~13ms/token = ~77 tok/s.
"""
import torch
import time
import os
import sys
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_float32_matmul_precision('high')

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    StaticCache,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
MAX_NEW = 128
MAX_CACHE = 512
PROMPT = "Explain quantum computing in simple terms:"

print(f"=== CUDA Graph Decode: {MODEL} NF4 ===\n")

# ── Load model ──
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map=DEV, torch_dtype=torch.float16,
)
model.eval()
print(f"VRAM after load: {torch.cuda.memory_allocated(0)/1e9:.1f} GB\n")

ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]

# ── Step 1: Check API ──
print("Step 1: Checking StaticCache API...")
import inspect
sc_params = list(inspect.signature(StaticCache.__init__).parameters.keys())
print(f"  StaticCache params: {sc_params}")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
qwen_params = list(inspect.signature(Qwen2Model.forward).parameters.keys())
print(f"  Qwen2Model.forward params: {qwen_params}")

# ── Step 2: Create StaticCache ──
print("\nStep 2: Creating StaticCache...")
cfg = model.config
static_cache = StaticCache(
    config=cfg,
    batch_size=1,
    max_cache_len=MAX_CACHE,
    device=torch.device(DEV),
    dtype=torch.float16,
)
print(f"  StaticCache created: max_cache_len={MAX_CACHE}")
vram_after_cache = torch.cuda.memory_allocated(0) / 1e9
print(f"  VRAM after cache: {vram_after_cache:.1f} GB")

# ── Step 3: Test prefill with StaticCache ──
print("\nStep 3: Testing prefill with StaticCache...")
try:
    cache_pos_prefill = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model.model(
            input_ids=ids,
            past_key_values=static_cache,
            cache_position=cache_pos_prefill,
            use_cache=True,
        )
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    next_token = logits.argmax(dim=-1)
    print(f"  Prefill OK: next_token={next_token.item()}")
    first_word = tokenizer.decode([next_token.item()])
    print(f"  First decoded: '{first_word}'")
except Exception as e:
    print(f"  PREFILL FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 4: Test decode step with StaticCache ──
print("\nStep 4: Testing decode with StaticCache...")
try:
    static_input = torch.zeros(1, 1, dtype=torch.long, device=DEV)
    static_pos = torch.zeros(1, dtype=torch.long, device=DEV)
    
    # Run a few decode steps eagerly
    pos = plen
    cur_token = next_token
    for i in range(5):
        static_input.copy_(cur_token.view(1, 1))
        static_pos.fill_(pos)
        with torch.no_grad():
            out = model.model(
                input_ids=static_input,
                past_key_values=static_cache,
                cache_position=static_pos,
                use_cache=True,
            )
        logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        cur_token = logits.argmax(dim=-1)
        pos += 1
        w = tokenizer.decode([cur_token.item()])
        print(f"  Decode step {i}: token={cur_token.item()} '{w}'")
    print("  Eager decode OK!")
except Exception as e:
    print(f"  DECODE FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 5: Benchmark eager decode with StaticCache ──
print("\nStep 5: Benchmark eager StaticCache decode...")
def run_eager_decode(n_tokens=MAX_NEW):
    """Run full generation with StaticCache (no CUDA Graph)."""
    # Reset cache
    sc = StaticCache(
        config=cfg, batch_size=1, max_cache_len=MAX_CACHE,
        device=torch.device(DEV), dtype=torch.float16,
    )
    cp = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=sc,
                         cache_position=cp, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        tok = logits.argmax(dim=-1)
    
    inp = torch.zeros(1, 1, dtype=torch.long, device=DEV)
    cpos = torch.zeros(1, dtype=torch.long, device=DEV)
    generated = [tok.item()]
    
    for i in range(n_tokens - 1):
        p = plen + i
        inp[0, 0] = tok.item()
        cpos[0] = p
        with torch.no_grad():
            out = model.model(input_ids=inp, past_key_values=sc,
                             cache_position=cpos, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            tok = logits.argmax(dim=-1)
        generated.append(tok.item())
        eos = getattr(tokenizer, 'eos_token_id', None)
        if eos and tok.item() == eos:
            break
    return generated

# Warmup
run_eager_decode(5)
torch.cuda.synchronize()

ts = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen = run_eager_decode(MAX_NEW)
    torch.cuda.synchronize()
    n = len(gen)
    ts.append(n / (time.perf_counter() - t0))
eager_static = sum(ts) / len(ts)
text = tokenizer.decode(gen, skip_special_tokens=True)
print(f"  Eager StaticCache: {eager_static:.1f} tok/s")
print(f'  "{text[:70]}..."')

# ── Step 6: CUDA Graph capture ──
print("\nStep 6: CUDA Graph capture...")
try:
    # Fresh cache for graph capture
    graph_cache = StaticCache(
        config=cfg, batch_size=1, max_cache_len=MAX_CACHE,
        device=torch.device(DEV), dtype=torch.float16,
    )
    
    # Prefill (NOT in graph)
    cp_pf = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=graph_cache,
                         cache_position=cp_pf, use_cache=True)
        logits_pf = model.lm_head(out.last_hidden_state[:, -1:, :])
        first_tok = logits_pf.argmax(dim=-1)
    
    # Static buffers for decode (graph I/O)
    g_input = torch.zeros(1, 1, dtype=torch.long, device=DEV)
    g_pos = torch.tensor([plen], dtype=torch.long, device=DEV)
    g_logits = torch.zeros(1, 1, cfg.vocab_size, dtype=torch.float16, device=DEV)
    
    # Warmup decode (allocate internal buffers)
    print("  Warmup decode (3 steps)...")
    g_input.copy_(first_tok.view(1, 1))
    for warmup_i in range(3):
        g_pos.fill_(plen + warmup_i)
        with torch.no_grad():
            out = model.model(input_ids=g_input, past_key_values=graph_cache,
                             cache_position=g_pos, use_cache=True)
            g_logits.copy_(model.lm_head(out.last_hidden_state).unsqueeze(0) 
                          if model.lm_head(out.last_hidden_state).dim() == 2 
                          else model.lm_head(out.last_hidden_state))
            g_input.copy_(g_logits[:, -1:, :].argmax(dim=-1).view(1, 1))
    
    torch.cuda.synchronize()
    
    # Reset cache and redo prefill for clean capture
    graph_cache = StaticCache(
        config=cfg, batch_size=1, max_cache_len=MAX_CACHE,
        device=torch.device(DEV), dtype=torch.float16,
    )
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=graph_cache,
                         cache_position=cp_pf, use_cache=True)
        logits_pf = model.lm_head(out.last_hidden_state[:, -1:, :])
        first_tok = logits_pf.argmax(dim=-1)
    
    g_input.copy_(first_tok.view(1, 1))
    g_pos.fill_(plen)
    
    # One more warmup step at the exact capture position
    with torch.no_grad():
        out = model.model(input_ids=g_input, past_key_values=graph_cache,
                         cache_position=g_pos, use_cache=True)
        lm_out = model.lm_head(out.last_hidden_state)
    
    g_pos.fill_(plen + 1)
    g_input.copy_(lm_out.argmax(dim=-1).view(1, 1))
    
    torch.cuda.synchronize()
    
    # ── CAPTURE ── 
    print("  Capturing CUDA Graph...")
    s = torch.cuda.Stream(device=DEV)
    s.wait_stream(torch.cuda.current_stream(DEV))
    
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = model.model(
                input_ids=g_input,
                past_key_values=graph_cache,
                cache_position=g_pos,
                use_cache=True,
            )
            g_out_logits = model.lm_head(out.last_hidden_state)
    
    torch.cuda.current_stream(DEV).wait_stream(s)
    torch.cuda.synchronize()
    print("  CUDA Graph captured!")
    
    # ── Step 7: Benchmark graphed decode ──
    print("\nStep 7: Benchmark CUDA Graph decode...")
    
    def run_graph_decode(n_tokens=MAX_NEW):
        gc2 = StaticCache(
            config=cfg, batch_size=1, max_cache_len=MAX_CACHE,
            device=torch.device(DEV), dtype=torch.float16,
        )
        # Copy cache into graph's cache (they share the same tensors)
        # Actually we need to use the SAME graph_cache object
        # Reset it
        for layer_idx in range(len(graph_cache.key_cache)):
            graph_cache.key_cache[layer_idx].zero_()
            graph_cache.value_cache[layer_idx].zero_()
        
        # Prefill
        cp = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = model.model(input_ids=ids, past_key_values=graph_cache,
                             cache_position=cp, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            tok = logits.argmax(dim=-1)
        
        generated = [tok.item()]
        g_input.copy_(tok.view(1, 1))
        
        for i in range(n_tokens - 1):
            g_pos.fill_(plen + i)
            graph.replay()
            tok_val = g_out_logits[:, -1:, :].argmax(dim=-1).item()
            generated.append(tok_val)
            g_input[0, 0] = tok_val
            eos = getattr(tokenizer, 'eos_token_id', None)
            if eos and tok_val == eos:
                break
        return generated
    
    # Warmup
    run_graph_decode(5)
    torch.cuda.synchronize()
    
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen = run_graph_decode(MAX_NEW)
        torch.cuda.synchronize()
        n = len(gen)
        ts.append(n / (time.perf_counter() - t0))
    graph_speed = sum(ts) / len(ts)
    text_g = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"  CUDA Graph decode: {graph_speed:.1f} tok/s")
    print(f'  "{text_g[:70]}..."')
    
except Exception as e:
    print(f"  CUDA Graph FAILED: {e}")
    import traceback; traceback.print_exc()
    graph_speed = 0

# ── Summary ──
print(f"\n{'='*55}")
print(f"Eager StaticCache:     {eager_static:6.1f} tok/s")
if graph_speed > 0:
    print(f"CUDA Graph decode:     {graph_speed:6.1f} tok/s ({graph_speed/eager_static:.2f}x)")
print(f"Previous HF baseline:  ~20.5 tok/s")
print(f"Ollama target:         ~60-100 tok/s")
