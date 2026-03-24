#!/usr/bin/env python3
"""CUDA Graph decode — minimal, focused on graph capture."""
import torch, time, os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
torch.set_float32_matmul_precision('high')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StaticCache

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
MAX_NEW = 128
MAX_CACHE = 512
PROMPT = "Explain quantum computing in simple terms:"

print(f"=== CUDA Graph Decode: {MODEL} NF4 ===")

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map=DEV, torch_dtype=torch.float16,
)
model.eval()
cfg = model.config
ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB, prompt_len={plen}")

# ── StaticCache + prefill ──
print("\n[1] StaticCache prefill...")
sc = StaticCache(config=cfg, batch_size=1, max_cache_len=MAX_CACHE,
                 device=torch.device(DEV), dtype=torch.float16)

cp = torch.arange(plen, device=DEV, dtype=torch.long)
with torch.no_grad():
    out = model.model(input_ids=ids, past_key_values=sc,
                     cache_position=cp, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    first_tok = logits.argmax(dim=-1)
print(f"    First token: {first_tok.item()} = '{tokenizer.decode([first_tok.item()])}'")

# ── Static decode buffers ──
g_inp = torch.zeros(1, 1, dtype=torch.long, device=DEV)
g_pos = torch.zeros(1, dtype=torch.long, device=DEV)

# ── Warmup 3 decode steps (allocate internal BnB/cuBLAS buffers) ──
print("[2] Warmup decode (3 steps)...")
g_inp.copy_(first_tok.view(1, 1))
for wi in range(3):
    g_pos.fill_(plen + wi)
    with torch.no_grad():
        out = model.model(input_ids=g_inp, past_key_values=sc,
                         cache_position=g_pos, use_cache=True)
        lm = model.lm_head(out.last_hidden_state)
        g_inp.copy_(lm.argmax(dim=-1).view(1, 1))
torch.cuda.synchronize()
print("    Warmup OK")

# ── CUDA Graph Capture ──
print("[3] Capturing CUDA Graph...")

# Reset cache, redo prefill
for i in range(len(sc.key_cache)):
    sc.key_cache[i].zero_()
    sc.value_cache[i].zero_()

cp2 = torch.arange(plen, device=DEV, dtype=torch.long)
with torch.no_grad():
    out = model.model(input_ids=ids, past_key_values=sc,
                     cache_position=cp2, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    first_tok2 = logits.argmax(dim=-1)

# One decode step warmup at the exact capture position
g_inp.copy_(first_tok2.view(1, 1))
g_pos.fill_(plen)
with torch.no_grad():
    out = model.model(input_ids=g_inp, past_key_values=sc,
                     cache_position=g_pos, use_cache=True)
    g_out_ref = model.lm_head(out.last_hidden_state)

torch.cuda.synchronize()

# Capture
try:
    g_pos.fill_(plen + 1)
    g_inp.copy_(g_out_ref.argmax(dim=-1).view(1, 1))
    
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.model(input_ids=g_inp, past_key_values=sc,
                         cache_position=g_pos, use_cache=True)
        g_out = model.lm_head(out.last_hidden_state)
    
    torch.cuda.synchronize()
    print("    CUDA Graph captured!")
    
except Exception as e:
    print(f"    CAPTURE FAILED: {e}")
    import traceback; traceback.print_exc()
    
    # Fallback: try torch.compile(reduce-overhead) with StaticCache
    print("\n[3b] Fallback: torch.compile(reduce-overhead) + StaticCache...")
    try:
        compiled_fwd = torch.compile(
            model.model.forward, mode="reduce-overhead", fullgraph=False
        )
        # Reset cache
        for i in range(len(sc.key_cache)):
            sc.key_cache[i].zero_()
            sc.value_cache[i].zero_()
        
        cp3 = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = compiled_fwd(input_ids=ids, past_key_values=sc,
                              cache_position=cp3, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            tok = logits.argmax(dim=-1)
        
        print("    reduce-overhead prefill OK, warming up...")
        for wi in range(5):
            g_inp[0,0] = tok.item()
            g_pos.fill_(plen + wi)
            with torch.no_grad():
                out = compiled_fwd(input_ids=g_inp, past_key_values=sc,
                                  cache_position=g_pos, use_cache=True)
                logits = model.lm_head(out.last_hidden_state)
                tok = logits.argmax(dim=-1)
            torch.cuda.synchronize()
        print("    reduce-overhead warmup OK!")
        
        # Benchmark reduce-overhead + StaticCache
        def run_compiled_static(n=MAX_NEW):
            for i in range(len(sc.key_cache)):
                sc.key_cache[i].zero_()
                sc.value_cache[i].zero_()
            
            cp = torch.arange(plen, device=DEV, dtype=torch.long)
            with torch.no_grad():
                out = compiled_fwd(input_ids=ids, past_key_values=sc,
                                  cache_position=cp, use_cache=True)
                logits = model.lm_head(out.last_hidden_state[:, -1:, :])
                t = logits.argmax(dim=-1)
            
            gen = [t.item()]
            for i in range(n-1):
                g_inp[0,0] = t.item()
                g_pos.fill_(plen + i)
                with torch.no_grad():
                    out = compiled_fwd(input_ids=g_inp, past_key_values=sc,
                                      cache_position=g_pos, use_cache=True)
                    logits = model.lm_head(out.last_hidden_state)
                    t = logits.argmax(dim=-1)
                gen.append(t.item())
                eos = getattr(tokenizer, 'eos_token_id', None)
                if eos and t.item() == eos:
                    break
            return gen
        
        # Warmup
        run_compiled_static(5); torch.cuda.synchronize()
        
        ts = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            gen = run_compiled_static(MAX_NEW)
            torch.cuda.synchronize()
            ts.append(len(gen) / (time.perf_counter() - t0))
        compiled_static = sum(ts)/len(ts)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"    reduce-overhead + StaticCache: {compiled_static:.1f} tok/s")
        print(f'    "{text[:70]}..."')
        
    except Exception as e2:
        print(f"    reduce-overhead ALSO FAILED: {e2}")
        import traceback; traceback.print_exc()
        compiled_static = 0
    
    # Also try just mode="default" with StaticCache (no CUDA Graphs)
    print("\n[3c] torch.compile(default) + StaticCache...")
    try:
        compiled_def = torch.compile(
            model.model.forward, mode="default", fullgraph=False
        )
        # Reset
        for i in range(len(sc.key_cache)):
            sc.key_cache[i].zero_()
            sc.value_cache[i].zero_()
        
        cp4 = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = compiled_def(input_ids=ids, past_key_values=sc,
                              cache_position=cp4, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            tok = logits.argmax(dim=-1)
        
        print("    Warmup compile...")
        for wi in range(10):
            g_inp[0,0] = tok.item()
            g_pos.fill_(plen + wi)
            with torch.no_grad():
                out = compiled_def(input_ids=g_inp, past_key_values=sc,
                                  cache_position=g_pos, use_cache=True)
                logits = model.lm_head(out.last_hidden_state)
                tok = logits.argmax(dim=-1)
            torch.cuda.synchronize()
        print("    Warmup done")
        
        def run_compiled_default(n=MAX_NEW):
            for i in range(len(sc.key_cache)):
                sc.key_cache[i].zero_()
                sc.value_cache[i].zero_()
            
            cp = torch.arange(plen, device=DEV, dtype=torch.long)
            with torch.no_grad():
                out = compiled_def(input_ids=ids, past_key_values=sc,
                                  cache_position=cp, use_cache=True)
                logits = model.lm_head(out.last_hidden_state[:, -1:, :])
                t = logits.argmax(dim=-1)
            
            gen = [t.item()]
            for i in range(n-1):
                g_inp[0,0] = t.item()
                g_pos.fill_(plen + i)
                with torch.no_grad():
                    out = compiled_def(input_ids=g_inp, past_key_values=sc,
                                      cache_position=g_pos, use_cache=True)
                    logits = model.lm_head(out.last_hidden_state)
                    t = logits.argmax(dim=-1)
                gen.append(t.item())
                eos = getattr(tokenizer, 'eos_token_id', None)
                if eos and t.item() == eos:
                    break
            return gen
        
        run_compiled_default(5); torch.cuda.synchronize()
        
        ts = []
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            gen = run_compiled_default(MAX_NEW)
            torch.cuda.synchronize()
            ts.append(len(gen) / (time.perf_counter() - t0))
        cd_speed = sum(ts)/len(ts)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        print(f"    compile(default) + StaticCache: {cd_speed:.1f} tok/s")
        print(f'    "{text[:70]}..."')
    except Exception as e3:
        print(f"    compile(default)+StaticCache FAILED: {e3}")
        import traceback; traceback.print_exc()
        cd_speed = 0
    
    sys.exit(0)

# ── If CUDA Graph capture succeeded, benchmark it ──
print("\n[4] Benchmark CUDA Graph decode...")

def run_graph(n=MAX_NEW):
    # Reset cache
    for i in range(len(sc.key_cache)):
        sc.key_cache[i].zero_()
        sc.value_cache[i].zero_()
    
    # Prefill (not graphed)
    cp = torch.arange(plen, device=DEV, dtype=torch.long)
    with torch.no_grad():
        out = model.model(input_ids=ids, past_key_values=sc,
                         cache_position=cp, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1:, :])
        tok = logits.argmax(dim=-1)
    
    gen = [tok.item()]
    g_inp.copy_(tok.view(1, 1))
    
    # Decode via graph replay
    for i in range(n - 1):
        g_pos.fill_(plen + i)
        graph.replay()
        tok_val = g_out[:, -1:, :].argmax(dim=-1).item()
        gen.append(tok_val)
        g_inp[0, 0] = tok_val
        eos = getattr(tokenizer, 'eos_token_id', None)
        if eos and tok_val == eos:
            break
    return gen

# Warmup
run_graph(5); torch.cuda.synchronize()

ts = []
for _ in range(5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    gen = run_graph(MAX_NEW)
    torch.cuda.synchronize()
    ts.append(len(gen) / (time.perf_counter() - t0))

graph_speed = sum(ts) / len(ts)
text = tokenizer.decode(gen, skip_special_tokens=True)
print(f"    CUDA Graph: {graph_speed:.1f} tok/s")
print(f'    "{text[:70]}..."')

print(f"\n{'='*55}")
print(f"CUDA Graph decode:     {graph_speed:6.1f} tok/s")
print(f"Previous HF baseline:  ~20.5 tok/s")
print(f"Ollama target:         ~60-100 tok/s")
