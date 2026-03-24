#!/usr/bin/env python3
"""CUDA Graph decode — v4 with correct transformers 5.3 StaticCache API."""
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

def reset_static_cache(sc):
    """Zero out all layers in a transformers 5.3 StaticCache."""
    for layer in sc.layers:
        if hasattr(layer, 'keys') and layer.is_initialized:
            layer.keys.zero_()
            layer.values.zero_()

print(f"=== CUDA Graph Decode v4: {MODEL} NF4 ===")

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

# ── Create StaticCache + prefill ──
print("\n[1] StaticCache + prefill...")
sc = StaticCache(config=cfg, max_cache_len=MAX_CACHE)

cp = torch.arange(plen, device=DEV, dtype=torch.long)
with torch.no_grad():
    out = model.model(input_ids=ids, past_key_values=sc,
                     cache_position=cp, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    first_tok = logits.argmax(dim=-1)
print(f"    OK: '{tokenizer.decode([first_tok.item()])}'")

# ── Warmup decode steps ──
print("[2] Warmup decode...")
g_inp = torch.zeros(1, 1, dtype=torch.long, device=DEV)
g_pos = torch.zeros(1, dtype=torch.long, device=DEV)

g_inp.copy_(first_tok.view(1, 1))
for wi in range(3):
    g_pos.fill_(plen + wi)
    with torch.no_grad():
        out = model.model(input_ids=g_inp, past_key_values=sc,
                         cache_position=g_pos, use_cache=True)
        lm = model.lm_head(out.last_hidden_state)
        g_inp.copy_(lm.argmax(dim=-1).view(1, 1))
torch.cuda.synchronize()
print("    OK")

# ── CUDA Graph Capture ──
print("[3] Capturing CUDA Graph...")

# Reset and redo prefill cleanly
reset_static_cache(sc)
cp2 = torch.arange(plen, device=DEV, dtype=torch.long)
with torch.no_grad():
    out = model.model(input_ids=ids, past_key_values=sc,
                     cache_position=cp2, use_cache=True)
    logits = model.lm_head(out.last_hidden_state[:, -1:, :])
    first_tok2 = logits.argmax(dim=-1)

# One decode step at capture start position
g_inp.copy_(first_tok2.view(1, 1))
g_pos.fill_(plen)
with torch.no_grad():
    out = model.model(input_ids=g_inp, past_key_values=sc,
                     cache_position=g_pos, use_cache=True)
    ref_logits = model.lm_head(out.last_hidden_state)

torch.cuda.synchronize()

# Setup for next step (graph will capture this position)
g_pos.fill_(plen + 1)
g_inp.copy_(ref_logits.argmax(dim=-1).view(1, 1))

graph_ok = False
graph_speed = 0
try:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = model.model(input_ids=g_inp, past_key_values=sc,
                         cache_position=g_pos, use_cache=True)
        g_out = model.lm_head(out.last_hidden_state)
    
    torch.cuda.synchronize()
    print("    CUDA Graph captured!")
    graph_ok = True
except Exception as e:
    print(f"    CAPTURE FAILED: {e}")
    import traceback; traceback.print_exc()

if graph_ok:
    # ── Benchmark CUDA Graph decode ──
    print("\n[4] Benchmark CUDA Graph decode...")
    
    def run_graph(n=MAX_NEW):
        reset_static_cache(sc)
        cp = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = model.model(input_ids=ids, past_key_values=sc,
                             cache_position=cp, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            tok = logits.argmax(dim=-1)
        
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
    print(f'    "{text[:80]}..."')

# ── Also test compile(default) + StaticCache ──
print("\n[5] compile(default) + StaticCache...")
comp_speed = 0
try:
    compiled_fwd = torch.compile(model.model.forward, mode="default", fullgraph=False)
    
    def run_compiled(n=MAX_NEW):
        reset_static_cache(sc)
        cp = torch.arange(plen, device=DEV, dtype=torch.long)
        with torch.no_grad():
            out = compiled_fwd(input_ids=ids, past_key_values=sc,
                              cache_position=cp, use_cache=True)
            logits = model.lm_head(out.last_hidden_state[:, -1:, :])
            t = logits.argmax(dim=-1)
        
        gen = [t.item()]
        for i in range(n - 1):
            g_inp[0, 0] = t.item()
            g_pos.fill_(plen + i)
            with torch.no_grad():
                out = compiled_fwd(input_ids=g_inp, past_key_values=sc,
                                  cache_position=g_pos, use_cache=True)
                logits = model.lm_head(out.last_hidden_state)
                t = logits.argmax(dim=-1)
            gen.append(t.item())
            if tokenizer.eos_token_id and t.item() == tokenizer.eos_token_id:
                break
        return gen
    
    print("    Warming up compile...")
    for _ in range(5):
        run_compiled(5)
        torch.cuda.synchronize()
    print("    Warmup done, benchmarking...")
    
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        gen = run_compiled(MAX_NEW)
        torch.cuda.synchronize()
        ts.append(len(gen) / (time.perf_counter() - t0))
    comp_speed = sum(ts) / len(ts)
    text = tokenizer.decode(gen, skip_special_tokens=True)
    print(f"    compile(default)+StaticCache: {comp_speed:.1f} tok/s")
    print(f'    "{text[:80]}..."')
except Exception as e:
    print(f"    compile(default)+StaticCache FAILED: {e}")
    import traceback; traceback.print_exc()

# ── Summary ──
print(f"\n{'='*55}")
if graph_speed > 0:
    print(f"CUDA Graph decode:             {graph_speed:6.1f} tok/s")
if comp_speed > 0:
    print(f"compile(default)+StaticCache:  {comp_speed:6.1f} tok/s")
print(f"Previous TurboEngine compiled:  ~29.6 tok/s")
print(f"Previous HF baseline:           ~20.5 tok/s")
print(f"Ollama target:                  ~60-100 tok/s")
