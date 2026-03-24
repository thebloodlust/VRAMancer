"""Test CUDA graphs on fp16 model — eliminate Python overhead entirely."""
import torch, time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
MAX_NEW = 128
PROMPT = "Explain quantum computing in simple terms:"

print(f"=== CUDA Graphs + fp16 {MODEL} ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("Loading fp16...")
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEV, torch_dtype=torch.float16)
model.eval()
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")

ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]

# Baseline: mode="default" (what we already know works)
print("\n1) compile mode='default' (known working)...")
import copy
model_default = model
model_default.model.forward = torch.compile(model_default.model.forward, mode="default", fullgraph=False)

from core.turbo_engine import TurboEngine
eng_def = TurboEngine(model_default, tokenizer, device=DEV, max_seq_len=512, compile=False)
for _ in range(5):
    eng_def.generate(PROMPT, max_new_tokens=10)
    torch.cuda.synchronize()

ts1 = []
for _ in range(5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o1 = eng_def.generate_ids(ids, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    ts1.append((o1.shape[1]-plen)/(time.perf_counter()-t0))
v1 = sum(ts1)/5
print(f"   {v1:.1f} tok/s")

# Test: can we use model.generate() with static cache + compile?
print("\n2) HF generate() with static cache + compile...")
del model_default
torch.cuda.empty_cache()
import gc; gc.collect()

model2 = AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEV, torch_dtype=torch.float16)
model2.eval()
model2.generation_config.cache_implementation = "static"
model2.forward = torch.compile(model2.forward, mode="reduce-overhead", fullgraph=True)

try:
    print("   Warmup (this may crash if graphs can't capture)...")
    with torch.no_grad():
        for _ in range(3):
            out2 = model2.generate(input_ids=ids, max_new_tokens=16, do_sample=False)
            torch.cuda.synchronize()
    
    ts2 = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out2 = model2.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        ts2.append((out2.shape[1]-plen)/(time.perf_counter()-t0))
    v2 = sum(ts2)/3
    text2 = tokenizer.decode(out2[0][plen:], skip_special_tokens=True)
    print(f"   {v2:.1f} tok/s (vs default: {v1:.1f})")
    print(f'   "{text2[:80]}..."')
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {str(e)[:200]}")
    v2 = 0

# Test: compile individual layers
print("\n3) Per-layer compile (reduce-overhead)...")
del model2
torch.cuda.empty_cache(); gc.collect()

model3 = AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEV, torch_dtype=torch.float16)
model3.eval()

for i, layer in enumerate(model3.model.layers):
    layer.forward = torch.compile(layer.forward, mode="reduce-overhead", fullgraph=False)

eng3 = TurboEngine(model3, tokenizer, device=DEV, max_seq_len=512, compile=False)
try:
    print("   Warmup (compiling all layers)...")
    for _ in range(5):
        eng3.generate(PROMPT, max_new_tokens=10)
        torch.cuda.synchronize()
    
    ts3 = []
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        o3 = eng3.generate_ids(ids, max_new_tokens=MAX_NEW)
        torch.cuda.synchronize()
        ts3.append((o3.shape[1]-plen)/(time.perf_counter()-t0))
    v3 = sum(ts3)/5
    text3 = tokenizer.decode(o3[0][plen:], skip_special_tokens=True)
    print(f"   {v3:.1f} tok/s (vs default: {v1:.1f})")
    print(f'   "{text3[:80]}..."')
except Exception as e:
    print(f"   FAILED: {type(e).__name__}: {str(e)[:200]}")
    v3 = 0

print(f"\n{'='*50}")
print(f"TurboEngine + default:        {v1:6.1f} tok/s")
if v2: print(f"HF static + reduce-overhead:  {v2:6.1f} tok/s")
if v3: print(f"Per-layer reduce-overhead:    {v3:6.1f} tok/s")
print(f"Ollama target:               ~60-100 tok/s")
