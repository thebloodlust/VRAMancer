"""TurboEngine benchmark: Qwen2.5-7B FP16 — the Ollama challenge."""
import torch, time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda:0"
MAX_NEW = 128
PROMPT = "Explain quantum computing in simple terms:"

print(f"=== {MODEL} FP16 + TurboEngine ===")
print("Loading fp16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=DEV, torch_dtype=torch.float16)
model.eval()
vram = torch.cuda.memory_allocated(0)/1e9
free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))/1e9
print(f"VRAM: {vram:.1f} GB (free: {free:.1f} GB)")

ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]

# 1. HF baseline
print("\n1) HF generate()...")
with torch.no_grad():
    _ = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False)
ts = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False)
    torch.cuda.synchronize()
    ts.append((out.shape[1]-plen)/(time.perf_counter()-t0))
base = sum(ts)/3
print(f"   {base:.1f} tok/s")

# 2. TurboEngine eager
print("\n2) TurboEngine (eager)...")
from core.turbo_engine import TurboEngine
eng = TurboEngine(model, tokenizer, device=DEV, max_seq_len=512, compile=False)
eng.generate(PROMPT, max_new_tokens=5); torch.cuda.synchronize()
ts2 = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o2 = eng.generate_ids(ids, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    ts2.append((o2.shape[1]-plen)/(time.perf_counter()-t0))
v2 = sum(ts2)/3
print(f"   {v2:.1f} tok/s ({v2/base:.2f}x)")

# 3. TurboEngine compiled
print("\n3) TurboEngine (compiled) — compiling...")
model.model.forward = torch.compile(model.model.forward, mode="default", fullgraph=False)
eng2 = TurboEngine(model, tokenizer, device=DEV, max_seq_len=512, compile=False)
# Warmup
for _ in range(5):
    eng2.generate(PROMPT, max_new_tokens=10)
    torch.cuda.synchronize()
print("   Warmup done, benchmarking...")
ts3 = []
for _ in range(5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o3 = eng2.generate_ids(ids, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    ts3.append((o3.shape[1]-plen)/(time.perf_counter()-t0))
v3 = sum(ts3)/5
text = tokenizer.decode(o3[0][plen:], skip_special_tokens=True)
print(f"   {v3:.1f} tok/s ({v3/base:.2f}x)")
print(f'   "{text[:80]}..."')

print(f"\n{'='*50}")
print(f"HF generate:              {base:6.1f} tok/s (baseline)")
print(f"TurboEngine (eager):      {v2:6.1f} tok/s ({v2/base:.2f}x)")
print(f"TurboEngine (compiled):   {v3:6.1f} tok/s ({v3/base:.2f}x)")
print(f"Ollama target:           ~60-100 tok/s")
