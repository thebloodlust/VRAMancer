"""Load GPTQ model directly via auto_gptq, bypassing transformers' broken quantizer."""
import torch, time, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

# Try loading directly
print("Checking auto_gptq...")
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEV = "cuda:0"
MAX_NEW = 128
PROMPT = "Explain quantum computing in simple terms:"

print(f"=== {MODEL} via auto_gptq ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("Loading model...")
model = AutoGPTQForCausalLM.from_quantized(
    MODEL,
    device=DEV,
    use_safetensors=True,
    inject_fused_attention=False,  # Don't inject fused attention
    inject_fused_mlp=False,        # Don't inject fused MLP  
)
model.eval()
vram = torch.cuda.memory_allocated(0)/1e9
print(f"VRAM: {vram:.1f} GB")

# Check linear types
linear_types = set()
for m in model.modules():
    tname = type(m).__name__
    if 'linear' in tname.lower() or 'quant' in tname.lower():
        linear_types.add(tname)
print(f"Quant layers: {linear_types}")

ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
plen = ids.shape[1]

# HF-style generate
print("\n1) model.generate()...")
with torch.no_grad():
    _ = model.generate(input_ids=ids, max_new_tokens=5)
ts = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    ts.append((out.shape[1]-plen)/(time.perf_counter()-t0))
base = sum(ts)/3
print(f"   {base:.1f} tok/s")

# TurboEngine
print("\n2) TurboEngine (eager)...")
# auto_gptq model wraps the actual HF model
actual_model = model.model if hasattr(model, 'model') else model
from core.turbo_engine import TurboEngine
eng = TurboEngine(actual_model, tokenizer, device=DEV, max_seq_len=512, compile=False)
eng.generate(PROMPT, max_new_tokens=5); torch.cuda.synchronize()
ts2 = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o2 = eng.generate_ids(ids, max_new_tokens=MAX_NEW)
    torch.cuda.synchronize()
    ts2.append((o2.shape[1]-plen)/(time.perf_counter()-t0))
v2 = sum(ts2)/3
print(f"   {v2:.1f} tok/s ({v2/base:.2f}x)")

# TurboEngine compiled
print("\n3) TurboEngine (compiled) — compiling...")
inner = actual_model.model if hasattr(actual_model, 'model') else actual_model
inner.forward = torch.compile(inner.forward, mode="default", fullgraph=False)
eng2 = TurboEngine(actual_model, tokenizer, device=DEV, max_seq_len=512, compile=False)
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
print(f"direct generate:          {base:6.1f} tok/s (baseline)")
print(f"TurboEngine (eager):      {v2:6.1f} tok/s ({v2/base:.2f}x)")
print(f"TurboEngine (compiled):   {v3:6.1f} tok/s ({v3/base:.2f}x)")
