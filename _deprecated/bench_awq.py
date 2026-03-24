#!/usr/bin/env python3
"""Benchmark AWQ INT4 vs BnB NF4 — optimized kernel comparison.

AWQ uses WQLinear-GEMM or Marlin kernels (hardware-aware, 3-4x faster).
BnB uses kgemm_4bit_inference_naive (literally named "naive").
"""
import torch, time, os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision('high')

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda:0"
MAX_NEW = 128
PROMPT = "Explain quantum computing in simple terms:"

# AWQ models (pre-quantized, fast kernels)
AWQ_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "TheBloke/Qwen2.5-7B-Instruct-AWQ",
    "casperhansen/qwen2.5-7b-instruct-awq",
]

def benchmark_model(model, tokenizer, label, n_runs=5, max_new=MAX_NEW):
    """Benchmark a model's raw decode speed."""
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
    plen = ids.shape[1]
    
    # Warmup
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()
    
    ts = []
    for _ in range(n_runs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False)
        torch.cuda.synchronize()
        n = out.shape[1] - plen
        ts.append(n / (time.perf_counter() - t0))
    
    avg = sum(ts) / len(ts)
    text = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
    print(f"  {label}: {avg:.1f} tok/s")
    print(f'    "{text[:70]}..."')
    return avg

def benchmark_turbo(model, tokenizer, label, n_runs=5, max_new=MAX_NEW):
    """Benchmark with TurboEngine (compiled)."""
    sys.path.insert(0, '.')
    from core.turbo_engine import TurboEngine
    
    engine = TurboEngine(model, tokenizer, device=DEV, max_seq_len=512, compile=True)
    
    # Warmup (JIT compilation)
    print(f"  Warming up {label}...")
    for _ in range(5):
        engine.generate(PROMPT, max_new_tokens=5)
        torch.cuda.synchronize()
    
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
    plen = ids.shape[1]
    
    ts = []
    for _ in range(n_runs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = engine.generate_ids(ids, max_new_tokens=max_new)
        torch.cuda.synchronize()
        n = out.shape[1] - plen
        ts.append(n / (time.perf_counter() - t0))
    
    avg = sum(ts) / len(ts)
    return avg

# ── Try AWQ models ──
print("=== AWQ INT4 Benchmark ===\n")

awq_model = None
awq_tokenizer = None
awq_name = None

for name in AWQ_MODELS:
    print(f"Trying {name}...")
    try:
        awq_tokenizer = AutoTokenizer.from_pretrained(name)
        awq_model = AutoModelForCausalLM.from_pretrained(
            name, device_map=DEV, torch_dtype=torch.float16,
        )
        awq_model.eval()
        awq_name = name
        vram = torch.cuda.memory_allocated(0) / 1e9
        print(f"  Loaded! VRAM: {vram:.1f} GB")
        
        # Check what linear type is used
        for m in awq_model.modules():
            t = type(m).__name__
            if 'linear' in t.lower() or 'wq' in t.lower():
                print(f"  Linear type: {t}")
                break
        break
    except Exception as e:
        print(f"  Failed: {e}")
        continue

if awq_model is None:
    print("\nNo AWQ model available. Testing with GPTQ instead...")
    GPTQ_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "TheBloke/Qwen2.5-7B-Instruct-GPTQ",
    ]
    for name in GPTQ_MODELS:
        print(f"Trying {name}...")
        try:
            awq_tokenizer = AutoTokenizer.from_pretrained(name)
            awq_model = AutoModelForCausalLM.from_pretrained(
                name, device_map=DEV, torch_dtype=torch.float16,
            )
            awq_model.eval()
            awq_name = name
            vram = torch.cuda.memory_allocated(0) / 1e9
            print(f"  Loaded! VRAM: {vram:.1f} GB")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

if awq_model is not None:
    print(f"\n── Benchmarking: {awq_name} ──\n")
    
    # 1. HF generate baseline
    base = benchmark_model(awq_model, awq_tokenizer, "HF generate()")
    
    # 2. TurboEngine compiled
    turbo = benchmark_turbo(awq_model, awq_tokenizer, "TurboEngine compiled")
    print(f"  TurboEngine compiled: {turbo:.1f} tok/s ({turbo/base:.2f}x)")
    
    print(f"\n{'='*55}")
    print(f"AWQ/GPTQ HF generate:      {base:6.1f} tok/s")
    print(f"AWQ/GPTQ TurboEngine:      {turbo:6.1f} tok/s ({turbo/base:.2f}x)")
    print(f"BnB NF4 HF baseline:       ~20.5 tok/s")
    print(f"BnB NF4 TurboEngine:       ~29.6 tok/s")
    print(f"Ollama target:             ~60-100 tok/s")
else:
    print("\nNo AWQ or GPTQ model could be loaded.")
