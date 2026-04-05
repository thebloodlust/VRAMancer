#!/usr/bin/env python3
"""Benchmark RTX 4060 8GB — small models that fit in 8 GB.

Run on PC portable with RTX 4060:
    pip install -e .
    pip install llama-cpp-python  # for GGUF
    python scripts/bench_rtx4060.py

Tests:
  1. GPT-2 124M (FP16) — baseline
  2. TinyLlama 1.1B (FP16) — fits in 8 GB
  3. Qwen2.5-7B NF4 — ~5 GB
  4. (if llama-cpp-python) GGUF Q4_K_M
"""

import os
import sys
import time
import gc

os.environ["VRM_TEST_MODE"] = "1"


def bench_model(model_name, dtype_str="float16", quant=None, max_new=50):
    """Load and benchmark a model, return tok/s."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Dtype: {dtype_str}, Quant: {quant or 'none'}")
    print(f"{'='*50}")

    device = "cuda:0" if torch.cuda.is_available() else "mps"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"trust_remote_code": True}
    if quant == "nf4":
        try:
            import bitsandbytes  # noqa: F401 — verify BnB itself loads
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        except (ImportError, RuntimeError) as e:
            print(f"  bitsandbytes not available: {e}")
            print("  Install: pip install bitsandbytes>=0.43.0")
            return None
    else:
        load_kwargs["torch_dtype"] = getattr(torch, dtype_str)
        load_kwargs["device_map"] = device

    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return None
    load_time = time.perf_counter() - t0
    model.eval()

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1e6
        print(f"  Load time: {load_time:.1f}s, VRAM: {vram_mb:.0f} MB")
    else:
        print(f"  Load time: {load_time:.1f}s")

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    if quant != "nf4":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Bench
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    elapsed = time.perf_counter() - t0

    new_toks = out.shape[-1] - inputs["input_ids"].shape[-1]
    tps = new_toks / elapsed
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    print(f"  Output: {text[:100]}...")
    print(f"  {new_toks} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

    if torch.cuda.is_available():
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tps


def bench_gguf():
    """Test llama.cpp GGUF if available."""
    print(f"\n{'='*50}")
    print("GGUF via llama-cpp-python")
    print(f"{'='*50}")

    try:
        from core.backends_llamacpp import LlamaCppBackend
    except ImportError:
        print("  llama-cpp-python not installed, SKIP")
        return None

    # Use a small GGUF model
    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    backend = LlamaCppBackend(model_name)
    try:
        backend.load_model(model_name, n_gpu_layers=-1)
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None

    prompt = "The future of artificial intelligence is"
    t0 = time.perf_counter()
    result = backend.generate(prompt, max_new_tokens=50)
    elapsed = time.perf_counter() - t0

    # Rough token count
    toks = len(result.split()) - len(prompt.split())
    tps = toks / elapsed if elapsed > 0 else 0

    print(f"  Output: {result[:100]}...")
    print(f"  ~{toks} tokens in {elapsed:.2f}s = ~{tps:.1f} tok/s")
    return tps


if __name__ == "__main__":
    import torch

    print("VRAMancer RTX 4060 Benchmark")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print()

    results = {}

    # 1. GPT-2
    results["gpt2_fp16"] = bench_model("gpt2", "float16")

    # 2. TinyLlama FP16
    results["tinyllama_fp16"] = bench_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "float16"
    )

    # 3. Qwen2.5-7B NF4 (needs ~5 GB)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 7:
            results["qwen7b_nf4"] = bench_model(
                "Qwen/Qwen2.5-7B", "float16", quant="nf4"
            )
        else:
            print(f"\nSkip Qwen 7B NF4 — need 7+ GB, have {vram_gb:.1f} GB")

    # 4. GGUF
    results["gguf"] = bench_gguf()

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, tps in results.items():
        if tps is not None:
            print(f"  {name:20s}: {tps:.1f} tok/s")
        else:
            print(f"  {name:20s}: SKIP")
