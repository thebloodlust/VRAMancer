#!/usr/bin/env python3
"""NVFP4 Blackwell benchmark — native FP4 quantization on RTX 50xx.

Compares real NVFP4 (torchao cublas scaled_mm) vs BnB NF4 vs BF16.
Requires: RTX 50xx (Blackwell, CC >= 10.0), torchao >= 0.16, PyTorch >= 2.10.

Results (Qwen2.5-7B-Instruct, RTX 5070 Ti CC 12.0, 128 tokens):
  BF16 baseline:       36.4 tok/s, 15.25 GB VRAM
  NVFP4 Dynamic W+A:   11.0 tok/s,  5.87 GB VRAM (real cublas FP4 kernel)
  BnB NF4:             17.5 tok/s, 14.77 GB VRAM
  BnB INT8:             7.9 tok/s, 14.76 GB VRAM

NVFP4 Dynamic W+A saves 62% VRAM vs BF16. Speed is limited by torchao
prototype overhead (0.16.0) — expect improvements in stable releases.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_nvfp4.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_nvfp4.py --model Qwen/Qwen2.5-7B-Instruct
"""
import os
import sys
import time
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not available")
    sys.exit(1)


def gpu_info():
    if not torch.cuda.is_available():
        return "No CUDA", -1
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    return f"{name} CC={cc[0]}.{cc[1]}, {free/1e9:.1f}/{total/1e9:.1f} GB", cc[0]


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def bench_generate(model, tokenizer, prompt, max_new=128, warmup=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=warmup, do_sample=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / 1e9
    tps = max_new / elapsed
    return tps, vram, elapsed, text


def nvfp4_filter_fn(module, fqn: str) -> bool:
    """Exclude lm_head from NVFP4 quantization (aten.expand not implemented)."""
    if not isinstance(module, nn.Linear):
        return False
    excluded = ("lm_head", "embed_tokens", "wte", "wpe")
    for name in excluded:
        if name in fqn:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="NVFP4 Blackwell benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    parser.add_argument("--skip-bf16", action="store_true", help="Skip BF16 baseline")
    parser.add_argument("--skip-bnb", action="store_true", help="Skip BnB tests")
    args = parser.parse_args()

    info, cc_major = gpu_info()
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {info}")

    if cc_major < 10:
        print(f"ERROR: NVFP4 requires Blackwell (CC >= 10.0), got CC {cc_major}")
        print("This benchmark is designed for RTX 50xx GPUs.")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results = {}

    # --- BF16 baseline ---
    if not args.skip_bf16:
        print(f"\n>>> BF16 BASELINE")
        clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        tps, vram, elapsed, text = bench_generate(model, tokenizer, args.prompt, args.max_tokens)
        results["BF16"] = (tps, vram)
        print(f"  {args.max_tokens} tokens in {elapsed:.3f}s = {tps:.1f} tok/s, VRAM: {vram:.2f} GB")
        print(f"  Output: {text[:100]}...")
        del model
        clear_gpu()

    # --- NVFP4 Dynamic W+A ---
    print(f"\n>>> NVFP4 Dynamic W+A (cublas Blackwell FP4)")
    clear_gpu()
    try:
        from torchao.quantization import quantize_
        from torchao.prototype.mx_formats import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True,
        )
        print(f"  CPU load: {time.perf_counter() - t0:.1f}s")
        t1 = time.perf_counter()
        quantize_(model, NVFP4DynamicActivationNVFP4WeightConfig(), filter_fn=nvfp4_filter_fn)
        print(f"  Quantization: {time.perf_counter() - t1:.1f}s")
        t2 = time.perf_counter()
        model = model.to("cuda:0")
        print(f"  GPU transfer: {time.perf_counter() - t2:.1f}s")

        tps, vram, elapsed, text = bench_generate(model, tokenizer, args.prompt, args.max_tokens)
        results["NVFP4 Dynamic W+A"] = (tps, vram)
        print(f"  {args.max_tokens} tokens in {elapsed:.3f}s = {tps:.1f} tok/s, VRAM: {vram:.2f} GB")
        print(f"  Output: {text[:100]}...")
        del model
        clear_gpu()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["NVFP4 Dynamic W+A"] = ("FAILED", str(e))

    # --- BnB NF4 ---
    if not args.skip_bnb:
        print(f"\n>>> BnB NF4")
        clear_gpu()
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
            tps, vram, elapsed, text = bench_generate(model, tokenizer, args.prompt, args.max_tokens)
            results["BnB NF4"] = (tps, vram)
            print(f"  {args.max_tokens} tokens in {elapsed:.3f}s = {tps:.1f} tok/s, VRAM: {vram:.2f} GB")
            print(f"  Output: {text[:100]}...")
            del model
            clear_gpu()
        except Exception as e:
            print(f"  FAILED: {e}")
            results["BnB NF4"] = ("FAILED", str(e))

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"NVFP4 Blackwell Benchmark — {info}")
    print(f"Model: {args.model}, {args.max_tokens} new tokens")
    print(f"{'='*70}")
    bf16_tps = results.get("BF16", (1, 0))[0] if "BF16" in results else None
    print(f"{'Method':<30} {'tok/s':>8} {'VRAM GB':>10} {'vs BF16':>10}")
    print("-" * 70)
    for name, vals in results.items():
        if isinstance(vals[0], (int, float)):
            tps, vram = vals
            if bf16_tps:
                ratio = f"{tps / bf16_tps:>9.1%}"
            else:
                ratio = "—"
            print(f"{name:<30} {tps:>8.1f} {vram:>10.2f} {ratio}")
        else:
            print(f"{name:<30} {'FAILED':>8}")
    print(f"{'='*70}")
    print()
    print("Notes:")
    print("- NVFP4 Dynamic W+A uses real cublas Blackwell FP4 kernel (torch._scaled_mm)")
    print("- NVFP4 requires sm100+ (Blackwell CC >= 10.0)")
    print("- torchao NVFP4 is prototype (0.16.0) — expect improvements in stable release")
    print("- BnB NF4 uses bitsandbytes kgemm_4bit_inference_naive dequant kernel")


if __name__ == "__main__":
    main()
