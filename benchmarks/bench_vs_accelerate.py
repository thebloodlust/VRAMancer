#!/usr/bin/env python3
"""VRAMancer vs accelerate — heterogeneous multi-GPU comparison.

Compares:
  1. accelerate device_map="balanced" — the standard multi-GPU approach
  2. VRAMancer pipeline parallel — VRAM-proportional split

Both methods load the same model across multiple GPUs and measure tok/s.
This is the benchmark that justifies VRAMancer's existence: for heterogeneous
GPUs (different VRAM sizes), accelerate's "balanced" split wastes the larger GPU.

Usage:
    python benchmarks/bench_vs_accelerate.py
    python benchmarks/bench_vs_accelerate.py --model Qwen/Qwen2.5-14B-Instruct
    python benchmarks/bench_vs_accelerate.py --model Qwen/Qwen2.5-7B-Instruct --quantization nf4
"""
import torch
import time
import os
import sys
import gc
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "The future of artificial intelligence is",
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list.",
]

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _gpu_report():
    """Print GPU VRAM status."""
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {props.name} — {free / 2**30:.1f} / {total / 2**30:.1f} GB free")


def _clear_gpu():
    """Force-clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _measure_tok_s(generate_fn, prompts, max_tokens, warmup=1):
    """Measure tokens/second over multiple prompts."""
    # Warmup
    for p in prompts[:warmup]:
        generate_fn(p, max_new_tokens=max(max_tokens // 4, 16))

    total_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_tokens = generate_fn(prompt, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tokens += n_tokens
        total_time += elapsed

    return total_tokens / total_time if total_time > 0 else 0.0


def bench_accelerate(model_name, prompts, max_tokens, quantization=None):
    """Baseline: accelerate device_map='balanced'."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n=== accelerate device_map='balanced' ===")
    _gpu_report()

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {
        "device_map": "balanced",
        "torch_dtype": torch.bfloat16,
    }

    if quantization == "nf4":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs.pop("torch_dtype", None)
        except ImportError:
            print("  WARNING: bitsandbytes not available, loading in bf16")
    elif quantization == "int8":
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs.pop("torch_dtype", None)
        except ImportError:
            print("  WARNING: bitsandbytes not available, loading in bf16")

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Show device map
    if hasattr(model, "hf_device_map"):
        devices_used = set(str(v) for v in model.hf_device_map.values())
        print(f"  Device map: {len(model.hf_device_map)} modules across {devices_used}")

    def generate_fn(prompt, max_new_tokens=128):
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return out.shape[1] - input_ids.shape[1]

    tok_s = _measure_tok_s(generate_fn, prompts, max_tokens)

    # VRAM usage
    vram_used = sum(
        (torch.cuda.memory_allocated(i)) / 2**30
        for i in range(torch.cuda.device_count())
    )

    print(f"  Result: {tok_s:.1f} tok/s, {vram_used:.1f} GB VRAM used")

    del model
    _clear_gpu()
    return {"method": "accelerate_balanced", "tok_s": tok_s, "vram_gb": vram_used}


def bench_vramancer(model_name, prompts, max_tokens, quantization=None):
    """VRAMancer: VRAM-proportional split across heterogeneous GPUs."""
    from core.inference_pipeline import InferencePipeline

    print("\n=== VRAMancer (VRAM-proportional split) ===")
    _gpu_report()

    if quantization:
        os.environ["VRM_QUANTIZATION"] = quantization

    num_gpus = torch.cuda.device_count()
    pipeline = InferencePipeline(backend_name="auto", verbose=True)
    pipeline.load(model_name, num_gpus=num_gpus)

    print(f"  Blocks: {len(pipeline.blocks)}, GPUs: {pipeline.num_gpus}")

    def generate_fn(prompt, max_new_tokens=128):
        result = pipeline.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        # Estimate tokens from output length (approximate)
        from core.utils import BasicTokenizer
        return len(BasicTokenizer().encode(result))

    tok_s = _measure_tok_s(generate_fn, prompts, max_tokens)

    vram_used = sum(
        (torch.cuda.memory_allocated(i)) / 2**30
        for i in range(torch.cuda.device_count())
    )

    print(f"  Result: {tok_s:.1f} tok/s, {vram_used:.1f} GB VRAM used")

    pipeline.shutdown()
    if quantization:
        os.environ.pop("VRM_QUANTIZATION", None)
    _clear_gpu()
    return {"method": "vramancer", "tok_s": tok_s, "vram_gb": vram_used}


def main():
    parser = argparse.ArgumentParser(description="VRAMancer vs accelerate benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["nf4", "int8"], help="Quantization mode")
    parser.add_argument("--skip-accelerate", action="store_true",
                        help="Skip accelerate baseline (e.g., if model OOMs)")
    parser.add_argument("--skip-vramancer", action="store_true",
                        help="Skip VRAMancer (for comparison only)")
    args = parser.parse_args()

    print("=" * 60)
    print("VRAMancer vs accelerate — Multi-GPU Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Quantization: {args.quantization or 'none (bf16)'}")
    print(f"GPUs: {torch.cuda.device_count()}")
    _gpu_report()

    results = []

    if not args.skip_accelerate:
        try:
            r = bench_accelerate(args.model, PROMPTS, args.max_tokens, args.quantization)
            results.append(r)
        except Exception as e:
            print(f"  accelerate FAILED: {e}")
            results.append({"method": "accelerate_balanced", "tok_s": 0, "error": str(e)})

    if not args.skip_vramancer:
        try:
            r = bench_vramancer(args.model, PROMPTS, args.max_tokens, args.quantization)
            results.append(r)
        except Exception as e:
            print(f"  VRAMancer FAILED: {e}")
            results.append({"method": "vramancer", "tok_s": 0, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in results:
        err = f" (ERROR: {r['error']})" if "error" in r else ""
        vram = f", {r.get('vram_gb', '?'):.1f} GB" if "vram_gb" in r else ""
        print(f"  {r['method']:30s} {r['tok_s']:8.1f} tok/s{vram}{err}")

    if len(results) == 2 and all(r["tok_s"] > 0 for r in results):
        accel = next(r for r in results if r["method"] == "accelerate_balanced")
        vrm = next(r for r in results if r["method"] == "vramancer")
        delta = (vrm["tok_s"] - accel["tok_s"]) / accel["tok_s"] * 100
        print(f"\n  VRAMancer vs accelerate: {delta:+.1f}%")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "results_vs_accelerate.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "quantization": args.quantization,
                    "max_tokens": args.max_tokens, "results": results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
