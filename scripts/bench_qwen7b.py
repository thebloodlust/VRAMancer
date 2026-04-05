#!/usr/bin/env python3
"""Direct Qwen-7B TurboQuant benchmark — skip expensive CPU microbenchmark."""
import os, sys, time, gc
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # RTX 3090
os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached model, no network
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ.pop("VRM_MINIMAL_TEST", None)
os.environ.pop("VRM_TEST_MODE", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def bench(model, config_name, env_overrides, prompts, max_tokens=64):
    for k, v in env_overrides.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)

    clear()
    from core.inference_pipeline import InferencePipeline, reset_pipeline
    reset_pipeline()

    try:
        pipeline = InferencePipeline()
        pipeline.load(model, num_gpus=1)

        # Warmup
        pipeline.generate("Hello world", max_new_tokens=8)
        torch.cuda.synchronize()
        clear()
        torch.cuda.reset_peak_memory_stats()

        tokenizer = getattr(pipeline.backend, "tokenizer", None)
        total_tokens = 0
        total_time = 0.0

        for prompt in prompts:
            t0 = time.perf_counter()
            result_text = pipeline.generate(prompt, max_new_tokens=max_tokens)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            if tokenizer:
                tokens = max(len(tokenizer.encode(result_text)), 1)
            else:
                tokens = len(result_text) // 4
            total_tokens += tokens
            total_time += elapsed

        avg_tps = total_tokens / total_time if total_time > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated(0) / 1e9

        print(f"  {config_name:<40s}  {avg_tps:>7.1f} tok/s  {peak_vram:>6.2f} GB")

        reset_pipeline()
        del pipeline
        clear()
        return avg_tps

    except Exception as e:
        print(f"  {config_name:<40s}  FAILED: {e}")
        try:
            from core.inference_pipeline import reset_pipeline
            reset_pipeline()
        except:
            pass
        clear()
        return 0

def main():
    model = "Qwen/Qwen2.5-7B-Instruct"
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of renewable energy?",
    ]

    print(f"VRAMancer TurboQuant Benchmark — {model}")
    print(f"GPU 0 (vis): {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    print(f"  {'Configuration':<40s}  {'Speed':>10s}  {'VRAM':>7s}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*7}")

    configs = [
        ("BF16 baseline", {"VRM_KV_COMPRESSION": "", "VRM_QUANTIZATION": ""}),
        ("TQ 3bit", {"VRM_KV_COMPRESSION": "turboquant", "VRM_QUANTIZATION": ""}),
        ("TQ 3bit + Sparse V 10%", {"VRM_KV_COMPRESSION": "turboquant", "VRM_SPARSE_V_RATIO": "0.1", "VRM_QUANTIZATION": ""}),
    ]

    for name, env in configs:
        bench(model, name, env, prompts, max_tokens=64)

if __name__ == "__main__":
    main()
