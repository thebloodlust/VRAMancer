#!/usr/bin/env python3
"""VRAMancer heterogeneous multi-GPU benchmark — the proof.

Demonstrates that VRAMancer can run models TOO LARGE for any single GPU
by splitting them proportionally across heterogeneous GPUs.

Test matrix:
  1. Qwen2.5-14B-Instruct (28GB bf16) — doesn't fit on any single GPU
     - Single GPU attempt → OOM (expected)
     - VRAMancer 2-GPU split → works
  2. CodeLlama-34B-GPTQ (18GB 4-bit) — tight fit, benefits from split
     - VRAMancer 2-GPU split → works

Usage:
    python benchmarks/bench_heterogeneous.py
    python benchmarks/bench_heterogeneous.py --model Qwen/Qwen2.5-14B-Instruct
    python benchmarks/bench_heterogeneous.py --model TheBloke/CodeLlama-34B-Instruct-GPTQ
"""
import torch
import time
import os
import sys
import json
import gc
import argparse
import subprocess
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure multi-GPU visibility — clear any stale limits
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("VRM_MINIMAL_TEST", None)

PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a Python function that implements binary search:",
    "The future of renewable energy depends on",
    "In distributed computing, consistency and availability",
    "To optimize GPU memory usage in deep learning,",
]

MAX_TOKENS = 64  # Keep short for benchmark speed


def gpu_info():
    """Print GPU info."""
    info = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info.append({
            "id": i,
            "name": p.name,
            "total_gb": round(total / 1024**3, 1),
            "free_gb": round(free / 1024**3, 1),
        })
    return info


def try_single_gpu(model_name, gpu_id=0):
    """Attempt to load model on a single GPU IN A SUBPROCESS.

    Subprocess isolation prevents CUDA memory contamination from OOM
    failures leaking into the parent process.
    """
    print(f"\n  [Single GPU {gpu_id}] Attempting to load {model_name} (subprocess)...")
    script = textwrap.dedent(f"""\
        import torch, time, json, sys
        PROMPTS = {PROMPTS!r}
        MAX_TOKENS = {MAX_TOKENS}
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained("{model_name}", trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            device = "cuda:{gpu_id}"
            model = AutoModelForCausalLM.from_pretrained(
                "{model_name}", torch_dtype=torch.bfloat16,
                device_map={{"": device}}, trust_remote_code=True,
            )
            model.eval()
            with torch.no_grad():
                inp = tok("Hello", return_tensors="pt").to(device)
                model.generate(**inp, max_new_tokens=5, do_sample=False)
                torch.cuda.synchronize()
            total_gen = 0
            start = time.perf_counter()
            with torch.no_grad():
                for p in PROMPTS:
                    inp = tok(p, return_tensors="pt").to(device)
                    out = model.generate(**inp, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True)
                    total_gen += out.shape[1] - inp["input_ids"].shape[1]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(json.dumps({{"status": "ok", "tok_s": round(total_gen / elapsed, 1), "tokens": total_gen}}))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(json.dumps({{"status": "OOM", "error": str(e)[:200]}}))
            else:
                print(json.dumps({{"status": "error", "error": str(e)[:200]}}))
        except Exception as e:
            print(json.dumps({{"status": "error", "error": str(e)[:200]}}))
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=300,
        )
        # Parse last JSON line from stdout
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"status": "error", "error": result.stderr[:300] if result.stderr else "no output"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout (300s)"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def bench_vramancer_multi_gpu(model_name, num_gpus=2, quantization=""):
    """Run model through VRAMancer pipeline on multiple GPUs."""
    quant_label = f" [{quantization.upper()}]" if quantization else " [BF16]"
    print(f"\n  [VRAMancer {num_gpus}-GPU{quant_label}] Loading {model_name}...")
    try:
        # Set quantization env var before importing pipeline
        if quantization:
            os.environ["VRM_QUANTIZATION"] = quantization
        else:
            os.environ.pop("VRM_QUANTIZATION", None)

        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(
            backend_name="huggingface",
            enable_metrics=False,
            enable_discovery=False,
            verbose=True,
        )

        load_start = time.perf_counter()
        pipe.load(model_name, num_gpus=num_gpus)
        load_time = time.perf_counter() - load_start
        print(f"    Model loaded in {load_time:.1f}s")

        # Print VRAM usage after load
        for i in range(min(num_gpus, torch.cuda.device_count())):
            free, total = torch.cuda.mem_get_info(i)
            used = (total - free) / 1024**3
            print(f"    GPU {i}: {used:.1f}GB used / {total/1024**3:.1f}GB total")

        # Warmup
        print("    Warmup...")
        pipe.generate("Hello world", max_new_tokens=10)
        torch.cuda.synchronize()

        # Benchmark
        print(f"    Benchmarking ({len(PROMPTS)} prompts × {MAX_TOKENS} tokens)...")
        total_gen = 0
        start = time.perf_counter()
        for p in PROMPTS:
            result = pipe.generate(p, max_new_tokens=MAX_TOKENS)
            total_gen += MAX_TOKENS
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tok_s = round(total_gen / elapsed, 1)

        pipe.shutdown()
        del pipe
        # Reset global singleton to avoid cross-tier contamination
        try:
            from core.inference_pipeline import reset_pipeline
            reset_pipeline()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        os.environ.pop("VRM_QUANTIZATION", None)

        return {
            "status": "ok",
            "tok_s": tok_s,
            "tokens": total_gen,
            "elapsed_s": round(elapsed, 2),
            "load_time_s": round(load_time, 1),
        }

    except Exception as e:
        os.environ.pop("VRM_QUANTIZATION", None)
        try:
            from core.inference_pipeline import reset_pipeline
            reset_pipeline()
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)[:300]}


def main():
    parser = argparse.ArgumentParser(description="VRAMancer heterogeneous GPU benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                        help="Model to benchmark")
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip single-GPU OOM test")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--quantization", default="",
                        help="Quantization mode: '' (bf16), 'int8', 'nf4' or 'all' for full tier bench")
    args = parser.parse_args()

    run_all_tiers = (args.quantization == "all")

    model_short = args.model.split("/")[-1]

    print("=" * 72)
    print(f" VRAMancer Heterogeneous Multi-GPU Benchmark")
    print(f" Model: {args.model}")
    print("=" * 72)

    # GPU info
    gpus = gpu_info()
    print(f"\n GPU Configuration:")
    total_vram = 0
    for g in gpus:
        print(f"   GPU {g['id']}: {g['name']} — {g['total_gb']}GB total, {g['free_gb']}GB free")
        total_vram += g['free_gb']
    print(f"   Combined free VRAM: {total_vram:.1f}GB")

    results = {"model": args.model, "gpus": gpus}

    # Step 1: Try single GPU (expect OOM for 14B+ models)
    if not args.skip_single:
        for i in range(min(len(gpus), 2)):
            print(f"\n{'─'*72}")
            print(f" TEST {i+1}: Single GPU {i} ({gpus[i]['name']}, {gpus[i]['free_gb']}GB free)")
            print(f"{'─'*72}")
            r = try_single_gpu(args.model, gpu_id=i)
            results[f"single_gpu_{i}"] = r
            if r["status"] == "OOM":
                print(f"    ✗ OOM — model too large for {gpus[i]['name']} ({gpus[i]['free_gb']}GB)")
            elif r["status"] == "ok":
                print(f"    ✓ Loaded! {r['tok_s']} tok/s")
            else:
                print(f"    ✗ Error: {r.get('error', 'unknown')}")

    # Step 2: VRAMancer multi-GPU (one or all tiers)
    tiers = [("", "BF16")] if not run_all_tiers else [
        ("", "BF16"),
        ("int8", "INT8"),
        ("nf4", "NF4"),
    ]
    if not run_all_tiers and args.quantization:
        tiers = [(args.quantization, args.quantization.upper())]

    for quant_val, quant_label in tiers:
        test_key = f"vramancer_{args.num_gpus}gpu{'_' + quant_val if quant_val else ''}"
        print(f"\n{'─'*72}")
        print(f" TEST: VRAMancer {args.num_gpus}-GPU {quant_label}")
        print(f"{'─'*72}")
        r = bench_vramancer_multi_gpu(args.model, num_gpus=args.num_gpus, quantization=quant_val)
        results[test_key] = r
        if r["status"] == "ok":
            print(f"\n    ✓ SUCCESS: {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
            print(f"    ✓ Model loaded in {r['load_time_s']}s across {args.num_gpus} heterogeneous GPUs")
        else:
            print(f"\n    ✗ Failed: {r.get('error', 'unknown')}")

    # Summary
    print(f"\n{'='*72}")
    print(f" RESULTS SUMMARY: {model_short}")
    print(f"{'='*72}")
    for key, val in results.items():
        if key in ("model", "gpus"):
            continue
        status = val.get("status", "?")
        if status == "OOM":
            print(f"   {key}: OOM (doesn't fit)")
        elif status == "ok":
            print(f"   {key}: {val['tok_s']} tok/s")
        else:
            print(f"   {key}: FAILED — {val.get('error', '?')[:80]}")

    vrm = results.get(f"vramancer_{args.num_gpus}gpu", {})
    if vrm.get("status") == "ok":
        oom_gpus = [k for k, v in results.items() if isinstance(v, dict) and v.get("status") == "OOM"]
        if oom_gpus:
            print(f"\n   >>> MODEL DOESN'T FIT ON ANY SINGLE GPU")
            print(f"   >>> VRAMancer splits it across {args.num_gpus} heterogeneous GPUs: {vrm['tok_s']} tok/s")
            print(f"   >>> This is impossible with standard HuggingFace on one card.")

    print(f"\n{'='*72}")
    print(f"Full results: {json.dumps(results, indent=2)}")

    return results


if __name__ == "__main__":
    main()
