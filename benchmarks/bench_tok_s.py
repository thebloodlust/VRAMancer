#!/usr/bin/env python3
"""VRAMancer tok/s benchmark — honest numbers.

Compares:
  1. Native HuggingFace generate() — baseline (best possible single-GPU)
  2. VRAMancer pipeline 1 GPU — overhead measurement
  3. VRAMancer pipeline 2 GPU — multi-GPU value (for larger models)

Usage:
    python benchmarks/bench_tok_s.py [--model gpt2] [--max-tokens 128] [--num-prompts 5]
"""
import torch
import time
import os
import sys
import json
import argparse

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land",
    "Python is a programming language that",
    "The best way to learn programming is",
    "In the year 2050 the world will",
    "Machine learning models have become",
    "The most important thing about software",
    "When we think about the universe",
]


def bench_native_hf(model_name, prompts, max_tokens):
    """Baseline: raw HuggingFace generate()."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).cuda()
    model.eval()

    # Warmup
    with torch.no_grad():
        inp = tok("Hello world", return_tensors="pt").to("cuda")
        model.generate(**inp, max_new_tokens=20, do_sample=False)
    torch.cuda.synchronize()

    total_gen = 0
    start = time.perf_counter()
    with torch.no_grad():
        for p in prompts:
            inp = tok(p, return_tensors="pt").to("cuda")
            out = model.generate(
                **inp, max_new_tokens=max_tokens, do_sample=False, use_cache=True
            )
            total_gen += out.shape[1] - inp["input_ids"].shape[1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    del model
    torch.cuda.empty_cache()
    return {"tokens": total_gen, "elapsed_s": round(elapsed, 2), "tok_s": round(total_gen / elapsed, 1)}


def bench_vramancer_sequential(model_name, prompts, max_tokens, num_gpus=1):
    """VRAMancer InferencePipeline sequential generate()."""
    from core.inference_pipeline import InferencePipeline

    pipe = InferencePipeline(
        backend_name="huggingface",
        enable_metrics=False,
        enable_discovery=False,
        verbose=False,
    )
    pipe.load(model_name, num_gpus=num_gpus)

    # Warmup
    pipe.generate("Hello world", max_new_tokens=20)
    torch.cuda.synchronize()

    total_gen = 0
    start = time.perf_counter()
    for p in prompts:
        pipe.generate(p, max_new_tokens=max_tokens)
        total_gen += max_tokens
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    pipe.shutdown()
    del pipe
    torch.cuda.empty_cache()
    return {"tokens": total_gen, "elapsed_s": round(elapsed, 2), "tok_s": round(total_gen / elapsed, 1)}


def _run_in_subprocess(func_name, *args):
    """Run benchmark function in a subprocess to get clean CUDA context."""
    import subprocess, tempfile, textwrap

    # Serialize args to pass to subprocess
    args_json = json.dumps(args)
    script = textwrap.dedent(f"""\
        import sys, json, os
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        from benchmarks.bench_tok_s import {func_name}, PROMPTS
        args = json.loads('{args_json}')
        result = {func_name}(*args)
        print("BENCH_RESULT:" + json.dumps(result))
    """)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        f.flush()
        proc = subprocess.run(
            [sys.executable, f.name],
            capture_output=True, text=True, timeout=600,
        )

    os.unlink(f.name)

    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])

    # If we got here, the subprocess failed
    print(f"STDERR:\n{proc.stderr[-2000:]}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--in-process", action="store_true", help="Run in same process (no subprocess isolation)")
    args = parser.parse_args()

    prompts = PROMPTS[: args.num_prompts]
    short = args.model.split("/")[-1]

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {short} | {args.max_tokens} tokens | {args.num_prompts} prompts | {args.num_gpus} GPU(s)")
    print(f"{'='*70}")

    results = {"model": short, "max_tokens": args.max_tokens, "num_prompts": args.num_prompts}
    run = (lambda fn, *a: fn(*a)) if args.in_process else (lambda fn, *a: _run_in_subprocess(fn.__name__, *a))

    if not args.skip_native and args.num_gpus == 1:
        print(f"\n[1/2] Native HuggingFace generate()...")
        r = run(bench_native_hf, args.model, prompts, args.max_tokens)
        if r:
            results["native_hf"] = r
            print(f"  => {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
        else:
            print("  => FAILED")

    label = f"{args.num_gpus}GPU"
    print(f"\n[2/2] VRAMancer pipeline ({label})...")
    r = run(bench_vramancer_sequential, args.model, prompts, args.max_tokens, args.num_gpus)
    if r:
        results[f"vramancer_{label}"] = r
        print(f"  => {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
    else:
        print("  => FAILED")

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(json.dumps(results, indent=2))

    if "native_hf" in results and f"vramancer_{label}" in results:
        overhead = (1 - results[f"vramancer_{label}"]["tok_s"] / results["native_hf"]["tok_s"]) * 100
        print(f"\nVRAMancer overhead vs native HF: {overhead:+.1f}%")

    return results


if __name__ == "__main__":
    main()
