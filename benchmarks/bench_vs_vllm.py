#!/usr/bin/env python3
"""VRAMancer vs vLLM benchmark — honest head-to-head comparison.

Compares on the same model/GPU/prompt/max_tokens:
  1. VRAMancer InferencePipeline (HuggingFace backend)
  2. vLLM offline inference (LLM class)
  3. Native HuggingFace (baseline)

Usage:
    # Single GPU (recommended: isolate GPU to avoid TDR)
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_vs_vllm.py --model mistralai/Mistral-7B-v0.1

    # Smaller model for quick test
    python benchmarks/bench_vs_vllm.py --model gpt2
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant land",
    "Python is a programming language that",
    "The best way to learn programming is",
    "In the year 2050 the world will",
]


def _run_in_subprocess(script_code, timeout=600, python_exe=None):
    """Run Python code in a clean subprocess, return parsed BENCH_RESULT."""
    exe = python_exe or sys.executable
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_code)
        f.flush()
        try:
            proc = subprocess.run(
                [exe, f.name],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            return None
        finally:
            os.unlink(f.name)

    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])

    print(f"  STDERR (last 2000 chars):\n{proc.stderr[-2000:]}")
    return None


def bench_native_hf(model_name, prompts, max_tokens):
    """Native HuggingFace — subprocess isolated."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(f"""\
        import sys, json, os, time, torch
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, "{root}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = {model_name!r}
        prompts = {prompts!r}
        max_tokens = {max_tokens}

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
        model.eval()

        # Warmup
        with torch.no_grad():
            inp = tok("Hello", return_tensors="pt").to("cuda")
            model.generate(**inp, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()

        total_gen = 0
        start = time.perf_counter()
        with torch.no_grad():
            for p in prompts:
                inp = tok(p, return_tensors="pt").to("cuda")
                out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, use_cache=True)
                total_gen += out.shape[1] - inp["input_ids"].shape[1]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print("BENCH_RESULT:" + json.dumps({{
            "engine": "native_hf",
            "tokens": total_gen, "elapsed_s": round(elapsed, 2),
            "tok_s": round(total_gen / elapsed, 1),
        }}))
    """)
    return _run_in_subprocess(script)


def bench_vramancer(model_name, prompts, max_tokens):
    """VRAMancer pipeline — subprocess isolated."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(f"""\
        import sys, json, os, time, torch
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        sys.path.insert(0, "{root}")
        from core.inference_pipeline import InferencePipeline

        model_name = {model_name!r}
        prompts = {prompts!r}
        max_tokens = {max_tokens}

        pipe = InferencePipeline(
            backend_name="huggingface",
            enable_metrics=False,
            enable_discovery=False,
            verbose=False,
        )
        pipe.load(model_name, num_gpus=1)

        # Warmup
        pipe.generate("Hello", max_new_tokens=10)
        torch.cuda.synchronize()

        total_gen = 0
        start = time.perf_counter()
        for p in prompts:
            pipe.generate(p, max_new_tokens=max_tokens)
            total_gen += max_tokens
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        pipe.shutdown()
        print("BENCH_RESULT:" + json.dumps({{
            "engine": "vramancer",
            "tokens": total_gen, "elapsed_s": round(elapsed, 2),
            "tok_s": round(total_gen / elapsed, 1),
        }}))
    """)
    return _run_in_subprocess(script)


def bench_vllm(model_name, prompts, max_tokens):
    """vLLM offline inference — subprocess isolated."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(f"""\
        import sys, json, os, time
        sys.path.insert(0, "{root}")
        model_name = {model_name!r}
        prompts = {prompts!r}
        max_tokens = {max_tokens}

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("BENCH_RESULT:" + json.dumps({{"engine": "vllm", "error": "vllm not installed"}}))
            sys.exit(0)

        llm = LLM(model=model_name, dtype="bfloat16", enforce_eager=True)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        # Warmup
        llm.generate(["Hello world"], params)

        start = time.perf_counter()
        outputs = llm.generate(prompts, params)
        elapsed = time.perf_counter() - start

        total_gen = sum(len(o.outputs[0].token_ids) for o in outputs)

        print("BENCH_RESULT:" + json.dumps({{
            "engine": "vllm",
            "tokens": total_gen, "elapsed_s": round(elapsed, 2),
            "tok_s": round(total_gen / elapsed, 1),
        }}))
    """)
    # Use .venv_vllm if available (vLLM needs different transformers version)
    vllm_python = os.path.join(root, ".venv_vllm", "bin", "python")
    python_exe = vllm_python if os.path.exists(vllm_python) else None
    return _run_in_subprocess(script, timeout=900, python_exe=python_exe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()

    prompts = PROMPTS[:args.num_prompts]
    short = args.model.split("/")[-1]

    print(f"\n{'='*70}")
    print(f"VRAMancer vs vLLM vs Native HF — {short}")
    print(f"Max tokens: {args.max_tokens} | Prompts: {args.num_prompts}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"{'='*70}")

    results = {"model": short, "max_tokens": args.max_tokens}

    # 1. Native HF
    if not args.skip_native:
        print(f"\n[1/3] Native HuggingFace...")
        r = bench_native_hf(args.model, prompts, args.max_tokens)
        if r and "error" not in r:
            results["native_hf"] = r
            print(f"  => {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
        else:
            print(f"  => FAILED: {r}")

    # 2. VRAMancer
    print(f"\n[2/3] VRAMancer pipeline...")
    r = bench_vramancer(args.model, prompts, args.max_tokens)
    if r and "error" not in r:
        results["vramancer"] = r
        print(f"  => {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
    else:
        print(f"  => FAILED: {r}")

    # 3. vLLM
    if not args.skip_vllm:
        print(f"\n[3/3] vLLM offline inference...")
        r = bench_vllm(args.model, prompts, args.max_tokens)
        if r and "error" not in r:
            results["vllm"] = r
            print(f"  => {r['tok_s']} tok/s ({r['tokens']} tokens in {r['elapsed_s']}s)")
        elif r and "error" in r:
            results["vllm"] = r
            print(f"  => SKIPPED: {r['error']}")
        else:
            print(f"  => FAILED")

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"{'Engine':<15} {'Tok/s':>8} {'Tokens':>8} {'Time(s)':>8} {'vs HF':>8}")
    print("-" * 50)

    hf_toks = results.get("native_hf", {}).get("tok_s", 0)
    for key in ["native_hf", "vramancer", "vllm"]:
        if key in results and "error" not in results[key]:
            r = results[key]
            delta = ""
            if hf_toks and key != "native_hf":
                pct = ((r["tok_s"] / hf_toks) - 1) * 100
                delta = f"{pct:+.1f}%"
            print(f"{r['engine']:<15} {r['tok_s']:>8.1f} {r['tokens']:>8} "
                  f"{r['elapsed_s']:>8.1f} {delta:>8}")

    print(f"\n{json.dumps(results, indent=2)}")
    return results


if __name__ == "__main__":
    main()
