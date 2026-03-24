#!/usr/bin/env python3
"""VRAMancer llama.cpp (GGUF) benchmark.

Compares GGUF quantized inference via llama-cpp-python against
BnB NF4 via HuggingFace, both raw and through VRAMancer.

Test matrix:
  1. Raw llama-cpp-python (Llama() direct) — single GPU, all layers offloaded
  2. VRAMancer LlamaCppBackend — single GPU
  3. VRAMancer LlamaCppBackend — 2-GPU tensor_split (heterogeneous)
  4. BnB NF4 via VRAMancer HuggingFace backend — single GPU (baseline)

Models tested (auto-downloaded from HuggingFace Hub):
  - Qwen2.5-7B-Instruct-GGUF (Q4_K_M) — fits single GPU
  - Qwen2.5-14B-Instruct-GGUF (Q4_K_M) — fits single GPU in Q4

Usage:
    python benchmarks/bench_llamacpp.py
    python benchmarks/bench_llamacpp.py --model bartowski/Qwen2.5-7B-Instruct-GGUF
    python benchmarks/bench_llamacpp.py --model bartowski/Qwen2.5-14B-Instruct-GGUF
    python benchmarks/bench_llamacpp.py --skip-nf4   # skip BnB NF4 comparison
"""
import time
import os
import sys
import json
import gc
import argparse
import subprocess
import textwrap
import socket

# Force IPv4 — IPv6 routing to HuggingFace CDN is broken on some networks
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    return [r for r in _orig_getaddrinfo(*args, **kwargs) if r[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("VRM_MINIMAL_TEST", None)

PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms:",
    "Write a Python function that implements binary search:",
    "The future of renewable energy depends on",
    "In distributed computing, consistency and availability",
    "To optimize GPU memory usage in deep learning,",
]

MAX_TOKENS = 128

# Default GGUF repos on HuggingFace Hub (bartowski is the reference uploader)
DEFAULT_MODELS = [
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
]


def gpu_info():
    """Print GPU info."""
    import torch
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


def print_gpu_vram(label=""):
    """Print current VRAM usage for all GPUs."""
    import torch
    if label:
        print(f"  [{label}]")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / 1024**3
        total_gb = total / 1024**3
        print(f"    GPU {i}: {used:.1f} GB used / {total_gb:.1f} GB total")


def resolve_gguf_path(repo_id):
    """Download a Q4_K_M GGUF file from HuggingFace Hub, return local path."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]

    # Prefer Q4_K_M > Q5_K_M > Q4_K_S > Q8_0 > first available
    for preferred in ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q8_0"]:
        for f in gguf_files:
            if preferred in f:
                print(f"  Downloading {f} from {repo_id}...")
                return hf_hub_download(repo_id, f)

    if gguf_files:
        print(f"  Downloading {gguf_files[0]} from {repo_id}...")
        return hf_hub_download(repo_id, gguf_files[0])

    raise FileNotFoundError(f"No .gguf files found in {repo_id}")


# ── Benchmark 1: Raw llama-cpp-python (direct Llama()) ──────────────

def bench_raw_llamacpp(gguf_path, num_gpus=1):
    """Benchmark raw llama-cpp-python without VRAMancer."""
    print(f"\n{'='*70}")
    print(f"  RAW llama-cpp-python — {os.path.basename(gguf_path)}")
    print(f"  GPUs: {num_gpus}")
    print(f"{'='*70}")

    import torch
    from llama_cpp import Llama

    # Compute tensor_split for multi-GPU
    tensor_split = None
    if num_gpus > 1:
        vram_values = []
        for i in range(min(num_gpus, torch.cuda.device_count())):
            free, _ = torch.cuda.mem_get_info(i)
            vram_values.append(free / 1024**3)
        total_vram = sum(vram_values)
        tensor_split = [v / total_vram for v in vram_values]
        print(f"  tensor_split: {[round(v, 3) for v in tensor_split]}")

    # Load model
    load_start = time.perf_counter()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        flash_attn=True,
        verbose=False,
        tensor_split=tensor_split,
    )
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time:.1f}s")
    print_gpu_vram("After load")

    # Warmup
    print("  Warmup...")
    llm.create_completion("Hello", max_tokens=10)

    # Benchmark
    print(f"  Generating {len(PROMPTS)} prompts × {MAX_TOKENS} tokens...")
    total_gen = 0
    ttft_list = []

    start = time.perf_counter()
    for prompt in PROMPTS:
        # Measure TTFT via streaming
        t0 = time.perf_counter()
        first_token_time = None
        tokens_this = 0

        for chunk in llm.create_completion(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
            stream=True,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
                ttft_list.append(first_token_time)
            tokens_this += 1

        total_gen += tokens_this

    elapsed = time.perf_counter() - start
    tok_s = round(total_gen / elapsed, 1)
    avg_ttft = round(sum(ttft_list) / len(ttft_list) * 1000, 1) if ttft_list else 0

    print(f"\n  ── Results ──")
    print(f"  Tokens generated: {total_gen}")
    print(f"  Total time:       {elapsed:.2f}s")
    print(f"  Throughput:       {tok_s} tok/s")
    print(f"  Avg TTFT:         {avg_ttft} ms")
    print_gpu_vram("Final VRAM")

    # Cleanup
    del llm
    gc.collect()

    return {
        "method": f"raw_llamacpp_{num_gpus}gpu",
        "gguf": os.path.basename(gguf_path),
        "num_gpus": num_gpus,
        "tok_s": tok_s,
        "ttft_ms": avg_ttft,
        "tokens": total_gen,
        "elapsed_s": round(elapsed, 2),
        "load_time_s": round(load_time, 1),
    }


# ── Benchmark 2: VRAMancer LlamaCppBackend ──────────────────────────

def bench_vramancer_llamacpp(gguf_repo, num_gpus=1):
    """Benchmark VRAMancer's LlamaCppBackend."""
    print(f"\n{'='*70}")
    print(f"  VRAMancer LlamaCppBackend — {gguf_repo}")
    print(f"  GPUs: {num_gpus}")
    print(f"{'='*70}")

    import torch
    from core.backends_llamacpp import LlamaCppBackend

    backend = LlamaCppBackend()

    # Load model
    load_start = time.perf_counter()
    backend.load_model(gguf_repo, n_gpu_layers=-1, n_ctx=4096, num_gpus=num_gpus)
    load_time = time.perf_counter() - load_start
    print(f"  Load time: {load_time:.1f}s")
    print_gpu_vram("After load")

    # Warmup
    print("  Warmup...")
    backend.generate("Hello", max_new_tokens=10)

    # Benchmark
    print(f"  Generating {len(PROMPTS)} prompts × {MAX_TOKENS} tokens...")
    total_gen = 0
    ttft_list = []

    start = time.perf_counter()
    for prompt in PROMPTS:
        t0 = time.perf_counter()
        first_token_time = None
        tokens_this = 0

        for chunk in backend.generate_stream(
            prompt,
            max_new_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.1,
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
                ttft_list.append(first_token_time)
            tokens_this += 1

        total_gen += tokens_this

    elapsed = time.perf_counter() - start
    tok_s = round(total_gen / elapsed, 1)
    avg_ttft = round(sum(ttft_list) / len(ttft_list) * 1000, 1) if ttft_list else 0

    print(f"\n  ── Results ──")
    print(f"  Tokens generated: {total_gen}")
    print(f"  Total time:       {elapsed:.2f}s")
    print(f"  Throughput:       {tok_s} tok/s")
    print(f"  Avg TTFT:         {avg_ttft} ms")
    print_gpu_vram("Final VRAM")

    # Cleanup
    del backend
    gc.collect()

    return {
        "method": f"vramancer_llamacpp_{num_gpus}gpu",
        "model": gguf_repo,
        "num_gpus": num_gpus,
        "tok_s": tok_s,
        "ttft_ms": avg_ttft,
        "tokens": total_gen,
        "elapsed_s": round(elapsed, 2),
        "load_time_s": round(load_time, 1),
    }


# ── Benchmark 3: BnB NF4 via HuggingFace (subprocess) ──────────────

def bench_nf4_subprocess(hf_model, gpu_id=0):
    """Benchmark BnB NF4 in a subprocess (isolates CUDA context)."""
    print(f"\n{'='*70}")
    print(f"  BnB NF4 baseline — {hf_model}")
    print(f"  GPU: {gpu_id}")
    print(f"{'='*70}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = textwrap.dedent(f"""\
        import torch, time, json, sys, gc, socket
        # Force IPv4 (IPv6 routing broken on some networks)
        _orig = socket.getaddrinfo
        def _ipv4(*a, **k):
            return [r for r in _orig(*a, **k) if r[0] == socket.AF_INET]
        socket.getaddrinfo = _ipv4
        sys.path.insert(0, "{project_root}")
        PROMPTS = {PROMPTS!r}
        MAX_TOKENS = {MAX_TOKENS}
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            tok = AutoTokenizer.from_pretrained("{hf_model}", trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            device = "cuda:{gpu_id}"

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            load_start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                "{hf_model}",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map={{"": device}},
                trust_remote_code=True,
            )
            load_time = time.perf_counter() - load_start

            # VRAM after load
            free, total = torch.cuda.mem_get_info({gpu_id})
            vram_used = round((total - free) / 1024**3, 1)

            # Warmup
            inp = tok("Hello", return_tensors="pt").to(device)
            model.generate(**inp, max_new_tokens=10, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()

            # Benchmark
            total_gen = 0
            start = time.perf_counter()
            for prompt in PROMPTS:
                inp = tok(prompt, return_tensors="pt").to(device)
                out = model.generate(
                    **inp, max_new_tokens=MAX_TOKENS,
                    temperature=0.7, top_p=0.95, top_k=40,
                    do_sample=True, pad_token_id=tok.pad_token_id,
                )
                total_gen += out.shape[1] - inp["input_ids"].shape[1]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            tok_s = round(total_gen / elapsed, 1)

            result = {{
                "method": "bnb_nf4",
                "model": "{hf_model}",
                "tok_s": tok_s,
                "tokens": total_gen,
                "elapsed_s": round(elapsed, 2),
                "load_time_s": round(load_time, 1),
                "vram_gb": vram_used,
            }}
            print("BENCH_RESULT:" + json.dumps(result))
        except Exception as e:
            print("BENCH_RESULT:" + json.dumps({{"method": "bnb_nf4", "error": str(e)}}))
    """)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=600,
            cwd=project_root,
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: NF4 subprocess exceeded 600s")
        return {"method": "bnb_nf4", "error": "timeout"}

    # Show subprocess output
    for line in result.stdout.splitlines():
        if not line.startswith("BENCH_RESULT:"):
            print(f"  {line}")

    # Parse result
    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            data = json.loads(line[len("BENCH_RESULT:"):])
            if "error" in data:
                print(f"  ERROR: {data['error']}")
            else:
                print(f"\n  ── Results ──")
                print(f"  Throughput: {data['tok_s']} tok/s")
                print(f"  Load time:  {data['load_time_s']}s")
                print(f"  VRAM used:  {data.get('vram_gb', '?')} GB")
            return data

    # If no result parsed
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[-500:]}")
    return {"method": "bnb_nf4", "error": "no result parsed"}


# ── Main ────────────────────────────────────────────────────────────

def main():
    global MAX_TOKENS
    parser = argparse.ArgumentParser(description="VRAMancer llama.cpp (GGUF) benchmark")
    parser.add_argument("--model", type=str, default=None,
                        help="HF repo for GGUF model (e.g. bartowski/Qwen2.5-7B-Instruct-GGUF)")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="HF model for NF4 comparison (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--skip-nf4", action="store_true",
                        help="Skip BnB NF4 comparison")
    parser.add_argument("--skip-multi", action="store_true",
                        help="Skip multi-GPU tests")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens per generation (default: {MAX_TOKENS})")
    args = parser.parse_args()

    MAX_TOKENS = args.max_tokens

    import torch
    print("=" * 70)
    print("  VRAMancer llama.cpp (GGUF) Benchmark")
    print("=" * 70)

    # GPU info
    gpus = gpu_info()
    num_gpus = len(gpus)
    print(f"\n  GPUs detected: {num_gpus}")
    for g in gpus:
        print(f"    GPU {g['id']}: {g['name']} — {g['free_gb']} GB free / {g['total_gb']} GB total")

    # Resolve models
    gguf_repos = [args.model] if args.model else DEFAULT_MODELS

    # Infer HF model name from GGUF repo for NF4 comparison
    hf_model = args.hf_model
    if not hf_model and not args.skip_nf4:
        # bartowski/Qwen2.5-7B-Instruct-GGUF → Qwen/Qwen2.5-7B-Instruct
        for repo in gguf_repos:
            name = repo.split("/")[-1].replace("-GGUF", "")
            # Common patterns: bartowski → Qwen, etc.
            hf_model = f"Qwen/{name}"
            break

    all_results = []

    for repo in gguf_repos:
        print(f"\n{'#'*70}")
        print(f"  Model: {repo}")
        print(f"{'#'*70}")

        # Step 1: Download GGUF file
        print("\n  Resolving GGUF file...")
        try:
            gguf_path = resolve_gguf_path(repo)
            gguf_size_gb = round(os.path.getsize(gguf_path) / 1024**3, 1)
            print(f"  GGUF file: {os.path.basename(gguf_path)} ({gguf_size_gb} GB)")
        except Exception as e:
            print(f"  ERROR resolving GGUF: {e}")
            continue

        # Step 2: Raw llama-cpp-python — single GPU
        try:
            r = bench_raw_llamacpp(gguf_path, num_gpus=1)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR raw llamacpp 1-GPU: {e}")

        torch.cuda.empty_cache()
        gc.collect()

        # Step 3: VRAMancer LlamaCppBackend — single GPU
        try:
            r = bench_vramancer_llamacpp(repo, num_gpus=1)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR VRAMancer llamacpp 1-GPU: {e}")

        torch.cuda.empty_cache()
        gc.collect()

        # Step 4: Multi-GPU (if available and not skipped)
        if num_gpus >= 2 and not args.skip_multi:
            try:
                r = bench_raw_llamacpp(gguf_path, num_gpus=num_gpus)
                all_results.append(r)
            except Exception as e:
                print(f"  ERROR raw llamacpp {num_gpus}-GPU: {e}")

            torch.cuda.empty_cache()
            gc.collect()

            try:
                r = bench_vramancer_llamacpp(repo, num_gpus=num_gpus)
                all_results.append(r)
            except Exception as e:
                print(f"  ERROR VRAMancer llamacpp {num_gpus}-GPU: {e}")

            torch.cuda.empty_cache()
            gc.collect()

        # Step 5: BnB NF4 comparison (subprocess)
        if not args.skip_nf4 and hf_model:
            r = bench_nf4_subprocess(hf_model, gpu_id=0)
            if r:
                all_results.append(r)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<35} {'tok/s':>8} {'TTFT':>10} {'Load':>8} {'Tokens':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['method']:<35} {'ERROR':>8}   {r.get('error', '')[:30]}")
            continue
        ttft = f"{r.get('ttft_ms', '-')} ms" if r.get('ttft_ms') else "-"
        print(f"  {r['method']:<35} {r['tok_s']:>8} {ttft:>10} {r.get('load_time_s', '-'):>7}s {r.get('tokens', '-'):>8}")

    # Compute speedup vs NF4 if both available
    nf4_result = next((r for r in all_results if r.get("method") == "bnb_nf4" and "tok_s" in r), None)
    gguf_1gpu = next((r for r in all_results if "raw_llamacpp_1gpu" in r.get("method", "") and "tok_s" in r), None)

    if nf4_result and gguf_1gpu:
        speedup = round(gguf_1gpu["tok_s"] / nf4_result["tok_s"], 2)
        print(f"\n  🔑 GGUF Q4_K_M vs BnB NF4: {speedup}x speedup")
        if speedup > 1:
            print(f"     llama.cpp dp4a kernels: {gguf_1gpu['tok_s']} tok/s")
            print(f"     BnB NF4 dequant→fp16:   {nf4_result['tok_s']} tok/s")

    # Save results
    results_file = os.path.join(os.path.dirname(__file__), "llamacpp_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()
