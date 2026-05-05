#!/usr/bin/env python3
"""Benchmark: P2P vs CPU-staged on all-reduce and tensor-parallel inference.

Quantifies the real tok/s impact of Proxmox P2P on:
  1. Raw all-reduce microbench (NCCL) at varying tensor sizes
  2. Tensor-parallel inference at varying batch sizes (Mistral-7B or Qwen-7B)

Usage:
    # Full bench (TP inference requires ~14 GB VRAM)
    source .venv/bin/activate
    python benchmarks/bench_p2p_impact.py

    # Microbench only (fast, no model loading)
    python benchmarks/bench_p2p_impact.py --allreduce-only

    # Specific model
    python benchmarks/bench_p2p_impact.py --model mistralai/Mistral-7B-v0.1
"""
import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. All-reduce microbench
# ─────────────────────────────────────────────────────────────────────────────

def bench_allreduce(n_iters: int = 50) -> dict:
    """Measure NCCL all-reduce latency and bandwidth at multiple tensor sizes.

    Tensor sizes are chosen to match real TP activations:
      hidden_size × batch × seq  (bfloat16)
      - 7B model, hidden=4096:  batch=1 → 8 KB, batch=8 → 64 KB, batch=32 → 256 KB
      - MLP intermediate (11008): batch=1 → 22 KB, batch=32 → 704 KB
    We also test large sizes (8 MB, 64 MB) to show theoretical P2P peak.
    """
    print("\n" + "=" * 64)
    print("  PART 1 — All-Reduce Microbench (NCCL, 2 GPUs)")
    print("=" * 64)

    if torch.cuda.device_count() < 2:
        print("  SKIP: need 2 GPUs")
        return {}

    # Check NCCL availability
    try:
        t0 = torch.zeros(4, device="cuda:0")
        t1 = torch.zeros(4, device="cuda:1")
        torch.cuda.nccl.all_reduce([t0, t1])
        del t0, t1
        nccl_ok = True
    except Exception as e:
        print(f"  NCCL not available: {e}")
        print("  Falling back to manual CPU-staged all-reduce timing")
        nccl_ok = False

    sizes = [
        ("8 KB   (7B batch=1)",   8 * 1024),
        ("64 KB  (7B batch=8)",   64 * 1024),
        ("256 KB (7B batch=32)",  256 * 1024),
        ("1 MB",                  1 * 1024 * 1024),
        ("8 MB",                  8 * 1024 * 1024),
        ("64 MB",                 64 * 1024 * 1024),
    ]

    results = {}
    print(f"\n  {'Tensor':25s}  {'Latency':>10s}  {'BW (GB/s)':>10s}  {'Source'}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*12}")

    for label, nbytes in sizes:
        numel = nbytes // 2  # bfloat16 = 2 bytes
        t0 = torch.randn(numel, dtype=torch.bfloat16, device="cuda:0")
        t1 = torch.randn(numel, dtype=torch.bfloat16, device="cuda:1")

        # Warmup
        for _ in range(5):
            if nccl_ok:
                torch.cuda.nccl.all_reduce([t0, t1])
            else:
                cpu = t0.to("cpu") + t1.to("cpu")
                t0.copy_(cpu.to("cuda:0"))
                t1.copy_(cpu.to("cuda:1"))
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if nccl_ok:
                torch.cuda.nccl.all_reduce([t0, t1])
            else:
                cpu = t0.to("cpu") + t1.to("cpu")
                t0.copy_(cpu.to("cuda:0"))
                t1.copy_(cpu.to("cuda:1"))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        p50_ms = sorted(times)[len(times) // 2] * 1000
        # All-reduce moves 2×nbytes (send + receive) across the link
        bw_gbps = (2 * nbytes / 1e9) / (avg_ms / 1000)
        src = "NCCL" if nccl_ok else "CPU-staged"

        print(f"  {label:25s}  {avg_ms:>8.3f} ms  {bw_gbps:>8.2f} GB/s  {src}")
        results[label] = {"avg_ms": round(avg_ms, 3), "p50_ms": round(p50_ms, 3),
                          "bw_gbps": round(bw_gbps, 2), "method": src}

        del t0, t1
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tensor-parallel inference bench
# ─────────────────────────────────────────────────────────────────────────────

def bench_tp_inference(model_name: str, batch_sizes: list, max_new_tokens: int = 32) -> dict:
    """Compare tensor-parallel throughput at different batch sizes.

    P2P matters when batch is large (bigger activations → bigger all-reduces).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.tensor_parallel import apply_tensor_parallel

    print("\n" + "=" * 64)
    print(f"  PART 2 — Tensor-Parallel Inference: {model_name.split('/')[-1]}")
    print("=" * 64)

    print(f"\n  Loading {model_name} on GPU 0 (bfloat16)...")
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
    )
    model.eval()
    load_time = time.perf_counter() - t_load
    print(f"  Loaded in {load_time:.1f}s")

    # Apply tensor parallel across 2 GPUs
    print("  Applying tensor-parallel sharding across cuda:0 + cuda:1...")
    tp_model = apply_tensor_parallel(model, devices=["cuda:0", "cuda:1"])
    torch.cuda.synchronize()
    print("  TP sharding done")

    prompt = "The future of artificial intelligence and distributed computing is"
    results = {}

    print(f"\n  {'Batch':>6s}  {'Tokens':>8s}  {'Time':>8s}  {'tok/s':>8s}  {'tok/s/GPU':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for bs in batch_sizes:
        prompts = [prompt] * bs
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")

        # Warmup
        try:
            with torch.no_grad():
                warmup_out = tp_model.generate(
                    **inputs, max_new_tokens=4, do_sample=False,
                )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  batch={bs}: warmup failed — {e}")
            continue

        # Measure
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = tp_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            n_new = (out.shape[1] - inputs["input_ids"].shape[1]) * bs
            tps = n_new / elapsed
            tps_per_gpu = tps / 2

            print(f"  {bs:>6d}  {n_new:>8d}  {elapsed:>6.2f}s  {tps:>8.1f}  {tps_per_gpu:>10.1f}")
            results[f"batch_{bs}"] = {
                "tokens": n_new, "elapsed_s": round(elapsed, 3),
                "tok_s": round(tps, 1), "tok_s_per_gpu": round(tps_per_gpu, 1),
            }
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={bs}: OOM")
            break
        except Exception as e:
            print(f"  batch={bs}: error — {e}")
            break

    del tp_model, model, tokenizer
    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transfer bandwidth comparison (P2P vs CPU-staged)
# ─────────────────────────────────────────────────────────────────────────────

def bench_transfer_comparison(n_iters: int = 20) -> dict:
    """Direct GPU-to-GPU copy bandwidth at sizes matching TP all-reduce."""
    print("\n" + "=" * 64)
    print("  PART 0 — Raw GPU Transfer Bandwidth")
    print("=" * 64)

    from core.transfer_manager import TransferManager
    tm = TransferManager(verbose=False)

    sizes_mb = [0.008, 0.064, 0.256, 1, 8, 64, 256]
    results = {}

    print(f"\n  {'Size':>12s}  {'BW (GB/s)':>10s}  {'Method':>12s}  {'Match TP?'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*12}")

    for size_mb in sizes_mb:
        numel = max(1, int(size_mb * 1024 * 1024 / 2))  # bfloat16
        src = torch.randn(numel, dtype=torch.bfloat16, device="cuda:0")

        # Warmup
        for _ in range(3):
            tm.send_activation(0, 1, src)
        torch.cuda.synchronize(1)

        times = []
        method = ""
        for _ in range(n_iters):
            t0 = time.perf_counter()
            r = tm.send_activation(0, 1, src)
            torch.cuda.synchronize(1)
            times.append(time.perf_counter() - t0)
            method = r.method.name

        avg = sum(times) / len(times)
        bw = (size_mb / 1024) / avg

        # TP context: is this a realistic all-reduce size?
        if size_mb < 0.1:
            tp_ctx = "7B batch=1-8"
        elif size_mb < 2:
            tp_ctx = "7B batch=32-128"
        elif size_mb < 32:
            tp_ctx = "14B batch=128+"
        else:
            tp_ctx = "large batch"

        print(f"  {size_mb*1024:>9.0f} KB  {bw:>10.2f}  {method:>12s}  {tp_ctx}")
        results[f"{size_mb}mb"] = {"bw_gbps": round(bw, 2), "method": method, "avg_ms": round(avg*1000, 3)}

        del src

    torch.cuda.empty_cache()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P2P impact benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="Model for TP inference bench")
    parser.add_argument("--batch-sizes", default="1,4,8,16",
                        help="Comma-separated batch sizes for TP bench")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--allreduce-only", action="store_true",
                        help="Skip TP inference (fast mode)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Iterations for microbench")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Show P2P status
    from core.hetero_config import probe_p2p
    p2p = probe_p2p(0, 1)
    print("\n" + "=" * 64)
    print("  VRAMancer P2P Impact Benchmark")
    print("=" * 64)
    print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
    print(f"  P2P status: {'ACTIVE (nvidia-smi topo OK)' if p2p else 'BLOCKED (CPU-staged)'}")
    print(f"  can_device_access_peer: {torch.cuda.can_device_access_peer(0, 1)}")
    print()
    print("  TIP: compare with VRM_TRANSFER_P2P=false to see CPU-staged baseline")

    all_results = {"p2p_active": p2p, "gpu0": torch.cuda.get_device_name(0),
                   "gpu1": torch.cuda.get_device_name(1)}

    # Part 0: raw transfer bandwidth
    all_results["transfer"] = bench_transfer_comparison(args.iters)

    # Part 1: all-reduce microbench
    all_results["allreduce"] = bench_allreduce(args.iters)

    # Part 2: TP inference
    if not args.allreduce_only:
        all_results["tp_inference"] = bench_tp_inference(
            args.model, batch_sizes, args.max_new_tokens,
        )

    # Summary
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    ar = all_results.get("allreduce", {})
    for label, v in ar.items():
        print(f"  all-reduce {label:25s}: {v['avg_ms']:>7.3f} ms  {v['bw_gbps']:>7.2f} GB/s")

    tp = all_results.get("tp_inference", {})
    if tp:
        print()
        for k, v in tp.items():
            print(f"  TP inference {k}: {v['tok_s']:.1f} tok/s ({v['tok_s_per_gpu']:.1f} tok/s/GPU)")

    # Save
    outfile = "bench_p2p_impact.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {outfile}")

    print("\n  To compare CPU-staged vs P2P, rerun with:")
    print("  VRM_TRANSFER_P2P=false python benchmarks/bench_p2p_impact.py --allreduce-only")


if __name__ == "__main__":
    main()
