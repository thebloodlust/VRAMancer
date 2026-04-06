#!/usr/bin/env python3
"""3-node heterogeneous benchmark: 2 CUDA GPUs + MacBook M4 (MLX via VTP).

Layer split for Qwen2.5-14B (48 layers):
  GPU 0 (RTX 3090):     embed + layers 0-29  + norm + lm_head  (~18 GB)
  GPU 1 (RTX 5070 Ti):  layers 30-41                           (~6.6 GB)
  MacBook M4 (MLX):     layers 42-47  via VTP on port 18951    (~0.8 GB 4-bit)

Usage:
    # 1) On MacBook:
    python3 mac_mlx --model mlx-community/Qwen2.5-14B-4bit --start-layer 42 --end-layer 48

    # 2) On Ubuntu:
    python benchmarks/bench_3node.py --mac-host 192.168.1.27

    # Custom split:
    python benchmarks/bench_3node.py --mac-host 192.168.1.27 \\
        --gpu0-layers 28 --gpu1-end 40 --mac-start 40
"""

import os
import sys
import time
import argparse
import json

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from core.cross_node import (
    VTPRemoteWorker, distributed_generate, get_model_layers,
)
from core.logger import get_logger

logger = get_logger("bench_3node")

MODEL = "Qwen/Qwen2.5-14B"


def build_device_map(config, gpu0_layers, gpu1_end, mac_start, num_layers):
    """Build a device_map dict for 2-GPU + remote Mac."""
    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.norm": "cuda:0",
        "model.rotary_emb": "cuda:0",
        "lm_head": "cuda:0",
    }
    for i in range(num_layers):
        if i < gpu0_layers:
            device_map[f"model.layers.{i}"] = "cuda:0"
        elif i < gpu1_end:
            device_map[f"model.layers.{i}"] = "cuda:1"
        else:
            # Mac layers — offload to disk (not used locally)
            device_map[f"model.layers.{i}"] = "disk"
    return device_map


def main():
    parser = argparse.ArgumentParser(description="3-node heterogeneous benchmark")
    parser.add_argument("--mac-host", required=True,
                        help="MacBook IP (e.g. 192.168.1.27)")
    parser.add_argument("--mac-port", type=int, default=18951)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--gpu0-layers", type=int, default=30,
                        help="Number of layers on GPU 0 (default: 30)")
    parser.add_argument("--gpu1-end", type=int, default=42,
                        help="Last layer on GPU 1 (exclusive, default: 42)")
    parser.add_argument("--mac-start", type=int, default=42,
                        help="First layer on Mac (default: 42)")
    parser.add_argument("--mac-end", type=int, default=48,
                        help="Last layer on Mac (exclusive, default: 48)")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms:")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ── GPU info ──────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"  VRAMancer 3-Node Heterogeneous Benchmark")
    print(f"{'='*60}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        free = torch.cuda.mem_get_info(i)[0] / 1e9
        print(f"  GPU {i}: {name} ({free:.1f} GB free)")
    print(f"  Mac:   {args.mac_host}:{args.mac_port} (MLX VTP)")
    print(f"  Model: {args.model}")
    print(f"{'='*60}\n")

    # ── Verify Mac VTP worker is reachable ────────────────────────
    import socket
    try:
        s = socket.create_connection((args.mac_host, args.mac_port), timeout=5)
        s.close()
        print(f"[OK] Mac VTP worker reachable at {args.mac_host}:{args.mac_port}")
    except Exception as e:
        print(f"[FAIL] Cannot reach Mac VTP worker: {e}")
        print(f"       Start it first: python3 mac_mlx --model mlx-community/Qwen2.5-14B-4bit "
              f"--start-layer {args.mac_start} --end-layer {args.mac_end}")
        sys.exit(1)

    # ── Load model config ─────────────────────────────────────────
    config = AutoConfig.from_pretrained(args.model)
    num_layers = config.num_hidden_layers
    print(f"Model: {num_layers} layers, hidden_size={config.hidden_size}")

    # ── Build device map ──────────────────────────────────────────
    device_map = build_device_map(
        config, args.gpu0_layers, args.gpu1_end,
        args.mac_start, num_layers)

    gpu0_count = sum(1 for k, v in device_map.items()
                     if v == "cuda:0" and "layers" in k)
    gpu1_count = sum(1 for k, v in device_map.items()
                     if v == "cuda:1" and "layers" in k)
    mac_count = args.mac_end - args.mac_start

    print(f"\nLayer split:")
    print(f"  GPU 0 (RTX 3090):    layers 0-{args.gpu0_layers-1} "
          f"({gpu0_count} layers) + embed/norm/head")
    print(f"  GPU 1 (RTX 5070 Ti): layers {args.gpu0_layers}-{args.gpu1_end-1} "
          f"({gpu1_count} layers)")
    print(f"  MacBook M4 (MLX):    layers {args.mac_start}-{args.mac_end-1} "
          f"({mac_count} layers)")

    import tempfile
    offload_dir = os.path.join(tempfile.gettempdir(), "vramancer_offload")
    os.makedirs(offload_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    print(f"\nLoading {args.model} on 2 GPUs ({args.dtype})...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # VRAM usage
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {alloc:.1f} / {total:.1f} GB allocated")

    # ── Create VTP remote worker for Mac ──────────────────────────
    print(f"\nConnecting VTP to Mac {args.mac_host}:{args.mac_port} "
          f"(layers {args.mac_start}-{args.mac_end-1})...")
    mac_worker = VTPRemoteWorker(
        host=args.mac_host,
        port=args.mac_port,
        start_layer=args.mac_start,
        end_layer=args.mac_end,
    )
    print("[OK] VTP connection established")

    # ── Create a simple backend-like object for distributed_generate ─
    class _Backend:
        pass
    backend = _Backend()
    backend.model = model
    backend.tokenizer = tokenizer

    # Local layers: 0 to gpu1_end (GPU 0 + GPU 1)
    local_range = (0, args.gpu1_end)

    # ── Warmup ────────────────────────────────────────────────────
    print(f"\nWarmup (3 tokens)...")
    try:
        warmup = distributed_generate(
            backend=backend,
            prompt="Hello",
            remote_workers=[mac_worker],
            local_layer_range=local_range,
            max_new_tokens=3,
            temperature=0.0,
        )
        print(f"  Warmup OK: \"{warmup['text'][:50]}\"")
    except Exception as e:
        print(f"  Warmup FAILED: {e}")
        import traceback
        traceback.print_exc()
        mac_worker.close()
        sys.exit(1)

    # ── Benchmark ─────────────────────────────────────────────────
    print(f"\nBenchmark: max_tokens={args.max_tokens}, temperature=0.7")
    print(f"Prompt: \"{args.prompt[:80]}\"")
    print()

    result = distributed_generate(
        backend=backend,
        prompt=args.prompt,
        remote_workers=[mac_worker],
        local_layer_range=local_range,
        max_new_tokens=args.max_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )

    # ── Results ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — 3-Node Heterogeneous Inference")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Tokens:     {result['tokens']}")
    print(f"  Time:       {result['total_seconds']:.2f}s")
    print(f"  Throughput: {result['tokens_per_second']:.2f} tok/s")
    print(f"  GPU 0:      RTX 3090 — {gpu0_count} layers ({args.dtype})")
    print(f"  GPU 1:      RTX 5070 Ti — {gpu1_count} layers ({args.dtype})")
    print(f"  MacBook:    M4 MLX — {mac_count} layers (4-bit)")
    print(f"{'='*60}")
    print(f"\nGenerated text:")
    print(f"  {result['text']}")

    # Save results
    bench_result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "nodes": 3,
        "gpu0": {"name": torch.cuda.get_device_name(0), "layers": gpu0_count,
                 "dtype": args.dtype},
        "gpu1": {"name": torch.cuda.get_device_name(1), "layers": gpu1_count,
                 "dtype": args.dtype},
        "mac": {"host": args.mac_host, "layers": mac_count, "dtype": "4-bit MLX"},
        "tokens": result["tokens"],
        "total_seconds": result["total_seconds"],
        "tokens_per_second": result["tokens_per_second"],
        "prompt": args.prompt,
        "text": result["text"],
    }
    outfile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bench_3node_result.json")
    with open(outfile, "w") as f:
        json.dump(bench_result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outfile}")

    mac_worker.close()


if __name__ == "__main__":
    main()
