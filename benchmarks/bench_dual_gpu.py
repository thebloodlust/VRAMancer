#!/usr/bin/env python3
"""VRAMancer — Dual-GPU diagnostic & benchmark script.

Runs on the Proxmox Q35 VM with RTX 5070 Ti + RTX 3090.
Steps:
  1. GPU detection & profile matching
  2. Heterogeneous auto-configuration
  3. P2P topology probe (reports VM/IOMMU status)
  4. Inter-GPU transfer benchmark (CPU-staged or P2P)
  5. Quick inference benchmark (GPT-2, then optional larger model)

Usage:
    # Source the VM config first
    set -a && source config/proxmox-q35-vm.env && set +a
    python benchmarks/bench_dual_gpu.py

    # Or with a specific model
    python benchmarks/bench_dual_gpu.py --model meta-llama/Llama-2-13b-hf
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import logging

# Ensure we're NOT in test mode
os.environ.pop("VRM_MINIMAL_TEST", None)
os.environ.pop("VRM_TEST_MODE", None)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("bench_dual_gpu")


def banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step_gpu_detection():
    """Step 1: Detect GPUs and match profiles."""
    banner("Step 1 — GPU Detection")
    from core.hetero_config import detect_gpus, lookup_gpu_profile
    gpus = detect_gpus()
    if not gpus:
        log.error("No CUDA GPUs detected. Check nvidia-smi and driver.")
        sys.exit(1)
    for gpu in gpus:
        p = gpu.profile
        arch = p.architecture if p else "unknown"
        fp16 = f"{p.fp16_tflops:.1f}" if p else "?"
        pcie = f"Gen{p.pcie_gen}" if p else "?"
        print(f"  GPU {gpu.index}: {gpu.name}")
        print(f"    VRAM:    {gpu.total_vram_gb:.1f} GB total, {gpu.free_vram_gb:.1f} GB free")
        print(f"    Arch:    {arch}, FP16: {fp16} TFLOPS, PCIe: {pcie}")
        print(f"    CC:      {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
    return gpus


def step_hetero_config():
    """Step 2: Auto-configure heterogeneous setup."""
    banner("Step 2 — Heterogeneous Auto-Configuration")
    from core.hetero_config import auto_configure
    config = auto_configure()
    print(config.summary())
    return config


def step_p2p_probe(gpus):
    """Step 3: Probe P2P topology and VM detection."""
    banner("Step 3 — P2P Topology & VM Detection")
    from core.hetero_config import probe_p2p, _detect_vm

    is_vm = _detect_vm()
    print(f"  VM detected: {'YES' if is_vm else 'No'}")
    p2p_forced = os.environ.get("VRM_TRANSFER_P2P", "").lower() in ("0", "false", "no")
    print(f"  VRM_TRANSFER_P2P forced off: {'YES' if p2p_forced else 'No'}")

    if len(gpus) >= 2:
        for i in range(len(gpus)):
            for j in range(i + 1, len(gpus)):
                can = probe_p2p(gpus[i].index, gpus[j].index)
                status = "P2P OK" if can else "BLOCKED (CPU-staged fallback)"
                print(f"  GPU {gpus[i].index} <-> GPU {gpus[j].index}: {status}")
                if not can and is_vm:
                    print(f"    -> Expected in VM: vfio-pci IOMMU groups isolate GPUs")
    return is_vm


def step_transfer_benchmark():
    """Step 4: Benchmark inter-GPU transfers."""
    banner("Step 4 — Transfer Benchmark")
    from core.transfer_manager import TransferManager

    try:
        import torch
    except ImportError:
        log.error("PyTorch not available — skipping transfer benchmark")
        return

    if torch.cuda.device_count() < 2:
        log.warning("Need 2 GPUs for transfer benchmark")
        return

    tm = TransferManager(verbose=False)
    sizes_mb = [1, 10, 50, 100, 500]
    print(f"\n  {'Size':>8s}  {'Time (ms)':>10s}  {'BW (GB/s)':>10s}  {'Method':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

    for size_mb in sizes_mb:
        numel = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        tensor = torch.randn(numel, device="cuda:0")

        # Warmup
        for _ in range(2):
            tm.send_activation(0, 1, tensor)

        # Measure
        times = []
        method_name = ""
        for _ in range(5):
            t0 = time.perf_counter()
            result = tm.send_activation(0, 1, tensor)
            torch.cuda.synchronize(1)
            times.append(time.perf_counter() - t0)
            method_name = result.method.name

        avg_time = sum(times) / len(times)
        bw_gbps = (size_mb / 1024) / avg_time if avg_time > 0 else 0
        print(f"  {size_mb:>6d} MB  {avg_time*1000:>8.2f} ms  {bw_gbps:>8.2f} GB/s  {method_name:>12s}")

        del tensor

    # Bidirectional test
    print(f"\n  --- Bidirectional (GPU 1 -> GPU 0) ---")
    tensor = torch.randn((100 * 1024 * 1024) // 4, device="cuda:1")
    for _ in range(2):
        tm.send_activation(1, 0, tensor)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = tm.send_activation(1, 0, tensor)
        torch.cuda.synchronize(0)
        times.append(time.perf_counter() - t0)
    avg_time = sum(times) / len(times)
    bw = (100 / 1024) / avg_time if avg_time > 0 else 0
    print(f"  100 MB: {avg_time*1000:.2f} ms, {bw:.2f} GB/s via {result.method.name}")
    del tensor


def step_inference_benchmark(model_name: str):
    """Step 5: Quick inference benchmark."""
    banner(f"Step 5 — Inference Benchmark ({model_name})")
    from core.inference_pipeline import get_pipeline

    try:
        pipeline = get_pipeline()
        pipeline.load(model_name, num_gpus=2)
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return

    prompts = [
        "The future of artificial intelligence is",
        "In a world where GPUs are shared across networks,",
        "The most important thing about multi-GPU inference is",
    ]

    print(f"\n  Generating with max_new_tokens=50 per prompt...\n")
    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        tokens = 0
        try:
            for chunk in pipeline.generate(prompt, max_new_tokens=50):
                tokens += 1
        except Exception as e:
            log.error(f"Generation failed: {e}")
            continue
        elapsed = time.perf_counter() - t0
        tps = tokens / elapsed if elapsed > 0 else 0
        total_tokens += tokens
        total_time += elapsed
        print(f"  [{i+1}/{len(prompts)}] {tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

    if total_time > 0:
        print(f"\n  Total: {total_tokens} tokens in {total_time:.2f}s "
              f"({total_tokens/total_time:.1f} tok/s overall)")


def main():
    parser = argparse.ArgumentParser(description="VRAMancer Dual-GPU Diagnostic & Benchmark")
    parser.add_argument("--model", default="gpt2", help="Model to benchmark (default: gpt2)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference benchmark")
    parser.add_argument("--skip-transfer", action="store_true", help="Skip transfer benchmark")
    args = parser.parse_args()

    banner("VRAMancer Dual-GPU Diagnostic")
    print(f"  Config: VRM_TRANSFER_P2P={os.environ.get('VRM_TRANSFER_P2P', 'not set')}")
    print(f"  Config: VRM_TRANSFER_METHOD={os.environ.get('VRM_TRANSFER_METHOD', 'not set')}")
    print(f"  Config: VRM_HETERO_STRATEGY={os.environ.get('VRM_HETERO_STRATEGY', 'not set')}")

    gpus = step_gpu_detection()
    step_hetero_config()
    step_p2p_probe(gpus)

    if not args.skip_transfer:
        step_transfer_benchmark()

    if not args.skip_inference:
        step_inference_benchmark(args.model)

    banner("Done")
    print("  All diagnostics complete.\n")


if __name__ == "__main__":
    main()
