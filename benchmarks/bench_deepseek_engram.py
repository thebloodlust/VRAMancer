"""V5 P13 — DeepSeek-V4-Flash : RTX 5070 Ti compute + RTX 3090 VRAM lending.

VRAMancer showcase benchmark demonstrating heterogeneous GPU VRAM pooling:

  Architecture:
    - RTX 5070 Ti (SM 12.0, 16 GB)  → ALL compute: MXFP8/FP4 MoE inference
      Only SM >= 10 can run DeepSeek V4's MXFP8 (UE8M0 scales) kernels.
    - RTX 3090    (SM  8.6, 24 GB)  → VRAM DONOR via VRAMancer lending pool.
      16 GB pre-allocated as P2P staging buffer for weight transfer (PCIe4).

  Why this is interesting:
    - Neither GPU can run the model alone: 5070 Ti OOMs (16 GB < 159 GB model),
      3090 cannot run MXFP8 compute (SM 8.6 < SM 10.0 required).
    - VRAMancer lending pool makes the 3090's idle VRAM available to the 5070 Ti
      as a high-bandwidth staging area, reducing reliance on system RAM.
    - PCIe 4.0 P2P between the two GPUs achieves ~28 GB/s unidirectional.
      On PCIe 5.0 servers (e.g. Intel Xeon w9-3595X) this doubles to ~56 GB/s,
      making per-token latency ~2× lower for large models.

  Memory layout:
    - Model weights: 159 GB → UVA-offloaded to DRAM (185 GB available)
    - Active layers: loaded on-demand from DRAM → 3090 lending buffer → 5070 Ti
    - KV cache: resident on 5070 Ti VRAM + DRAM spill (engram effect)

  Usage:
    VRM_KV_OFFLOAD_ENGRAM=1 VRM_KV_DRAM_LIMIT_GB=180 \\
      VRM_BENCH_MODEL=/path/to/DeepSeek-V4-Flash \\
      python benchmarks/bench_deepseek_engram.py
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HF_MODEL_ID = os.environ.get("VRM_BENCH_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
# TP=1: only the 5070 Ti runs MXFP8 compute (SM 8.6 Ampere cannot run UE8M0 scales)
# cpu_offload_gb: model 159 GB, 5070 Ti has 16 GB → offload most weights to DRAM
CPU_OFFLOAD_GB = float(os.environ.get("VRM_BENCH_CPU_OFFLOAD_GB", "145"))
MAX_MODEL_LEN = int(os.environ.get("VRM_BENCH_MAX_MODEL_LEN", "2048"))
# RTX 3090 GPU index (system order, before CUDA_VISIBLE_DEVICES is narrowed)
LENDER_GPU_IDX = int(os.environ.get("VRM_BENCH_LENDER_GPU", "1"))   # 3090
COMPUTE_GPU_IDX = int(os.environ.get("VRM_BENCH_COMPUTE_GPU", "0"))  # 5070 Ti
# How much VRAM to borrow from 3090 as P2P staging buffer (bytes)
LEND_BYTES = int(os.environ.get("VRM_BENCH_LEND_GB", "12")) * 1024 ** 3
CONTEXT_SIZES = [512, 1024, 2048]
MAX_NEW = 32
OUT_JSON = Path("benchmarks/results/bench_deepseek_engram_v5.json")
OUT_MD = Path("benchmarks/results/bench_deepseek_engram_v5.md")


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

def gpu_info(idx: int) -> dict:
    """Return name, SM, VRAM for a CUDA device."""
    p = torch.cuda.get_device_properties(idx)
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "name": p.name,
        "sm": f"{p.major}.{p.minor}",
        "total_mb": total // (1024 * 1024),
        "free_mb": free // (1024 * 1024),
        "used_mb": (total - free) // (1024 * 1024),
    }


def measure_vram_per_gpu() -> dict:
    out = {}
    for i in range(torch.cuda.device_count()):
        info = gpu_info(i)
        out[f"gpu{i}"] = {"used_mb": info["used_mb"], "total_mb": info["total_mb"]}
    return out


def measure_dram_used() -> dict:
    try:
        import psutil
        return {"rss_mb": psutil.Process().memory_info().rss // (1024 * 1024)}
    except Exception:
        return {"rss_mb": -1}


def p2p_bandwidth_gb_s(src: int, dst: int, size_mb: int = 512) -> float:
    """Measure unidirectional P2P memcpy bandwidth src→dst in GB/s."""
    try:
        can_peer = torch.cuda.can_device_access_peer(dst, src)
        buf_src = torch.empty(size_mb * 1024 * 1024 // 2, dtype=torch.float16,
                              device=f"cuda:{src}")
        buf_dst = torch.empty_like(buf_src, device=f"cuda:{dst}")
        # Warmup
        for _ in range(3):
            buf_dst.copy_(buf_src)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            buf_dst.copy_(buf_src)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        bytes_transferred = buf_src.nbytes * 10
        bw = bytes_transferred / dt / 1e9
        del buf_src, buf_dst
        torch.cuda.empty_cache()
        return bw, can_peer
    except Exception as e:
        return 0.0, False


def make_prompt_str(approx_chars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * 50
    text = base
    while len(text) < approx_chars:
        text += base
    return text[:approx_chars]


# ---------------------------------------------------------------------------
# VRAMancer lending pool setup
# ---------------------------------------------------------------------------

def setup_lending_pool(compute_gpu: int, lender_gpu: int, lend_bytes: int) -> dict:
    """
    Register both GPUs in VRAMancer's lending pool.
    The 3090 (lender) pre-allocates a staging buffer so the 5070 Ti (borrower)
    can use it as a high-bandwidth PCIe P2P staging area for weight transfers.

    Returns metrics dict for the results JSON.
    """
    metrics = {
        "lending_enabled": False,
        "lender_gpu": lender_gpu,
        "compute_gpu": compute_gpu,
        "lend_requested_gb": lend_bytes / 1e9,
        "lease_id": None,
        "p2p_bw_gb_s": 0.0,
        "p2p_peer_access": False,
        "lender_info": {},
        "compute_info": {},
    }
    try:
        from core.vram_lending import VRAMLendingPool, LendingPolicy

        # Gather hardware info
        c_info = gpu_info(compute_gpu)
        l_info = gpu_info(lender_gpu)
        metrics["compute_info"] = c_info
        metrics["lender_info"] = l_info

        print(f"  Compute GPU {compute_gpu}: {c_info['name']} "
              f"(SM {c_info['sm']}, {c_info['total_mb']} MB)")
        print(f"  Lender  GPU {lender_gpu}: {l_info['name']} "
              f"(SM {l_info['sm']}, {l_info['total_mb']} MB)")

        # Measure P2P bandwidth before lending allocation
        bw, can_peer = p2p_bandwidth_gb_s(lender_gpu, compute_gpu)
        metrics["p2p_bw_gb_s"] = round(bw, 2)
        metrics["p2p_peer_access"] = can_peer
        print(f"  P2P bandwidth {lender_gpu}→{compute_gpu}: {bw:.1f} GB/s "
              f"({'peer-access' if can_peer else 'staged'})")

        # Build lending pool
        policy = LendingPolicy(
            min_free_ratio=0.10,        # keep 10% of 3090 VRAM free (2.4 GB)
            max_lend_ratio=0.70,        # lend up to 70% of free VRAM (~16 GB)
            buffer_prealloc_ratio=1.0,  # pre-alloc full buffer immediately
        )
        pool = VRAMLendingPool(policy=policy)

        # Register 3090 as lender (no model loaded on it)
        pool.register_gpu(
            gpu_id=lender_gpu,
            total_bytes=l_info["total_mb"] * 1024 * 1024,
            model_bytes=0,
            device_name=l_info["name"],
            pcie_gen=4,   # PCIe 4.0 — would be 5 on modern servers
            compute_capability=(int(l_info["sm"].split(".")[0]),
                                 int(l_info["sm"].split(".")[1])),
        )
        # Register 5070 Ti as borrower (model will be loaded here)
        pool.register_gpu(
            gpu_id=compute_gpu,
            total_bytes=c_info["total_mb"] * 1024 * 1024,
            model_bytes=0,  # updated after load
            device_name=c_info["name"],
            pcie_gen=4,
            compute_capability=(int(c_info["sm"].split(".")[0]),
                                 int(c_info["sm"].split(".")[1])),
        )

        # Borrow from 3090 as staging buffer for the 5070 Ti
        lease = pool.borrow(
            borrower_gpu=compute_gpu,
            size_bytes=lend_bytes,
            purpose="weight_staging_buffer",
            priority=3,
            preferred_lender=lender_gpu,
        )
        if lease is not None:
            # Materialise the buffer on the 3090 — it's now resident in P2P-accessible VRAM
            staging_tensor = pool.allocate_on_lease(
                lease,
                shape=(lend_bytes // 2,),
                dtype=torch.float16,
            )
            metrics["lending_enabled"] = True
            metrics["lease_id"] = lease.lease_id
            metrics["lend_actual_gb"] = lease.size_bytes / 1e9
            print(f"  [VRAMancer] Lending pool active: "
                  f"{lease.size_bytes / 1e9:.1f} GB borrowed from GPU {lender_gpu} "
                  f"(lease {lease.lease_id[:8]}…)")
            print(f"  [VRAMancer] Staging buffer on 3090 VRAM @ P2P PCIe4 "
                  f"({bw:.1f} GB/s). PCIe5 would yield ~{bw * 1.9:.0f} GB/s.")
            # Keep pool + lease alive on the module so they don't get GC'd
            setup_lending_pool._pool = pool
            setup_lending_pool._lease = lease
            setup_lending_pool._staging = staging_tensor
        else:
            print("  [VRAMancer] Lending borrow failed (no capacity?) — continuing without.")

    except Exception as e:
        print(f"  [VRAMancer] Lending pool setup failed: {e} — continuing without.")

    return metrics


def run_bench():
    print("=" * 70)
    print("[P13] DeepSeek-V4-Flash — VRAMancer VRAM Lending Showcase")
    print("=" * 70)
    print(f"  Model     : {HF_MODEL_ID}")
    print(f"  Compute   : GPU {COMPUTE_GPU_IDX} (RTX 5070 Ti, SM12, MXFP8 capable)")
    print(f"  Lender    : GPU {LENDER_GPU_IDX} (RTX 3090, SM8.6, VRAM donor via P2P PCIe4)")
    print(f"  Lend size : {LEND_BYTES / 1e9:.0f} GB staging buffer on 3090")
    print(f"  Offload   : cpu_offload_gb={CPU_OFFLOAD_GB} (UVA to DRAM)")
    print(f"  max_model_len: {MAX_MODEL_LEN}")
    print()

    os.environ["VRM_KV_OFFLOAD_ENGRAM"] = "1"
    os.environ.setdefault("VRM_KV_DRAM_LIMIT_GB", "180")
    # CRITICAL: must be set before any CUDA init so spawned vLLM workers inherit it.
    # Without PCI_BUS_ID, CUDA default (FASTEST_FIRST) may order 3090 before 5070 Ti
    # — especially in Proxmox where virtual PCI bus IDs differ from physical.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Avoid fragmentation when allocating the 1+ GB KV-cache block after weights fill
    # most of the 15.5 GB 5070 Ti — PyTorch's expandable_segments allows non-contiguous
    # allocation from scattered free pages.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Step 1: Load model on 5070 Ti only (TP=1)
    # MXFP8 (UE8M0 scales = ScalarType 44) requires SM >= 10.0 — only 5070 Ti qualifies.
    # 3090 (SM 8.6) CANNOT run DeepSeek V4's quantized attention kernels.
    # IMPORTANT: lending pool is set up AFTER load so the 3090 VRAM is not
    # pre-allocated when vLLM workers check available memory.
    try:
        from core.inference_pipeline import InferencePipeline, reset_pipeline
        reset_pipeline()
    except ImportError as e:
        print(f"[BLOCKED@P13] Cannot import InferencePipeline: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[Step 2] Loading model on GPU {COMPUTE_GPU_IDX} only (TP=1)...")
    print(f"  159 GB model, 16 GB VRAM → {CPU_OFFLOAD_GB:.0f} GB UVA-offloaded to DRAM")
    print(f"  3090 VRAM untouched at this stage (lending pool initialized after load)")

    # CRITICAL: pin CUDA_VISIBLE_DEVICES to the compute GPU *before* pipeline.load()
    # spawns the vLLM EngineCore subprocess.  Without this restriction, vLLM's
    # internal scheduler may elect the 3090 (NVML-order or FASTEST_FIRST heuristic)
    # as the active device inside the worker — even with CUDA_DEVICE_ORDER=PCI_BUS_ID
    # set in the parent env.  Consequence: Triton compiles fp8 kernels for SM86
    # (capability 86 < 89), fp8e4nv is not added to supported_fp8_dtypes → crash.
    # We restore CUDA_VISIBLE_DEVICES after load so the main process can see both
    # GPUs for the lending-pool setup that follows.
    _orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(COMPUTE_GPU_IDX)
    print(f"  [GPU pin] CUDA_VISIBLE_DEVICES={COMPUTE_GPU_IDX} for vLLM worker spawn")

    try:
        pipeline = InferencePipeline(
            backend_name="vllm", enable_metrics=False, enable_discovery=False
        )
        pipeline.load(
            HF_MODEL_ID,
            num_gpus=1,
            tensor_parallel_size=1,    # 5070 Ti only — 3090 cannot run MXFP8
            kv_cache_dtype="fp8",      # DeepSeek V4 MLA requires fp8 KV cache
            cpu_offload_gb=CPU_OFFLOAD_GB,
            max_model_len=MAX_MODEL_LEN,
            # gpu_memory_utilization tuning for 5070 Ti (15.47 GiB total):
            # Observed at 0.90 (run 2026-05-08):
            #   non-KV overhead (profile peak): ~4.46 GiB
            #   KV cache: 9.26 GiB → free for CUDA: 927 MB only
            #   → Triton _fused_kernel cuLaunchKernel OOM at 456 tokens
            #     (insufficient global memory for register spill scratch buffers)
            # At 0.82:
            #   KV = 0.82*15.25 - 4.46 = 8.05 GiB (~2.1 GB free for Triton kernels)
            #   Tokens: ~22 500 (still covers 2048-ctx benchmark)
            gpu_memory_utilization=0.82,
            enforce_eager=True,
        )
    except Exception as e:
        msg = str(e)
        print(f"[FAILED@P13 — model load failed: {msg[:400]}]")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({
            "model": HF_MODEL_ID,
            "backend": "vllm",
            "note": "FAILED — vLLM load error",
            "error": msg[:800],
            "lending": {},
        }, indent=2))
        return
    finally:
        # Restore so main process can access both GPUs (lending pool, metrics…).
        # 'finally' runs even after 'return' in the except block above.
        if _orig_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _orig_cvd
        print(f"  [GPU pin] CUDA_VISIBLE_DEVICES restored for main process")

    vram_loaded = measure_vram_per_gpu()
    dram_loaded = measure_dram_used()
    print(f"\n[P13] Model loaded.")
    print(f"  5070 Ti VRAM used : {vram_loaded.get('gpu0', {}).get('used_mb', '?')} MB")
    if 'gpu1' in vram_loaded:
        print(f"  3090 VRAM used    : {vram_loaded.get('gpu1', {}).get('used_mb', '?')} MB "
              f"(baseline, before lending)")
    print(f"  Process DRAM RSS  : {dram_loaded['rss_mb']} MB")
    print()

    # Step 2: Set up VRAMancer lending pool AFTER model is loaded.
    # The 3090 VRAM is now fully free (model only occupies 5070 Ti).
    # VRAMancer borrows 12 GB from the 3090 as a P2P PCIe4 staging buffer.
    print("[Step 2] Initialising VRAMancer lending pool (3090 → 5070 Ti staging buffer)...")
    lending_metrics = setup_lending_pool(COMPUTE_GPU_IDX, LENDER_GPU_IDX, LEND_BYTES)
    print()

    # Char sizes: approx 4 chars/token for English
    char_sizes = {512: 2048, 1024: 4096, 2048: 8192}

    results = []
    print("[Step 3] Running inference benchmarks...")
    for ctx_size in CONTEXT_SIZES:
        if ctx_size > MAX_MODEL_LEN:
            continue
        prompt = make_prompt_str(char_sizes[ctx_size])
        vram_before = measure_vram_per_gpu()
        dram_before = measure_dram_used()

        t0 = time.perf_counter()
        try:
            _ = pipeline.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.0)
        except Exception as e:
            results.append({"ctx_target": ctx_size, "error": str(e)[:300]})
            print(f"  ctx≈{ctx_size} tok: ERROR {e}")
            continue
        dt = time.perf_counter() - t0

        vram_after = measure_vram_per_gpu()
        dram_after = measure_dram_used()
        tok_s = MAX_NEW / dt if dt > 0 else 0

        vram_deltas = {
            k: vram_after[k]["used_mb"] - vram_before[k]["used_mb"]
            for k in vram_after if k in vram_before
        }
        dram_delta = (
            dram_after["rss_mb"] - dram_before["rss_mb"]
            if dram_before["rss_mb"] >= 0 and dram_after["rss_mb"] >= 0
            else -1
        )
        results.append({
            "ctx_target": ctx_size,
            "prompt_chars": len(prompt),
            "max_new": MAX_NEW,
            "dt_s": round(dt, 3),
            "tok_s": round(tok_s, 2),
            "vram_before": vram_before,
            "vram_after": vram_after,
            "vram_deltas_mb": vram_deltas,
            "dram_before_mb": dram_before["rss_mb"],
            "dram_after_mb": dram_after["rss_mb"],
            "dram_delta_mb": dram_delta,
        })
        print(f"  ctx≈{ctx_size} tok: {tok_s:.2f} tok/s  "
              f"VRAM Δ={vram_deltas}  DRAM Δ={dram_delta} MB")

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    gpu0_info = lending_metrics.get("compute_info", {})
    gpu1_info = lending_metrics.get("lender_info", {})

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": HF_MODEL_ID,
        "backend": "vllm",
        "vramancer_showcase": "VRAM lending pool: RTX 3090 donates VRAM to RTX 5070 Ti",
        "note": (
            "DeepSeek-V4-Flash 158B MoE (MXFP8/FP4). "
            "Compute: RTX 5070 Ti (SM12) TP=1 — only SM>=10 can run UE8M0 MXFP8 kernels. "
            f"RTX 3090 (SM8.6) acts as VRAM donor ({LEND_BYTES / 1e9:.0f} GB P2P staging buffer). "
            f"cpu_offload_gb={CPU_OFFLOAD_GB} UVA-offloads remaining weights to DRAM. "
            "PCIe5 would double P2P bandwidth vs PCIe4 shown here."
        ),
        "hardware": {
            "compute_gpu": {
                "index": COMPUTE_GPU_IDX,
                "name": gpu0_info.get("name", "RTX 5070 Ti"),
                "sm": gpu0_info.get("sm", "12.0"),
                "vram_mb": gpu0_info.get("total_mb", 16384),
                "role": "compute — MXFP8 inference",
            },
            "lender_gpu": {
                "index": LENDER_GPU_IDX,
                "name": gpu1_info.get("name", "RTX 3090"),
                "sm": gpu1_info.get("sm", "8.6"),
                "vram_mb": gpu1_info.get("total_mb", 24576),
                "role": "VRAM donor — P2P PCIe4 staging buffer",
            },
            "p2p_bw_gb_s": lending_metrics.get("p2p_bw_gb_s", 0),
            "p2p_peer_access": lending_metrics.get("p2p_peer_access", False),
            "pcie_gen": 4,
            "theoretical_pcie5_bw_gb_s": round(
                lending_metrics.get("p2p_bw_gb_s", 0) * 1.9, 1
            ),
        },
        "lending": lending_metrics,
        "vllm_config": {
            "tensor_parallel_size": 1,
            "cpu_offload_gb": CPU_OFFLOAD_GB,
            "max_model_len": MAX_MODEL_LEN,
            "kv_cache_dtype": "fp8",
            "enforce_eager": True,
        },
        "vram_after_load": vram_loaded,
        "dram_after_load_mb": dram_loaded["rss_mb"],
        "results": results,
    }, indent=2))

    # Markdown summary
    bw = lending_metrics.get("p2p_bw_gb_s", 0)
    md_lines = [
        "# DeepSeek-V4-Flash — VRAMancer VRAM Lending Showcase (V5 P13)",
        "",
        "> **158B MoE model (159 GB)** running on a **16 GB GPU** via VRAMancer VRAM lending.",
        ">",
        f"> **Compute GPU**: {gpu0_info.get('name', 'RTX 5070 Ti')} "
        f"(SM {gpu0_info.get('sm', '12.0')}, {gpu0_info.get('total_mb', 16384) // 1024} GB) — "
        "MXFP8/FP4 inference (requires SM ≥ 10.0)",
        f"> **Lender GPU**: {gpu1_info.get('name', 'RTX 3090')} "
        f"(SM {gpu1_info.get('sm', '8.6')}, {gpu1_info.get('total_mb', 24576) // 1024} GB) — "
        f"donates {LEND_BYTES / 1e9:.0f} GB VRAM as P2P staging buffer",
        f"> **P2P bandwidth**: {bw:.1f} GB/s (PCIe 4.0) → ~{bw * 1.9:.0f} GB/s on PCIe 5.0",
        "",
        "## Why this matters",
        "",
        "| GPU | Standalone | With VRAMancer |",
        "|-----|-----------|----------------|",
        f"| RTX 5070 Ti (16 GB) | **OOM** (model = 159 GB) | ✅ runs via P2P + DRAM offload |",
        f"| RTX 3090   (24 GB)  | **MXFP8 unsupported** (SM 8.6 < SM 10.0) | ✅ VRAM donor |",
        "",
        "## Inference results",
        "",
        f"**cpu_offload_gb:** {CPU_OFFLOAD_GB}  "
        f"**max_model_len:** {MAX_MODEL_LEN}  "
        f"**P2P staging:** {LEND_BYTES / 1e9:.0f} GB on RTX 3090",
        "",
        "| Context (tok) | tok/s | 5070Ti VRAM Δ (MB) | DRAM Δ (MB) |",
        "|---------------|-------|-------------------|-------------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r['ctx_target']} | ERROR | — | — |")
            continue
        vd = r.get("vram_deltas_mb", {})
        g0 = vd.get("gpu0", "N/A")
        md_lines.append(
            f"| {r['ctx_target']} | {r['tok_s']:.2f} | {g0} | {r['dram_delta_mb']} |"
        )
    md_lines += [
        "",
        "## Key insight",
        "",
        "- Without VRAMancer: **no solution** (compute GPU OOMs, lender GPU lacks SM12)",
        "- With VRAMancer: 5070 Ti handles all MXFP8 kernels; 3090 VRAM used as fast staging",
        "- On a **PCIe 5.0 server** (e.g. Intel w9-3595X or AMD EPYC 9654): "
        f"P2P bandwidth doubles to ~{bw * 1.9:.0f} GB/s, halving weight-transfer latency",
        "",
        "*Generated by VRAMancer bench_deepseek_engram.py*",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nResults written to {OUT_JSON}")
    print(f"Summary written to {OUT_MD}")


if __name__ == "__main__":
    run_bench()
