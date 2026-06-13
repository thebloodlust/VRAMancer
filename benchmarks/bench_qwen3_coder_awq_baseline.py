"""V6.D-bis — Qwen3-Coder-30B-A3B AWQ Int4 baseline on RTX 5070 Ti (Blackwell sm_120).

Companion to bench_qwen3_coder_lending.py. Same model family, same hardware,
but using the AWQ Int4 quantization (~17 GB on disk) which fits entirely on
the RTX 5070 Ti's 16 GB VRAM with KV cache budget — no cpu_offload, no
lending pool, no PCIe weight prefetch on the hot path. Goal: measure the
achievable upper bound on Blackwell sm_120 single-GPU throughput, so we can
quantify what the lending pool buys back when we move to FP8 (V6.D) or wire
the V6.E expert-prefetch path.

Design choices:
  - quantization auto-detected from the AWQ model config
  - cpu_offload_gb=0 (model fits on GPU)
  - enforce_eager=False (let vLLM capture CUDA graphs — biggest single win)
  - VRM_VLLM_TARGET_GPU=0 honoured (pin to 5070 Ti, not the 24 GB 3090)
  - Same 3-context grid (512 / 1024 / 2048) and prompt format as the
    lending bench so numbers are directly comparable.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Same env hygiene as bench_qwen3_coder_lending.py — see that file's docstring
# and project_vllm_cvd_pitfall memory for why each one matters.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402

HF_MODEL_ID = os.environ.get(
    "VRM_BENCH_MODEL", "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
)
MAX_MODEL_LEN = int(os.environ.get("VRM_BENCH_MAX_MODEL_LEN", "4096"))
COMPUTE_GPU_IDX = int(os.environ.get("VRM_BENCH_COMPUTE_GPU", "0"))
CONTEXT_SIZES = [512, 1024, 2048]
MAX_NEW = 32
GPU_MEMORY_UTILIZATION = float(os.environ.get("VRM_BENCH_GPU_UTIL", "0.92"))
ENFORCE_EAGER = os.environ.get("VRM_BENCH_ENFORCE_EAGER", "0") == "1"
# AWQ Int4 dequantizes to ~14.9 GB on GPU + ~3 GB KV cache = >17 GB → won't
# fit on the 5070 Ti's 16 GB even at gpu_util=0.92. Push 4 GB of cold-expert
# weights to DRAM via UVA cpu_offload — minimal perf hit since the bulk
# stays on GPU, but enough headroom for KV cache + activations.
CPU_OFFLOAD_GB = float(os.environ.get("VRM_BENCH_CPU_OFFLOAD_GB", "4"))
OUT_JSON = Path("benchmarks/results/bench_qwen3_coder_awq_baseline_v6.json")
OUT_MD = Path("benchmarks/results/bench_qwen3_coder_awq_baseline_v6.md")


# ---------------------------------------------------------------------------
# Hardware probes (NVML so they survive CVD narrowing)
# ---------------------------------------------------------------------------

_NVML_INIT = False


def _nvml() -> Any:
    global _NVML_INIT
    try:
        import pynvml  # noqa: F401
    except ImportError:
        return None
    if not _NVML_INIT:
        try:
            import pynvml
            pynvml.nvmlInit()
            _NVML_INIT = True
        except Exception:
            return None
    import pynvml
    return pynvml


def measure_vram_per_gpu() -> dict:
    pynvml = _nvml()
    out: dict = {}
    if pynvml is None:
        return out
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[f"gpu{i}"] = {
                "used_mb": mem.used // (1024 * 1024),
                "total_mb": mem.total // (1024 * 1024),
            }
    except Exception:
        pass
    return out


def measure_dram_used() -> dict:
    try:
        import psutil
        return {"rss_mb": psutil.Process().memory_info().rss // (1024 * 1024)}
    except Exception:
        return {"rss_mb": -1}


def gpu_info(idx: int) -> dict:
    pynvml = _nvml()
    if pynvml is None:
        return {}
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
    return {
        "name": name,
        "sm": f"{major}.{minor}",
        "total_mb": mem.total // (1024 * 1024),
        "used_mb": mem.used // (1024 * 1024),
    }


def make_prompt_str(approx_chars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * 50
    text = base
    while len(text) < approx_chars:
        text += base
    return text[:approx_chars]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_bench():
    print("=" * 70)
    print("[V6.D-bis] Qwen3-Coder-30B-A3B AWQ Int4 — Blackwell sm_120 baseline")
    print("=" * 70)
    print(f"  Model         : {HF_MODEL_ID}")
    print(f"  Compute GPU   : {COMPUTE_GPU_IDX}")
    print(f"  max_model_len : {MAX_MODEL_LEN}")
    print(f"  gpu_mem_util  : {GPU_MEMORY_UTILIZATION}")
    print(f"  enforce_eager : {ENFORCE_EAGER}  (CUDA graphs ON if False)")
    print()

    if "VRM_VLLM_TARGET_GPU" not in os.environ:
        os.environ["VRM_VLLM_TARGET_GPU"] = str(COMPUTE_GPU_IDX)
        print(f"  [auto] VRM_VLLM_TARGET_GPU={COMPUTE_GPU_IDX}")

    g0 = gpu_info(COMPUTE_GPU_IDX)
    print(f"  Pinning to GPU {COMPUTE_GPU_IDX}: {g0.get('name')} "
          f"(SM {g0.get('sm')}, {g0.get('total_mb')} MB)")

    try:
        from core.inference_pipeline import InferencePipeline, reset_pipeline
        reset_pipeline()
    except ImportError as e:
        print(f"[BLOCKED] Cannot import InferencePipeline: {e}", file=sys.stderr)
        sys.exit(1)

    log_path = OUT_JSON.parent / "bench_qwen3_coder_awq_baseline_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush(); sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    load_exc: Exception | None = None
    pipeline = None
    t_load_0 = time.perf_counter()
    try:
        pipeline = InferencePipeline(
            backend_name="vllm", enable_metrics=False, enable_discovery=False
        )
        pipeline.load(
            HF_MODEL_ID,
            num_gpus=1,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enforce_eager=ENFORCE_EAGER,
            cpu_offload_gb=CPU_OFFLOAD_GB,
        )
    except Exception as e:
        load_exc = e
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

    t_load = time.perf_counter() - t_load_0

    if load_exc is not None:
        msg = str(load_exc)
        log_tail = ""
        try:
            with open(log_path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 8000))
                log_tail = fh.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"[FAILED — load error: {msg[:400]}]")
        print(f"[log tail from {log_path}:]")
        print(log_tail[-4000:])
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({
            "model": HF_MODEL_ID,
            "backend": "vllm",
            "note": "FAILED — vLLM load error",
            "error": msg[:800],
            "log_path": str(log_path),
            "log_tail": log_tail[-6000:],
        }, indent=2))
        return

    vram_loaded = measure_vram_per_gpu()
    dram_loaded = measure_dram_used()
    print(f"\n[Loaded in {t_load:.1f}s]")
    for gid, m in vram_loaded.items():
        print(f"  {gid}: {m['used_mb']} / {m['total_mb']} MB used")
    print(f"  Process DRAM RSS: {dram_loaded['rss_mb']} MB")
    print()

    # Sanity: model should be on COMPUTE_GPU_IDX (>1 GB growth there).
    compute_after = vram_loaded.get(f"gpu{COMPUTE_GPU_IDX}", {}).get("used_mb", 0)
    if compute_after < 4096:
        print(f"[WARNING] gpu{COMPUTE_GPU_IDX} only at {compute_after} MB after "
              f"load — model likely not on the intended GPU.")

    char_sizes = {512: 2048, 1024: 4096, 2048: 8192}
    results = []
    print("[Step 3] Running inference benchmarks...")
    for ctx_size in CONTEXT_SIZES:
        if ctx_size > MAX_MODEL_LEN:
            continue
        prompt = make_prompt_str(char_sizes[ctx_size])
        vram_before = measure_vram_per_gpu()

        t0 = time.perf_counter()
        try:
            _ = pipeline.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.0)
        except Exception as e:
            results.append({"ctx_target": ctx_size, "error": str(e)[:300]})
            print(f"  ctx≈{ctx_size}: ERROR {e}")
            continue
        dt = time.perf_counter() - t0
        vram_after = measure_vram_per_gpu()
        tok_s = MAX_NEW / dt if dt > 0 else 0
        results.append({
            "ctx_target": ctx_size,
            "prompt_chars": len(prompt),
            "max_new": MAX_NEW,
            "dt_s": round(dt, 3),
            "tok_s": round(tok_s, 2),
            "vram_before": vram_before,
            "vram_after": vram_after,
        })
        print(f"  ctx≈{ctx_size}: {tok_s:.2f} tok/s  ({dt:.2f} s for {MAX_NEW} tokens)")

    # ---- Write JSON ----
    g0_post = gpu_info(COMPUTE_GPU_IDX)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": HF_MODEL_ID,
        "backend": "vllm",
        "scope": (
            "Single-GPU AWQ Int4 baseline on Blackwell sm_120. NO lending "
            "pool, NO cpu_offload. Goal: ceiling for what the 5070 Ti can "
            "achieve standalone, to quantify what V6.D lending + V6.E "
            "expert prefetch buy back when running larger / higher-precision "
            "variants of the same model family."
        ),
        "compute_gpu": {
            "index": COMPUTE_GPU_IDX,
            "name": g0.get("name"),
            "sm": g0.get("sm"),
            "vram_mb": g0.get("total_mb"),
        },
        "vllm_config": {
            "tensor_parallel_size": 1,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": ENFORCE_EAGER,
            "cpu_offload_gb": CPU_OFFLOAD_GB,
            "quantization": "awq (auto-detected)",
        },
        "load_seconds": round(t_load, 2),
        "vram_after_load": vram_loaded,
        "dram_after_load_mb": dram_loaded["rss_mb"],
        "results": results,
    }, indent=2))

    # ---- Write Markdown ----
    md_lines = [
        "# Qwen3-Coder-30B-A3B AWQ Int4 — Blackwell sm_120 single-GPU baseline (V6.D-bis)",
        "",
        "> Companion to `bench_qwen3_coder_lending_v6.md`. Same model family,",
        "> same hardware, but **AWQ Int4** instead of FP8 — fits entirely on",
        "> the 5070 Ti's 16 GB, no cpu_offload, no lending. This bench measures",
        "> the **upper bound** on what the 5070 Ti can do standalone, so the",
        "> V6.D lending + V6.E expert-prefetch wins are quantifiable.",
        "",
        "## Setup",
        "",
        f"- **Compute GPU**: {g0.get('name')} (SM {g0.get('sm')}, "
        f"{g0.get('total_mb')} MB)",
        f"- **Model**: `{HF_MODEL_ID}` (~17 GB on disk, AWQ 4-bit)",
        f"- **vLLM**: TP=1, `cpu_offload_gb={CPU_OFFLOAD_GB}`, "
        f"`gpu_memory_utilization={GPU_MEMORY_UTILIZATION}`, "
        f"`enforce_eager={ENFORCE_EAGER}` "
        f"({'CUDA graphs OFF' if ENFORCE_EAGER else 'CUDA graphs ON'})",
        f"- **Lending pool**: NOT used (single-GPU baseline). Note: even AWQ "
        f"Int4 needed {CPU_OFFLOAD_GB} GB cpu_offload to fit on the 5070 Ti — "
        f"the 30B-A3B's dequantized weights are ~15 GB, leaving no room for "
        f"the KV cache. This is exactly the case the V6.D lending pool addresses.",
        "",
        "## Inference results",
        "",
        f"**Load time**: {t_load:.1f} s   "
        f"**5070 Ti VRAM after load**: "
        f"{vram_loaded.get(f'gpu{COMPUTE_GPU_IDX}', {}).get('used_mb', '?')} MB",
        "",
        "| Context (tok) | tok/s | dt (s) |",
        "|---------------|-------|--------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r['ctx_target']} | ERROR | — |")
            continue
        md_lines.append(
            f"| {r['ctx_target']} | {r['tok_s']:.2f} | {r['dt_s']:.2f} |"
        )
    md_lines += [
        "",
        "## Comparison to V6.D FP8 + lending",
        "",
        "When the V6.D bench ran with FP8 + cpu_offload_gb=24 + lending lease,",
        "steady-state was ~5 tok/s (DRAM-bound on PCIe 4.0 expert fetches).",
        "If this AWQ baseline lands clearly above that, the gap quantifies the",
        "PCIe-fetch tax that V6.E expert prefetch through the 3090 lending",
        "buffer is meant to claw back — same model family, same GPU, only the",
        "memory placement strategy changes.",
        "",
        "*Generated by VRAMancer bench_qwen3_coder_awq_baseline.py*",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"\nResults: {OUT_JSON}")
    print(f"Summary: {OUT_MD}")


if __name__ == "__main__":
    run_bench()
