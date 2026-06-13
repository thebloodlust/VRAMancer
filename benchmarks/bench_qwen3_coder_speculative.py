"""V6.E speculative — Qwen3-Coder-30B-A3B + Qwen3-0.6B draft, on RTX 5070 Ti.

Uses vLLM 0.20.1's built-in speculative_config to run a small draft model
(Qwen3-0.6B BF16, ~1.2 GB) alongside the target (Qwen3-Coder-30B-A3B AWQ,
~14.9 GB) on the same Blackwell sm_120 GPU. The draft proposes K tokens
per step, the target verifies them in a single batched forward pass.

Same family / vocabulary => high acceptance rate expected (60-80% on code
generation). Speedup roughly = 1 + (K * accept_rate), so K=4 with 70%
acceptance ≈ 1 + 2.8 = ~3.8× theoretical (offset by per-step overhead).

This is the "VRAMancer compute pool" angle: even if the 3090 stays as a
VRAM donor (Phase 1, both models on 5070 Ti), the speculative scheme
turns the per-token cost from 1× target-forward into ~1× target-forward-
batched, exploiting the underused parallelism on Blackwell.

Phase 2 follow-up (separate session): place draft on 3090 explicitly so
the 3090's idle Ampere compute becomes a co-processor (vLLM 0.20.1 does
NOT expose draft device pinning out of the box — would require a
patched executor).

Usage:
    .venv/bin/python benchmarks/bench_qwen3_coder_speculative.py
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Same env hygiene as other benches.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 5070 Ti

import torch  # noqa: E402

TARGET_MODEL = os.environ.get(
    "VRM_BENCH_TARGET_MODEL", "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
)
DRAFT_MODEL = os.environ.get(
    "VRM_BENCH_DRAFT_MODEL", "Qwen/Qwen3-0.6B"
)
NUM_SPEC_TOKENS = int(os.environ.get("VRM_BENCH_SPEC_K", "4"))
MAX_MODEL_LEN = int(os.environ.get("VRM_BENCH_MAX_MODEL_LEN", "4096"))
GPU_UTIL = float(os.environ.get("VRM_BENCH_GPU_UTIL", "0.94"))
CPU_OFFLOAD_GB = float(os.environ.get("VRM_BENCH_CPU_OFFLOAD_GB", "6"))
CONTEXT_SIZES = [512, 1024, 2048]
MAX_NEW = int(os.environ.get("VRM_BENCH_MAX_NEW", "64"))  # bigger so the
                                                          # speculative win
                                                          # has room to amortize
OUT_JSON = Path("benchmarks/results/bench_qwen3_coder_speculative_v6.json")
OUT_MD = Path("benchmarks/results/bench_qwen3_coder_speculative_v6.md")


_NVML_INIT = False


def _nvml() -> Any:
    global _NVML_INIT
    try:
        import pynvml
    except ImportError:
        return None
    if not _NVML_INIT:
        try:
            pynvml.nvmlInit()
            _NVML_INIT = True
        except Exception:
            return None
    return pynvml


def measure_vram_per_gpu() -> dict:
    pynvml = _nvml()
    if pynvml is None:
        return {}
    try:
        out = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[f"gpu{i}"] = {
                "used_mb": mem.used // (1024 * 1024),
                "total_mb": mem.total // (1024 * 1024),
            }
        return out
    except Exception:
        return {}


def make_prompt_str(approx_chars: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * 50
    text = base
    while len(text) < approx_chars:
        text += base
    return text[:approx_chars]


def main():
    print("=" * 70)
    print("[V6.E spec] Qwen3-Coder-30B-A3B + Qwen3-0.6B draft (K=" +
          str(NUM_SPEC_TOKENS) + ")")
    print("=" * 70)
    print(f"  Target  : {TARGET_MODEL}")
    print(f"  Draft   : {DRAFT_MODEL}")
    print(f"  K       : {NUM_SPEC_TOKENS}")
    print(f"  max_len : {MAX_MODEL_LEN}, gpu_util={GPU_UTIL}, "
          f"cpu_offload={CPU_OFFLOAD_GB} GB")
    print()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[BLOCKED] vLLM unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    # Capture vLLM stderr to disk in case of failure.
    log_path = OUT_JSON.parent / "bench_qwen3_coder_speculative_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush(); sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    load_exc: Exception | None = None
    llm = None
    t0 = time.perf_counter()
    try:
        llm = LLM(
            model=TARGET_MODEL,
            tensor_parallel_size=1,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_UTIL,
            cpu_offload_gb=CPU_OFFLOAD_GB,
            enforce_eager=False,  # CUDA graphs ON
            trust_remote_code=True,
            speculative_config={
                "model": DRAFT_MODEL,
                "num_speculative_tokens": NUM_SPEC_TOKENS,
            },
        )
    except Exception as e:
        load_exc = e
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)

    t_load = time.perf_counter() - t0

    if load_exc is not None:
        msg = str(load_exc)[:600]
        log_tail = ""
        try:
            with open(log_path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 8000))
                log_tail = fh.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"[FAILED — load error: {msg}]")
        print(f"[log tail from {log_path}:]")
        print(log_tail[-4000:])
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({
            "target_model": TARGET_MODEL,
            "draft_model": DRAFT_MODEL,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "note": "FAILED — vLLM load error",
            "error": msg,
            "log_path": str(log_path),
            "log_tail": log_tail[-6000:],
        }, indent=2))
        return

    vram_loaded = measure_vram_per_gpu()
    print(f"\n[Loaded in {t_load:.1f}s]")
    for gid, m in vram_loaded.items():
        print(f"  {gid}: {m['used_mb']} / {m['total_mb']} MB")
    print()

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    char_sizes = {512: 2048, 1024: 4096, 2048: 8192}
    results = []
    print("[Inference] Running benchmarks...")
    for ctx_size in CONTEXT_SIZES:
        if ctx_size > MAX_MODEL_LEN:
            continue
        prompt = make_prompt_str(char_sizes[ctx_size])
        # Two warmup runs so per-prompt JIT/graph capture doesn't dominate.
        for _ in range(2):
            try:
                _ = llm.generate([prompt], sampling, use_tqdm=False)
            except Exception:
                pass
        t_run0 = time.perf_counter()
        try:
            outs = llm.generate([prompt], sampling, use_tqdm=False)
        except Exception as e:
            results.append({"ctx_target": ctx_size, "error": str(e)[:300]})
            continue
        dt = time.perf_counter() - t_run0
        if not outs or not outs[0].outputs:
            results.append({"ctx_target": ctx_size, "error": "empty output"})
            continue
        out = outs[0].outputs[0]
        n_out = len(out.token_ids)
        tok_s = n_out / dt if dt > 0 else 0
        results.append({
            "ctx_target": ctx_size,
            "prompt_chars": len(prompt),
            "max_new_requested": MAX_NEW,
            "tokens_generated": n_out,
            "dt_s": round(dt, 3),
            "tok_s": round(tok_s, 2),
        })
        print(f"  ctx≈{ctx_size}: {tok_s:.2f} tok/s  "
              f"({n_out} tok in {dt:.2f} s)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "vllm_config": {
            "tensor_parallel_size": 1,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_memory_utilization": GPU_UTIL,
            "cpu_offload_gb": CPU_OFFLOAD_GB,
            "enforce_eager": False,
        },
        "load_seconds": round(t_load, 2),
        "vram_after_load": vram_loaded,
        "results": results,
    }, indent=2))

    md = [
        "# Qwen3-Coder-30B-A3B + Qwen3-0.6B speculative — V6.E PoC",
        "",
        f"- **Target**: `{TARGET_MODEL}` on RTX 5070 Ti (Blackwell sm_120)",
        f"- **Draft**: `{DRAFT_MODEL}` (~1.2 GB BF16, same Qwen3 vocab)",
        f"- **K (num_speculative_tokens)**: {NUM_SPEC_TOKENS}",
        f"- **Both on 5070 Ti** (Phase 1 — Phase 2 would split target/draft "
        "across 5070 Ti / 3090, but vLLM 0.20.1 does not expose draft "
        "device pinning natively)",
        f"- **Load time**: {t_load:.1f} s",
        "",
        "## Inference results",
        "",
        "| Context (tok) | tok/s | tokens generated | dt (s) |",
        "|---------------|-------|------------------|--------|",
    ]
    for r in results:
        if "error" in r:
            md.append(f"| {r['ctx_target']} | ERROR | — | — |")
            continue
        md.append(
            f"| {r['ctx_target']} | {r['tok_s']:.2f} | "
            f"{r['tokens_generated']} | {r['dt_s']:.2f} |"
        )
    md += [
        "",
        "## Comparison",
        "",
        "| Config                                            | tok/s steady |",
        "|---------------------------------------------------|--------------|",
        "| FP8 + cpu_offload=24 + lending (V6.D)             | ~5  |",
        "| AWQ Int4 + cpu_offload=4 + CUDA graphs (V6.D-bis) | ~30 |",
        "| **AWQ Int4 + speculative K=" + str(NUM_SPEC_TOKENS) +
        " (this run)**            | **see table above** |",
        "",
        "*Generated by VRAMancer bench_qwen3_coder_speculative.py*",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"\nResults: {OUT_JSON}")
    print(f"Summary: {OUT_MD}")


if __name__ == "__main__":
    main()
