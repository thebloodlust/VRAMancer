"""Subprocess worker — load vLLM, run 3 generations, emit JSON to stdout."""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
KV_DTYPE = os.environ.get("KV_DTYPE", "auto")
CTX = int(os.environ.get("CTX", "2048"))
CPU_OFFLOAD = float(os.environ.get("CPU_OFFLOAD_GB", "14"))
MAX_NEW = 32

result = {"status": "starting"}
try:
    f0, _ = torch.cuda.mem_get_info(0)
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        kv_cache_dtype=KV_DTYPE,
        cpu_offload_gb=CPU_OFFLOAD,
        max_model_len=CTX,
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    load_t = time.perf_counter() - t0
    f1, _ = torch.cuda.mem_get_info(0)
    # KV blocks: f0 - f1 - model — but easier: report free after load.
    used_mb_after_load = (f0 - f1) / 1e6

    # Use a chunky prompt close to ctx limit to stress KV.
    target_prompt_tokens = min(CTX - MAX_NEW - 64, 1500)
    prompt = "Document " + (" word" * target_prompt_tokens) + ". Summary:"
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    iters = []
    for i in range(3):
        ts = time.perf_counter()
        out = llm.generate([prompt], sp, use_tqdm=False)
        dt = time.perf_counter() - ts
        gen = len(out[0].outputs[0].token_ids)
        iters.append({"i": i, "gen": gen, "dt": round(dt, 3), "tok_s": round(gen / dt, 2) if dt else 0})
    avg = sum(it["tok_s"] for it in iters[1:]) / max(1, len(iters) - 1)
    result = {
        "status": "ok",
        "load_s": round(load_t, 1),
        "model_mb": round(used_mb_after_load, 0),
        "iters": iters,
        "avg_tok_s": round(avg, 2),
    }
except Exception as e:
    result = {"status": "load_failed", "err": str(e)[:200]}

print(json.dumps(result), flush=True)
sys.exit(0)
