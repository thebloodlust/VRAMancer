#!/usr/bin/env python3
"""T7.9 step 1 -- Pre-measurement of the pipeline-parallel "bubble".

Hypothesis: in VRAMancer's 2-GPU pipeline-parallel decode loop
(HuggingFaceBackend._infer_with_kv_cache_impl, "Path 2"), each GPU sits
idle while the other computes its block -- roughly 50% idleness per GPU
for a single request.

Method:
- Load Qwen2.5-7B-Instruct WITHOUT accelerate device_map (device_map=None),
  so split_model() takes VRAMancer's manual pipeline-parallel path
  (self.blocks with len==2, KVCacheBlock per GPU) instead of the
  accelerate/native-generate path used by T7.1.
- Instrument HuggingFaceBackend.infer() with torch.cuda.Event timers
  bracketing each block's forward pass (per-GPU compute time) and the
  cross-GPU transfer in between.
- Run 50 greedy decode steps (1 request), sum per-GPU busy time, and
  compute idleness_i = 1 - busy_time_i / wall_time.

This is a disposable measurement script (T7.9 step 1) -- gates whether
step 2 (2-slot interleaving scheduler) is worth implementing. If
idleness < 30%, report the number and STOP before implementing the
scheduler (per the task's stop condition).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.backends import HuggingFaceBackend

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAX_NEW = 50
OUT_JSON = Path("benchmarks/results/phase7/T7.9_bubble_probe.json")
OUT_MD = Path("benchmarks/results/phase7/T7.9_bubble_probe.md")


def main():
    backend = HuggingFaceBackend()
    backend.model_name = MODEL
    backend.tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if backend.tokenizer.pad_token_id is None:
        backend.tokenizer.pad_token_id = backend.tokenizer.eos_token_id

    print("Loading model without accelerate device_map (device_map=None)...")
    backend.model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=None,
    )
    assert not (hasattr(backend.model, "hf_device_map") and backend.model.hf_device_map), \
        "model still has hf_device_map -- split_model() would take the accelerate path"

    blocks = backend.split_model(num_gpus=2)
    print(f"split_model -> {len(blocks)} blocks, devices={backend.block_devices}, "
          f"components={'yes' if backend._components else 'no'}")
    if len(blocks) <= 1:
        raise SystemExit("Manual split produced <=1 block -- Path 2 not active, "
                          "T7.9 as scoped does not apply. Report this.")

    # Per-GPU busy-time accounting via CUDA events bracketing each block.
    n_gpus = len(set(backend.block_devices))
    busy_ms = {dev: 0.0 for dev in set(backend.block_devices)}

    orig_block_forward = {}
    for idx, block in enumerate(blocks):
        dev = backend.block_devices[idx]

        def make_wrapper(orig_forward, dev=dev):
            def wrapped(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(dev)
                start.record(torch.cuda.current_stream(dev))
                out = orig_forward(*args, **kwargs)
                end.record(torch.cuda.current_stream(dev))
                torch.cuda.synchronize(dev)
                busy_ms[dev] += start.elapsed_time(end)
                return out
            return wrapped

        orig_block_forward[idx] = block.forward
        block.forward = make_wrapper(block.forward)

    inputs = backend.tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs.input_ids

    import time
    t0 = time.perf_counter()
    generated = input_ids
    past = None
    for step in range(MAX_NEW):
        step_input = generated[:, -1:] if past is not None else generated
        result = backend.infer(step_input, past_key_values=past, use_cache=True)
        logits, past = result if isinstance(result, tuple) else (result, None)
        next_logits = logits[:, -1, :] if logits.dim() >= 2 else logits
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True).to(generated.device)
        generated = torch.cat([generated, next_token], dim=-1)
        if backend.tokenizer.eos_token_id is not None and next_token.item() == backend.tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    n_steps = generated.shape[1] - input_ids.shape[1]

    wall_ms = wall_s * 1000
    idleness = {f"cuda:{dev}": round(1 - (busy_ms[dev] / wall_ms), 4) for dev in busy_ms}
    busy_pct = {f"cuda:{dev}": round(busy_ms[dev] / wall_ms, 4) for dev in busy_ms}

    result_data = {
        "model": MODEL,
        "n_decode_steps": n_steps,
        "wall_time_ms": round(wall_ms, 2),
        "busy_ms_per_gpu": {f"cuda:{k}": round(v, 2) for k, v in busy_ms.items()},
        "busy_fraction_per_gpu": busy_pct,
        "idleness_per_gpu": idleness,
        "tok_s": round(n_steps / wall_s, 2),
    }
    print(json.dumps(result_data, indent=2))

    max_idle = max(idleness.values())
    verdict = "PEU_DE_GAIN_ATTENDU" if max_idle < 0.3 else "BULLE_SIGNIFICATIVE"
    result_data["verdict"] = verdict

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result_data, indent=2))

    lines = [
        "# T7.9 step 1 -- Pipeline bubble pre-measurement",
        "",
        f"Model: `{MODEL}`, {n_steps} greedy decode steps, 1 request, 2 GPUs "
        f"(manual VRAMancer split, Path 2).",
        "",
        f"Wall time: {wall_ms:.1f} ms ({result_data['tok_s']} tok/s)",
        "",
        "| GPU | busy (ms) | busy % | idle % |",
        "|---|---|---|---|",
    ]
    for dev in busy_ms:
        lines.append(f"| cuda:{dev} | {busy_ms[dev]:.1f} | "
                      f"{busy_pct[f'cuda:{dev}']*100:.1f}% | "
                      f"{idleness[f'cuda:{dev}']*100:.1f}% |")
    lines.append("")
    lines.append(f"Verdict: **{verdict}** (max idleness = {max_idle*100:.1f}%, "
                  f"threshold = 30%)")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
