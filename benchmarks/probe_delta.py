#!/usr/bin/env python3
"""T7.5 -- Feasibility probe: temporal delta of boundary activations (1 day MAX).

Hypothesis (uncertain, kill fast if false): hidden states at the GPU
boundary (the activation handed off from the last layer on GPU0 to the
first layer on GPU1) change little from one decode step to the next;
transmitting a quantized *delta* could cost less than the full activation.

Method: load Qwen2.5-7B across 2 GPUs (accelerate device_map="auto"),
locate the boundary layer (first decoder layer placed on cuda:1 via
hf_device_map), register a forward pre-hook capturing its input
hidden_states at every decode step, generate 100 tokens greedy, and
compute r_t = norm(h_t - h_{t-1}) / norm(h_t) for t=2..100.

Verdict: median(r_t) < 0.3 -> hypothesis alive, open a follow-up task.
         median(r_t) >= 0.3 -> close definitively, report the number.

This is a disposable feasibility script -- not a benchmark to be reused.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAX_NEW = 100
OUT_JSON = Path("benchmarks/results/phase7/T7.5_delta_probe.json")
OUT_MD = Path("benchmarks/results/phase7/T7.5_delta_probe.md")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    dev_map = getattr(model, "hf_device_map", {})
    print("hf_device_map sample:", {k: v for k, v in list(dev_map.items())[:5]}, "...")

    # Find decoder layers and the device of each, to locate the boundary:
    # the first layer index whose device differs from layer 0's device.
    layer_devices = {}
    for k, v in dev_map.items():
        if ".layers." in k:
            idx = int(k.split(".layers.")[1].split(".")[0])
            layer_devices.setdefault(idx, v)

    layer0_dev = layer_devices[0]
    boundary_idx = None
    for idx in sorted(layer_devices):
        if layer_devices[idx] != layer0_dev:
            boundary_idx = idx
            break

    if boundary_idx is None:
        raise SystemExit(
            f"Model fits entirely on one device ({layer0_dev}) -- "
            "no GPU boundary to probe. hf_device_map: " + str(dev_map)
        )

    print(f"Boundary layer: model.model.layers[{boundary_idx}] "
          f"({layer_devices[boundary_idx-1]} -> {layer_devices[boundary_idx]})")

    boundary_layer = model.model.layers[boundary_idx]

    captured: list[torch.Tensor] = []

    def hook(module, args, kwargs):
        hs = kwargs.get("hidden_states") if kwargs else None
        if hs is None and args:
            hs = args[0]
        # hs shape: [batch, seq, hidden]. During incremental decode seq==1.
        captured.append(hs.detach().float().reshape(-1).cpu())

    handle = boundary_layer.register_forward_pre_hook(hook, with_kwargs=True)

    msgs = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True,
    ).to(f"cuda:{layer0_dev}" if isinstance(layer0_dev, int) else layer0_dev)

    with torch.no_grad():
        model.generate(
            **inputs, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True,
        )

    handle.remove()

    print(f"Captured {len(captured)} boundary activations "
          f"(prompt forward + {MAX_NEW} decode steps expected)")

    # Drop the first capture (full-prompt prefill, seq_len > 1) -- we only
    # want the per-token decode steps (seq_len == 1) for h_t.
    decode_acts = [h for h in captured if h.numel() == captured[-1].numel()]
    print(f"Decode-step activations (seq_len==1): {len(decode_acts)}")

    r_values = []
    for t in range(1, len(decode_acts)):
        h_t = decode_acts[t]
        h_prev = decode_acts[t - 1]
        delta_norm = torch.norm(h_t - h_prev).item()
        h_norm = torch.norm(h_t).item()
        r_t = delta_norm / h_norm if h_norm > 0 else float("nan")
        r_values.append(r_t)

    r_min = min(r_values)
    r_med = statistics.median(r_values)
    r_max = max(r_values)

    verdict = "HYPOTHESE_VIVANTE" if r_med < 0.3 else "FERME"

    result = {
        "model": MODEL,
        "boundary_layer_idx": boundary_idx,
        "boundary_devices": [str(layer_devices[boundary_idx - 1]), str(layer_devices[boundary_idx])],
        "n_decode_steps": len(decode_acts),
        "n_r_values": len(r_values),
        "r_min": r_min,
        "r_median": r_med,
        "r_max": r_max,
        "verdict": verdict,
    }
    print(json.dumps(result, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    OUT_MD.write_text(
        "# T7.5 -- Delta probe (boundary activations)\n\n"
        f"Model: `{MODEL}`, boundary layer: `model.model.layers[{boundary_idx}]` "
        f"({layer_devices[boundary_idx-1]} -> {layer_devices[boundary_idx]}), "
        f"{len(r_values)} decode steps (greedy, max_new_tokens={MAX_NEW}).\n\n"
        "r_t = norm(h_t - h_{t-1}) / norm(h_t)\n\n"
        f"| min | median | max | verdict |\n"
        f"|---|---|---|---|\n"
        f"| {r_min:.4f} | {r_med:.4f} | {r_max:.4f} | {verdict} |\n"
    )
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
