"""V6.E profiler — Qwen3-Coder-30B-A3B expert usage histogram.

Goal: identify the "hot" experts (top-K most frequently routed) for a
representative coding workload, so we can pin them in the 5070 Ti's VRAM
while keeping cold experts in the 3090 lending buffer (P2P prefetch on
demand). This profiler runs the model on a varied prompt mix and dumps
a per-layer-per-expert activation histogram.

Uses vLLM 0.20.1's built-in `enable_return_routed_experts=True`, which
exposes the routed expert IDs at every layer for every generated token
in CompletionOutput.routed_experts (shape [seq_len, num_layers, topk]).
No monkey-patching, no model surgery.

Outputs:
  - benchmarks/results/qwen3_coder_expert_histogram.json
      {
        "model": "...",
        "num_layers": int,
        "num_experts_per_layer": int,
        "topk": int,
        "total_tokens_observed": int,
        "histogram": [[count, count, ...], ...]   # [layer][expert] -> count
        "hot_experts_top20pct": [[exp_ids...], ...],
        "hot_share_top20pct": float,    # fraction of all activations that hit hot experts
      }
  - benchmarks/results/qwen3_coder_expert_histogram.md (human summary)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Same env hygiene as the bench scripts.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 5070 Ti

import numpy as np  # noqa: E402

HF_MODEL_ID = os.environ.get(
    "VRM_PROFILE_MODEL", "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
)
MAX_MODEL_LEN = int(os.environ.get("VRM_PROFILE_MAX_LEN", "2048"))
GPU_UTIL = float(os.environ.get("VRM_PROFILE_GPU_UTIL", "0.92"))
CPU_OFFLOAD_GB = float(os.environ.get("VRM_PROFILE_CPU_OFFLOAD", "4"))
MAX_NEW = int(os.environ.get("VRM_PROFILE_MAX_NEW", "128"))
OUT_JSON = Path("benchmarks/results/qwen3_coder_expert_histogram.json")
OUT_MD = Path("benchmarks/results/qwen3_coder_expert_histogram.md")


# A varied mix: code, prose, math, list, comments — covers most expert
# specialisations the MoE router is likely to dispatch on.
PROMPTS = [
    # Code-heavy
    "Write a Python function that implements a binary search tree with "
    "insert, delete, and balance operations. Include type hints, docstrings, "
    "and a brief example of usage at the end:\n\n",
    "Refactor this Rust code to use Result<T, E> instead of unwrap() "
    "everywhere, and explain the trade-offs:\n\nfn parse(s: &str) -> u32 {\n"
    "    s.parse::<u32>().unwrap()\n}\n\n",
    "Given a SQL schema with users, orders, and products tables, write an "
    "optimized query that returns the top 10 customers by total spend in "
    "the last 90 days, broken down by product category:\n\n",
    # System-design / prose
    "Explain the trade-offs between eventual consistency and strong "
    "consistency in a distributed key-value store, with concrete examples "
    "of when each is appropriate.\n\n",
    "Describe the architecture of a high-throughput log ingestion pipeline "
    "that handles 1M events per second with under 1s p99 latency. List the "
    "components and their responsibilities.\n\n",
    # Math / reasoning
    "Prove that the sum of the first n odd positive integers equals n^2, "
    "using induction. Show every step.\n\n",
    "If a fair coin is flipped 10 times, what is the probability of getting "
    "exactly 6 heads? Walk through the calculation.\n\n",
    # Mixed
    "List five common pitfalls when implementing a thread-safe LRU cache "
    "in C++, and for each, show a minimal code snippet that demonstrates "
    "the bug:\n\n",
    "Write a comprehensive bash script that monitors a Linux server's CPU, "
    "memory, and disk I/O, alerts on thresholds, and rotates its log file "
    "daily:\n\n",
    "Given the function f(x) = x^3 - 2x + 1, find all real roots and "
    "describe the shape of the graph:\n\n",
]


def main():
    print("=" * 70)
    print("[V6.E profile] Qwen3-Coder-30B-A3B expert usage histogram")
    print("=" * 70)
    print(f"  Model       : {HF_MODEL_ID}")
    print(f"  Prompts     : {len(PROMPTS)}")
    print(f"  Max new tok : {MAX_NEW} per prompt")
    print(f"  Goal tokens : ~{MAX_NEW * len(PROMPTS)}")
    print()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"[BLOCKED] vLLM not importable: {e}", file=sys.stderr)
        sys.exit(1)

    print("[1/3] Loading model...")
    llm = LLM(
        model=HF_MODEL_ID,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        cpu_offload_gb=CPU_OFFLOAD_GB,
        enforce_eager=True,  # eager makes per-token expert capture cleaner
        trust_remote_code=True,
        enable_return_routed_experts=True,
    )
    print("[1/3] Model loaded.")

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW,
    )

    print(f"\n[2/3] Generating {MAX_NEW} tokens for each of {len(PROMPTS)} prompts...")
    outputs = llm.generate(PROMPTS, sampling)

    # Aggregate the per-token routed_experts across all prompts.
    # routed_experts shape per prompt: [seq_len, num_layers, topk]
    histogram = None  # will be Counter[(layer_idx, expert_id)] -> count
    num_layers = None
    num_experts = None  # max expert id seen + 1
    topk = None
    total_tokens = 0

    for i, out in enumerate(outputs):
        if not out.outputs:
            print(f"  [warn] prompt {i}: empty output")
            continue
        co = out.outputs[0]  # we only sampled n=1
        re_arr = co.routed_experts
        if re_arr is None:
            print(f"  [warn] prompt {i}: routed_experts is None — "
                  f"enable_return_routed_experts may not be wired for this model")
            continue
        # re_arr is [seq_len, layer_num, topk]
        if histogram is None:
            num_layers = re_arr.shape[1]
            topk = re_arr.shape[2]
            histogram = [Counter() for _ in range(num_layers)]
            print(f"  routed_experts shape: {re_arr.shape}  "
                  f"(layers={num_layers}, topk={topk})")
        seq_len = re_arr.shape[0]
        total_tokens += seq_len
        for tok_idx in range(seq_len):
            for layer in range(num_layers):
                for k in range(topk):
                    eid = int(re_arr[tok_idx, layer, k])
                    histogram[layer][eid] += 1
                    if num_experts is None or eid + 1 > num_experts:
                        num_experts = eid + 1
        print(f"  prompt {i}: {seq_len} tokens captured")

    if histogram is None:
        print("[ABORT] No expert routing data captured.")
        return

    print(f"\n[2/3] Captured {total_tokens} tokens × "
          f"{num_layers} layers × {topk} topk = "
          f"{total_tokens * num_layers * topk} expert activations.")

    # Build dense histogram array: [num_layers][num_experts]
    print("\n[3/3] Computing hot/cold split (top 20% experts per layer)...")
    dense_hist = np.zeros((num_layers, num_experts), dtype=np.int64)
    for layer in range(num_layers):
        for eid, count in histogram[layer].items():
            if eid < num_experts:
                dense_hist[layer, eid] = count

    hot_count = max(1, int(num_experts * 0.20))
    hot_experts_per_layer = []
    hot_activations = 0
    total_activations = int(dense_hist.sum())
    for layer in range(num_layers):
        # Top hot_count expert IDs by count for this layer
        top_idx = np.argsort(dense_hist[layer])[-hot_count:][::-1]
        hot_experts_per_layer.append([int(x) for x in top_idx])
        hot_activations += int(dense_hist[layer][top_idx].sum())

    hot_share = hot_activations / total_activations if total_activations else 0.0

    # Per-layer hot share for sanity
    per_layer_hot_share = []
    for layer in range(num_layers):
        layer_total = int(dense_hist[layer].sum())
        layer_hot = int(dense_hist[layer][hot_experts_per_layer[layer]].sum())
        per_layer_hot_share.append(
            layer_hot / layer_total if layer_total else 0.0
        )

    # ---- Write JSON ----
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": HF_MODEL_ID,
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts,
        "topk": topk,
        "total_tokens_observed": total_tokens,
        "total_activations": total_activations,
        "histogram": dense_hist.tolist(),
        "hot_experts_top20pct": hot_experts_per_layer,
        "hot_count_per_layer": hot_count,
        "hot_share_top20pct": round(hot_share, 4),
        "per_layer_hot_share": [round(x, 4) for x in per_layer_hot_share],
        "prompts_count": len(PROMPTS),
    }, indent=2))

    # ---- Write Markdown ----
    md = [
        "# Qwen3-Coder-30B-A3B — expert usage histogram (V6.E profile)",
        "",
        f"- **Model**: `{HF_MODEL_ID}`",
        f"- **Layers**: {num_layers}",
        f"- **Experts per layer**: {num_experts}",
        f"- **topk per token**: {topk}",
        f"- **Tokens profiled**: {total_tokens} "
        f"(across {len(PROMPTS)} varied prompts: code / prose / math / mixed)",
        f"- **Total expert activations**: {total_activations:,}",
        "",
        "## Pareto check: top 20% experts capture this share of activations",
        "",
        f"- **Top-{hot_count} experts per layer hit "
        f"{hot_share*100:.1f}% of all activations** "
        f"(vs uniform = 20%).",
        "- If this is well above 20%, a hot/cold pinning scheme has clear "
        "headroom: pin top-K to GPU VRAM, route the rest via the 3090 "
        "lending buffer (P2P 11.7 GB/s) instead of DRAM (UVA ~25 GB/s "
        "with higher latency on small fetches).",
        "- If it's near 20%, expert usage is roughly uniform and the "
        "lending pool's value is in raw bandwidth (DRAM→3090→GPU) rather "
        "than caching.",
        "",
        "## Per-layer hot share distribution",
        "",
        "| Layer range | Mean hot share | Min | Max |",
        "|-------------|---------------|-----|-----|",
    ]
    pls = per_layer_hot_share
    if pls:
        # Bucket layers into thirds for readability
        third = max(1, num_layers // 3)
        for label, sl in [
            ("early (first third)", pls[:third]),
            ("middle", pls[third:2 * third]),
            ("late (last third)", pls[2 * third:]),
        ]:
            if sl:
                md.append(
                    f"| {label} | {sum(sl)/len(sl)*100:.1f}% | "
                    f"{min(sl)*100:.1f}% | {max(sl)*100:.1f}% |"
                )
    md += [
        "",
        "## Implication for V6.E expert pinning",
        "",
        f"- **Hot experts per layer**: {hot_count} (top 20%)",
        f"- **Hot expert weight footprint** (rough est, 30B/{num_experts}≈"
        f"{30/num_experts:.2f} GB/expert in BF16, "
        f"{30/num_experts/2:.2f} GB in FP8, "
        f"{30/num_experts/4:.2f} GB in Int4): ~"
        f"{hot_count * num_layers * 30/num_experts/2:.1f} GB FP8 / "
        f"~{hot_count * num_layers * 30/num_experts/4:.1f} GB Int4 across "
        f"all layers.",
        "- These hot weights, pinned to 5070 Ti VRAM, would be reused on "
        f"~{hot_share*100:.1f}% of expert activations with zero PCIe cost.",
        "- The remaining 80% of experts (cold) sit in the 3090 lending "
        "buffer and are P2P-fetched on demand — still PCIe but lower "
        "latency than DRAM via UVA, and overlap-able with the next layer's "
        "router compute.",
        "",
        "*Generated by VRAMancer benchmarks/profile_qwen3_experts.py*",
    ]
    OUT_MD.write_text("\n".join(md))

    print(f"\nHistogram JSON: {OUT_JSON}")
    print(f"Summary MD    : {OUT_MD}")
    print(f"\nKey number  : top-20%-experts capture {hot_share*100:.1f}% "
          f"of all activations.")


if __name__ == "__main__":
    main()
