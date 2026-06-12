#!/usr/bin/env python3
"""T7.6... no wait T7.0 -- Prompt/prefix caching for the llama.cpp backend.

Hypothesis: coding agents resend the same long prefix (system prompt +
repo context) on every call. llama-cpp-python's ``Llama.generate()``
already does longest-common-prefix KV cache reuse against the *previous*
call's tokens (``llama_cpp/llama.py`` -- ``kv_cache_seq_rm`` on prefix
match), as long as the same ``Llama`` instance is reused across requests
(true here: VRAMancer's llama.cpp backend keeps one persistent ``self.model``
for the lifetime of the server process).

This script measures TTFT (time-to-first-token) for a second request that
shares a ~4000-token prefix with the first, with vs without an intervening
"eviction" request that breaks the prefix match -- against a LIVE
VRAMancer API server (no model load here; reuses whatever is already
running, e.g. Qwen3.6-35B-A3B via llama.cpp).

Usage:
    python benchmarks/bench_prefix_cache_llamacpp.py --base-url http://localhost:5031
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = Path("benchmarks/results/phase7/T7.0_prefix_cache.json")
OUT_MD = Path("benchmarks/results/phase7/T7.0_prefix_cache.md")


def _read_chars(path: Path, n_chars: int) -> str:
    text = path.read_text(errors="ignore")
    return text[:n_chars]


def _completion(base_url: str, prompt: str, max_tokens: int, stream: bool, timeout: int = 300):
    """POST /v1/completions. Returns (ttft_s_or_None, total_s, text, prompt_tokens)."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if not stream:
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        total = time.perf_counter() - t0
        text = data["choices"][0]["text"]
        prompt_tokens = data.get("usage", {}).get("prompt_tokens")
        return None, total, text, prompt_tokens

    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/completions", json=payload, stream=True, timeout=timeout)
    r.raise_for_status()
    ttft = None
    text_parts = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        raw = line[len("data: "):]
        if raw.strip() == "[DONE]":
            break
        chunk = json.loads(raw)
        choice = chunk["choices"][0]
        token_text = choice.get("text", "")
        if token_text and ttft is None:
            ttft = time.perf_counter() - t0
        if token_text:
            text_parts.append(token_text)
    total = time.perf_counter() - t0
    return ttft, total, "".join(text_parts), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:5031")
    ap.add_argument("--prefix-chars", type=int, default=14000,
                    help="chars of source code used as the shared prefix (~4000 tokens for code)")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    prefix = _read_chars(ROOT / "core" / "inference_pipeline.py", args.prefix_chars)
    evictor_prefix = _read_chars(ROOT / "core" / "backends.py", args.prefix_chars)

    suffix_a = "\n\n# TASK A: Add a one-line docstring to the function above.\n"
    suffix_b = "\n\n# TASK B: Add type hints to the function above.\n"

    results = {"base_url": args.base_url, "prefix_chars": args.prefix_chars,
               "max_tokens": args.max_tokens}

    print("=== req1: warm the cache (PREFIX + SUFFIX_A) ===")
    _, total1, text1, prompt_tokens1 = _completion(
        args.base_url, prefix + suffix_a, args.max_tokens, stream=False,
    )
    results["req1_prompt_tokens"] = prompt_tokens1
    results["req1_total_s"] = round(total1, 3)
    print(f"prompt_tokens={prompt_tokens1}, total={total1:.3f}s")

    print("\n=== req2 WITH cache: PREFIX + SUFFIX_B (TTFT) ===")
    ttft_with, total_with, text_with, _ = _completion(
        args.base_url, prefix + suffix_b, args.max_tokens, stream=True,
    )
    results["ttft_with_cache_s"] = round(ttft_with, 3) if ttft_with is not None else None
    results["total_with_cache_s"] = round(total_with, 3)
    print(f"TTFT={ttft_with:.3f}s, total={total_with:.3f}s")

    print("\n=== eviction request: EVICTOR_PREFIX + SUFFIX_A (breaks prefix match) ===")
    _, total_evict, _, prompt_tokens_evict = _completion(
        args.base_url, evictor_prefix + suffix_a, 16, stream=False,
    )
    results["evict_prompt_tokens"] = prompt_tokens_evict
    results["evict_total_s"] = round(total_evict, 3)
    print(f"prompt_tokens={prompt_tokens_evict}, total={total_evict:.3f}s")

    print("\n=== req2 WITHOUT cache: PREFIX + SUFFIX_B again (TTFT) ===")
    ttft_without, total_without, text_without, _ = _completion(
        args.base_url, prefix + suffix_b, args.max_tokens, stream=True,
    )
    results["ttft_without_cache_s"] = round(ttft_without, 3) if ttft_without is not None else None
    results["total_without_cache_s"] = round(total_without, 3)
    print(f"TTFT={ttft_without:.3f}s, total={total_without:.3f}s")

    results["outputs_identical"] = (text_with == text_without)
    results["text_with_cache"] = text_with
    results["text_without_cache"] = text_without
    if ttft_with and ttft_without:
        results["ttft_reduction_pct"] = round(100 * (1 - ttft_with / ttft_without), 1)

    print(f"\noutputs_identical={results['outputs_identical']}")
    if "ttft_reduction_pct" in results:
        print(f"TTFT reduction: {results['ttft_reduction_pct']}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    lines = [
        "# T7.0 -- Prompt/prefix cache (llama.cpp backend)",
        "",
        f"Live server: `{args.base_url}` (model: whatever is currently loaded).",
        f"Shared prefix: ~{results.get('req1_prompt_tokens', '?')} tokens "
        f"(first {args.prefix_chars} chars of `core/inference_pipeline.py`), "
        f"differing only in the last ~10-15 tokens (TASK A vs TASK B suffix).",
        "",
        "| Metric | WITH cache (req2 right after req1) | WITHOUT cache "
        "(req2 after an evicting request) |",
        "|---|---|---|",
        f"| TTFT (s) | {results.get('ttft_with_cache_s')} | {results.get('ttft_without_cache_s')} |",
        f"| total (s) | {results.get('total_with_cache_s')} | {results.get('total_without_cache_s')} |",
        "",
        f"**TTFT reduction**: {results.get('ttft_reduction_pct', 'n/a')}% "
        f"(acceptance threshold: >= 50%).",
        "",
        f"**Greedy outputs identical (with vs without cache)**: "
        f"{results['outputs_identical']}.",
        "",
        "## Raw outputs",
        "",
        "### WITH cache",
        "```",
        text_with,
        "```",
        "",
        "### WITHOUT cache",
        "```",
        text_without,
        "```",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
