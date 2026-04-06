#!/usr/bin/env python3
"""Benchmark: GPT-2 inference via browser WebNN ("WebNPU").

Usage:
    python benchmarks/bench_webnpu_gpt2.py

Same as bench_webgpu_gpt2.py but directs the user to open webnpu.html
which auto-detects and uses WebNN (NPU→GPU→CPU) with WebGPU fallback.

The server-side protocol is identical — only the browser-side changes.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("ERROR: torch required. pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers required. pip install transformers")
    sys.exit(1)

from core.webgpu_backend import WebGPUBackend


def main():
    model_name = os.environ.get("VRM_WEBGPU_MODEL", "gpt2")
    prompt = os.environ.get("VRM_WEBGPU_PROMPT", "The meaning of life is")
    max_tokens = int(os.environ.get("VRM_WEBGPU_MAX_TOKENS", "50"))

    print("=" * 55)
    print("  VRAMancer WebNPU Inference Benchmark")
    print("  WebNN → NPU / GPU / CPU (with WebGPU fallback)")
    print("=" * 55)
    print(f"Model: {model_name}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print()

    # --- Load model locally ---
    print("[1/5] Loading model locally...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()
    load_time = time.time() - t0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {total_params/1e6:.0f}M params in {load_time:.1f}s")

    # --- Start server ---
    print("\n[2/5] Starting WebGPU/WebNN backend server...")
    backend = WebGPUBackend()
    http_proto = "https" if backend._ssl_context else "http"
    port = backend._ws_port

    print(f"  Server: {http_proto}://0.0.0.0:{port}")
    print()
    print(f"  >>> Open in browser: {http_proto}://localhost:{port}/webnpu.html")
    print(f"  >>> Or any device:   {http_proto}://<server-ip>:{port}/webnpu.html")
    print()
    print(f"  The page auto-detects: WebNN NPU → WebNN GPU → WebGPU fallback")
    print(f"  Waiting for browser to connect...")

    try:
        backend.wait_for_workers(1, timeout=300)
    except TimeoutError:
        print("ERROR: No browser connected within 5 minutes.")
        backend.shutdown()
        sys.exit(1)

    rtt = backend.ping_workers()
    for addr, ms in rtt.items():
        print(f"  Worker {addr}: RTT {ms}ms")

    # --- Push model to browser ---
    print("\n[3/5] Uploading GPT-2 weights to browser...")
    t0 = time.time()
    backend.push_gpt2_model(model, tokenizer)
    upload_time = time.time() - t0
    total_mb = total_params * 4 / 1024 / 1024
    print(f"  Uploaded {total_mb:.0f} MB in {upload_time:.1f}s "
          f"({total_mb/upload_time:.0f} MB/s)")

    # --- Local PyTorch reference ---
    print(f"\n[4/5] Local PyTorch reference generation...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        local_output = model.generate(
            input_ids, max_new_tokens=max_tokens,
            do_sample=False, temperature=1.0,
        )
    local_time = time.time() - t0
    local_tokens = local_output[0].tolist()
    local_text = tokenizer.decode(local_tokens, skip_special_tokens=True)
    local_new_toks = len(local_tokens) - input_ids.shape[1]
    local_tps = local_new_toks / local_time if local_time > 0 else 0
    print(f"  Local: {local_new_toks} tokens in {local_time:.2f}s "
          f"({local_tps:.1f} tok/s)")
    print(f"  Text: {local_text!r}")

    # --- Browser WebNN generation ---
    print(f"\n[5/5] Browser WebNN (WebNPU) generation...")
    prompt_ids = tokenizer.encode(prompt)
    tokens_received = []

    def on_token(token_id, step, time_ms):
        tokens_received.append(token_id)
        text = tokenizer.decode([token_id])
        print(f"  Token {step}: {token_id} ({text!r}) {time_ms:.1f}ms",
              flush=True)

    t0 = time.time()
    result = backend.generate_browser(
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        token_callback=on_token,
    )
    wall_time = time.time() - t0

    print(f"\n{'=' * 55}")
    print(f"  RESULTS — WebNPU Benchmark")
    print(f"{'=' * 55}")
    print(f"Prompt: {prompt!r}")
    if result.get("text"):
        print(f"WebNPU output: {result['text']!r}")
    print(f"Local output:  {local_text!r}")
    print()
    print(f"WebNPU: {result['total_tokens']} tokens, "
          f"{result['total_ms']:.0f}ms total, "
          f"{result.get('tok_per_s', 0):.1f} tok/s (decode)")
    print(f"Local:  {local_new_toks} tokens, "
          f"{local_time*1000:.0f}ms total, "
          f"{local_tps:.1f} tok/s")

    # Check token match
    webnn_tokens = result["tokens"]
    local_new_only = local_tokens[input_ids.shape[1]:]
    match_len = min(len(webnn_tokens), len(local_new_only))
    matches = sum(1 for a, b in zip(webnn_tokens[:match_len],
                                      local_new_only[:match_len]) if a == b)

    print(f"\nToken match: {matches}/{match_len} "
          f"({'PASS' if matches == match_len else 'MISMATCH'})")

    if matches < match_len:
        print(f"  WebNPU tokens: {webnn_tokens[:10]}...")
        print(f"  Local tokens:  {local_new_only[:10]}...")

    backend.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
