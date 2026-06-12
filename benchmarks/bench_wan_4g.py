#!/usr/bin/env python3
"""Benchmark: GPT-2 inference via WAN (4G / internet) through DDNS.

Proves VRAMancer AITP works over hostile networks.

Server side (this script):
    VRM_DDNS_HOST=jeje1.synology.me python benchmarks/bench_wan_4g.py

Client side (phone browser over 4G):
    Open https://jeje1.synology.me:55555/webnpu.html

Router must forward port 55555 TCP to server LAN IP.

Env vars:
    VRM_DDNS_HOST       DDNS hostname (required, e.g. jeje1.synology.me)
    VRM_AITP_PORT       AITP base port (default 55555)
    VRM_WEBGPU_MODEL    Model name (default gpt2)
    VRM_WEBGPU_PROMPT   Prompt text
    VRM_WEBGPU_MAX_TOKENS  Max tokens (default 30)
"""

import sys
import os
import time
import json
import socket
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force WebSocket port = AITP port for single-port WAN forwarding
_PORT = int(os.environ.get("VRM_AITP_PORT", "55555"))
os.environ.setdefault("VRM_WEBGPU_WS_PORT", str(_PORT))

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


def check_ddns(hostname: str) -> str | None:
    """Resolve DDNS and return public IP, or None on failure."""
    try:
        ip = socket.gethostbyname(hostname)
        return ip
    except socket.gaierror:
        return None


def check_port_open(ip: str, port: int, timeout: float = 3.0) -> bool:
    """Quick TCP connect test."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


def get_public_ip() -> str | None:
    """Get our public IP via simple DNS trick."""
    try:
        result = subprocess.run(
            ["dig", "+short", "myip.opendns.com", "@resolver1.opendns.com"],
            capture_output=True, text=True, timeout=5,
        )
        ip = result.stdout.strip()
        return ip if ip else None
    except Exception:
        return None


def main():
    ddns_host = os.environ.get("VRM_DDNS_HOST", "")
    model_name = os.environ.get("VRM_WEBGPU_MODEL", "gpt2")
    prompt = os.environ.get("VRM_WEBGPU_PROMPT", "The meaning of life is")
    max_tokens = int(os.environ.get("VRM_WEBGPU_MAX_TOKENS", "30"))

    print("=" * 60)
    print("  VRAMancer WAN/4G Inference Benchmark")
    print("  AITP over Internet — DDNS + WebNPU")
    print("=" * 60)
    print(f"DDNS host:  {ddns_host or '(not set)'}")
    print(f"Port:       {_PORT}")
    print(f"Model:      {model_name}")
    print(f"Prompt:     {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print()

    # --- Pre-flight checks ---
    print("[0/6] Pre-flight WAN checks...")

    our_ip = get_public_ip()
    if our_ip:
        print(f"  Public IP: {our_ip}")
    else:
        print("  Public IP: (could not determine)")

    if ddns_host:
        resolved = check_ddns(ddns_host)
        if resolved:
            print(f"  DDNS {ddns_host} -> {resolved}")
            if our_ip and resolved == our_ip:
                print(f"  DDNS matches our public IP")
            elif our_ip:
                print(f"  WARNING: DDNS resolves to {resolved}, "
                      f"our public IP is {our_ip}")
        else:
            print(f"  WARNING: Cannot resolve {ddns_host}")
    else:
        print("  WARNING: VRM_DDNS_HOST not set. Set it for WAN test.")
        print("  Falling back to LAN-only mode.")

    # --- Load model ---
    print(f"\n[1/6] Loading model locally...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    )
    model.eval()
    load_time = time.time() - t0
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {total_params / 1e6:.0f}M params in {load_time:.1f}s")

    # --- Local reference ---
    print(f"\n[2/6] Local PyTorch reference generation...")
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

    # --- Start server ---
    print(f"\n[3/6] Starting WebGPU/WebNN server on port {_PORT}...")
    backend = WebGPUBackend()
    http_proto = "https" if backend._ssl_context else "http"

    print(f"  Server: {http_proto}://0.0.0.0:{_PORT}")
    print()
    if ddns_host:
        print(f"  >>> WAN URL:  {http_proto}://{ddns_host}:{_PORT}/webnpu.html")
    print(f"  >>> LAN URL:  {http_proto}://<server-ip>:{_PORT}/webnpu.html")
    print(f"  >>> Local:    {http_proto}://localhost:{_PORT}/webnpu.html")
    print()
    print(f"  On your phone (4G, WiFi OFF):")
    if ddns_host:
        print(f"    1. Open {http_proto}://{ddns_host}:{_PORT}/webnpu.html")
    print(f"    2. Accept self-signed certificate warning")
    print(f"    3. Page auto-connects via WebSocket")
    print()
    print(f"  Waiting for browser connection (10 min timeout for WAN)...")

    try:
        backend.wait_for_workers(1, timeout=600)
    except TimeoutError:
        print("ERROR: No browser connected within 10 minutes.")
        backend.shutdown()
        sys.exit(1)

    # --- WAN RTT measurement ---
    print(f"\n[4/6] Measuring WAN RTT...")
    rtt = backend.ping_workers()
    for addr, ms in rtt.items():
        print(f"  Worker {addr}: RTT {ms:.1f}ms")
        if ms > 100:
            print(f"    (High latency — typical for 4G/WAN)")

    # --- Push model ---
    print(f"\n[5/6] Uploading GPT-2 weights to browser...")
    t0 = time.time()
    backend.push_gpt2_model(model, tokenizer)
    upload_time = time.time() - t0
    total_mb = total_params * 4 / 1024 / 1024
    bw = total_mb / upload_time if upload_time > 0 else 0
    print(f"  Uploaded {total_mb:.0f} MB in {upload_time:.1f}s ({bw:.0f} MB/s)")

    # --- Browser generation ---
    print(f"\n[6/6] Browser WebNPU generation over WAN...")
    prompt_ids = tokenizer.encode(prompt)
    token_times = []

    def on_token(token_id, step, time_ms):
        token_times.append(time_ms)
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

    # --- Results ---
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — WAN/4G Benchmark")
    print(f"{'=' * 60}")
    print(f"DDNS:       {ddns_host or 'N/A'}")
    print(f"Port:       {_PORT}")
    print(f"Prompt:     {prompt!r}")
    if result.get("text"):
        print(f"WAN output: {result['text']!r}")
    print(f"Local ref:  {local_text!r}")
    print()

    # WAN-specific metrics
    wan_tps = result.get("tok_per_s", 0)
    total_ms = result.get("total_ms", wall_time * 1000)
    ttft = token_times[0] if token_times else 0
    avg_itl = (sum(token_times[1:]) / len(token_times[1:])
               if len(token_times) > 1 else 0)
    p99_itl = (sorted(token_times[1:])[int(len(token_times[1:]) * 0.99)]
               if len(token_times) > 1 else 0)

    print(f"WAN decode:   {result.get('total_tokens', 0)} tokens, "
          f"{total_ms:.0f}ms total, {wan_tps:.1f} tok/s")
    print(f"Local ref:    {local_new_toks} tokens, "
          f"{local_time * 1000:.0f}ms total, {local_tps:.1f} tok/s")
    print()
    print(f"WAN RTT:      {list(rtt.values())[0]:.0f}ms" if rtt else "WAN RTT: N/A")
    print(f"WAN TTFT:     {ttft:.0f}ms")
    print(f"WAN avg ITL:  {avg_itl:.1f}ms")
    print(f"WAN p99 ITL:  {p99_itl:.1f}ms")
    print(f"Upload:       {total_mb:.0f} MB in {upload_time:.1f}s ({bw:.0f} MB/s)")

    # Token match
    webnn_tokens = result.get("tokens", [])
    local_new_only = local_tokens[input_ids.shape[1]:]
    match_len = min(len(webnn_tokens), len(local_new_only))
    matches = sum(1 for a, b in zip(webnn_tokens[:match_len],
                                     local_new_only[:match_len]) if a == b)
    print(f"\nToken match:  {matches}/{match_len} "
          f"({'PASS' if matches == match_len else 'MISMATCH'})")

    # Save results
    bench_data = {
        "benchmark": "wan_4g",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ddns": ddns_host,
        "port": _PORT,
        "model": model_name,
        "prompt": prompt,
        "wan_tok_per_s": wan_tps,
        "wan_total_ms": total_ms,
        "wan_ttft_ms": ttft,
        "wan_avg_itl_ms": round(avg_itl, 1),
        "wan_p99_itl_ms": round(p99_itl, 1),
        "wan_rtt_ms": list(rtt.values())[0] if rtt else None,
        "upload_mb": round(total_mb, 1),
        "upload_seconds": round(upload_time, 1),
        "upload_mbps": round(bw, 1),
        "local_tok_per_s": round(local_tps, 1),
        "token_match": f"{matches}/{match_len}",
        "tokens_generated": result.get("total_tokens", 0),
    }

    out_file = f"bench_wan_4g_{ddns_host.replace('.', '_') or 'local'}.json"
    with open(out_file, "w") as f:
        json.dump(bench_data, f, indent=2)
    print(f"\nSaved: {out_file}")

    backend.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
