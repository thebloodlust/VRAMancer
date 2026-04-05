#!/usr/bin/env python3
"""VRAMancer WebGPU Demo — End-to-end matmul via browser (or headless worker).

This demo:
1. Starts the WebGPU backend (WebSocket server + HTTP UI server)
2. Optionally spawns a headless Python worker that emulates the browser protocol
3. Dispatches real matmul operations (including model weight slices)
4. Validates correctness against PyTorch reference
5. Measures throughput

Usage:
    # With headless worker (no browser needed):
    python examples/demo_webgpu_matmul.py --headless

    # With real browser:
    python examples/demo_webgpu_matmul.py
    # Then open http://localhost:8766/index.html in Chrome/Edge
"""

import argparse
import asyncio
import struct
import sys
import time
import threading
import numpy as np

sys.path.insert(0, ".")

OP_MATMUL = 0x01
OP_PING = 0x02
OP_SHUTDOWN = 0xFF


# ── Headless Worker (emulates browser WebGPU protocol in Python) ─────

class HeadlessWorker:
    """Python WebSocket client that speaks the same binary protocol as
    dashboard/worker/worker.js. Computes matmuls with numpy."""

    def __init__(self, ws_url: str = "ws://localhost:8765"):
        self.ws_url = ws_url
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="headless-worker"
        )
        self._thread.start()

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self):
        import websockets
        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    max_size=256 * 1024 * 1024,
                    ping_interval=None,
                    write_limit=2 ** 22,
                ) as ws:
                    print(f"[headless] Connected to {self.ws_url}")
                    await self._serve(ws)
            except Exception as e:
                if self._running:
                    print(f"[headless] Connection error: {e}, retrying...")
                    await asyncio.sleep(1.0)

    async def _serve(self, ws):
        while self._running:
            try:
                msg = await ws.recv()
            except Exception:
                break

            if isinstance(msg, str):
                msg = msg.encode()

            op = msg[0]

            if op == OP_MATMUL:
                M = struct.unpack_from("<I", msg, 1)[0]
                N = struct.unpack_from("<I", msg, 5)[0]
                K = struct.unpack_from("<I", msg, 9)[0]

                header_size = 13
                a_size = M * K * 4
                b_size = N * K * 4

                A = np.frombuffer(
                    msg[header_size:header_size + a_size], dtype=np.float32
                ).reshape(M, K)
                B = np.frombuffer(
                    msg[header_size + a_size:header_size + a_size + b_size],
                    dtype=np.float32,
                ).reshape(N, K)

                # C = A @ B^T (same as WGSL shader)
                C = A @ B.T

                resp_header = struct.pack("<BII", OP_MATMUL, M, N)
                await ws.send(resp_header + C.tobytes())

            elif op == OP_PING:
                await ws.send(struct.pack("B", OP_PING))

            elif op == OP_SHUTDOWN:
                print("[headless] Shutdown received")
                self._running = False
                break

    def stop(self):
        self._running = False


# ── Demo ─────────────────────────────────────────────────────────────

def demo_basic_matmul(backend):
    """Test basic matmul correctness."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic matmul correctness")
    print("=" * 60)

    sizes = [
        (4, 8, 4, "tiny"),
        (32, 64, 32, "small"),
        (128, 256, 128, "medium"),
        (512, 1024, 512, "large"),
    ]

    for M, N, K, label in sizes:
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)

        # Reference: A @ B^T
        ref = A @ B.T

        t0 = time.monotonic()
        result = backend.matmul(A, B)
        dt = (time.monotonic() - t0) * 1000

        max_err = np.max(np.abs(result - ref))
        rel_err = max_err / (np.max(np.abs(ref)) + 1e-10)
        ok = rel_err < 1e-5

        gflops = 2.0 * M * N * K / 1e9
        print(
            f"  [{label:>6}] {M}x{K} @ {N}x{K}^T → {M}x{N}  "
            f"err={rel_err:.2e}  {dt:.1f}ms  "
            f"{gflops / (dt / 1000):.1f} GFLOPS  "
            f"{'OK' if ok else 'FAIL'}"
        )
        if not ok:
            print(f"    WARNING: max_err={max_err:.6f}")


def demo_torch_matmul(backend):
    """Test torch tensor round-trip."""
    print("\n" + "=" * 60)
    print("TEST 2: PyTorch tensor round-trip")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("  SKIP: torch not available")
        return

    M, N, K = 256, 512, 256
    torch.manual_seed(42)
    A = torch.randn(M, K)
    B = torch.randn(N, K)

    ref = A @ B.T

    t0 = time.monotonic()
    result = backend.matmul(A, B)
    dt = (time.monotonic() - t0) * 1000

    max_err = (result - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-10)
    ok = rel_err < 1e-5

    print(
        f"  torch [{M}x{K} @ {N}x{K}^T]  "
        f"type={result.dtype}  err={rel_err:.2e}  {dt:.1f}ms  "
        f"{'OK' if ok else 'FAIL'}"
    )


def demo_model_layer(backend):
    """Dispatch a real model linear layer through WebGPU."""
    print("\n" + "=" * 60)
    print("TEST 3: Real model linear layer via WebGPU")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: torch/transformers not available")
        return

    model_name = "gpt2"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    model.eval()

    # Get the first MLP's fc weight: [intermediate, hidden]
    # GPT-2: h.0.mlp.c_fc is a Conv1D with weight [hidden, intermediate]
    # Actually GPT-2 uses Conv1D, weight shape is [hidden, 4*hidden] = [768, 3072]
    fc = model.transformer.h[0].mlp.c_fc
    W = fc.weight.data  # [768, 3072] for GPT-2 Conv1D

    # Simulate: activation [1, 768] @ weight [768, 3072] → [1, 3072]
    # Since our protocol does C = A @ B^T, and W is [768, 3072]:
    #   A = activation [1, 768], B = W^T [3072, 768]
    #   C = A @ B^T = [1, 768] @ [3072, 768]^T = [1, 768] @ [768, 3072] = [1, 3072]
    text = "Hello world"
    tokens = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        hidden = model.transformer.wte(tokens["input_ids"])  # [1, seq, 768]
    activation = hidden[0, -1:, :]  # [1, 768] last token

    # Reference: activation @ W (GPT-2 Conv1D does input @ weight)
    ref = activation @ W  # [1, 3072]

    # Via WebGPU: A=[1,768], B=W^T=[3072,768], C = A @ B^T = A @ W
    B_transposed = W.T.contiguous()  # [3072, 768]

    t0 = time.monotonic()
    result = backend.matmul(activation, B_transposed)
    dt = (time.monotonic() - t0) * 1000

    max_err = (result - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-10)
    ok = rel_err < 1e-4  # f32 matmul tolerance

    print(
        f"  GPT-2 h.0.mlp.c_fc: [1,768] @ [3072,768]^T → [1,3072]"
    )
    print(
        f"  err={rel_err:.2e}  {dt:.1f}ms  "
        f"{'OK' if ok else 'FAIL'}"
    )
    if ok:
        print(f"  First 5 values (WebGPU): {result[0,:5].tolist()}")
        print(f"  First 5 values (ref):    {ref[0,:5].tolist()}")


def demo_bandwidth(backend):
    """Measure throughput across matrix sizes."""
    print("\n" + "=" * 60)
    print("TEST 4: Bandwidth sweep")
    print("=" * 60)

    sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for M, N, K in sizes:
        np.random.seed(0)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(N, K).astype(np.float32)

        # Warmup
        backend.matmul(A, B)

        n_iter = 5
        t0 = time.monotonic()
        for _ in range(n_iter):
            backend.matmul(A, B)
        dt = (time.monotonic() - t0) / n_iter

        gflops = 2.0 * M * N * K / 1e9
        data_mb = (M * K + N * K + M * N) * 4 / 1e6
        print(
            f"  {M:>4}x{K:>4} @ {N:>4}x{K:>4}  "
            f"{dt * 1000:>7.1f}ms  "
            f"{gflops / dt:>7.1f} GFLOPS  "
            f"{data_mb / dt:>7.1f} MB/s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="VRAMancer WebGPU matmul demo"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Use Python headless worker (no browser needed)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765,
        help="WebSocket server port"
    )
    parser.add_argument(
        "--http-port", type=int, default=8766,
        help="HTTP server port for worker UI"
    )
    parser.add_argument(
        "--skip-model", action="store_true",
        help="Skip model layer test (requires ~500MB download)"
    )
    args = parser.parse_args()

    from core.webgpu_backend import WebGPUBackend

    print("=" * 60)
    print("VRAMancer WebGPU Matmul Demo")
    print("=" * 60)

    backend = WebGPUBackend(
        ws_port=args.ws_port,
        http_port=args.http_port,
        serve_ui=True,
    )

    headless = None
    if args.headless:
        print("\nStarting headless worker (numpy-based, emulates browser)...")
        headless = HeadlessWorker(f"ws://localhost:{args.ws_port}")
        headless.start()
        time.sleep(0.5)  # Let it connect
    else:
        print(
            f"\nOpen http://localhost:{args.http_port}/index.html "
            f"in Chrome/Edge and click Connect."
        )
        print("Waiting for browser worker...")

    try:
        backend.wait_for_workers(n=1, timeout=120)
    except TimeoutError as e:
        print(f"\nERROR: {e}")
        return

    # Ping
    pings = backend.ping_workers()
    for addr, rtt in pings.items():
        print(f"  Worker {addr}: RTT = {rtt} ms")

    # Run tests
    demo_basic_matmul(backend)
    demo_torch_matmul(backend)
    if not args.skip_model:
        demo_model_layer(backend)
    demo_bandwidth(backend)

    # Status
    print("\n" + "=" * 60)
    print("BACKEND STATUS")
    print("=" * 60)
    status = backend.status()
    for k, v in status.items():
        if k != "workers":
            print(f"  {k}: {v}")
    for w in status.get("workers", []):
        print(f"  worker {w['addr']}: {w['ops']} ops, {w['gflops']} GFLOPS")

    # Cleanup
    print("\nShutting down...")
    backend.shutdown()
    if headless:
        headless.stop()
    time.sleep(0.5)
    print("Done.")


if __name__ == "__main__":
    main()
