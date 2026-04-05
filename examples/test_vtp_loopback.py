#!/usr/bin/env python3
"""VTP Loopback Test — validate TCP tensor transport GPU0 → GPU1.

Spins up a VTPServer + client on localhost, sends tensors of various
dtypes through TCP, and validates correctness.
"""
import os, sys, time, threading
sys.path.insert(0, "/home/jeremie/VRAMancer/VRAMancer")
os.environ["VRM_MINIMAL_TEST"] = ""  # need real torch

import torch
from core.network.llm_transport import (
    LLMTransport, VTPServer, TensorHeader, VTPOpcode,
    _DTYPE_TO_CODE, _CODE_TO_DTYPE, _HEADER_PAD,
)

def test_header_roundtrip():
    """Test TensorHeader encode/decode."""
    hdr = TensorHeader(
        opcode=VTPOpcode.TENSOR,
        flags=0,
        payload_bytes=1024,
        layer_id=12,
        seq_id=42,
        src_gpu=0,
        dst_gpu=1,
        ndim=3,
        dtype_code=1,  # float16
        shape=(2, 4, 128),
    )
    raw = hdr.encode()
    assert len(raw) == 64, f"Header size: {len(raw)}"
    dec = TensorHeader.decode(raw)
    assert dec.opcode == VTPOpcode.TENSOR
    assert dec.payload_bytes == 1024
    assert dec.layer_id == 12
    assert dec.seq_id == 42
    assert dec.src_gpu == 0
    assert dec.dst_gpu == 1
    assert dec.ndim == 3
    assert dec.dtype_code == 1
    assert dec.shape == (2, 4, 128)
    print("✓ TensorHeader roundtrip OK")

def test_tcp_loopback():
    """Test full VTP TCP loopback: server ← client tensor transfer."""
    # -- Server side --
    server_transport = LLMTransport(node_id="server-node")
    server = VTPServer(server_transport, host="0.0.0.0", port=0)  # random port
    actual_port = server.start()
    print(f"  VTP server on port {actual_port}")

    time.sleep(0.3)  # let server thread start

    # -- Client side --
    client_transport = LLMTransport(node_id="client-node")
    ok = client_transport.connect_peer_tcp("server-node", "127.0.0.1", actual_port)
    assert ok, "TCP connect failed"
    print("  Client connected, handshake done")

    time.sleep(0.5)  # let server-side handshake + recv_loop start

    # -- Test 1: float32 tensor --
    t_f32 = torch.randn(4, 8, dtype=torch.float32, device="cuda:0")
    result = client_transport.send_tensor(t_f32, "server-node", dst_gpu=1, layer_id=0)
    print(f"  Sent float32 {tuple(t_f32.shape)}: {result['method']}, "
          f"{result.get('bytes', 0)} bytes, "
          f"{result.get('bandwidth_gbps', 0):.2f} Gbps")
    time.sleep(0.3)

    # Check server received it
    if hasattr(server_transport, '_recv_queue'):
        key = ("client-node", 0)
        if key in server_transport._recv_queue:
            recv_t, recv_h = server_transport._recv_queue[key]
            print(f"  Received: shape={tuple(recv_t.shape)}, dtype={recv_t.dtype}, "
                  f"device={recv_t.device}")
            # Move both to CPU for comparison
            orig_cpu = t_f32.cpu()
            recv_cpu = recv_t.cpu().to(torch.float32)
            if torch.allclose(orig_cpu, recv_cpu, atol=1e-6):
                print("  ✓ float32 roundtrip EXACT MATCH")
            else:
                maxdiff = (orig_cpu - recv_cpu).abs().max().item()
                print(f"  ✗ float32 MISMATCH maxdiff={maxdiff}")
        else:
            print(f"  ✗ float32 not in recv_queue, keys={list(server_transport._recv_queue.keys())}")
    else:
        print("  ✗ No _recv_queue on server transport")

    # -- Test 2: float16 tensor --
    t_f16 = torch.randn(16, 128, dtype=torch.float16, device="cuda:0")
    result = client_transport.send_tensor(t_f16, "server-node", dst_gpu=1, layer_id=1)
    print(f"  Sent float16 {tuple(t_f16.shape)}: {result['method']}, {result.get('bytes', 0)} bytes")
    time.sleep(0.3)

    if hasattr(server_transport, '_recv_queue'):
        key = ("client-node", 1)
        if key in server_transport._recv_queue:
            recv_t, recv_h = server_transport._recv_queue[key]
            orig_cpu = t_f16.cpu()
            recv_cpu = recv_t.cpu().to(torch.float16)
            if torch.allclose(orig_cpu, recv_cpu, atol=1e-3):
                print("  ✓ float16 roundtrip MATCH")
            else:
                maxdiff = (orig_cpu - recv_cpu).abs().max().item()
                print(f"  ✗ float16 MISMATCH maxdiff={maxdiff}")
        else:
            print(f"  ✗ float16 not in recv_queue")

    # -- Test 3: bfloat16 tensor (known bug: numpy can't handle bf16) --
    t_bf16 = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda:0")
    result = client_transport.send_tensor(t_bf16, "server-node", dst_gpu=1, layer_id=2)
    print(f"  Sent bfloat16 {tuple(t_bf16.shape)}: {result['method']}, {result.get('bytes', 0)} bytes")
    time.sleep(0.3)

    if hasattr(server_transport, '_recv_queue'):
        key = ("client-node", 2)
        if key in server_transport._recv_queue:
            recv_t, recv_h = server_transport._recv_queue[key]
            orig_cpu = t_bf16.cpu().to(torch.float32)
            recv_cpu = recv_t.cpu().to(torch.float32)
            if torch.allclose(orig_cpu, recv_cpu, atol=1e-2):
                print("  ✓ bfloat16 roundtrip MATCH")
            else:
                maxdiff = (orig_cpu - recv_cpu).abs().max().item()
                print(f"  ✗ bfloat16 MISMATCH maxdiff={maxdiff}")
        else:
            print(f"  ✗ bfloat16 not in recv_queue")

    # -- Test 4: Large tensor (activation-sized) --
    t_large = torch.randn(1, 1, 5120, dtype=torch.float16, device="cuda:0")
    t0 = time.perf_counter()
    result = client_transport.send_tensor(t_large, "server-node", dst_gpu=1, layer_id=3)
    dt = time.perf_counter() - t0
    nbytes = t_large.nelement() * t_large.element_size()
    bw = nbytes / dt / 1e9
    print(f"  Sent large {tuple(t_large.shape)} ({nbytes/1024:.1f} KB): "
          f"{dt*1000:.2f} ms, {bw:.2f} GB/s")
    time.sleep(0.3)

    if hasattr(server_transport, '_recv_queue'):
        key = ("client-node", 3)
        if key in server_transport._recv_queue:
            recv_t, recv_h = server_transport._recv_queue[key]
            orig_cpu = t_large.cpu()
            recv_cpu = recv_t.cpu().to(torch.float16)
            if torch.allclose(orig_cpu, recv_cpu, atol=1e-3):
                print(f"  ✓ large tensor roundtrip MATCH (on {recv_t.device})")
            else:
                maxdiff = (orig_cpu - recv_cpu).abs().max().item()
                print(f"  ✗ large tensor MISMATCH maxdiff={maxdiff}")

    # -- Test 5: Really large tensor (full hidden state) --
    t_hidden = torch.randn(1, 512, 5120, dtype=torch.float16, device="cuda:0")
    t0 = time.perf_counter()
    result = client_transport.send_tensor(t_hidden, "server-node", dst_gpu=1, layer_id=4)
    dt = time.perf_counter() - t0
    nbytes = t_hidden.nelement() * t_hidden.element_size()
    bw = nbytes / dt / 1e9
    print(f"  Sent hidden-state {tuple(t_hidden.shape)} ({nbytes/1024/1024:.1f} MB): "
          f"{dt*1000:.1f} ms, {bw:.2f} GB/s")
    time.sleep(0.5)

    if hasattr(server_transport, '_recv_queue'):
        key = ("client-node", 4)
        if key in server_transport._recv_queue:
            recv_t, recv_h = server_transport._recv_queue[key]
            orig_cpu = t_hidden.cpu()
            recv_cpu = recv_t.cpu().to(torch.float16)
            if torch.allclose(orig_cpu, recv_cpu, atol=1e-3):
                print(f"  ✓ hidden-state roundtrip MATCH ({nbytes/1024/1024:.1f} MB)")
            else:
                maxdiff = (orig_cpu - recv_cpu).abs().max().item()
                print(f"  ✗ hidden-state MISMATCH maxdiff={maxdiff}")

    # -- Stats --
    print(f"\n  Client stats: {client_transport.stats()}")
    print(f"  Server stats: {server_transport.stats()}")

    # Cleanup
    server.stop()
    client_transport.close()
    server_transport.close()

def test_bandwidth_sweep():
    """Send increasing tensor sizes to measure TCP bandwidth."""
    server_transport = LLMTransport(node_id="bw-server")
    server = VTPServer(server_transport, host="0.0.0.0", port=0)
    actual_port = server.start()
    time.sleep(0.3)

    client_transport = LLMTransport(node_id="bw-client")
    ok = client_transport.connect_peer_tcp("bw-server", "127.0.0.1", actual_port)
    assert ok
    time.sleep(0.5)

    print("\n  Size (KB)    Time (ms)    BW (GB/s)")
    print("  " + "-" * 40)
    sizes = [1, 4, 16, 64, 256, 1024, 4096, 10240]  # KB
    for size_kb in sizes:
        numel = size_kb * 1024 // 2  # float16 = 2 bytes
        t = torch.randn(numel, dtype=torch.float16, device="cuda:0")
        # Warmup
        client_transport.send_tensor(t, "bw-server", layer_id=99)
        time.sleep(0.1)
        # Measure
        nbytes = t.nelement() * t.element_size()
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            client_transport.send_tensor(t, "bw-server", layer_id=99)
            times.append(time.perf_counter() - t0)
            time.sleep(0.05)
        avg_t = sum(times) / len(times)
        bw = nbytes / avg_t / 1e9
        print(f"  {size_kb:>8}    {avg_t*1000:>8.2f}    {bw:>8.2f}")

    server.stop()
    client_transport.close()
    server_transport.close()

if __name__ == "__main__":
    print("=" * 60)
    print("VTP Loopback Test")
    print("=" * 60)

    print("\n[1] TensorHeader roundtrip")
    test_header_roundtrip()

    print("\n[2] TCP loopback (GPU0 → TCP → GPU1)")
    test_tcp_loopback()

    print("\n[3] TCP bandwidth sweep")
    test_bandwidth_sweep()

    print("\n" + "=" * 60)
    print("Done")
