#!/usr/bin/env python3
"""Mesure cross-nœud : le transfert d'activation entre couches est-il viable sur réseau ?

On ne dispose que d'1 machine -> on mesure le PLANCHER logiciel (socket TCP loopback)
pour le transfert d'une activation, puis on extrapole par débit de transport (1 GbE,
10 GbE, USB4/Thunderbolt ~20 Gbps). Le vrai LAN est forcément >= loopback.

3 modes cross-nœud à juger :
  - PAR-COUCHE interleavé (design naïf) : ~N crossings/token -> mort.
  - PIPELINE-parallel contigu (A=couches 0..k, B=k+1..N) : 1 crossing/token.
  - DATA-parallel (requête entière par nœud) : 0 crossing.

Question décidée : 1 crossing/token coûte-t-il cher vs le calcul/token (~27 ms mesuré) ?

Usage: python benchmarks/bench_cross_node_transfer.py
"""
import os, sys, socket, struct, threading, time, statistics

# Profil modèle (Qwen2.5-14B : hidden=5120, 48 couches). Override via argv.
HIDDEN = int(sys.argv[1]) if len(sys.argv) > 1 else 5120
LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
SEQ = 216
DTYPE_BYTES = 2  # bf16
COMPUTE_MS_PER_TOKEN = 27.0  # mesuré (bench_disagg, 1.5B) — ordre de grandeur

DECODE_BYTES = 1 * 1 * HIDDEN * DTYPE_BYTES        # activation 1 token
PREFILL_BYTES = 1 * SEQ * HIDDEN * DTYPE_BYTES     # activation prompt


def _echo_server(port, ready):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port)); s.listen(1); ready.set()
    conn, _ = s.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        hdr = conn.recv(4)
        if not hdr: break
        n = struct.unpack("!I", hdr)[0]
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk: break
            buf += chunk
        conn.sendall(b"\x01")  # ack
    conn.close(); s.close()


def measure_rtt(payload_bytes, iters=50):
    port = 53900 + (payload_bytes % 500)
    ready = threading.Event()
    t = threading.Thread(target=_echo_server, args=(port, ready), daemon=True)
    t.start(); ready.wait()
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    c.connect(("127.0.0.1", port))
    data = os.urandom(payload_bytes)
    hdr = struct.pack("!I", payload_bytes)
    # warmup
    for _ in range(5):
        c.sendall(hdr + data); c.recv(1)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c.sendall(hdr + data); c.recv(1)
        times.append(time.perf_counter() - t0)
    c.close()
    return statistics.median(times)


def main():
    print(f"[profil] hidden={HIDDEN} couches={LAYERS} seq={SEQ} bf16", flush=True)
    print(f"  activation décode = {DECODE_BYTES/1024:.1f} KB | prefill = {PREFILL_BYTES/1e6:.2f} MB\n", flush=True)

    dec_rtt = measure_rtt(DECODE_BYTES)
    pre_rtt = measure_rtt(PREFILL_BYTES)
    # one-way ~ rtt/2 (ack 1 octet négligeable)
    dec_ow_ms = dec_rtt / 2 * 1000
    pre_ow_ms = pre_rtt / 2 * 1000
    print(f"[loopback] transfert 1-way décode {DECODE_BYTES/1024:.1f}KB : {dec_ow_ms:.3f} ms "
          f"(plancher logiciel, latency-bound)")
    print(f"[loopback] transfert 1-way prefill {PREFILL_BYTES/1e6:.2f}MB : {pre_ow_ms:.3f} ms "
          f"({PREFILL_BYTES/ (pre_ow_ms/1000) /1e9:.1f} GB/s)\n")

    # Extrapolation par transport : temps ~ plancher_latence + octets/débit
    transports = [
        ("1 GbE", 0.125e9, 0.2),       # 125 MB/s, ~0.2ms base
        ("10 GbE", 1.25e9, 0.05),
        ("USB4/Thunderbolt", 2.5e9, 0.05),  # ~20 Gbps effectif
        ("loopback (réf)", PREFILL_BYTES/(pre_ow_ms/1000), 0.0),
    ]
    print("Coût d'1 crossing (pipeline-parallel, 1/token) par transport :")
    print(f"{'transport':18s} {'décode/token':>14s} {'% du calcul':>12s}   {'prefill (1x)':>12s}")
    for name, bw, base_ms in transports:
        dec_ms = base_ms + DECODE_BYTES / bw * 1000
        pre_ms = base_ms + PREFILL_BYTES / bw * 1000
        pct = 100 * dec_ms / COMPUTE_MS_PER_TOKEN
        print(f"{name:18s} {dec_ms:>11.3f}ms {pct:>11.1f}% {pre_ms:>11.2f}ms")

    print(f"\n(calcul/token de référence ~{COMPUTE_MS_PER_TOKEN} ms ; décode autorégressif = 1 crossing/token)")
    print("\n=== VERDICT ===")
    print("• PAR-COUCHE interleavé : ~N crossings/token -> mort (réseau domine).")
    print("• PIPELINE contigu (1 crossing/token) : le TRANSFERT est petit vs le calcul")
    print("  (décode = activation 10KB, latency-bound). MAIS décode autorégressif = SÉRIEL :")
    print("  nœud B attend A -> 0 speedup single-req, juste split mémoire + latence en plus.")
    print("  Gain seulement en MULTI-requêtes (micro-batches pipeline).")
    print("• DATA-parallel (requête entière/nœud) : 0 crossing -> le plus simple, le meilleur")
    print("  débit quand le modèle tient sur 1 nœud.")
    print("• Thunderbolt/USB4 (~20 Gbps) rapproche le transfert de la vitesse locale")
    print("  (CPU-staged 11-25 GB/s) -> seul transport qui rend le cross-nœud non-stupide.")
    print("  1 GbE (0.125 GB/s) = 100x plus lent -> prefill 2MB = ~16ms, lourd.")


if __name__ == "__main__":
    main()
