#!/usr/bin/env python3
"""Banc VTP (core/network/llm_transport.py) sur son chemin RÉEL entre machines : le repli TCP.

Sans NIC RDMA (Mellanox + nvidia_peermem), c'est ce chemin qui servirait en multi-nœud.
Mesure : 1) intégrité bit à bit et débit GPU → TCP loopback → GPU par taille ;
2) connexion inactive 35 s ; 3) deux tenseurs de la même couche avant consommation.
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.ERROR)
import torch  # noqa: E402

from core.network.llm_transport import LLMTransport, VTPServer  # noqa: E402

srv_t = LLMTransport(node_id="srv")
srv = VTPServer(srv_t, host="127.0.0.1", port=0)
port = srv.start()
time.sleep(0.3)
cli = LLMTransport(node_id="cli")
assert cli.connect_peer_tcp("srv", "127.0.0.1", port), "connexion TCP impossible"
time.sleep(0.5)


def pop(layer, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        q = getattr(srv_t, "_recv_queue", {})
        if ("cli", layer) in q:
            return q.pop(("cli", layer))
        time.sleep(0.0002)
    return None


print("=== 1. intégrité et débit (fp16, cuda:0 → TCP loopback → cuda:0) ===")
layer = 0
for n_elem in (4 * 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024):
    t = torch.randn(n_elem, dtype=torch.float16, device="cuda:0")
    times, ok = [], True
    for _ in range(3):
        layer += 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = cli.send_tensor(t, "srv", dst_gpu=0, layer_id=layer)
        rec = pop(layer)
        dt = time.perf_counter() - t0
        if rec is None or r.get("method") == "tcp_failed":
            ok = False
            break
        ok = ok and torch.equal(rec[0].to("cuda:0").to(torch.float16), t)
        times.append(dt)
    if times:
        best = min(times)
        print(f"  {n_elem * 2 / 1e6:8.2f} Mo : {n_elem * 2 / best / 1e9:6.2f} Go/s · "
              f"aller complet {best * 1000:8.2f} ms · intègre={ok}")
    else:
        print(f"  {n_elem * 2 / 1e6:8.2f} Mo : ÉCHEC")

print("\n=== 2. connexion inactive 35 s ===")
time.sleep(35)
r = cli.send_tensor(torch.ones(8, dtype=torch.float16, device="cuda:0"), "srv", dst_gpu=0, layer_id=5000)
print("  envoi après 35 s :", "reçu" if pop(5000, 10) is not None else "PERDU",
      "(reconnecté)" if r.get("reconnected") else "(même connexion)")

print("\n=== 3. deux tenseurs de la MÊME couche, envoyés avant consommation ===")
a = torch.full((1024,), 1.0, dtype=torch.float16, device="cuda:0")
b = torch.full((1024,), 2.0, dtype=torch.float16, device="cuda:0")
before = srv_t._stats.get("tensors_recv", 0)
cli.send_tensor(a, "srv", dst_gpu=0, layer_id=999)
cli.send_tensor(b, "srv", dst_gpu=0, layer_id=999)
time.sleep(1.0)
arrived = srv_t._stats.get("tensors_recv", 0) - before
got = []
for _ in range(2):
    item = srv_t.pop_received("cli", 999, timeout_s=2.0) if hasattr(srv_t, "pop_received") else None
    if item is not None:
        got.append(float(item[0][0]))
print(f"  arrivés sur le fil : {arrived} · récupérés via pop_received() : {len(got)} → valeurs {got}")
print("  →", "OK : les deux tenseurs, dans l'ordre d'envoi" if got == [1.0, 2.0] else
      "DÉFAUT : tenseur perdu ou désordonné")
srv.stop() if hasattr(srv, "stop") else None
