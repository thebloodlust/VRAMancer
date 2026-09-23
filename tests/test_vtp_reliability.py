#!/usr/bin/env python3
"""Fiabilité du repli TCP de VTP (core/network/llm_transport.py) — défauts du 2026-09-23.

1. deux tenseurs de la même couche arrivés avant consommation : le premier était
   écrasé (perdu sans erreur) → file FIFO + pop_received() ;
2. une connexion inactive 30 s était coupée par le serveur (le heartbeat prévu
   n'était jamais envoyé) → testé ici en réduisant le délai via un faux timeout.
Tensors CPU : aucun GPU requis.
"""
import os
import socket
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")

from core.network.llm_transport import LLMTransport, VTPServer  # noqa: E402

import core.network.llm_transport as _vtp  # noqa: E402


@pytest.fixture(autouse=True)
def _real_tcp(monkeypatch):
    """La suite tourne avec VRM_MINIMAL_TEST=1, qui met VTP en mode stub (rien n'est
    envoyé) : VTP n'était donc jamais exercé pour de vrai. Le repli TCP n'a besoin ni
    de GPU ni de NIC RDMA — on le fait tourner réellement ici, en loopback."""
    monkeypatch.setattr(_vtp, "_STUB_MODE", False)


@pytest.fixture
def pair():
    srv_t = LLMTransport(node_id="srv")
    srv = VTPServer(srv_t, host="127.0.0.1", port=0)
    port = srv.start()
    time.sleep(0.2)
    cli = LLMTransport(node_id="cli")
    assert cli.connect_peer_tcp("srv", "127.0.0.1", port)
    time.sleep(0.3)
    yield srv_t, cli, srv
    srv.stop()


def test_same_layer_tensors_are_not_overwritten(pair):
    srv_t, cli, _ = pair
    for v in (1.0, 2.0, 3.0):
        cli.send_tensor(torch.full((16,), v, dtype=torch.float16), "srv", layer_id=7)
    got = [float(srv_t.pop_received("cli", 7, timeout_s=3)[0][0]) for _ in range(3)]
    assert got == [1.0, 2.0, 3.0]                       # tous, dans l'ordre


def test_pop_received_times_out_cleanly(pair):
    srv_t, _, _ = pair
    t0 = time.time()
    assert srv_t.pop_received("cli", 12345, timeout_s=0.3) is None
    assert time.time() - t0 < 2


def test_recv_exact_distinguishes_idle_from_closed():
    """Un délai dépassé sans aucun octet = inactif (relevé), pas « fermé »."""
    a, b = socket.socketpair()
    try:
        a.settimeout(0.1)
        with pytest.raises(socket.timeout):
            VTPServer._tcp_recv_exact_static(a, 8, raise_idle_timeout=True)
        assert VTPServer._tcp_recv_exact_static(a, 8) is None   # ancien contrat conservé
        b.close()
        a.settimeout(1.0)
        assert VTPServer._tcp_recv_exact_static(a, 8, raise_idle_timeout=True) is None
    finally:
        a.close()
