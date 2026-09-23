#!/usr/bin/env python3
"""Banc d'essai du protocole AITP (experimental/aitp_protocol.py) — mesures, pas de mocks.

Vérifie quatre choses qu'on ne peut pas trancher en lisant le code :
  1. débit du Reed-Solomon (FastFEC, Python pur) en encodage et décodage ;
  2. taille maximale de tenseur transmissible, avec et sans FEC (UDP = 64 Ko/datagramme) ;
  3. récupération réelle après perte de fragments (jusqu'à `parity` pertes) ;
  4. collision : deux tenseurs successifs de la MÊME couche en vol en même temps.

Usage : python benchmarks/bench_aitp_protocol.py   (loopback IPv6, ~1 min)
"""
import os
import random
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VRM_EXPERIMENTAL", "1")
os.environ.setdefault("VRM_CLUSTER_SECRET", "bench-aitp")

from experimental.aitp_fec import FastFEC            # noqa: E402
from experimental.aitp_protocol import AITPProtocol  # noqa: E402


def section(t):
    print(f"\n=== {t} ===")


# ── 1. Débit du codeur Reed-Solomon ──────────────────────────────────────────
section("1. Débit Reed-Solomon (FastFEC 10+2, Python pur)")
fec = FastFEC(data_shards=10, parity_shards=2)
for size in (10 * 1024, 100 * 1024, 1024 * 1024):
    data = os.urandom(size)
    t0 = time.perf_counter()
    shards = fec.encode(data)
    t_enc = time.perf_counter() - t0
    avail = {i: s for i, s in enumerate(shards)}
    for lost in (0, 3):                       # retire 2 fragments de données
        avail.pop(lost, None)
    t0 = time.perf_counter()
    out = fec.decode(avail, size)
    t_dec = time.perf_counter() - t0
    ok = out == data
    print(f"  {size // 1024:>5} Ko : encode {size / t_enc / 1e6:8.2f} Mo/s · "
          f"decode (2 pertes) {size / t_dec / 1e6:8.2f} Mo/s · intègre={ok}")

# ── 2. Taille maximale transmissible ─────────────────────────────────────────
section("2. Taille max d'un tenseur (loopback IPv6)")
PORT = 47001
rx = AITPProtocol(port=PORT)
got = []
th = threading.Thread(target=rx.recv_loop, kwargs={"callback": lambda *a: got.append(a)},
                      daemon=True)
th.start()
tx = AITPProtocol(port=PORT + 1)


def try_send(proto, size, fec_on):
    try:
        proto.send_anycast("::1", 1, os.urandom(size))
        return "envoyé"
    except OSError as e:                    # v1 : erreur brute du noyau
        return f"ÉCHEC ({e.strerror or e})"
    except ValueError:                      # v2 : refus explicite avant l'envoi
        return "REFUSÉ (message explicite : trop gros pour UDP, utiliser TCP)"


for size in (8 * 1024, 60 * 1024, 70 * 1024, 640 * 1024, 700 * 1024, 5 * 1024 * 1024):
    print(f"  sans FEC {size // 1024:>5} Ko : {try_send(tx, size, False)}")
tx.enable_fec(10, 2)
for size in (60 * 1024, 640 * 1024, 700 * 1024, 5 * 1024 * 1024):
    print(f"  avec FEC {size // 1024:>5} Ko : {try_send(tx, size, True)}")
activation = 512 * 5120 * 2
print(f"  (repère : activation de prefill 512 tokens × 5120 × fp16 = {activation // 1024} Ko)")
rx.stop_recv()
time.sleep(1.2)

# ── 3. Récupération après pertes (réseau réel, fragments supprimés à l'envoi) ─
section("3. Pertes de fragments : FEC 10+2 récupère-t-il vraiment ?")
PORT = 47011
rx = AITPProtocol(port=PORT)
rx.enable_fec(10, 2)
received = {}
deliveries = []


def _on_tensor(lid, d, f, a):
    received[lid] = d
    deliveries.append((lid, d))


th = threading.Thread(target=rx.recv_loop, kwargs={"callback": _on_tensor}, daemon=True)
th.start()
tx = AITPProtocol(port=PORT + 1)
tx.enable_fec(10, 2)
# send_anycast() vise TOUJOURS self.port : le protocole suppose que tous les nœuds
# écoutent sur le même port (vrai entre machines). Sur une seule machine, on fait
# donc viser au sender le port du récepteur, comme s'il était un nœud distant.
tx.port = PORT
real_sock = tx.sock


class _LossySock:
    """Proxy de socket : laisse passer tout sauf les fragments désignés comme perdus."""

    def __init__(self, sock):
        self._s = sock
        self.drop = set()
        self.i = 0

    def sendto(self, pkt, addr):
        i = self.i
        self.i += 1
        return len(pkt) if i in self.drop else self._s.sendto(pkt, addr)

    def __getattr__(self, name):
        return getattr(self._s, name)


lossy_sock = _LossySock(real_sock)
tx.sock = lossy_sock
for n_lost in (0, 1, 2, 3):
    payload = os.urandom(50 * 1024)
    lossy_sock.drop = set(random.sample(range(12), n_lost))
    lossy_sock.i = 0
    lid = 100 + n_lost
    tx.send_anycast("::1", lid, payload)
    time.sleep(0.4)
    got_d = received.get(lid)
    verdict = ("reçu intact" if got_d == payload else
               "reçu CORROMPU" if got_d is not None else "non reçu")
    print(f"  {n_lost} fragment(s) perdu(s) sur 12 : {verdict}")
tx.sock = real_sock

# ── 4. Collision : deux tenseurs de la même couche en vol ────────────────────
section("4. Collision : 2 tenseurs successifs de la MÊME couche")
received.clear()
got_all = []
rx_cb_orig = None


class _CaptureSock:
    """Capture les paquets émis au lieu de les envoyer, pour les entrelacer ensuite."""

    def __init__(self, sock):
        self._s = sock
        self.captured = []

    def sendto(self, pkt, addr):
        self.captured.append((pkt, addr))
        return len(pkt)

    def __getattr__(self, name):
        return getattr(self._s, name)


a, b = os.urandom(50 * 1024), os.urandom(50 * 1024)
cap_a, cap_b = _CaptureSock(real_sock), _CaptureSock(real_sock)
tx.sock = cap_a
tx.send_anycast("::1", 7, a)          # vrai chemin d'envoi (méta FEC de la version courante)
tx.sock = cap_b
tx.send_anycast("::1", 7, b)
tx.sock = real_sock
# entrelace A0 B0 A1 B1 … comme deux envois concurrents sur le même lien
deliveries = []
th_cb = rx  # le récepteur de la section 3 tourne toujours
received_list = []
deliveries.clear()
for (pa, ad), (pb, _) in zip(cap_a.captured, cap_b.captured):
    real_sock.sendto(pa, ad)
    real_sock.sendto(pb, ad)
time.sleep(0.8)
got7 = [d for lid, d in deliveries if lid == 7]
labels = ["A intact" if d == a else "B intact" if d == b else "CORROMPU" for d in got7]
print(f"  tenseurs livrés pour la couche 7 : {labels or ['aucun']}")
verdict = ("OK — les deux tenseurs sont livrés intacts" if sorted(labels) == ["A intact", "B intact"]
           else "DÉFAUT — données mélangées ou perdues")
print(f"  → {verdict}")
rx.stop_recv()
