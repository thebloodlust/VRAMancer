"""Fibre / USB4 / RDMA fast‑path abstrait.

Objectif : fournir une interface unifiée pour un transport sans surcouche
TCP/IP (ou en contournement) avec latence minimale, permettant de plugger
un backend spécifique (driver kernel, C-extension, RDMA verbs, etc.).

Actuellement : stubs + simulation latence + autosensing des interfaces.
"""
from __future__ import annotations
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Callable
import mmap
import hashlib

from core.logger import LoggerAdapter
from core.metrics import FASTPATH_BYTES, FASTPATH_LATENCY

_BENCH_CACHE: dict[str, tuple[float, float]] = {}
_BENCH_TTL = float(os.environ.get('VRM_FASTPATH_BENCH_TTL', '30'))  # secondes

_RDMA_AVAILABLE = False
try:  # Détection légère de pyverbs ou rdma-core python
    import pyverbs  # type: ignore
    _RDMA_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        import rdma  # type: ignore
        _RDMA_AVAILABLE = True
    except Exception:
        _RDMA_AVAILABLE = False

log = LoggerAdapter("fibre")

def detect_fast_interfaces():
    candidates = []
    # USB4 mount points simulés
    for i in range(1,5):
        path = f"/mnt/usb4_share_{i}"
        if os.path.exists(path):
            candidates.append({"type":"usb4","path":path})
    # Fibre / SFP+ (stub) : on pourrait lire /sys/class/net pour détecter des noms comme "enp1s0f0"
    try:
        nets = os.listdir('/sys/class/net')
        fibre_like = [n for n in nets if 'enp' in n or 'eth' in n]
        for n in fibre_like:
            candidates.append({"type":"sfp","if":n})
    except Exception:
        pass
    if _RDMA_AVAILABLE:
        candidates.append({"type":"rdma","if":"verbs0"})
    # Interface spécifique demandée ? (ex: export VRM_FASTPATH_IF=eth1,usb4,rdma)
    prefer_if = os.environ.get('VRM_FASTPATH_IF')
    if prefer_if:
        # Remonter l'interface demandée en tête si trouvée
        for i, it in enumerate(candidates):
            if it.get('if') == prefer_if or it.get('type') == prefer_if:
                if i != 0:
                    candidates.insert(0, candidates.pop(i))
                break
    return candidates

@dataclass
class FastHandle:
    kind: str
    meta: dict
    latency_us: int = 40  # valeur par défaut simulée
    shm_path: Optional[str] = None
    _last_sent_len: int = 0
    def capabilities(self):  # simple introspection
        return {
            'kind': self.kind,
            'latency_us': self.latency_us,
            'zero_copy': self.kind in {'rdma','usb4','sfp'},
            'rdma_available': _RDMA_AVAILABLE,
        }

    def _ensure_segment(self, size: int):
        if not self.shm_path:
            name = f"/tmp/vramancer_fast_{hashlib.sha1(str(self.meta).encode()).hexdigest()[:8]}"
            self.shm_path = name
        # Crée/étend le fichier mmap
        with open(self.shm_path, 'ab') as f:
            if f.tell() < size:
                f.truncate(size)

    def send(self, data: bytes) -> int:
        size = len(data)
        start = time.perf_counter()
        self._ensure_segment(size)
        with open(self.shm_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            mm.seek(0)
            mm.write(data)
            mm.flush()
            mm.close()
        self._last_sent_len = size
        FASTPATH_BYTES.labels(self.kind, "send").inc(size)
        FASTPATH_LATENCY.labels(self.kind, "send").observe(time.perf_counter()-start)
        return size

    def recv(self) -> Optional[bytes]:
        if not self.shm_path or self._last_sent_len == 0:
            return None
        start = time.perf_counter()
        try:
            with open(self.shm_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), self._last_sent_len, access=mmap.ACCESS_READ)
                buf = mm.read(self._last_sent_len)
                mm.close()
            FASTPATH_BYTES.labels(self.kind, "recv").inc(len(buf))
            FASTPATH_LATENCY.labels(self.kind, "recv").observe(time.perf_counter()-start)
            return buf
        except FileNotFoundError:
            return None


def open_low_latency_channel(prefer: Optional[str] = None) -> Optional[FastHandle]:
    interfaces = detect_fast_interfaces()
    if not interfaces:
        log.warning("Aucune interface fast‑path détectée (stub mode)")
        return FastHandle(kind="stub", meta={}, latency_us=120)
    if prefer:
        for it in interfaces:
            if it['type'] == prefer:
                if it['type'] == 'rdma' and _RDMA_AVAILABLE:
                    return FastHandle(kind='rdma', meta=it, latency_us=20)
                return FastHandle(kind=it['type'], meta=it, latency_us=50 if it['type']=="usb4" else 70)
    it = interfaces[0]
    if it['type'] == 'rdma' and _RDMA_AVAILABLE:
        return FastHandle(kind='rdma', meta=it, latency_us=20)
    return FastHandle(kind=it['type'], meta=it, latency_us=50 if it['type']=="usb4" else 70)

__all__ = ["open_low_latency_channel", "FastHandle", "detect_fast_interfaces"]

def benchmark_interfaces(sample_size: int = 3, force: bool = False):
    now = time.time()
    results = []
    interfaces = detect_fast_interfaces()
    for it in interfaces:
        ident = it.get('if') or it.get('path') or it['type']
        cached = _BENCH_CACHE.get(ident)
        if (not force) and cached and (now - cached[0] < _BENCH_TTL):
            results.append({"interface": ident, "kind": it['type'], "latency_s": cached[1], "cached": True})
            continue
        # Simule latence
        lat = 0.0
        payload = b'x'*4096
        fh = FastHandle(kind=it['type'], meta=it)
        for _ in range(sample_size):
            start = time.perf_counter()
            fh.send(payload)
            fh.recv()
            lat += (time.perf_counter()-start)
        avg = lat / sample_size if sample_size else 0.0
        _BENCH_CACHE[ident] = (now, avg)
        results.append({"interface": ident, "kind": it['type'], "latency_s": avg, "cached": False})
    return results
