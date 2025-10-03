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
    return candidates

@dataclass
class FastHandle:
    kind: str
    meta: dict
    latency_us: int = 40  # valeur par défaut simulée
    shm_path: Optional[str] = None
    _last_sent_len: int = 0

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
        self._ensure_segment(size)
        with open(self.shm_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            mm.seek(0)
            mm.write(data)
            mm.flush()
            mm.close()
        self._last_sent_len = size
        return size

    def recv(self) -> Optional[bytes]:
        if not self.shm_path or self._last_sent_len == 0:
            return None
        try:
            with open(self.shm_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), self._last_sent_len, access=mmap.ACCESS_READ)
                buf = mm.read(self._last_sent_len)
                mm.close()
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
                return FastHandle(kind=it['type'], meta=it, latency_us=50 if it['type']=="usb4" else 70)
    it = interfaces[0]
    return FastHandle(kind=it['type'], meta=it, latency_us=50 if it['type']=="usb4" else 70)

__all__ = ["open_low_latency_channel", "FastHandle", "detect_fast_interfaces"]
