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

    def send(self, data: bytes) -> int:
        # Simulation latence
        target = time.perf_counter_ns() + self.latency_us * 1000
        while time.perf_counter_ns() < target:
            pass
        return len(data)

    def recv(self) -> Optional[bytes]:
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
