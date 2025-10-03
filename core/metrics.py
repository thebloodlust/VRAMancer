"""Métriques Prometheus pour VRAMancer.

Usage :
    from core.metrics import metrics_server_start, INFER_REQUESTS
    metrics_server_start()
    INFER_REQUESTS.inc()
"""
from __future__ import annotations
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import threading
import os

INFER_REQUESTS = Counter("vramancer_infer_total", "Nombre total de requêtes d'inférence")
INFER_ERRORS = Counter("vramancer_infer_errors_total", "Erreurs d'inférence")
INFER_LATENCY = Histogram("vramancer_infer_latency_seconds", "Latence inférence (s)")
GPU_MEMORY_USED = Gauge("vramancer_gpu_memory_used_bytes", "Mémoire GPU utilisée (bytes)", ["gpu"])
MEMORY_PROMOTIONS = Counter("vramancer_memory_promotions_total", "Promotions mémoire (tiers)", ["from","to"])
MEMORY_DEMOTIONS  = Counter("vramancer_memory_demotions_total", "Démotions mémoire (tiers)", ["from","to"])

_started = False

def metrics_server_start(port: int | None = None):
    global _started
    if _started:
        return
    p = port or int(os.environ.get("VRM_METRICS_PORT", 9108))
    # Démarrage non bloquant
    t = threading.Thread(target=start_http_server, args=(p,), daemon=True)
    t.start()
    _started = True

__all__ = [
    "INFER_REQUESTS",
    "INFER_ERRORS",
    "INFER_LATENCY",
    "GPU_MEMORY_USED",
    "MEMORY_PROMOTIONS",
    "MEMORY_DEMOTIONS",
    "metrics_server_start",
]