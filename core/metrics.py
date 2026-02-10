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
MEMORY_EVICTIONS = Counter("vramancer_memory_evictions_total", "Evictions planifiées (hotness)", ["from","to"])
FASTPATH_BYTES    = Counter("vramancer_fastpath_bytes_total", "Octets transférés fastpath", ["method","direction"])  # direction=send|recv
FASTPATH_LATENCY  = Histogram("vramancer_fastpath_latency_seconds", "Latence opérations fastpath", ["method","op"])  # op=send|recv
GPU_TRANSFER_OPS  = Counter("vramancer_gpu_transfer_ops_total", "Transferts GPU-to-GPU", ["method","src","dst"])  # method=p2p|nccl|cpu_staged
GPU_TRANSFER_BW   = Histogram("vramancer_gpu_transfer_bandwidth_gbps", "Bande passante transferts GPU (Gbps)", ["method"])
TASKS_SUBMITTED   = Counter("vramancer_tasks_submitted_total", "Tâches soumises")
TASKS_COMPLETED   = Counter("vramancer_tasks_completed_total", "Tâches complétées")
TASKS_FAILED      = Counter("vramancer_tasks_failed_total", "Tâches en erreur")
TASKS_RUNNING     = Gauge("vramancer_tasks_running", "Tâches en cours d'exécution")
TASKS_PER_RESOURCE= Gauge("vramancer_tasks_resource_running", "Tâches en cours par ressource", ["resource"])  # resource ex: cuda:0, mps:0, cpu:0
TELEMETRY_PACKETS = Counter("vramancer_telemetry_packets_total", "Paquets de télémétrie servis / ingérés", ["direction"])  # direction=out|in
DEVICE_INFO       = Gauge("vramancer_device_info", "Informations device (valeur=1)", ["backend","name","index"])  # always set to 1
TASK_DURATION     = Histogram("vramancer_task_duration_seconds", "Durée des tâches (s)", ["priority","status"])  # status=completed|failed|cancelled
BLOCK_HOTNESS     = Gauge("vramancer_block_hotness", "Score d'activité (hotness) des blocs", ["block","tier"])  # score hybride LRU/LFU
TASK_PCT          = Gauge("vramancer_task_duration_percentile", "Percentiles durées tâches (s)", ["priority","status","percentile"])  # p50/p95/p99
API_LATENCY       = Histogram("vramancer_api_latency_seconds", "Latence endpoints API", ["path","method","status"])
FASTPATH_IF_LATENCY = Gauge("vramancer_fastpath_interface_latency_seconds", "Latence benchmark fastpath interface", ["interface","kind"])
HA_JOURNAL_ROTATIONS = Counter("vramancer_ha_journal_rotations_total", "Rotations du journal HA")
HA_JOURNAL_SIZE = Gauge("vramancer_ha_journal_size_bytes", "Taille actuelle journal HA")
ORCH_PLACEMENTS = Counter("vramancer_orch_placements_total", "Placements de blocs orchestrateur", ["level"])
ORCH_MIGRATIONS = Counter("vramancer_orch_migrations_total", "Migrations inter-GPU orchestrateur")
ORCH_REBALANCE  = Counter("vramancer_orch_rebalance_total", "Cycles de rééquilibrage")
ORCH_HIERARCHY_MOVE = Counter("vramancer_orch_hierarchy_moves_total", "Migrations hiérarchie mémoire", ["to_level"])  # dram|nvme|network
ENV_ENDPOINT_HITS = Counter("vramancer_env_endpoint_hits_total", "Accès à /api/env (diagnostic)")

# VTP (LLM Transport Protocol) metrics
VTP_TENSORS_SENT = Counter("vramancer_vtp_tensors_sent_total", "Tensors envoyés via VTP", ["tier"])
VTP_TENSORS_RECV = Counter("vramancer_vtp_tensors_recv_total", "Tensors reçus via VTP", ["tier"])
VTP_BYTES_SENT = Counter("vramancer_vtp_bytes_sent_total", "Octets envoyés via VTP", ["tier"])
VTP_KV_CACHE_OPS = Counter("vramancer_vtp_kv_cache_ops_total", "Opérations KV cache VTP", ["direction"])
VTP_GPUDIRECT_OPS = Counter("vramancer_vtp_gpudirect_ops_total", "Transferts GPUDirect RDMA réels")

# VRAM Lending Pool metrics
LENDING_BORROWS = Counter("vramancer_lending_borrows_total", "VRAM borrows from lending pool", ["borrower", "lender"])
LENDING_RECLAIMS = Counter("vramancer_lending_reclaims_total", "VRAM reclaims from lending pool", ["urgency"])
LENDING_ACTIVE_LEASES = Gauge("vramancer_lending_active_leases", "Active VRAM lending leases")
LENDING_BYTES_LENT = Gauge("vramancer_lending_bytes_lent_total", "Total bytes currently lent across GPUs")
LENDING_POOL_CAPACITY_GB = Gauge("vramancer_lending_pool_capacity_gb", "Total lendable VRAM in pool (GB)")

# Continuous Batcher metrics
BATCHER_BATCH_SIZE = Histogram("vramancer_batcher_batch_size", "Requests per batch")
BATCHER_QUEUE_DEPTH = Gauge("vramancer_batcher_queue_depth", "Pending requests in batcher queue")
BATCHER_THROUGHPUT = Gauge("vramancer_batcher_throughput_tok_s", "Batcher throughput (tokens/s)")

# PagedKV Cache metrics
PAGED_KV_USED_PAGES = Gauge("vramancer_paged_kv_used_pages", "Used KV cache pages", ["device"])
PAGED_KV_FREE_PAGES = Gauge("vramancer_paged_kv_free_pages", "Free KV cache pages", ["device"])
PAGED_KV_BORROWED_PAGES = Gauge("vramancer_paged_kv_borrowed_pages", "Borrowed overflow pages from lending pool")

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
    "MEMORY_EVICTIONS",
    "FASTPATH_BYTES",
    "FASTPATH_LATENCY",
    "GPU_TRANSFER_OPS",
    "GPU_TRANSFER_BW",
    "TASKS_SUBMITTED",
    "TASKS_COMPLETED",
    "TASKS_FAILED",
    "TASKS_RUNNING",
    "TASKS_PER_RESOURCE",
    "TELEMETRY_PACKETS",
    "DEVICE_INFO",
    "BLOCK_HOTNESS",
    "TASK_PCT",
    "metrics_server_start",
    "ORCH_PLACEMENTS",
    "ORCH_MIGRATIONS",
    "ORCH_REBALANCE",
    "ORCH_HIERARCHY_MOVE",
    "ENV_ENDPOINT_HITS",
    "VTP_TENSORS_SENT",
    "VTP_TENSORS_RECV",
    "VTP_BYTES_SENT",
    "VTP_KV_CACHE_OPS",
    "VTP_GPUDIRECT_OPS",
    "LENDING_BORROWS",
    "LENDING_RECLAIMS",
    "LENDING_ACTIVE_LEASES",
    "LENDING_BYTES_LENT",
    "LENDING_POOL_CAPACITY_GB",
    "BATCHER_BATCH_SIZE",
    "BATCHER_QUEUE_DEPTH",
    "BATCHER_THROUGHPUT",
    "PAGED_KV_USED_PAGES",
    "PAGED_KV_FREE_PAGES",
    "PAGED_KV_BORROWED_PAGES",
]

def counter_value(counter) -> float:
    """Retourne la valeur totale d'un Counter prometheus client.
    Compatible avec tests sans accéder à attribut privé interne (API stable).
    """
    try:  # prometheus_client 0.19+
        samples = counter.collect()[0].samples
        # Cherche sample sans labels suffix _total
        for s in samples:
            if s.name.endswith('_total') and not s.labels:
                return float(s.value)
        # fallback: premier sample
        if samples:
            return float(samples[0].value)
    except Exception:
        pass
    return 0.0


def publish_device_info(devices):
    """Publie les devices sous forme de Gauge=1 (idempotent)."""
    for d in devices:
        DEVICE_INFO.labels(d.get('backend'), d.get('name'), str(d.get('index'))).set(1)

def publish_task_percentiles(percentiles: dict):
    for key, stats in percentiles.items():
        # key format: priority_status
        try:
            prio, status = key.split('_',1)
        except ValueError:
            continue
        for pct_label in ["p50","p95","p99"]:
            if pct_label in stats:
                TASK_PCT.labels(prio, status, pct_label).set(stats[pct_label])

