"""Benchmark hiérarchique des tiers mémoire L1→L6.

Objectif : mesurer latence (µs), bande passante (GB/s) et coût d'accès effectif
pour orienter la répartition initiale et la promotion/demotion dynamique
des blocs (poids de modèles ou caches intermédiaires).

Méthodologie minimale :
 - L1 : accès tensor en VRAM GPU primaire
 - L2 : accès VRAM GPU secondaire (via torch.cuda.memcpy_peer si dispo)
 - L3 : accès RAM host (tensor CPU)
 - L4 : stub (future RAM distante RDMA / fibre)
 - L5 : NVMe (lecture binaire séquentielle bloc simulé)
 - L6 : stub stockage objet (latence simulée configurable)

Les résultats sont renvoyés sous forme de dict et peuvent être exportés vers
Prometheus (labels tier) ou vers un plan de placement initial.
"""
from __future__ import annotations
import os
import time
import mmap
import random
import statistics as stats
from pathlib import Path
from typing import Dict, Any

import torch

from core.logger import LoggerAdapter

class MemoryTierBenchmark:
    def __init__(self, tmp_dir: str = ".hm_cache", sample_mb: int = 8, repeats: int = 5):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)
        self.sample_mb = sample_mb
        self.repeats = repeats
        self.log = LoggerAdapter("bench")

    def _timeit(self, fn, label: str):
        lat = []
        for _ in range(self.repeats):
            t0 = time.perf_counter_ns()
            fn()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = (time.perf_counter_ns() - t0) / 1000.0  # µs
            lat.append(dt)
        return {
            "latency_us_p50": stats.median(lat),
            "latency_us_p95": sorted(lat)[int(len(lat)*0.95)-1],
            "latency_us_min": min(lat),
            "latency_us_max": max(lat),
            "samples": len(lat)
        }

    def run(self) -> Dict[str, Any]:
        size = self.sample_mb * 1024 * 1024 // 4  # float32 elements
        results: Dict[str, Any] = {}

        # L1 VRAM primaire
        if torch.cuda.is_available():
            t_primary = torch.randn(size, device="cuda:0")
            def read_l1():
                _ = t_primary[::1024].sum().item()
            results["L1"] = self._timeit(read_l1, "L1") | {"bandwidth_gbps": (self.sample_mb/ (results.get("L1", {}).get("latency_us_p50",1)/1e6)) / 1024}
        else:
            results["L1"] = {"error": "CUDA non disponible"}

        # L2 VRAM secondaire (peer)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            t_secondary = torch.randn(size, device="cuda:1")
            def read_l2():
                _ = t_secondary[::1024].sum().item()
            results["L2"] = self._timeit(read_l2, "L2")
        else:
            results["L2"] = {"info": "GPU secondaire absent"}

        # L3 RAM host
        t_host = torch.randn(size, device="cpu")
        def read_l3():
            _ = t_host[::1024].sum().item()
        results["L3"] = self._timeit(read_l3, "L3")

        # L5 NVMe (simulate lecture bloc binaire)
        nvme_path = self.tmp_dir / "sample_block.bin"
        if not nvme_path.exists():
            with nvme_path.open("wb") as f:
                f.write(os.urandom(self.sample_mb * 1024 * 1024))
        def read_l5():
            with nvme_path.open("rb") as f:
                _ = f.read(4096)
        results["L5"] = self._timeit(read_l5, "L5")

        # L4 / L6 stubs (latence synthétique)
        def synthetic(lat_us: int):
            def _r():
                # Busy-wait approximative pour simuler latence basse (L4) vs haute (L6)
                target = time.perf_counter_ns() + lat_us*1000
                while time.perf_counter_ns() < target:
                    pass
            return _r
        results["L4"] = self._timeit(synthetic(80), "L4")  # futur RDMA/fibre
        results["L6"] = self._timeit(synthetic(800), "L6") # stockage objet distant

        # Classement (tiers par p50 croissant)
        sortable = [ (tier, meta.get("latency_us_p50", 1e12)) for tier, meta in results.items() if "latency_us_p50" in meta ]
        order = [t for t,_ in sorted(sortable, key=lambda x: x[1])]
        results["ranking"] = order
        return results

def benchmark_and_rank()-> Dict[str, Any]:
    bench = MemoryTierBenchmark()
    return bench.run()

__all__ = ["MemoryTierBenchmark", "benchmark_and_rank"]
