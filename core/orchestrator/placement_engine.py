"""Placement Engine (Production-ready minimal)

Responsabilités:
- Décider du placement initial des blocs / tâches selon métriques (latence, pression VRAM, affinité backend)
- Exposer une interface stable pour des stratégies pluggables

API:
    engine = PlacementEngine(metrics_provider)
    decision = engine.place(block_meta) -> {level, gpu_id}
    engine.register_strategy(name, callable)

Extensible: ajouter des stratégies (cost aware, energy, multi-cloud) via register_strategy.
"""
from __future__ import annotations
from typing import Callable, Dict, Any
try:
    from core.metrics import ORCH_PLACEMENTS
except Exception:  # metrics pas initialisées
    class _Dummy:
        def labels(self,*a,**k): return self
        def inc(self,*a,**k): return None
    ORCH_PLACEMENTS = _Dummy()

class PlacementEngine:
    def __init__(self, metrics_provider=None):
        self._strategies: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._default = self._heuristic
        self.metrics_provider = metrics_provider

    def register_strategy(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self._strategies[name] = fn

    def place(self, block: Dict[str, Any], strategy: str | None = None) -> Dict[str, Any]:
        fn = self._strategies.get(strategy or "", self._default)
        decision = fn(block)
        try:
            lvl = decision.get('level','unknown')
            ORCH_PLACEMENTS.labels(lvl).inc()
        except Exception:
            pass
        return decision

    # ---- heuristique par défaut ----
    def _heuristic(self, block: Dict[str, Any]) -> Dict[str, Any]:
        size = block.get("size_mb", 128)
        # Exemple simple: petites tailles en L1 sinon L3 (host pinned) pour éviter fragmentation GPU
        level = "L1" if size <= 256 else "L3"
        return {"level": level, "gpu_id": 0}

__all__ = ["PlacementEngine"]
