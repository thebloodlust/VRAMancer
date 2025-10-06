"""XAI Dashboard (Production-ready minimal stub)

Objectif: exposer une interface pour expliquer décisions / allocations mémoire / placements modèles.

Fonctions:
    - register_explainer(name, fn)
    - explain(kind, payload) -> dict

Un explainer est une fonction prenant un payload (dict) et retournant un dict {explanation:str, meta:...}.
"""
from __future__ import annotations
from typing import Callable, Dict, Any
try:
    from core.metrics import Counter as _Counter
except Exception:
    class _Counter:  # type: ignore
        def __init__(self,*a,**k): pass
        def inc(self,*a,**k): pass

XAI_REQUESTS = _Counter('vramancer_xai_requests_total', 'Requêtes XAI', ['kind'])

class XAIDashboard:
    def __init__(self):
        self._explainers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        # Enregistre un explainer par défaut "feature_attrib" (approx)
        self.register_explainer("feature_attrib", self._feature_attrib)

    def _feature_attrib(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Explainer très simplifié: calcule importance relative normalisée.
        Attend payload {features:[float...]}
        """
        feats = payload.get("features", [])
        if not feats:
            return {"explanation": "no features", "attribution": []}
        total = sum(abs(f) for f in feats) or 1.0
        attrib = [abs(f)/total for f in feats]
        return {"explanation": "relative L1 attribution", "attribution": attrib}

    def register_explainer(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self._explainers[name] = fn

    def explain(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._explainers.get(kind)
        if not fn:
            return {"error": "no_explainer", "kind": kind}
        try:
            out = fn(payload)
            XAI_REQUESTS.labels(kind).inc()
            return out
        except Exception as e:  # pragma: no cover - résilience
            return {"error": str(e), "kind": kind}

__all__ = ["XAIDashboard"]
