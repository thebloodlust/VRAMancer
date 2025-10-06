"""Gestion hiérarchique mémoire multi-niveau (L1→L6).

Niveaux proposés :
  L1 : VRAM GPU primaire (accès direct)
  L2 : VRAM GPUs secondaires (cache blocs)
  L3 : RAM système (host)
  L4 : RAM distante (autres nœuds / RDMA futur)
  L5 : Stockage rapide local (NVMe / SSD)
  L6 : Stockage réseau / objet (fallback, latent)

Ce module fournit un orchestrateur simple de placement & migration.
"""
from __future__ import annotations
import os
import time
import pickle
from pathlib import Path
from typing import Any, Literal
from core.logger import LoggerAdapter
from core.metrics import MEMORY_PROMOTIONS, MEMORY_DEMOTIONS, MEMORY_EVICTIONS
from core.tracing import get_tracer
from core.memory_block import MemoryBlock

Tier = Literal["L1","L2","L3","L4","L5","L6"]

class HierarchicalMemoryManager:
    def __init__(self, nvme_dir: str = ".hm_cache", max_nvme_mb: int = 2048, decay_half_life_s: float = 60.0):
        self.log = LoggerAdapter("hmem")
        self.nvme_dir = Path(nvme_dir)
        self.nvme_dir.mkdir(exist_ok=True)
        self.max_nvme_mb = max_nvme_mb
        # Tables de suivi
        self.registry: dict[str, dict[str, Any]] = {}
        # Hotness hybride (LRU + LFU avec décroissance exponentielle)
        self._hot_scores: dict[str, float] = {}
        self._last_touch: dict[str, float] = {}
        self._decay_half_life = decay_half_life_s
        # Autosave thread
        import threading as _thr
        if os.environ.get('VRM_AUTOSAVE_MEMORY','1') == '1':
            def _autosave():  # pragma: no cover
                while True:
                    try:
                        self.save_state()
                    except Exception:
                        pass
                    time.sleep(int(os.environ.get('VRM_AUTOSAVE_INTERVAL','30')))
            _thr.Thread(target=_autosave, daemon=True).start()

    def register_block(self, block: MemoryBlock, tier: Tier):
        self.registry[block.id] = {
            "tier": tier,
            "size_mb": block.size_mb,
            "ts": time.time(),
            "access": 0,
            "last_access": None,
            "meta": {},
        }
        self.log.debug(f"Register {block.id[:8]} @ {tier}")

    def get_tier(self, block_id: str) -> Tier | None:
        info = self.registry.get(block_id)
        return info["tier"] if info else None

    # --- Migration logique ---
    def migrate(self, block: MemoryBlock, target: Tier):
        prev = self.get_tier(block.id)
        if prev == target:
            return
        self.registry[block.id]["tier"] = target
        self.registry[block.id]["ts"] = time.time()
        # Metrics
        if prev and target:
            if self._is_promotion(prev, target):
                MEMORY_PROMOTIONS.labels(prev, target).inc()
            else:
                MEMORY_DEMOTIONS.labels(prev, target).inc()
        tracer = get_tracer()
        with tracer.start_as_current_span("memory.migrate") as span:
            span.set_attribute("block.id", block.id)
            span.set_attribute("from", prev or "?")
            span.set_attribute("to", target)
            self.log.info(f"Migration bloc {block.id[:8]} {prev} → {target}")

    def _is_promotion(self, prev: Tier, target: Tier) -> bool:
        order = ["L6","L5","L4","L3","L2","L1"]  # plus lent → plus rapide
        return order.index(target) > order.index(prev)

    # --- Accès (pour promotion) ---
    def touch(self, block: MemoryBlock):
        if block.id in self.registry:
            meta = self.registry[block.id]
            meta["access"] += 1
            meta["last_access"] = time.time()
            now = meta["last_access"]
            prev_score = self._hot_scores.get(block.id, 0.0)
            last_t = self._last_touch.get(block.id, now)
            dt = max(0.0, now - last_t)
            # Décroissance exponentielle: score *= 0.5^(dt/half_life)
            if self._decay_half_life > 0:
                decay_factor = 0.5 ** (dt / self._decay_half_life)
            else:
                decay_factor = 1.0
            score = prev_score * decay_factor + 1.0  # ajout d'un événement d'accès
            self._hot_scores[block.id] = score
            self._last_touch[block.id] = now
            # Mettre à jour Gauge (lazy import pour éviter cycle)
            try:
                from core.metrics import BLOCK_HOTNESS
                BLOCK_HOTNESS.labels(block.id[:8], self.registry[block.id]["tier"]).set(score)
            except Exception:
                pass

    def promote_policy(self, block: MemoryBlock):
        meta = self.registry.get(block.id)
        if not meta:
            return
        tier = meta["tier"]
        acc = meta["access"]
        score = self._hot_scores.get(block.id, 0.0)
        # Règle heuristique : si un bloc stocké hors VRAM (>=L3) est accédé
        # plus de X fois dans une fenêtre, on le remonte progressivement.
        # Ajoute dimension score (pénalise ancienneté, favorise accès récents).
        if tier in {"L5","L4"} and (acc >= 3 or score > 3):
            self.migrate(block, "L3")
        elif tier == "L3" and (acc >= 5 or score > 6):
            self.migrate(block, "L2")
        elif tier == "L2" and (acc >= 8 or score > 10):
            self.migrate(block, "L1")
        # Reset partiel pour éviter promotions infinies trop rapides
        if acc >= 8:
            meta["access"] = 0

    # --- NVMe spill (L5) ---
    def spill_to_nvme(self, block: MemoryBlock, payload: Any):
        path = self.nvme_dir / f"{block.id}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        self.migrate(block, "L5")
        self.log.debug(f"Spill bloc {block.id[:8]} vers NVMe ({path})")

    def load_from_nvme(self, block: MemoryBlock) -> Any | None:
        path = self.nvme_dir / f"{block.id}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as f:
            data = pickle.load(f)
        self.migrate(block, "L3")  # retour RAM
        self.log.debug(f"Reload bloc {block.id[:8]} depuis NVMe")
        return data

    # --- Politique simple de tiering ---
    def policy_demote_if_needed(self, block: MemoryBlock, gpu_over_pct: float):
        tier = self.get_tier(block.id)
        if tier == "L1" and gpu_over_pct > 90:
            # Descendre en VRAM secondaire (simulation)
            self.migrate(block, "L2")
        elif tier in {"L2","L3"} and gpu_over_pct > 95:
            # Spill NVMe
            self.spill_to_nvme(block, {"dummy":"weights"})

    # --- Eviction planner (lot B) ---
    def eviction_cycle(self, target_free_pct: float = 10.0, vram_pressure: float | None = None):
        """Applique une politique d'éviction basée sur le hotness.
        Objectif: libérer de la VRAM L1/L2 quand la pression est trop forte.
        Heuristique simple:
          - Calcule un score relatif (hotness) pour chaque bloc en L1/L2
          - Trie ascendant et déplace les X% plus froids vers un niveau inférieur
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("memory.eviction_cycle"):
            l12 = [bid for bid, meta in self.registry.items() if meta['tier'] in {'L1','L2'}]
            if not l12:
                return []
            scores = []
            for bid in l12:
                scores.append((self._hot_scores.get(bid, 0.0), bid))
            scores.sort(key=lambda x: x[0])  # froid → chaud
            # Ajuste le pourcentage si pression VRAM forte
            ratio = 0.2
            if vram_pressure and vram_pressure > 0.9:  # >90% utilisé
                ratio = 0.4
            elif vram_pressure and vram_pressure > 0.8:
                ratio = 0.3
            k = max(1, int(len(scores)*ratio))
            evicted = []
            for _, bid in scores[:k]:
                tier = self.registry[bid]['tier']
                dummy_block = MemoryBlock(id=bid, size_mb=self.registry[bid]['size_mb'])
                if tier == 'L1':
                    self.migrate(dummy_block, 'L2')
                    evicted.append((bid,'L1','L2'))
                    MEMORY_EVICTIONS.labels('L1','L2').inc()
                elif tier == 'L2':
                    # Vers RAM ou NVMe selon taille
                    if self.registry[bid]['size_mb'] > 512:
                        self.spill_to_nvme(dummy_block, {"dummy":"weights"})
                        evicted.append((bid,'L2','L5'))
                        MEMORY_EVICTIONS.labels('L2','L5').inc()
                    else:
                        self.migrate(dummy_block, 'L3')
                        evicted.append((bid,'L2','L3'))
                        MEMORY_EVICTIONS.labels('L2','L3').inc()
            return evicted

    def summary(self) -> dict[str, Any]:
        tiers: dict[str,int] = {k:0 for k in ["L1","L2","L3","L4","L5","L6"]}
        for meta in self.registry.values():
            tiers[meta["tier"]] += 1
        return {"tiers": tiers, "count": len(self.registry)}

    # --- Persistence (prod stricte) ---
    def save_state(self, path: str = ".hm_state.pkl"):
        data = {
            'registry': self.registry,
            'hot': self._hot_scores,
            'last_touch': self._last_touch,
            'decay_half_life': self._decay_half_life,
            'ts': time.time()
        }
        try:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:  # pragma: no cover
            self.log.warning(f"save_state fail: {e}")

    def load_state(self, path: str = ".hm_state.pkl"):
        if not os.path.exists(path):
            return False
        try:
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.registry = data.get('registry', {})
            self._hot_scores = data.get('hot', {})
            self._last_touch = data.get('last_touch', {})
            self._decay_half_life = data.get('decay_half_life', self._decay_half_life)
            return True
        except Exception as e:  # pragma: no cover
            self.log.warning(f"load_state fail: {e}")
            return False

    # --- Benchmark initial (optionnel) ---
    def run_initial_benchmark(self):
        try:
            from core.memory_benchmark import benchmark_and_rank
            res = benchmark_and_rank()
            self.log.info(f"Benchmark tiers (p50 asc) : {res.get('ranking')}")
            return res
        except Exception as e:
            self.log.warning(f"Benchmark tiers impossible: {e}")
            return None

__all__ = ["HierarchicalMemoryManager"]