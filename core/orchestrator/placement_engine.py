"""Placement Engine — choix du GPU pour un bloc, et plan de placement d'un modèle.

PÉRIMÈTRE HONNÊTE (relu ligne à ligne le 2026-09-22) : ce module n'a aujourd'hui
AUCUN consommateur dans le chemin de service — seuls des tests l'appellent. Il ne
décide donc encore rien en production ; le traiter comme une brique en attente de
câblage, pas comme un composant éprouvé.

Deux niveaux, à ne pas confondre :

`place(block)` — placement d'UN bloc, par HEURISTIQUE (pas de DP, pas de mesure
de latence) :
  - "profiled" (défaut) : score = (VRAM libre / VRAM totale) × facteur de compute
    × 1.5 si la couche est de l'attention. Le facteur de compute vaut 1.0 tant
    qu'aucun profil GPU n'a été calculé — dans ce cas la stratégie se réduit au
    ratio de VRAM libre. Le nom « profiled » est donc optimiste.
  - "vram"     : VRAM libre pondérée par le score Connectome (santé du lien réseau).
  - "balanced" : round-robin entre GPU (ignore l'hétérogénéité).
  - custom     : register_strategy(name, callable).

`place_model(model)` — plan couche→GPU pour un modèle complet. Celui-ci mesure
réellement (LayerProfiler : latence, mémoire, FLOPS par couche + benchmark des
GPU) puis résout par programmation dynamique (compute_optimal_placement). Si
LayerProfiler est indisponible, renvoie un plan factice {"strategy":
"vram_fallback"} qui ne contient AUCUNE affectation — vérifier le retour.

Les latences renvoyées par le plan sont des ESTIMATIONS du modèle de coût, pas
des mesures de bout en bout.

API:
    engine = PlacementEngine(metrics_provider)
    decision = engine.place(block_meta)            -> {level, gpu_id, strategy}
    plan = engine.place_model(model, num_gpus)     -> PlacementPlan | dict fallback
    engine.register_strategy(name, callable)
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Any, List, Optional

_logger = logging.getLogger("vramancer.placement_engine")

try:
    from core.metrics import ORCH_PLACEMENTS
except Exception:
    class _Dummy:
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): return None
    ORCH_PLACEMENTS = _Dummy()

try:
    from core.layer_profiler import (
        LayerProfiler,
        LayerProfile,
        GPUProfile,
        PlacementPlan,
        compute_optimal_placement,
    )
    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False
    LayerProfiler = None
    PlacementPlan = None

try:
    from core.monitor import GPUMonitor
except ImportError:
    GPUMonitor = None

try:
    from core.utils import enumerate_devices, detect_backend
except ImportError:
    def enumerate_devices():
        return [{"id": "cpu:0", "backend": "cpu", "index": 0, "name": "CPU"}]
    def detect_backend():
        return "cpu"


class PlacementEngine:
    """Production placement engine with profiling-based optimization.

    Provides both block-level placement (place()) and model-level
    optimal assignment (place_model()).
    """

    def __init__(self, metrics_provider=None, monitor: Optional[Any] = None):
        self._strategies: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._default = self._strategy_profiled
        self.metrics_provider = metrics_provider
        self.monitor = monitor

        # Cache profiling results
        self._gpu_profiles: Optional[List] = None
        self._profiler: Optional[Any] = None

        if _PROFILER_AVAILABLE:
            self._profiler = LayerProfiler()

        # Register built-in strategies
        self._strategies["profiled"] = self._strategy_profiled
        self._strategies["vram"] = self._strategy_vram
        self._strategies["balanced"] = self._strategy_balanced

    def register_strategy(
        self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """Register a custom placement strategy."""
        self._strategies[name] = fn
        _logger.info(f"Registered placement strategy: {name}")

    # ------------------------------------------------------------------
    # Block-level placement (backward-compatible API)
    # ------------------------------------------------------------------

    def place(
        self, block: Dict[str, Any], strategy: str | None = None
    ) -> Dict[str, Any]:
        """Place a single block on the optimal GPU/tier.

        Args:
            block: {"size_mb": int, "layer_type": str, "priority": int, ...}
            strategy: Strategy name or None for default ("profiled").

        Returns:
            {"level": str, "gpu_id": int, "reason": str}
        """
        fn = self._strategies.get(strategy or "", self._default)
        decision = fn(block)
        try:
            lvl = decision.get("level", "unknown")
            ORCH_PLACEMENTS.labels(lvl).inc()
        except Exception:
            _logger.debug("Metrics placement counter failed", exc_info=True)
        return decision

    # ------------------------------------------------------------------
    # Connectome health weighting
    # ------------------------------------------------------------------

    def _apply_connectome_health_score(self, base_score: float, gpu_id: int) -> float:
        """Weight a placement score by the node's measured Connectome health.

        The Connectome tracks per-node link health (adaptive weight from
        measured latency/reliability). Local GPUs map to weight 1.0; remote
        nodes are scaled by their current health weight.
        """
        try:
            from core.network.connectome import global_connectome
            # GPU IDs map to local nodes here; in a real cluster this maps to a node IP.
            node_id = f"gpu_{gpu_id}"
            strength = global_connectome.get_synaptic_weight(node_id)
            return base_score * strength
        except Exception:
            return base_score

    # ------------------------------------------------------------------
    # Model-level placement (new API)
    # ------------------------------------------------------------------

    def place_model(
        self,
        model: Any,
        num_gpus: int = 0,
        transfer_bandwidth_gbps: float = 0.0,
    ) -> Any:
        """Compute a layer-to-GPU placement plan for a full model.

        Measures each layer (latency, memory, FLOPS) and benchmarks each GPU,
        then solves via DP. "Optimal" is optimal FOR THE COST MODEL — the
        returned latency is an estimate, not an end-to-end measurement.
        Without LayerProfiler, returns the {"strategy": "vram_fallback"} dict,
        which carries NO layer assignment at all.

        Args:
            model: Loaded nn.Module (HuggingFace or custom).
            num_gpus: Number of GPUs (0 = auto-detect).
            transfer_bandwidth_gbps: Inter-GPU bandwidth (0 = auto-detect).

        Returns:
            PlacementPlan with (layer_index, gpu_index) assignments.
        """
        if not _PROFILER_AVAILABLE:
            _logger.warning("LayerProfiler not available, falling back to VRAM-based")
            return self._vram_fallback_plan(model, num_gpus)

        # Profile layers
        _logger.info("Profiling model layers...")
        layer_profiles = self._profiler.profile_model(model)

        # Profile GPUs (cached)
        if self._gpu_profiles is None:
            _logger.info("Benchmarking GPUs...")
            self._gpu_profiles = self._profiler.profile_gpus()

        gpu_profiles = self._gpu_profiles
        if num_gpus > 0:
            gpu_profiles = gpu_profiles[:num_gpus]

        if not gpu_profiles:
            _logger.warning("No GPUs profiled")
            return PlacementPlan()

        # Compute optimal placement via DP
        _logger.info(
            f"Computing optimal placement: {len(layer_profiles)} layers "
            f"across {len(gpu_profiles)} GPUs"
        )
        plan = compute_optimal_placement(
            layer_profiles, gpu_profiles, transfer_bandwidth_gbps
        )

        _logger.info(
            f"Optimal plan: latency={plan.estimated_latency_ms:.1f}ms "
            f"(transfer overhead={plan.estimated_transfer_overhead_ms:.1f}ms), "
            f"GPU util={plan.gpu_utilization}"
        )

        return plan

    def _vram_fallback_plan(self, model: Any, num_gpus: int) -> Dict[str, Any]:
        """Fallback sans LayerProfiler — NE CONTIENT AUCUNE affectation de couche.

        Marqueur signalant que le placement n'a pas pu être calculé ; l'appelant
        doit le détecter (clé "strategy" == "vram_fallback") et décider lui-même.
        """
        return {"strategy": "vram_fallback", "num_gpus": num_gpus}

    # ------------------------------------------------------------------
    # Built-in strategies for block-level placement
    # ------------------------------------------------------------------

    def _strategy_profiled(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristique : VRAM libre × compute × poids de type de couche.

        Utilise les benchmarks GPU s'ils ont déjà été calculés (get_gpu_profiles /
        place_model) ; sinon le facteur de compute vaut 1.0 et il ne reste que le
        ratio de VRAM libre. Aucun profilage n'est déclenché ici.
        """
        size_mb = block.get("size_mb", 128)
        layer_type = block.get("layer_type", "unknown")

        # Get GPU with most free VRAM and best compute score
        gpu_id, level = self._select_best_gpu(size_mb, layer_type)

        return {
            "level": level,
            "gpu_id": gpu_id,
            "strategy": "profiled",
            "reason": f"Best GPU for {size_mb}MB {layer_type} block",
        }

    def _strategy_vram(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """VRAM-proportional strategy with Connectome weights."""
        size_mb = block.get("size_mb", 128)

        gpu_id = 0
        max_free = 0

        if self.monitor:
            try:
                devs = enumerate_devices()
                for dev in devs:
                    if dev["backend"] in ("cuda", "rocm", "mps"):
                        free = self.monitor.get_free_memory(dev["index"])
                        
                        # -> Adaptive scoring heuristic
                        # We discount the free VRAM based on the strength of the connection.
                        # Meaning a GPU with lots of VRAM but a terrible connection will look "full".
                        effective_free = self._apply_connectome_health_score(free, dev["index"])
                        
                        if effective_free > max_free:
                            max_free = effective_free
                            gpu_id = dev["index"]
            except Exception:
                _logger.debug("GPU memory enumeration failed", exc_info=True)

        # Since max_free is degraded by adaptive scoring, if the network is terrible, we fall to L3 instead of L1.
        level = "L1" if max_free >= size_mb * 1024 * 1024 else "L3"
        return {
            "level": level,
            "gpu_id": gpu_id,
            "strategy": "vram_adaptive",
        }

    def _strategy_balanced(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Balanced strategy: round-robin across GPUs."""
        if not hasattr(self, "_rr_counter"):
            self._rr_counter = 0

        devs = enumerate_devices()
        gpu_devs = [d for d in devs if d["backend"] in ("cuda", "rocm", "mps")]
        if not gpu_devs:
            return {"level": "L3", "gpu_id": 0, "strategy": "balanced"}

        idx = self._rr_counter % len(gpu_devs)
        self._rr_counter += 1

        return {
            "level": "L1",
            "gpu_id": gpu_devs[idx]["index"],
            "strategy": "balanced",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_best_gpu(
        self, size_mb: int, layer_type: str
    ) -> tuple:
        """Select best GPU considering free VRAM and compute capability.

        Returns:
            (gpu_id, tier_level)
        """
        devs = enumerate_devices()
        gpu_devs = [d for d in devs if d["backend"] in ("cuda", "rocm", "mps")]

        if not gpu_devs:
            level = "L1" if size_mb <= 256 else "L3"
            return 0, level

        best_gpu = 0
        best_score = -1.0

        for dev in gpu_devs:
            idx = dev["index"]
            total_mem = dev.get("total_memory", 0) or 0
            total_mb = total_mem / (1024 * 1024) if total_mem else 8192

            # Free VRAM from monitor
            free_mb = total_mb
            if self.monitor:
                try:
                    free_bytes = self.monitor.get_free_memory(idx)
                    free_mb = free_bytes / (1024 * 1024)
                except Exception:
                    _logger.debug("GPU free memory query failed", exc_info=True)

            if free_mb < size_mb:
                continue  # Won't fit

            # Score: free_vram * compute_factor
            compute_factor = 1.0
            if self._gpu_profiles:
                for gp in self._gpu_profiles:
                    if gp.index == idx:
                        max_throughput = max(
                            (p.compute_throughput_gflops for p in self._gpu_profiles),
                            default=1.0,
                        )
                        compute_factor = gp.compute_throughput_gflops / max(
                            max_throughput, 0.01
                        )
                        break

            # Attention layers benefit more from fast GPUs
            type_weight = 1.5 if layer_type == "attention" else 1.0
            score = (free_mb / total_mb) * compute_factor * type_weight

            if score > best_score:
                best_score = score
                best_gpu = idx

        level = "L1"  # On GPU VRAM
        return best_gpu, level

    def get_gpu_profiles(self) -> Optional[list]:
        """Return cached GPU profiles, running benchmark if needed."""
        if self._gpu_profiles is None and self._profiler:
            self._gpu_profiles = self._profiler.profile_gpus()
        return self._gpu_profiles

    def invalidate_cache(self):
        """Force re-benchmark on next call."""
        self._gpu_profiles = None


__all__ = ["PlacementEngine"]
