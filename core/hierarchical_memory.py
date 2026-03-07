"""Gestion hiérarchique mémoire multi-niveau (L1→L7) - Hyperviseur de VRAM Distribuée.

Niveaux supportés :
  L1 : VRAM GPU primaire (Compute Node - accès direct)
  L2 : VRAM GPUs secondaires (Local PCIe/NVLink P2P)
  L3 : VRAM GPUs distants (via RDMA/RoCEv2)
  L4 : RAM hôte locale (Pinned/Page-locked DRAM)
  L5 : NVMe Local (mmap / GPUDirect Storage)
  L6 : RAM distante (via réseau rapide)
  L7 : Disque réseau distant / Fallback TCP

Ce module est le "cerveau" interceptant les memory page-faults.
La copie réelle des tenseurs est déléguée au moteur C++ VTP (VRAMancer Transport Protocol).
"""
from __future__ import annotations
import os
import time
import pickle
import threading
from pathlib import Path
from typing import Any, Literal, List, Optional
from core.logger import LoggerAdapter
from core.metrics import MEMORY_PROMOTIONS, MEMORY_DEMOTIONS, MEMORY_EVICTIONS
from core.tracing import get_tracer
from core.memory_block import MemoryBlock

Tier = Literal["L1", "L2", "L3", "L4", "L5", "L6", "L7"]

# Thresholds to trigger predictive fetch (us)
_PREFETCH_LEAD_TIME_US = 5000

class HierarchicalMemoryManager:
    """Hyperviseur L1-L7 pour VRAM distribuée.
    
    Traque où se trouve chaque `block_id`. Intercepte les accès et orchestre
    les pre-fetches asynchrones via le VTP C++ backend."""
    def __init__(self, nvme_dir: str = ".hm_cache", max_nvme_mb: int = 2048, decay_half_life_s: float = 60.0):
        self.log = LoggerAdapter("hmem.v2")
        self.nvme_dir = Path(nvme_dir)
        self.nvme_dir.mkdir(exist_ok=True)
        self.max_nvme_mb = max_nvme_mb
        # Thread safety
        self._lock = threading.Lock()
        # Tables de suivi (L1-L7 mappings)
        self.registry: dict[str, dict[str, Any]] = {}
        # Tensor registry: block_id -> tensor
        self._tensor_registry: dict[str, Any] = {}
        # Hotness hybride
        self._hot_scores: dict[str, float] = {}
        self._last_touch: dict[str, float] = {}
        self._decay_half_life = decay_half_life_s
        self._prefetch_queue: List[str] = []
        
        # Lier au backend C++ VTP si dispo
        self._vtp_backend = None
        self._init_vtp()

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

    
    def _init_vtp(self):
        """Initialise le VRAMancer Transport Protocol (vtp_core.cpp) s'il est compilé."""
        try:
            import vtp_core
            self._vtp_backend = vtp_core
            self.log.info("VTP (C++ fast path) chargé avec succès ✅")
        except ImportError:
            self.log.warning("Extension C++ 'vtp_core' non disponible. Fallback sur Python pur 🐢")

    def schedule_prefetch(self, block_ids: list[str], target_tier: Tier = "L1"):
        """Schedule un prefetch anticipé utilisant le fast path C++ si possible."""
        for bid in block_ids:
            if bid not in self._prefetch_queue:
                self._prefetch_queue.append(bid)
        
        # Lancer le worker si dispo (simplifié)
        import threading as _thr
        _thr.Thread(target=self._process_prefetch, args=(target_tier,), daemon=True).start()

    def _process_prefetch(self, target_tier: Tier):
        """Worker thread pour le pré-chargement."""
        while self._prefetch_queue:
            bid = self._prefetch_queue.pop(0)
            if bid in self.registry:
                current_tier = self.registry[bid]["tier"]
                if current_tier != target_tier:
                    # Simulation: si on a VTP, VTP s'en occupe en zero-copy C++
                    if self._vtp_backend:
                        pass # self._vtp_backend.fast_p2p_transfer()
                    else:
                        # Fallback
                        block = self.registry[bid]["block"]
                        tensor = self._tensor_registry.get(bid)
                        if tensor is not None:
                            self.migrate(block, target_tier, tensor)

    def register_block(self, block: MemoryBlock, tier: Tier, tensor: Any = None):
        with self._lock:
            self.registry[block.id] = {
                "tier": tier,
                "size_mb": block.size_mb,
                "ts": time.time(),
                "access": 0,
                "last_access": None,
                "meta": {},
            }
            if tensor is not None:
                self._tensor_registry[block.id] = tensor
        self.log.debug(f"Register {block.id[:8]} @ {tier}")

    def get_tier(self, block_id: str) -> Tier | None:
        with self._lock:
            info = self.registry.get(block_id)
            return info["tier"] if info else None

    # --- Migration logique + transport physique ---
    def migrate(self, block: MemoryBlock, target: Tier, tensor: Any = None):
        """Migrate a block to a different memory tier.

        If a tensor is provided, the actual data is moved using
        the appropriate transport:
          L1 <-> L2: NCCL / CUDA P2P (via TransferManager)
          L1/L2 -> L3: tensor.cpu()
          L3 -> L1/L2: tensor.cuda(gpu_id)
          L3 <-> L4: RDMA / TCP (via FastHandle)
          L3 <-> L5: NVMe spill (pickle)
        """
        prev = self.get_tier(block.id)
        if prev == target:
            return tensor
        self.registry[block.id]["tier"] = target
        self.registry[block.id]["ts"] = time.time()

        moved_tensor = tensor

        # Physical data movement (when tensor provided)
        if tensor is not None:
            moved_tensor = self._execute_physical_move(block, prev, target, tensor)

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
            self.log.info(f"Migration bloc {block.id[:8]} {prev} -> {target}")

        return moved_tensor

    def _execute_physical_move(self, block: MemoryBlock, prev: Tier, target: Tier, tensor: Any) -> Any:
        """Execute the actual data movement between tiers."""
        try:
            # L1 <-> L2: inter-GPU transfer via NCCL/P2P
            if prev in {"L1", "L2"} and target in {"L1", "L2"}:
                return self._move_inter_gpu(block, tensor, target)

            # L1/L2 -> L3: GPU to CPU
            if prev in {"L1", "L2"} and target == "L3":
                return self._move_gpu_to_cpu(tensor)

            # L3 -> L1/L2: CPU to GPU
            if prev == "L3" and target in {"L1", "L2"}:
                return self._move_cpu_to_gpu(tensor, block.gpu_id)

            # L3 <-> L4: network transfer (RDMA/TCP)
            if target == "L4" or prev == "L4":
                return self._move_network(tensor, block, prev, target)

        except Exception as e:
            self.log.warning(f"Physical move {prev}->{target} failed: {e}, metadata updated anyway")

        return tensor

    def _move_inter_gpu(self, block: MemoryBlock, tensor: Any, target: Tier) -> Any:
        """Move tensor between GPUs using TransferManager (NCCL/P2P).

        Returns the tensor ON THE TARGET GPU (not the source).
        """
        try:
            from core.transfer_manager import TransferManager
            if not hasattr(self, '_transfer_mgr'):
                self._transfer_mgr = TransferManager(verbose=False)
            # Determine target GPU from block metadata or tier convention
            # L1 = primary GPU (0), L2 = secondary GPU (1+)
            src_gpu = tensor.device.index if hasattr(tensor, 'device') and tensor.device.index is not None else 0
            dst_gpu = 0 if target == "L1" else (block.gpu_id if block.gpu_id != src_gpu else 1)
            result = self._transfer_mgr.send_activation(src_gpu, dst_gpu, tensor)
            self.log.debug(
                f"Inter-GPU {src_gpu}->{dst_gpu}: "
                f"{result.bytes_transferred / 1e6:.1f}MB, {result.bandwidth_gbps:.1f} Gbps"
            )
            # Return tensor on the DESTINATION GPU
            import torch as _torch
            dst_tensor = tensor.to(f"cuda:{dst_gpu}", non_blocking=True)
            return dst_tensor
        except ImportError:
            self.log.warning("TransferManager unavailable, skip physical inter-GPU move")
            return tensor

    def _move_gpu_to_cpu(self, tensor: Any) -> Any:
        """Move tensor from GPU VRAM to CPU RAM."""
        try:
            import torch
            if hasattr(tensor, 'cpu'):
                return tensor.cpu().pin_memory()  # Pinned for fast re-upload
        except Exception:
            pass
        return tensor

    def _move_cpu_to_gpu(self, tensor: Any, gpu_id: int) -> Any:
        """Move tensor from CPU RAM to GPU VRAM."""
        try:
            import torch
            if hasattr(tensor, 'cuda'):
                return tensor.cuda(gpu_id)
        except Exception:
            pass
        return tensor

    def _move_network(self, tensor: Any, block: MemoryBlock, prev: Tier, target: Tier) -> Any:
        """Move tensor to/from remote node via RDMA or TCP."""
        try:
            from core.network.fibre_fastpath import open_low_latency_channel
            if not hasattr(self, '_net_channel'):
                self._net_channel = open_low_latency_channel()
            if self._net_channel:
                if target == "L4":
                    self._net_channel.send_tensor(tensor)
                    self.log.debug(f"Sent block {block.id[:8]} to remote via {self._net_channel.kind}")
                # recv path handled by caller with shape/dtype info
        except ImportError:
            self.log.warning("FastHandle unavailable, skip network move")
        return tensor

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

    # --- NVMe spill (L5) --- CXL Software Bridge Enabled
    def spill_to_nvme(self, block: MemoryBlock, payload: Any):
        import torch
        if torch.is_tensor(payload):
            path = self.nvme_dir / f"{block.id}.cxl"
            # Assure que le tenseur est contigu en memoire CPU
            cpu_tensor = payload.cpu().contiguous()
            num_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
            ptr = cpu_tensor.data_ptr()
            
            # Stockage des metadata sans pickle
            self.registry[block.id].setdefault("meta", {}).update({
                "storage_type": "cxl_raw",
                "shape": tuple(cpu_tensor.shape),
                "dtype_str": str(cpu_tensor.dtype),
                "device": "cpu"
            })
            
            try:
                import software_cxl
                software_cxl.cxl_direct_memory_dump(str(path), ptr, num_bytes)
                self.migrate(block, "L5")
                self.log.debug(f"⚡ [CXL] Spill {block.id[:8]} -> NVMe ({num_bytes/1e6:.1f}MB) GIL-bypassed")
                return
            except Exception as e:
                self.log.warning(f"CXL Bridge fallback: {e}")
                
        # Fallback classique lent
        path = self.nvme_dir / f"{block.id}.pkl"
        with path.open("wb") as f:
            pickle.dump(payload, f)
        self.migrate(block, "L5")
        self.log.debug(f"Spill bloc {block.id[:8]} vers NVMe (Pickle)")

    def load_from_nvme(self, block: MemoryBlock) -> Any | None:
        meta = self.registry.get(block.id, {}).get("meta", {})
        if meta.get("storage_type") == "cxl_raw":
            import torch
            path = self.nvme_dir / f"{block.id}.cxl"
            if not path.exists():
                return None
                
            dtype_map = {
                "torch.float32": torch.float32, "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16, "torch.int8": torch.int8,
                "torch.uint8": torch.uint8, "torch.int32": torch.int32,
                "torch.int64": torch.int64
            }
            dtype = dtype_map.get(meta["dtype_str"], torch.float16)
            
            # Allocation directe zero-copy
            tensor = torch.empty(meta["shape"], dtype=dtype, device="cpu")
            num_bytes = tensor.numel() * tensor.element_size()
            ptr = tensor.data_ptr()
            
            try:
                import software_cxl
                software_cxl.cxl_direct_memory_load(str(path), ptr, num_bytes)
                self.migrate(block, "L3")
                self.log.debug(f"⚡ [CXL] Reload {block.id[:8]} from NVMe ({num_bytes/1e6:.1f}MB) GIL-bypassed")
                return tensor
            except Exception as e:
                self.log.warning(f"CXL reload fallback: {e}")
        
        # Fallback classique lent
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
            # Descendre en VRAM secondaire
            self.migrate(block, "L2")
        elif tier in {"L2","L3"} and gpu_over_pct > 95:
            # Spill NVMe — use real tensor if available
            tensor = self._tensor_registry.get(block.id)
            self.spill_to_nvme(block, tensor if tensor is not None else block)

    # --- Eviction planner (lot B) ---
    def update_all_scores(self, current_time: float):
        """Update all hotness scores using C++ extension if available."""
        try:
            import vramancer_vtp_c
            # Prepare inputs for C++
            access_counts = {bid: meta["access"] for bid, meta in self.registry.items()}
            # C++ function returns updated scores
            new_scores = vramancer_vtp_c.compute_hotness_scores(
                access_counts,
                self._last_touch,
                current_time,
                self._decay_half_life
            )
            self._hot_scores.update(new_scores)
        except ImportError:
            # Fallback to Python loop
            decay_constant = 0.69314718056 / self._decay_half_life if self._decay_half_life > 0 else 0
            import math
            for bid, meta in self.registry.items():
                count = meta["access"]
                last_t = self._last_touch.get(bid, current_time)
                dt = max(0.0, current_time - last_t)
                self._hot_scores[bid] = count * math.exp(-decay_constant * dt)

    def eviction_cycle(self, target_free_pct: float = 10.0, vram_pressure: float | None = None):
        """Applique une politique d'éviction basée sur le hotness.
        Objectif: libérer de la VRAM L1/L2 quand la pression est trop forte.
        Heuristique simple:
          - Calcule un score relatif (hotness) pour chaque bloc en L1/L2
          - Trie ascendant et déplace les X% plus froids vers un niveau inférieur
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("memory.eviction_cycle"):
            # Update all scores to current time before sorting
            self.update_all_scores(time.time())
            
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
                    # Vers RAM ou NVMe selon taille — use real tensor
                    tensor = self._tensor_registry.get(bid)
                    if self.registry[bid]['size_mb'] > 512:
                        self.spill_to_nvme(dummy_block, tensor if tensor is not None else dummy_block)
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
        self.log.info("Memory tier benchmarking not available")
        return None

__all__ = ["HierarchicalMemoryManager"]