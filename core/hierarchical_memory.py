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
from core.memory_block import MemoryBlock

Tier = Literal["L1","L2","L3","L4","L5","L6"]

class HierarchicalMemoryManager:
    def __init__(self, nvme_dir: str = ".hm_cache", max_nvme_mb: int = 2048):
        self.log = LoggerAdapter("hmem")
        self.nvme_dir = Path(nvme_dir)
        self.nvme_dir.mkdir(exist_ok=True)
        self.max_nvme_mb = max_nvme_mb
        # Tables de suivi
        self.registry: dict[str, dict[str, Any]] = {}

    def register_block(self, block: MemoryBlock, tier: Tier):
        self.registry[block.id] = {
            "tier": tier,
            "size_mb": block.size_mb,
            "ts": time.time(),
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
        self.log.info(f"Migration bloc {block.id[:8]} {prev} → {target}")

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

    def summary(self) -> dict[str, Any]:
        tiers: dict[str,int] = {k:0 for k in ["L1","L2","L3","L4","L5","L6"]}
        for meta in self.registry.values():
            tiers[meta["tier"]] += 1
        return {"tiers": tiers, "count": len(self.registry)}

__all__ = ["HierarchicalMemoryManager"]