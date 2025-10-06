from __future__ import annotations
import typing as T
import logging
from collections import OrderedDict

from core.utils import assign_block_to_device
from core.scheduler import SimpleScheduler

class MemoryBalancer:
    """
    Gère le rééquilibrage de mémoire et le partage de blocs entre tous les GPUs/accélérateurs.
    Supporte CUDA, ROCm, MPS (Apple), CPU, et n'importe quel nombre de GPU secondaires.
    """
    def __init__(self, scheduler: "SimpleScheduler", logger: logging.Logger, cache_size: int = 2):
        self.scheduler = scheduler
        self.logger    = logger
        self._gpu_state = {
            gpu["id"]: {"used": 0, "total": gpu["total_vram_mb"]}
            for gpu in scheduler.get_available_gpus()
        }
        # Cache LRU de blocs par GPU secondaire : {gpu_id: OrderedDict{block_id: block}}
        self._gpu_cache: dict[int, OrderedDict[str, object]] = {
            gpu_id: OrderedDict() for gpu_id in self._gpu_state
        }
        self._cache_size = cache_size  # nombre max de blocs en cache par GPU

    def allocate_block(self, block, target_gpu: int) -> object:
        """
        Alloue un bloc sur le GPU cible, ou le récupère du cache si possible.
        """
        cache = self._gpu_cache[target_gpu]
        if hasattr(block, 'id') and block.id in cache:
            # Bloc déjà en cache sur ce GPU
            self.logger.info(f"[MemoryBalancer] Bloc {block.id} récupéré du cache GPU{target_gpu}")
            cache.move_to_end(block.id)
            return cache[block.id]
        # Déplacer le bloc sur le bon device
        block_on_device = assign_block_to_device(block, target_gpu)
        # Mettre à jour le cache (LRU)
        if hasattr(block, 'id'):
            cache[block.id] = block_on_device
            if len(cache) > self._cache_size:
                evicted_id, _ = cache.popitem(last=False)
                self.logger.info(f"[MemoryBalancer] Bloc {evicted_id} évincé du cache GPU{target_gpu}")
        self._gpu_state[target_gpu]["used"] += getattr(block, 'size_mb', 0)
        return block_on_device

    def release_block(self, block, gpu_id: int):
        """
        Libère un bloc de la VRAM du GPU (ou du cache), mais peut le garder en cache sur un GPU secondaire.
        """
        cache = self._gpu_cache[gpu_id]
        if hasattr(block, 'id') and block.id in cache:
            del cache[block.id]
            self.logger.info(f"[MemoryBalancer] Bloc {block.id} libéré du cache GPU{gpu_id}")
        self._gpu_state[gpu_id]["used"] = max(0, self._gpu_state[gpu_id]["used"] - getattr(block, 'size_mb', 0))

    def migrate_block(self, block, src_gpu: int, dst_gpu: int) -> object:
        """
        Migre un bloc d'un GPU à un autre (ou CPU/MPS/ROCm), en utilisant le cache si possible.
        """
        self.release_block(block, src_gpu)
        return self.allocate_block(block, dst_gpu)

    def balance(self, blocks: list, usage: dict[int, float], threshold: float = 90.0):
        """
        Rééquilibre les blocs entre GPUs selon l'utilisation (en %).
        Si un GPU dépasse le seuil, migre des blocs vers les moins chargés.
        """
        overloaded = [gpu for gpu, u in usage.items() if u > threshold]
        underloaded = [gpu for gpu, u in usage.items() if u < threshold - 20]
        for gpu_id in overloaded:
            for block in blocks:
                if getattr(block, 'gpu_id', None) == gpu_id:
                    if underloaded:
                        dst_gpu = underloaded[0]
                        self.logger.info(f"[MemoryBalancer] Migration bloc {block.id} GPU{gpu_id} → GPU{dst_gpu}")
                        self.migrate_block(block, gpu_id, dst_gpu)
                        break

    def get_memory_state(self) -> dict[int, dict[str, int]]:
        """
        Renvoie un dictionnaire `{gpu_id: {"used": X, "total": Y}}`.
        Peut être enrichi avec des stats live du scheduler.
        """
        return self._gpu_state

    def get_cache_state(self) -> dict[int, list[str]]:
        """
        Renvoie l'état du cache par GPU (liste des block ids).
        """
        return {gpu: list(cache.keys()) for gpu, cache in self._gpu_cache.items()}
