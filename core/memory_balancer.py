from __future__ import annotations
import typing as T
import logging

class MemoryBalancer:
    """
    Gère le rééquilibrage de mémoire entre les GPUs.
    """

    def __init__(self, scheduler: "Scheduler", logger: logging.Logger):
        self.scheduler = scheduler
        self.logger    = logger
        # On garde un état interne que `get_memory_state` retournera
        self._gpu_state = {
            gpu["id"]: {"used": 0, "total": gpu["total_vram_mb"]}
            for gpu in scheduler.get_available_gpus()
        }

    # --------------------------------------------------------------------
    # Méthodes métier existantes (balance, predictive_balance, …) …
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Méthode d’extension – utilisée par le dashboard
    # --------------------------------------------------------------------
    def get_memory_state(self) -> dict[int, dict[str, int]]:
        """
        Renvoie un dictionnaire `{gpu_id: {"used": X, "total": Y}}`.
        Si votre Scheduler peut fournir les chiffres de mémoire en temps réel,
        appelez‑le ici ; sinon, on renvoie l’état interne mis à jour dans
        les méthodes `balance`, `release_layer`, ….
        """
        # Exemple d’appel (à adapter à votre implémentation ):
        # for gpu_id in self._gpu_state:
        #     usage = self.scheduler.get_gpu_usage(gpu_id)  # {"used": ..., "total": ...}
        #     self._gpu_state[gpu_id]["used"] = usage["used"]
        # return self._gpu_state

        # Si le Scheduler n’expose pas de `get_gpu_usage`, on renvoie l’état interne
        return self._gpu_state

    # --------------------------------------------------------------------
    # Méthodes déjà présentes dans votre classe (balance, predictive_balance, …)
    # --------------------------------------------------------------------
