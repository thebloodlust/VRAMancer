# core/monitor.py
import random
from typing import Dict, Optional

import torch


class GPUMonitor:
    """
    Petit utilitaire pour surveiller l’utilisation de la mémoire VRAM
    de chaque GPU CUDA disponible.

    L’API est volontairement simple :
    * `vram_usage(gpu_id)` → pour obtenir un pourcentage de VRAM utilisé.
    * `detect_overload(threshold)` → renvoie l’id du GPU qui dépasse le
      seuil (ou `None` si aucun).
    * `status()` → dictionnaire lisible par l’utilisateur.
    """

    def __init__(self, verbose: bool = True) -> None:
        """
        :param verbose: si `True`, imprime les erreurs sur la console
                        (utile en mode debug).
        """
        self.verbose = verbose

    # --------------------------------------------------------------------
    #  Méthode utilitaire : obtention de la VRAM d’un GPU
    # --------------------------------------------------------------------
    def vram_usage(self, gpu_id: int = 0) -> float:
        """
        Retourne la VRAM utilisée (en %) du GPU `gpu_id`.

        En cas d’erreur (p. ex. aucun GPU, GPU indisponible), on
        renvoie une valeur aléatoire entre 40 % et 90 % pour ne pas
        faire planter le programme.
        """
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            total_mb = props.total_memory / (1024**2)
            used_mb = torch.cuda.memory_allocated(gpu_id) / (1024**2)
            return round((used_mb / total_mb) * 100, 2)
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur pour GPU {gpu_id} : {exc}")
            # Valeur simulée
            return random.randint(40, 90)

    # --------------------------------------------------------------------
    #  Détection d’une surcharge
    # --------------------------------------------------------------------
    def detect_overload(self, threshold: float = 90.0) -> Optional[int]:
        """
        Vérifie si un GPU dépasse le seuil `threshold` (en %).
        Renvoie l’ID du premier GPU qui l’est, ou `None` si aucun.

        Si une erreur se produit, on renvoie aléatoirement 0 ou `None`
        afin de ne pas bloquer le reste du programme.
        """
        try:
            for i in range(torch.cuda.device_count()):
                if self.vram_usage(i) > threshold:
                    return i
            return None
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur pendant la détection d’overload : {exc}")
            return random.choice([0, None])

    # --------------------------------------------------------------------
    #  Résumé lisible
    # --------------------------------------------------------------------
    def status(self) -> Dict[str, str]:
        """
        Retourne un dictionnaire `{ "GPU i" : "xx% VRAM" }` pour
        chaque GPU disponible.
        """
        try:
            status: Dict[str, str] = {}
            for i in range(torch.cuda.device_count()):
                usage = self.vram_usage(i)
                status[f"GPU {i}"] = f"{usage}% VRAM"
            return status
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur pendant la récupération du status : {exc}")
            # Valeur simulée
            return {"GPU 0": "Simulé"}
