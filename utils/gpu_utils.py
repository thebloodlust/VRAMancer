# utils/gpu_utils.py
import os
import subprocess
import json
from typing import List, Dict

def _nvml_query() -> List[Dict]:
    """
    Utilise nvml (via python‑pynvml) pour obtenir
    l’état de chaque GPU : nom, mémoire totale, mémoire utilisée.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            total = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 ** 2)  # MB
            used  = pynvml.nvmlDeviceGetMemoryInfo(handle).used  // (1024 ** 2)  # MB
            gpus.append({
                "name": name,
                "total_vram_mb": total,
                "used_vram_mb": used,
                "is_available": True  # on peut ajouter une logique de disponibilité
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception as e:
        # fallback: on retourne une liste vide
        print(f"[GPU Utils] Erreur lors de la récupération via nvml : {e}")
        return []

def get_available_gpus() -> List[Dict]:
    """Retourne une liste de dictionnaires GPU."""
    return _nvml_query()
