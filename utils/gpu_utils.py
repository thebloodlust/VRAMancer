# utils/gpu_utils.py
import subprocess
import json
from typing import List, Dict

def get_available_gpus() -> List[Dict]:
    """
    Utilise pynvml (ou nvml) pour récupérer les infos GPU.
    Renvoie une liste de dict : {name, total_vram_mb, used_vram_mb, is_available}
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpus.append({
                "name": name,
                "total_vram_mb": mem.total // (1024**2),
                "used_vram_mb": mem.used  // (1024**2),
                "is_available": True     # on pourra affiner la logique
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception as exc:
        print(f"[GPU Utils] Erreur : {exc}")
        return []
