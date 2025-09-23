# core/memory_monitor.py

import psutil

def get_ram_status() -> tuple[int, int]:
    """
    Retourne la RAM disponible et totale en octets.
    """
    mem = psutil.virtual_memory()
    return mem.available, mem.total

def is_ram_saturated(threshold_gb: int = 2) -> bool:
    """
    Indique si la RAM disponible est inférieure au seuil (en Go).
    """
    available, _ = get_ram_status()
    return available < threshold_gb * 1024**3
