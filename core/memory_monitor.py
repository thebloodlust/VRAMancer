"""Monitoring memoire RAM pour VRAMancer.

Fournit des helpers pour surveiller l'utilisation RAM du systeme
et declencher des alertes ou des evictions quand la memoire est basse.
"""
from __future__ import annotations
import logging
from typing import Optional

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

_log = logging.getLogger("vramancer.memory_monitor")


def get_ram_status() -> tuple[int, int]:
    """Retourne (RAM disponible, RAM totale) en octets."""
    if not _HAS_PSUTIL:
        return (0, 0)
    mem = psutil.virtual_memory()
    return mem.available, mem.total


def is_ram_saturated(threshold_gb: float = 2.0) -> bool:
    """Indique si la RAM disponible est inferieure au seuil (en Go)."""
    available, _ = get_ram_status()
    return available < threshold_gb * 1024 ** 3


def get_ram_percent() -> float:
    """Retourne le pourcentage de RAM utilisee."""
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.virtual_memory().percent


def get_swap_status() -> tuple[int, int]:
    """Retourne (swap utilise, swap total) en octets."""
    if not _HAS_PSUTIL:
        return (0, 0)
    swap = psutil.swap_memory()
    return swap.used, swap.total


def get_process_memory_mb() -> float:
    """Retourne la memoire RSS du process actuel en MB."""
    if not _HAS_PSUTIL:
        return 0.0
    import os
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def check_memory_pressure() -> dict:
    """Retourne un rapport complet de pression memoire.

    Returns
    -------
    dict avec keys: available_gb, total_gb, percent_used, swap_used_gb,
    process_mb, pressure_level (low/medium/high/critical)
    """
    available, total = get_ram_status()
    available_gb = available / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    percent = get_ram_percent()
    swap_used, swap_total = get_swap_status()
    proc_mb = get_process_memory_mb()

    if percent > 95:
        level = "critical"
    elif percent > 85:
        level = "high"
    elif percent > 70:
        level = "medium"
    else:
        level = "low"

    return {
        "available_gb": round(available_gb, 2),
        "total_gb": round(total_gb, 2),
        "percent_used": round(percent, 1),
        "swap_used_gb": round(swap_used / (1024 ** 3), 2),
        "process_mb": round(proc_mb, 1),
        "pressure_level": level,
    }
