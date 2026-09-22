"""Détection des GPU AMD par sysfs amdgpu — sans ROCm, sans torch-hip.

Pourquoi ce module existe : `pynvml` et `torch.cuda` ne voient QUE les cartes
NVIDIA. Sur une machine AMD, tout l'outillage VRAMancer (dashboard, `doctor`,
inventaire hétérogène) annonçait donc « aucun GPU détecté » alors qu'une RX 7900 XT
faisait tourner un 35B à 37.8 tok/s sous Vulkan (mesuré le 2026-09-22).

Ce module lit ce que le noyau expose déjà — les mêmes compteurs que `radeontop` :
VRAM totale/utilisée, occupation, température, puissance. Il ne dépend de RIEN
(stdlib seulement) et renvoie une liste vide sur une machine sans carte AMD.

Limites assumées : sysfs ne donne pas le nom commercial (résolu via la base
pci.ids du système quand elle est présente), et ces chiffres décrivent la carte,
pas ce que VRAMancer peut en faire — aucun backend de calcul n'est impliqué ici.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

_PCI_IDS_PATHS = ("/usr/share/misc/pci.ids", "/usr/share/hwdata/pci.ids")
AMD_VENDOR_ID = "0x1002"


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _read_int(path: str) -> Optional[int]:
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def pci_device_name(pci_id: str) -> Optional[str]:
    """Nom commercial depuis la base pci.ids système ('1002:744C' -> 'Navi 31 […]')."""
    if not pci_id or ":" not in pci_id:
        return None
    vendor, device = (x.lower() for x in pci_id.split(":", 1))
    for db in _PCI_IDS_PATHS:
        try:
            in_vendor = False
            with open(db, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    if not line.startswith("\t"):
                        in_vendor = line.split()[0].lower() == vendor
                    elif in_vendor and not line.startswith("\t\t"):
                        parts = line.strip().split(None, 1)
                        if parts and parts[0].lower() == device:
                            return parts[1].strip() if len(parts) > 1 else None
        except Exception:
            continue
    return None


def amd_gpus() -> List[Dict[str, Any]]:
    """Cartes AMD vues par le noyau. Liste vide si aucune (jamais d'exception)."""
    out: List[Dict[str, Any]] = []
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        if _read(os.path.join(dev, "vendor")) != AMD_VENDOR_ID:
            continue
        total = _read_int(os.path.join(dev, "mem_info_vram_total"))
        if total is None:                      # amdgpu absent/non chargé
            continue
        used = _read_int(os.path.join(dev, "mem_info_vram_used")) or 0
        pci_id = ""
        uevent = _read(os.path.join(dev, "uevent")) or ""
        for line in uevent.splitlines():
            if line.startswith("PCI_ID="):
                pci_id = line.split("=", 1)[1]
        info: Dict[str, Any] = {
            "index": len(out),
            "name": pci_device_name(pci_id) or f"AMD GPU ({pci_id or 'amdgpu'})",
            "pci_id": pci_id,
            "backend": "amdgpu",
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": max(0, total - used),
            "busy_percent": _read_int(os.path.join(dev, "gpu_busy_percent")),
            "runtime_status": _read(os.path.join(dev, "power", "runtime_status")),
        }
        # Capteurs (hwmon) — absents sur certaines cartes/VM : tout est optionnel.
        for hw in sorted(glob.glob(os.path.join(dev, "hwmon", "hwmon*"))):
            t = _read_int(os.path.join(hw, "temp1_input"))
            p = _read_int(os.path.join(hw, "power1_average"))
            if t is not None:
                info["temp_c"] = round(t / 1000)
            if p is not None:
                info["power_w"] = round(p / 1_000_000)
            break
        out.append(info)
    return out


def has_amd_gpu() -> bool:
    """Au moins une carte AMD pilotée par amdgpu."""
    return bool(amd_gpus())


__all__ = ["amd_gpus", "has_amd_gpu", "pci_device_name", "AMD_VENDOR_ID"]
