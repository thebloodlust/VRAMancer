# core/utils.py
"""
Helper utilities pour la gestion des back‑ends GPU.
"""

import torch
import torch.backends.mps

__all__ = ["get_device_type", "assign_block_to_device"]


def get_device_type(device_index: int) -> str:
    """
    Renvoie le type de GPU pour l’indice donné.

    - "cuda"  → NVIDIA (ou un build ROCm = CUDA)
    - "rocm"  → AMD ROCm (si torch.version.hip est défini)
    - "mps"   → Apple Silicon
    - "cpu"   → aucun GPU visible
    """
    # 1. CUDA / ROCm
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device_index)
            if torch.version.hip:        # build ROCm
                return "rocm" if "AMD" in props.name else "cuda"
            return "cuda"
        except RuntimeError:
            pass

    # 2. Apple MPS
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def assign_block_to_device(block: torch.nn.Module, device_index: int):
    """
    Place un bloc de modèle sur le GPU le plus adapté.
    """
    device_type = get_device_type(device_index)
    if device_type in ("cuda", "rocm"):
        return block.to(f"{device_type}:{device_index}")
    elif device_type == "mps":
        return block.to("mps")
    else:
        # Pas de GPU → CPU
        return block
