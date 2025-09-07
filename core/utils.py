# core/utils.py
"""
Utility helpers for GPU‑aware inference.
"""

from __future__ import annotations

import torch
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# 1️⃣  Helpers
# ----------------------------------------------------------------------


def get_device_type(idx: int) -> torch.device:
    """
    Return a torch device that corresponds to *idx* in the current
    CUDA/ROCm/MPS environment.

    Parameters
    ----------
    idx : int
        GPU index to probe.  The first GPU (0) is usually a CUDA device,
        the second (1) a ROCm device, the third (2) a MPS device, etc.
        If the index is out of range, ``torch.device('cpu')`` is returned
        and a warning is emitted.

    Returns
    -------
    torch.device
        The inferred device.
    """
    try:
        # 1️⃣  CUDA
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            return torch.device(f"cuda:{idx}")

        # 2️⃣  ROCm
        if getattr(torch, "hip", None) is not None and idx < torch.cuda.device_count():
            return torch.device(f"hip:{idx}")

        # 3️⃣  MPS
        if getattr(torch, "mps", None) is not None and idx == "mps":
            return torch.device("mps")

        # 4️⃣  Fallback
        return torch.device("cpu")

    except Exception as exc:
        raise RuntimeError(f"Could not determine device for index {idx}") from exc


def assign_block_to_device(block: torch.nn.Module, idx: int) -> torch.nn.Module:
    """
    Move *block* to the GPU that matches *idx*.

    Parameters
    ----------
    block : torch.nn.Module
        Any PyTorch module (or ModuleList/Sequential etc.).
    idx : int
        Logical index of the block – we map it to a GPU if available.

    Returns
    -------
    torch.nn.Module
        The block moved to the appropriate device.
    """
    device = get_device_type(idx)
    if device.type == "cpu":
        return block  # nothing to do

    # Deep‑copy the module to avoid side‑effects on the original instance
    new_block = type(block)()
    new_block.load_state_dict(block.state_dict())
    new_block.to(device)
    return new_block
