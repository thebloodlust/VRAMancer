# core/utils.py
"""
Utility helpers for GPU‑aware inference.
"""

from __future__ import annotations

import torch
from typing import Iterable, Sequence
from transformers import AutoTokenizer
from transformers.utils import is_tokenizers_available


# --------------------------------------------------------------
# 1️⃣  Helpers
# --------------------------------------------------------------
def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load a Hugging‑Face tokenizer.

    Parameters
    ----------
    model_name : str
        The name or path of the model (e.g. "gpt2", "EleutherAI/gpt-j-6B").

    Returns
    -------
    AutoTokenizer
        Instantiated tokenizer ready for use.
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[Tokenizer] Fast tokenizer indisponible ({e}). Fallback slow...")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_device_type(idx: int) -> torch.device:
    """
    Return the proper PyTorch device for the given GPU index.
    Detects CUDA, ROCm, or Apple M‑series.

    Parameters
    ----------
    idx : int
        GPU index.

    Returns
    -------
    torch.device
        The device string (`cuda:{idx}`, `mps:{idx}`, `cpu:{idx}`).
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > idx:
        return torch.device(f"cuda:{idx}")
    elif hasattr(torch, "hip") and torch.hip.device_count() > idx:
        return torch.device(f"hip:{idx}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # M‑series have a single device, ignore idx
        return torch.device("mps")
    else:
        return torch.device(f"cpu:{idx}")


def assign_block_to_device(block: torch.nn.Module, idx: int) -> torch.nn.Module:
    """
    Move a sub‑module (`block`) to the device corresponding to GPU index `idx`.
    The function returns the same module reference, but moved in‑place.

    Parameters
    ----------
    block : torch.nn.Module
        Sub‑module to be moved.
    idx : int
        Target GPU index.

    Returns
    -------
    torch.nn.Module
        The module moved to the desired device.
    """
    device = get_device_type(idx)
    return block.to(device)
