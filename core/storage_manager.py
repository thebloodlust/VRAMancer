# core/storage_manager.py

import torch
import os

def load_block_from_disk(path: str) -> torch.nn.Module:
    """
    Charge un bloc depuis le disque avec fallback CPU.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bloc introuvable : {path}")
    return torch.load(path, map_location="cpu")

def save_block_to_disk(block: torch.nn.Module, path: str) -> None:
    """
    Sauvegarde un bloc sur le disque.
    """
    torch.save(block, path)
