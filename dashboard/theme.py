# core/__init__.py
"""
core – Package d’utilitaires de base pour le projet.

Contient :
* utils.py       – gestion des GPU hétérogènes
* tokenizer.py   – wrapper & cache du tokenizer HuggingFace
* monitor.py     – statistiques de mémoire GPU
* scheduler.py   – exécution du modèle bloc‑par‑bloc
"""

# Exporter les symboles les plus couramment utilisés
from .utils import get_device_type, assign_block_to_device
from .tokenizer import get_tokenizer
from .monitor import GPUMonitor
from .scheduler import SimpleScheduler

__all__ = [
    "get_device_type",
    "assign_block_to_device",
    "get_tokenizer",
    "GPUMonitor",
    "SimpleScheduler",
]
