# core/__init__.py
"""
Package core – expose les composants principaux.

Ce fichier ne doit pas importer toute la logique (cela augmente le temps de chargement
et crée un cycle d’import).  Nous n’exposons que les API publiques
qui seront utilisées à l’extérieur du répertoire `core`.
"""

# API publique
from .utils import get_device_type, assign_block_to_device
from .monitor import GPUMonitor
from .scheduler import SimpleScheduler

__all__ = [
    "get_device_type",
    "assign_block_to_device",
    "GPUMonitor",
    "SimpleScheduler",
]
