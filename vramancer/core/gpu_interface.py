# DEPRECATED: Ce fichier est un doublon obsolète.
# Le vrai code vit dans core/gpu_interface.py — ce fichier sera supprimé.
# Re-export pour compatibilité.
try:
    from core.gpu_interface import get_available_gpus, print_gpu_summary  # noqa: F401
except ImportError:
    def get_available_gpus(): return []  # pragma: no cover
    def print_gpu_summary(): pass  # pragma: no cover

_DEPRECATED = True

