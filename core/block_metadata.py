"""Metadonnees de blocs pour le scheduler VRAMancer.

Fournit des metadonnees statiques et dynamiques pour chaque bloc
d'un modele (taille, importance, type de couche, etc.).
"""
from __future__ import annotations
from typing import Any


# Metadonnees par defaut par type de couche
_LAYER_DEFAULTS: dict[str, dict[str, Any]] = {
    "embedding": {"estimated_size_mb": 200, "importance": "critical", "compute_intensity": "low"},
    "attention":  {"estimated_size_mb": 800, "importance": "critical", "compute_intensity": "high"},
    "mlp":        {"estimated_size_mb": 600, "importance": "normal",   "compute_intensity": "high"},
    "layernorm":  {"estimated_size_mb": 1,   "importance": "normal",   "compute_intensity": "low"},
    "lm_head":    {"estimated_size_mb": 200, "importance": "critical", "compute_intensity": "medium"},
    "unknown":    {"estimated_size_mb": 500, "importance": "normal",   "compute_intensity": "medium"},
}

# Cache dynamique : peut etre mis a jour par le profiler
_dynamic_metadata: dict[int, dict[str, Any]] = {}


def get_block_metadata(index: int, layer_type: str = "unknown") -> dict[str, Any]:
    """Retourne les metadonnees d'un bloc.

    Cherche d'abord dans le cache dynamique, puis utilise les defaults
    bases sur le type de couche.

    Parameters
    ----------
    index : int
        Index du bloc dans le modele.
    layer_type : str
        Type de couche (embedding, attention, mlp, layernorm, lm_head).
    """
    if index in _dynamic_metadata:
        return _dynamic_metadata[index]
    base = _LAYER_DEFAULTS.get(layer_type, _LAYER_DEFAULTS["unknown"]).copy()
    base["index"] = index
    base["layer_type"] = layer_type
    return base


def set_block_metadata(index: int, metadata: dict[str, Any]) -> None:
    """Met a jour les metadonnees dynamiques d'un bloc (via profiler)."""
    _dynamic_metadata[index] = metadata


def clear_metadata_cache() -> None:
    """Efface le cache dynamique."""
    _dynamic_metadata.clear()


def get_all_metadata() -> dict[int, dict[str, Any]]:
    """Retourne toutes les metadonnees dynamiques."""
    return dict(_dynamic_metadata)
