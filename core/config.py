"""Configuration centrale VRAMancer.

Ordre de résolution :
1. Variables d'environnement (préfixe VRM_)
2. config.yaml à la racine (facultatif)
3. Valeurs par défaut ci-dessous
"""
from __future__ import annotations
import os
import yaml
from pathlib import Path
from functools import lru_cache

DEFAULTS = {
    "backend": "auto",
    "model": "gpt2",
    "num_gpus": None,
    "net_mode": "auto",
    "log_level": "INFO",
}

CONFIG_PATHS = [Path("config.yaml"), Path("./release_bundle/config.yaml")]

def _load_yaml() -> dict:
    for p in CONFIG_PATHS:
        if p.exists():
            try:
                with p.open("r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}

def _env_overrides() -> dict:
    overrides = {}
    prefix = "VRM_"
    for k in DEFAULTS.keys():
        env_key = prefix + k.upper()
        if env_key in os.environ:
            overrides[k] = os.environ[env_key]
    return overrides

@lru_cache
def get_config() -> dict:
    cfg = DEFAULTS.copy()
    yaml_cfg = _load_yaml()
    cfg.update({k: v for k, v in yaml_cfg.items() if v is not None})
    cfg.update(_env_overrides())
    return cfg

__all__ = ["get_config"]