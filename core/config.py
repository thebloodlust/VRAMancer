"""Configuration centrale VRAMancer — production-ready.

Ordre de resolution (derniere valeur gagne) :
1. Valeurs par defaut ci-dessous
2. Fichier YAML (config.yaml ou chemin multi-OS)
3. Variables d'environnement (prefixe VRM_)

Multi-OS:
  - Linux   : $XDG_CONFIG_HOME/vramancer/config.yaml  (defaut ~/.config/)
  - macOS   : ~/Library/Application Support/vramancer/config.yaml
  - Windows : %APPDATA%/vramancer/config.yaml

Hot-reload: appeler ``reload_config()`` pour forcer le rechargement.
"""
from __future__ import annotations

import os
import sys
import copy
import platform
import threading
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

_logger = logging.getLogger("vramancer.config")

# -------------------------------------------------------------------------
# Defaults — toutes les cles reconnues par VRAMancer
# -------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # --- inference ---
    "backend":          "auto",        # auto | huggingface | vllm | ollama
    "model":            "gpt2",
    "num_gpus":         None,          # None = auto-detect
    "max_batch_size":   8,
    "max_seq_len":      2048,
    "dtype":            "float16",     # float16 | bfloat16 | float32 | int8 | int4

    # --- reseau ---
    "net_mode":         "auto",        # auto | local | cluster | disabled
    "cluster_port":     55555,
    "api_port":         5000,
    "metrics_port":     9108,
    "discovery_method": "udp",         # udp | mdns | static

    # --- memoire hierarchique ---
    "memory_tiers":     6,             # L1(VRAM)->L6(network)
    "nvme_cache_path":  None,          # None = auto-detect
    "nvme_cache_max_gb": 50,
    "dram_cache_max_gb": None,         # None = 80% dispo

    # --- transport ---
    "transfer_method":  "auto",        # auto | nccl | p2p | cpu_staged
    "rdma_enabled":     False,
    "gpudirect_enabled": False,

    # --- securite ---
    "api_token":        None,          # None = env VRM_API_TOKEN
    "hmac_secret":      None,
    "rate_limit_rps":   100,
    "require_auth":     True,

    # --- observabilite ---
    "log_level":        "INFO",
    "log_json":         False,
    "tracing_enabled":  False,
    "metrics_enabled":  True,

    # --- persistence ---
    "sqlite_path":      None,

    # --- modes speciaux ---
    "read_only":        False,
    "production":       False,
    "minimal_test":     False,
    "test_mode":        False,

    # --- compression ---
    "compression_strategy": "adaptive",  # none | light | adaptive | aggressive
    "quantization":     None,            # None | int8 | int4 | gptq | awq
}

# Mapping env var -> config key (prefix VRM_ + uppercase key)
# Cles booleennes pour conversion auto
_BOOL_KEYS = {
    "rdma_enabled", "gpudirect_enabled", "require_auth",
    "log_json", "tracing_enabled", "metrics_enabled",
    "read_only", "production", "minimal_test", "test_mode",
}

_INT_KEYS = {
    "num_gpus", "max_batch_size", "max_seq_len", "cluster_port",
    "api_port", "metrics_port", "memory_tiers", "nvme_cache_max_gb",
    "rate_limit_rps",
}

_FLOAT_KEYS = {
    "dram_cache_max_gb",
}

# -------------------------------------------------------------------------
# Multi-OS config paths
# -------------------------------------------------------------------------

def _os_config_paths() -> List[Path]:
    """Retourne les chemins de config selon l'OS, du plus specifique au plus general."""
    paths: List[Path] = []
    system = platform.system().lower()

    if system == "linux":
        xdg = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        paths.append(Path(xdg) / "vramancer" / "config.yaml")
    elif system == "darwin":
        paths.append(Path.home() / "Library" / "Application Support" / "vramancer" / "config.yaml")
    elif system == "windows":
        appdata = os.environ.get("APPDATA", os.path.expanduser("~/AppData/Roaming"))
        paths.append(Path(appdata) / "vramancer" / "config.yaml")

    # Toujours chercher dans le repertoire courant et le bundle
    paths.append(Path("config.yaml"))
    paths.append(Path("./release_bundle/config.yaml"))

    return paths


CONFIG_PATHS = _os_config_paths()

# -------------------------------------------------------------------------
# Internal state (thread-safe)
# -------------------------------------------------------------------------
_lock = threading.Lock()
_cached_config: Optional[Dict[str, Any]] = None


def _load_yaml() -> dict:
    """Charge le premier fichier YAML trouve parmi les chemins configures."""
    if yaml is None:
        return {}
    for p in CONFIG_PATHS:
        try:
            exists = p.exists()
        except OSError:
            continue
        if exists:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                _logger.debug("Config YAML chargee depuis %s", p)
                return data
            except Exception as exc:
                _logger.warning("Impossible de charger %s: %s", p, exc)
    return {}


def _coerce(key: str, raw: str) -> Any:
    """Convertit une valeur string (env var) vers le type attendu."""
    if key in _BOOL_KEYS:
        return raw.lower() in ("1", "true", "yes", "on")
    if key in _INT_KEYS:
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    if key in _FLOAT_KEYS:
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    if raw.lower() == "none":
        return None
    return raw


def _env_overrides() -> dict:
    """Lit toutes les VRM_* env vars et les mappe sur les cles de config."""
    overrides: Dict[str, Any] = {}
    prefix = "VRM_"

    for k in DEFAULTS:
        env_key = prefix + k.upper()
        if env_key in os.environ:
            overrides[k] = _coerce(k, os.environ[env_key])

    # Cles speciales avec noms historiques
    _special = {
        "VRM_BACKEND":           "backend",
        "VRM_MODEL":             "model",
        "VRM_API_TOKEN":         "api_token",
        "VRM_LOG_JSON":          "log_json",
        "VRM_TRACING":           "tracing_enabled",
        "VRM_SQLITE_PATH":       "sqlite_path",
        "VRM_READ_ONLY":         "read_only",
        "VRM_PRODUCTION":        "production",
        "VRM_MINIMAL_TEST":      "minimal_test",
        "VRM_TEST_MODE":         "test_mode",
        "VRM_DISABLE_RATE_LIMIT":"rate_limit_rps",
        "VRM_METRICS_PORT":      "metrics_port",
    }
    for env_key, cfg_key in _special.items():
        if env_key in os.environ and cfg_key not in overrides:
            val = os.environ[env_key]
            if env_key == "VRM_DISABLE_RATE_LIMIT" and val in ("1", "true"):
                overrides[cfg_key] = 0  # 0 = disabled
            else:
                overrides[cfg_key] = _coerce(cfg_key, val)

    return overrides


def _validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Valide et normalise les valeurs de configuration."""
    errors: List[str] = []

    # backend
    valid_backends = {"auto", "huggingface", "vllm", "ollama"}
    if cfg["backend"] not in valid_backends:
        errors.append(f"backend={cfg['backend']!r} invalide, attendu: {valid_backends}")

    # dtype
    valid_dtypes = {"float16", "bfloat16", "float32", "int8", "int4", None}
    if cfg["dtype"] not in valid_dtypes:
        errors.append(f"dtype={cfg['dtype']!r} invalide, attendu: {valid_dtypes}")

    # net_mode
    valid_net = {"auto", "local", "cluster", "disabled"}
    if cfg["net_mode"] not in valid_net:
        errors.append(f"net_mode={cfg['net_mode']!r} invalide, attendu: {valid_net}")

    # ports
    for port_key in ("cluster_port", "api_port", "metrics_port"):
        v = cfg.get(port_key)
        if v is not None and (not isinstance(v, int) or v < 1 or v > 65535):
            errors.append(f"{port_key}={v!r} doit etre un entier 1-65535")

    # compression
    valid_comp = {"none", "light", "adaptive", "aggressive"}
    if cfg["compression_strategy"] not in valid_comp:
        errors.append(f"compression_strategy={cfg['compression_strategy']!r} invalide")

    # quantization
    valid_quant = {None, "int8", "int4", "gptq", "awq"}
    if cfg["quantization"] not in valid_quant:
        errors.append(f"quantization={cfg['quantization']!r} invalide")

    if errors:
        for e in errors:
            _logger.error("Config validation: %s", e)
        if cfg.get("production"):
            raise ValueError(
                f"Configuration invalide en mode production: {'; '.join(errors)}"
            )

    return cfg


def _build_config() -> Dict[str, Any]:
    """Construit la configuration complete (defaults -> yaml -> env)."""
    cfg = copy.deepcopy(DEFAULTS)
    yaml_cfg = _load_yaml()
    cfg.update({k: v for k, v in yaml_cfg.items() if v is not None})
    cfg.update(_env_overrides())
    return _validate(cfg)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def get_config() -> Dict[str, Any]:
    """Retourne la configuration courante (cache thread-safe).

    Premier appel: charge defaults -> YAML -> env vars.
    Appels suivants: retourne le cache (appeler reload_config() pour forcer).
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    with _lock:
        if _cached_config is not None:  # double-check locking
            return _cached_config
        _cached_config = _build_config()
    return _cached_config


def reload_config() -> Dict[str, Any]:
    """Force le rechargement complet de la configuration."""
    global _cached_config
    with _lock:
        _cached_config = _build_config()
    _logger.info("Configuration rechargee")
    return _cached_config


def get(key: str, default: Any = None) -> Any:
    """Raccourci: get_config()[key] avec default."""
    return get_config().get(key, default)


def config_path() -> Optional[Path]:
    """Retourne le chemin du premier fichier config trouve, ou None."""
    for p in CONFIG_PATHS:
        if p.exists():
            return p
    return None


__all__ = ["get_config", "reload_config", "get", "config_path", "DEFAULTS"]