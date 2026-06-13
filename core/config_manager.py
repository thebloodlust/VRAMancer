"""Centralized configuration manager for VRAMancer.

Sources priority (highest first):
  1. Environment variables (``VRM_*``)
  2. ``config.yaml`` (resolved via core.config)
  3. Hardcoded defaults
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


def _yaml_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_yaml_path() -> Optional[Path]:
    candidates = []
    try:
        from core.config import get_config_path  # type: ignore
        p = get_config_path()
        if p:
            candidates.append(Path(p))
    except Exception:
        _logger.debug("Config path env lookup failed", exc_info=True)
    candidates.append(Path.cwd() / "config.yaml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return default


@dataclass
class Config:
    production: bool = False
    minimal_test: bool = False
    debug: bool = False
    backend: str = "auto"
    model: str = "gpt2"
    quantization: str = ""
    parallel_mode: str = "pp"
    trust_remote_code: bool = False
    continuous_batching: bool = False
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "vramancer")
    data_dir: Path = field(default_factory=Path.cwd)
    api_token: Optional[str] = None
    auth_secret: Optional[str] = None


_LOCK = threading.RLock()
_CONFIG: Optional[Config] = None


def _build() -> Config:
    yaml_path = _resolve_yaml_path()
    yaml_data: Dict[str, Any] = _yaml_load(yaml_path) if yaml_path else {}

    def pick(env_key: str, yaml_key: str, default: Any) -> Any:
        if env_key in os.environ:
            return os.environ[env_key]
        if yaml_key in yaml_data:
            return yaml_data[yaml_key]
        return default

    return Config(
        production=_bool(pick("VRM_PRODUCTION", "production", False)),
        minimal_test=_bool(pick("VRM_MINIMAL_TEST", "minimal_test", False)),
        debug=_bool(pick("VRM_DEBUG", "debug", False)),
        backend=str(pick("VRM_BACKEND", "backend", "auto")),
        model=str(pick("VRM_MODEL", "model", "gpt2")),
        quantization=str(pick("VRM_QUANTIZATION", "quantization", "")).lower(),
        parallel_mode=str(pick("VRM_PARALLEL_MODE", "parallel_mode", "pp")).lower(),
        trust_remote_code=_bool(pick("VRM_TRUST_REMOTE_CODE", "trust_remote_code", False)),
        continuous_batching=_bool(pick("VRM_CONTINUOUS_BATCHING", "continuous_batching", False)),
        cache_dir=Path(str(pick("VRM_CACHE_DIR", "cache_dir",
                                str(Path.home() / ".cache" / "vramancer")))),
        data_dir=Path(str(pick("VRM_DATA_DIR", "data_dir", str(Path.cwd())))),
        api_token=os.environ.get("VRM_API_TOKEN") or yaml_data.get("api_token"),
        auth_secret=os.environ.get("VRM_AUTH_SECRET") or yaml_data.get("auth_secret"),
    )


def get_config() -> Config:
    global _CONFIG
    with _LOCK:
        if _CONFIG is None:
            _CONFIG = _build()
        return _CONFIG


def reload_config() -> Config:
    global _CONFIG
    with _LOCK:
        _CONFIG = _build()
        return _CONFIG


__all__ = ["Config", "get_config", "reload_config"]
