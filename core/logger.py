"""Logger unifié pour VRAMancer.

Fournit :
- configuration centralisée
- niveaux (DEBUG/INFO/WARNING/ERROR)
- format structuré (timestamp + module + niveau)
"""
from __future__ import annotations
import logging
import os
import json
from datetime import datetime
from typing import Optional, Any, Dict

_LOGGER: Optional[logging.Logger] = None

def get_logger(name: str = "vramancer", level: str = None) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER.getChild(name)

    base = logging.getLogger("vramancer")
    lvl = (level or os.environ.get("VRAMANCER_LOG", "INFO")).upper()
    base.setLevel(getattr(logging, lvl, logging.INFO))
    handler = None
    if os.environ.get("VRM_LOG_JSON", "0") in {"1", "true", "TRUE"}:
        class JsonFormatter(logging.Formatter):  # pragma: no cover
            def format(self, record: logging.LogRecord) -> str:
                payload: Dict[str, Any] = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "lvl": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler
            handler = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        except ImportError:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            
    if handler and not base.handlers:
        base.addHandler(handler)
    base.propagate = False
    _LOGGER = base
    return base.getChild(name)

class LoggerAdapter:
    def __init__(self, component: str, level: str = None):
        self._log = get_logger(component, level)

    def debug(self, msg: str, *a, **kw):
        self._log.debug(msg, *a, **kw)
    def info(self, msg: str, *a, **kw):
        self._log.info(msg, *a, **kw)
    def warning(self, msg: str, *a, **kw):
        self._log.warning(msg, *a, **kw)
    def error(self, msg: str, *a, **kw):
        self._log.error(msg, *a, **kw)
    def exception(self, msg: str, *a, **kw):
        self._log.exception(msg, *a, **kw)

__all__ = ["get_logger", "LoggerAdapter"]
