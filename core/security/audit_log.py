"""Persistent audit log for VRAMancer API.

Writes a row per authenticated (or rejected) request to a SQLite database.
Designed to be cheap (synchronous insert on a single connection, WAL mode,
~30µs per row on local SSD) and crash-safe.

Enabled by setting ``VRM_AUDIT_LOG=1`` (or providing ``VRM_AUDIT_LOG_PATH``).
Disabled by default to keep the test surface clean.

Schema (v1)
-----------
::

    CREATE TABLE audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_utc      REAL    NOT NULL,             -- Unix timestamp
        method      TEXT    NOT NULL,             -- HTTP method
        path        TEXT    NOT NULL,             -- request path
        status      INTEGER,                       -- HTTP status code
        ip          TEXT,                          -- remote_addr
        user_agent  TEXT,
        token_hash  TEXT,                          -- sha256(token)[:16] — never raw
        role        TEXT,                          -- admin|user|readonly|anon
        result      TEXT NOT NULL,                 -- allow|deny|error
        reason      TEXT,                          -- denial reason
        latency_ms  REAL
    );

Token hash, not raw token, is stored. IPs and UAs are stored as-is (compliance
disclaimer: review your DPA before deploying in EU/UK).
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc      REAL    NOT NULL,
    method      TEXT    NOT NULL,
    path        TEXT    NOT NULL,
    status      INTEGER,
    ip          TEXT,
    user_agent  TEXT,
    token_hash  TEXT,
    role        TEXT,
    result      TEXT    NOT NULL,
    reason      TEXT,
    latency_ms  REAL
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts_utc);
CREATE INDEX IF NOT EXISTS idx_audit_ip ON audit_log(ip);
CREATE INDEX IF NOT EXISTS idx_audit_result ON audit_log(result);
"""


class AuditLog:
    """Thread-safe SQLite-backed audit log.

    Single shared connection guarded by a lock — sufficient for the typical
    Flask gunicorn-1-worker-per-GPU deployment. For multi-process, configure
    each worker with its own DB path (``VRM_AUDIT_LOG_PATH=/var/log/vrm/audit-${WORKER_ID}.db``).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        with self._lock:
            self._conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA_V1)

    def record(
        self,
        *,
        method: str,
        path: str,
        status: Optional[int],
        ip: Optional[str],
        user_agent: Optional[str],
        token: Optional[str],
        role: Optional[str],
        result: str,
        reason: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Write one row. Never raises (best-effort)."""
        try:
            token_hash = None
            if token:
                token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
            with self._lock:
                self._conn.execute(
                    "INSERT INTO audit_log "
                    "(ts_utc, method, path, status, ip, user_agent, token_hash, role, result, reason, latency_ms) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        time.time(),
                        method, path[:512], status,
                        ip, (user_agent or "")[:256], token_hash, role,
                        result, reason, latency_ms,
                    ),
                )
        except Exception:
            logger.debug("audit_log insert failed", exc_info=True)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


_singleton_lock = threading.Lock()
_singleton: Optional[AuditLog] = None


def get_audit_log() -> Optional[AuditLog]:
    """Return the process-wide AuditLog instance, or None if disabled.

    Enabled iff ``VRM_AUDIT_LOG=1`` OR ``VRM_AUDIT_LOG_PATH`` is set.
    Default path: ``$VRM_DATA_DIR/audit.db`` or ``~/.vramancer/audit.db``.
    """
    global _singleton
    if _singleton is not None:
        return _singleton

    enabled = os.environ.get("VRM_AUDIT_LOG", "0") in ("1", "true", "yes")
    custom_path = os.environ.get("VRM_AUDIT_LOG_PATH")
    if not enabled and not custom_path:
        return None

    if custom_path:
        path = custom_path
    else:
        data_dir = os.environ.get("VRM_DATA_DIR") or os.path.join(
            os.path.expanduser("~"), ".vramancer"
        )
        path = os.path.join(data_dir, "audit.db")

    with _singleton_lock:
        if _singleton is None:
            try:
                _singleton = AuditLog(path)
                logger.info("audit_log enabled: %s", path)
            except Exception as e:
                logger.warning("audit_log init failed (%s); disabled", e)
                return None
    return _singleton


def reset_audit_log() -> None:
    """For tests: close and forget the singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
            _singleton = None


__all__ = ["AuditLog", "get_audit_log", "reset_audit_log"]
