"""Persistence légère (SQLite) pour workflows.

Activation via variable d'env:
  VRM_SQLITE_PATH=state.db

API minimaliste:
  save_workflow(dict)
  load_workflow(id)
  list_workflows()

Schema versioning:
  Version 1: workflows(id TEXT PK, data TEXT)
  Version 2: + schema_version table, + created_at column

Objectif: fournir un socle simple sans engager une ORM lourde.
"""
from __future__ import annotations
import os, json, sqlite3, threading, time
from typing import Any, Dict, List

_lock = threading.Lock()
_DB_PATH = os.environ.get("VRM_SQLITE_PATH")

CURRENT_SCHEMA_VERSION = 2

def _conn():  # pragma: no cover - I/O simple
    if not _DB_PATH:
        raise RuntimeError("VRM_SQLITE_PATH non défini")
    c = sqlite3.connect(_DB_PATH)
    return c


def _get_schema_version(c) -> int:
    try:
        cur = c.cursor()
        cur.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet — check if workflows exists (legacy v1)
        cur = c.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflows'")
        return 1 if cur.fetchone() else 0


def _apply_migrations(c, current: int):
    if current < 1:
        c.execute("CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, data TEXT)")
        current = 1

    if current < 2:
        c.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY, applied_at REAL)")
        # Add created_at column if missing
        try:
            c.execute("ALTER TABLE workflows ADD COLUMN created_at REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Record both v1 and v2
        c.execute("INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(1, ?)", (time.time(),))
        c.execute("INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(2, ?)", (time.time(),))
        current = 2

    # Future migrations go here:
    # if current < 3: ...


def _ensure():  # pragma: no cover
    if not _DB_PATH:
        return
    with _lock:
        with _conn() as c:
            ver = _get_schema_version(c)
            if ver < CURRENT_SCHEMA_VERSION:
                _apply_migrations(c, ver)


def persistence_enabled() -> bool:
    return bool(_DB_PATH)


def get_schema_version() -> int:
    if not persistence_enabled():
        return 0
    _ensure()
    with _conn() as c:
        return _get_schema_version(c)


def save_workflow(wf: Dict[str, Any]):  # pragma: no cover
    if not persistence_enabled():
        return
    _ensure()
    with _lock:
        with _conn() as c:
            c.execute(
                "REPLACE INTO workflows(id, data, created_at) VALUES(?, ?, ?)",
                (wf['id'], json.dumps(wf), time.time()),
            )

def load_workflow(wid: str) -> Dict[str, Any] | None:  # pragma: no cover
    if not persistence_enabled(): return None
    _ensure()
    with _conn() as c:
        cur = c.cursor()
        cur.execute("SELECT data FROM workflows WHERE id=?", (wid,))
        row = cur.fetchone()
    if not row: return None
    return json.loads(row[0])

def list_workflows(limit:int=100) -> List[Dict[str,Any]]:  # pragma: no cover
    if not persistence_enabled(): return []
    _ensure()
    with _conn() as c:
        cur = c.cursor()
        cur.execute("SELECT data FROM workflows ORDER BY rowid DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
    return [json.loads(r[0]) for r in rows]

__all__ = [
    'persistence_enabled','save_workflow','load_workflow','list_workflows',
    'get_schema_version', 'CURRENT_SCHEMA_VERSION',
]