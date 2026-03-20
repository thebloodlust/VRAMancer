"""Persistence légère (SQLite) pour workflows.

Activation via variable d'env:
  VRM_SQLITE_PATH=state.db

API minimaliste:
  save_workflow(dict)
  load_workflow(id)
  list_workflows()

Objectif: fournir un socle simple sans engager une ORM lourde.
"""
from __future__ import annotations
import os, json, sqlite3, threading
from typing import Any, Dict, List

_lock = threading.Lock()
_DB_PATH = os.environ.get("VRM_SQLITE_PATH")

def _conn():  # pragma: no cover - I/O simple
    if not _DB_PATH:
        raise RuntimeError("VRM_SQLITE_PATH non défini")
    c = sqlite3.connect(_DB_PATH)
    return c

def _ensure():  # pragma: no cover
    if not _DB_PATH:
        return
    with _lock:
        with _conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, data TEXT)")

def persistence_enabled() -> bool:
    return bool(_DB_PATH)

def save_workflow(wf: Dict[str, Any]):  # pragma: no cover
    if not persistence_enabled():
        return
    _ensure()
    with _lock:
        with _conn() as c:
            c.execute("REPLACE INTO workflows(id,data) VALUES(?,?)", (wf['id'], json.dumps(wf)))

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
]