"""M3 — historique local léger des requêtes (SQLite, stdlib, zéro dépendance).

Garde les N dernières requêtes (défaut 1000) : tendances tok/s, requêtes OOM/timeout,
dashboard "24h" sans Prometheus externe. Prompt JAMAIS stocké en clair (longueur seule).

    from core.request_history import record, recent, stats
    record(model="Qwen2.5-14B", prompt_tokens=512, generated_tokens=128, duration_ms=2300)
"""
from __future__ import annotations
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_DB = os.environ.get("VRM_HISTORY_DB", os.path.expanduser("~/.vramancer/history.db"))
_MAX = int(os.environ.get("VRM_HISTORY_MAX", "1000"))
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB) or ".", exist_ok=True)
    c = sqlite3.connect(_DB, check_same_thread=False)
    c.execute(
        """CREATE TABLE IF NOT EXISTS requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, model TEXT,
            prompt_tokens INTEGER, generated_tokens INTEGER, duration_ms REAL,
            tok_s REAL, gpu0_vram_mb REAL, gpu1_vram_mb REAL, status TEXT)"""
    )
    return c


def record(model: str, prompt_tokens: int = 0, generated_tokens: int = 0,
           duration_ms: float = 0.0, status: str = "ok",
           vram_mb: Optional[List[float]] = None) -> float:
    """Enregistre une requête, élague à _MAX. Renvoie le tok/s calculé."""
    tok_s = round(generated_tokens / (duration_ms / 1000.0), 2) if duration_ms > 0 else 0.0
    g0 = vram_mb[0] if vram_mb and len(vram_mb) > 0 else None
    g1 = vram_mb[1] if vram_mb and len(vram_mb) > 1 else None
    try:
        with _lock:
            c = _conn()
            c.execute(
                "INSERT INTO requests(ts,model,prompt_tokens,generated_tokens,"
                "duration_ms,tok_s,gpu0_vram_mb,gpu1_vram_mb,status) VALUES(?,?,?,?,?,?,?,?,?)",
                (time.time(), model, prompt_tokens, generated_tokens, duration_ms, tok_s, g0, g1, status),
            )
            c.execute("DELETE FROM requests WHERE id NOT IN "
                      "(SELECT id FROM requests ORDER BY id DESC LIMIT ?)", (_MAX,))
            c.commit(); c.close()
    except Exception:
        pass  # l'historique ne doit jamais casser l'inférence
    return tok_s


_COLS = ["ts", "model", "prompt_tokens", "generated_tokens", "duration_ms", "tok_s", "status"]


def recent(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        with _lock:
            c = _conn()
            rows = c.execute(
                f"SELECT {','.join(_COLS)} FROM requests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            c.close()
        return [dict(zip(_COLS, r)) for r in rows]
    except Exception:
        return []


def stats() -> Dict[str, Any]:
    try:
        with _lock:
            c = _conn()
            ok = c.execute("SELECT COUNT(*),AVG(tok_s),AVG(duration_ms) FROM requests WHERE status='ok'").fetchone()
            bad = c.execute("SELECT COUNT(*) FROM requests WHERE status!='ok'").fetchone()[0]
            c.close()
        return {"count_ok": ok[0] or 0,
                "avg_tok_s": round(ok[1], 2) if ok[1] else 0.0,
                "avg_duration_ms": round(ok[2], 1) if ok[2] else 0.0,
                "count_error": bad}
    except Exception as e:
        return {"error": str(e)}
