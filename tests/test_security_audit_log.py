"""Tests for core/security/audit_log.py — persistent audit trail."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import pytest

from core.security import audit_log as al


@pytest.fixture
def fresh_log(tmp_path, monkeypatch):
    monkeypatch.setenv("VRM_AUDIT_LOG", "1")
    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("VRM_AUDIT_LOG_PATH", str(db_path))
    al.reset_audit_log()
    log = al.get_audit_log()
    assert log is not None
    yield log, db_path
    al.reset_audit_log()


def _rec(log, **overrides):
    payload = dict(
        method="GET", path="/x", status=200, ip="127.0.0.1",
        user_agent="pytest", token=None, role="anon", result="ok",
    )
    payload.update(overrides)
    log.record(**payload)


def _rows(db_path: Path):
    with sqlite3.connect(str(db_path)) as conn:
        return list(conn.execute(
            "SELECT method, path, status, token_hash, role, result, ip FROM audit_log"
        ))


def test_audit_log_records_basic_request(fresh_log):
    log, db_path = fresh_log
    _rec(log, method="GET", path="/health", status=200, ip="1.2.3.4")
    rows = _rows(db_path)
    assert len(rows) == 1
    assert rows[0][0] == "GET"
    assert rows[0][1] == "/health"
    assert rows[0][2] == 200
    assert rows[0][5] == "ok"


def test_audit_log_hashes_token_never_stores_raw(fresh_log):
    log, db_path = fresh_log
    raw = "supersecret-token-xyz"
    _rec(log, method="POST", path="/v1/chat", token=raw, role="user")
    rows = _rows(db_path)
    assert len(rows) == 1
    stored = rows[0][3]
    assert stored is not None
    assert raw not in stored
    expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
    assert stored == expected
    assert len(stored) == 16


def test_audit_log_record_never_raises_on_bad_input(fresh_log):
    log, _ = fresh_log
    log.record(
        method="GET", path="/x", status=200, ip="1.1.1.1",
        user_agent="x", token=object(), role="anon", result="ok",  # type: ignore[arg-type]
    )


def test_audit_log_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("VRM_AUDIT_LOG", raising=False)
    monkeypatch.delenv("VRM_AUDIT_LOG_PATH", raising=False)
    al.reset_audit_log()
    log = al.get_audit_log()
    assert log is None
    al.reset_audit_log()


def test_audit_log_threadsafe_concurrent_writes(fresh_log):
    log, db_path = fresh_log
    N = 50
    THREADS = 8

    def worker(tid):
        for i in range(N):
            _rec(log, path=f"/t{tid}/{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = _rows(db_path)
    assert len(rows) == N * THREADS


def test_audit_log_indexes_present(fresh_log):
    log, db_path = fresh_log
    _rec(log)
    with sqlite3.connect(str(db_path)) as conn:
        idx = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )}
    assert any("ts" in i for i in idx), idx
    assert any("ip" in i for i in idx), idx
    assert any("result" in i for i in idx), idx
