#!/usr/bin/env python3
"""M3+M4 — tests historique SQLite + format alertes (déterministe, sans réseau)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DB temporaire AVANT import du module
_TMP = tempfile.mkdtemp(prefix="vrm_hist_")
os.environ["VRM_HISTORY_DB"] = os.path.join(_TMP, "h.db")
os.environ["VRM_HISTORY_MAX"] = "5"


def test_history_record_recent_stats():
    import core.request_history as h
    for i in range(3):
        h.record(model="m", prompt_tokens=100, generated_tokens=50, duration_ms=1000, status="ok")
    h.record(model="m", prompt_tokens=10, generated_tokens=0, duration_ms=200, status="oom")
    r = h.recent(10)
    assert len(r) == 4
    assert r[0]["status"] == "oom"          # plus récent en tête
    assert abs(r[1]["tok_s"] - 50.0) < 0.1  # 50 tok / 1s
    st = h.stats()
    assert st["count_ok"] == 3 and st["count_error"] == 1
    assert abs(st["avg_tok_s"] - 50.0) < 0.1


def test_history_prune():
    import core.request_history as h
    for i in range(10):
        h.record(model="m", generated_tokens=1, duration_ms=100)
    assert len(h.recent(100)) <= 5  # élagué à VRM_HISTORY_MAX=5


def test_alert_format():
    import core.alerts as a
    p, _ = a._format("https://api.telegram.org/botX/sendMessage", "hi", "warn")
    assert "text" in p and p["text"].startswith("⚠️")
    p, _ = a._format("https://discord.com/api/webhooks/1/2", "hi", "error")
    assert p["content"].startswith("🚨")
    p, _ = a._format("https://hooks.slack.com/x", "hi", "info")
    assert "text" in p
    p, _ = a._format("https://example.com/hook", "hi", "info")
    assert p["level"] == "info" and p["message"] == "hi"


def test_alert_noop_when_unconfigured():
    import core.alerts as a
    os.environ.pop("VRM_ALERT_WEBHOOK", None)
    res = a.notify("test")
    assert res["ok"] is False and "non défini" in res["reason"]


def _run():
    fails = 0
    for fn in (test_history_record_recent_stats, test_history_prune,
               test_alert_format, test_alert_noop_when_unconfigured):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print("TOUS OK" if not fails else f"{fails} ÉCHECS")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
