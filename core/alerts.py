"""M4 — alertes webhook (Telegram / Discord / Slack / générique), stdlib only.

Configurer : `export VRM_ALERT_WEBHOOK=https://...` (et `VRM_ALERT_CHAT_ID` pour Telegram).
Le format du payload est auto-détecté depuis l'URL.

    from core.alerts import notify
    notify("GPU0 à 95% VRAM, évacuation KV en cours", level="warn")
"""
from __future__ import annotations
import json
import os
import urllib.request
from typing import Any, Dict, Tuple

_PREFIX = {"info": "ℹ️", "warn": "⚠️", "error": "🚨"}


def webhook_url() -> str:
    return os.environ.get("VRM_ALERT_WEBHOOK", "").strip()


def is_configured() -> bool:
    return bool(webhook_url())


def _format(url: str, message: str, level: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    text = f"{_PREFIX.get(level, '')} VRAMancer: {message}".strip()
    u = url.lower()
    if "telegram" in u:
        return {"chat_id": os.environ.get("VRM_ALERT_CHAT_ID", ""), "text": text}, {}
    if "discord" in u:
        return {"content": text}, {}
    if "slack" in u or "hooks.slack" in u:
        return {"text": text}, {}
    return {"text": text, "level": level, "message": message}, {}


def notify(message: str, level: str = "info", timeout: float = 5.0) -> Dict[str, Any]:
    """Envoie une alerte au webhook configuré. No-op silencieux si non configuré."""
    url = webhook_url()
    if not url:
        return {"ok": False, "reason": "VRM_ALERT_WEBHOOK non défini"}
    payload, headers = _format(url, message, level)
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            code = getattr(r, "status", r.getcode())
            return {"ok": 200 <= code < 300, "status": code}
    except Exception as e:
        return {"ok": False, "error": str(e)}
