#!/usr/bin/env python
"""Détection manuelle de l'URL API VRAMancer.

Usage:
  python scripts/api_autodetect.py                # scan ports connus
  VRM_API_PORT=5010 python scripts/api_autodetect.py
  python scripts/api_autodetect.py --json

Algorithme:
  1. Si VRM_API_BASE défini -> validation directe (/api/health)
  2. Sinon tente successivement (avec HEAD ou GET):
     - http://localhost:${VRM_API_PORT or 5030}
     - http://127.0.0.1:${VRM_API_PORT or 5030}
     - http://localhost:5030
     - http://127.0.0.1:5030
     - http://localhost:5010
     - http://127.0.0.1:5010
  3. Retourne la première base répondant status 200 JSON {status:'ok'} ou `{}`.

Retour code:
  0 si succès (une base trouvée) sinon 2.
"""
from __future__ import annotations
import os, sys, json, time
try:
    import requests  # type: ignore
except Exception:
    print("[ERROR] Le module requests est requis pour ce script.")
    sys.exit(1)

def check(base: str, timeout: float = 0.8):
    url = base.rstrip('/') + '/api/health'
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return True, r.json()
    except Exception:
        return False, None
    return False, None

def main():
    want_json = '--json' in sys.argv
    env_base = os.environ.get('VRM_API_BASE')
    port_env = os.environ.get('VRM_API_PORT', '5030')
    tried = []
    if env_base:
        candidates = [env_base]
    else:
        candidates = [
            f"http://localhost:{port_env}",
            f"http://127.0.0.1:{port_env}",
            "http://localhost:5030",
            "http://127.0.0.1:5030",
            "http://localhost:5010",
            "http://127.0.0.1:5010",
        ]
    for base in candidates:
        ok, data = check(base)
        tried.append({"base": base, "ok": ok, "resp": data})
        if ok:
            if want_json:
                print(json.dumps({"selected": base, "health": data, "tried": tried}, indent=2))
            else:
                print(f"[OK] API détectée: {base} | data={data}")
            return 0
    if want_json:
        print(json.dumps({"selected": None, "tried": tried}, indent=2))
    else:
        print("[FAIL] Aucune API détectée. Lancer: python -m core.api.unified_api")
    return 2

if __name__ == '__main__':
    sys.exit(main())
