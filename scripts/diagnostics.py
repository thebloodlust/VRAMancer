"""Script de diagnostic rapide VRAMancer.

Collecte:
 - Version
 - Variables d'env clefs
 - Présence persistence
 - Endpoints critiques santé
 - Métriques sélectionnées (si serveur metrics lancé)
"""
from __future__ import annotations
import os, json, time, socket
from urllib.request import urlopen

CRITICAL_ENV = [
    'VRM_API_PORT','VRM_METRICS_PORT','VRM_HA_REPLICATION','VRM_UNIFIED_API_QUOTA','VRM_READ_ONLY'
]

def fetch(url: str, timeout=1.5):
    try:
        with urlopen(url, timeout=timeout) as r:
            return r.read().decode()[:5000]
    except Exception as e:
        return f"ERR:{e}"

def main():
    from core import __version__
    api_port = int(os.environ.get('VRM_API_PORT','5010'))
    metrics_port = int(os.environ.get('VRM_METRICS_PORT','9108'))
    report = {
        'version': __version__,
        'timestamp': time.time(),
        'env': {k: os.environ.get(k) for k in CRITICAL_ENV},
        'persistence': bool(os.environ.get('VRM_SQLITE_PATH')),
        'health': fetch(f'http://localhost:{api_port}/api/health'),
        'info': fetch(f'http://localhost:{api_port}/api/info'),
        'metrics_sample': fetch(f'http://localhost:{metrics_port}/metrics').split('\n')[:30],
    }
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()