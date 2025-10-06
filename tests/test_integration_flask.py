import os, threading, time, requests
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
from core.network.supervision_api import app

# Test d'intégration : démarrage serveur Flask + requêtes réelles.

def run_app():  # pragma: no cover
    app.run(port=5099)

def test_integration_endpoints():
    t = threading.Thread(target=run_app, daemon=True)
    t.start()
    # attendre le boot
    time.sleep(1.5)
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    base = "http://127.0.0.1:5099"
    r = requests.get(f"{base}/api/health")
    assert r.status_code == 200
    r2 = requests.get(f"{base}/api/nodes")
    assert r2.status_code == 200 and isinstance(r2.json(), list)
    r3 = requests.get(f"{base}/api/fastpath/capabilities")
    assert r3.status_code == 200
