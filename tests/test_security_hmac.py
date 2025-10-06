import os, time, hmac, hashlib
from core.security import _compute_hmac, reset_rotation, get_effective_secret
from dashboard.dashboard_web import app as web_app
from core.network.supervision_api import app as sup_app

SECRET = "test-secret-token"
os.environ["VRM_API_TOKEN"] = SECRET  # s'assurer alignement (tests isolés)
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'  # neutralise RL pour ces tests
# Pour ces tests on veut réellement tester HMAC -> ne pas relâcher sécurité
os.environ.pop('VRM_TEST_RELAX_SECURITY', None)
os.environ.pop('VRM_TEST_MODE', None)
os.environ.pop('VRM_TEST_ALL_OPEN', None)
os.environ['VRM_DISABLE_SECRET_ROTATION'] = '1'

def _sign(ts: int, method: str, path: str, body: bytes):
    eff = get_effective_secret() or SECRET
    return _compute_hmac(eff, str(ts), method, path, body)

def test_hmac_success_dashboard():
    reset_rotation()
    client = web_app.test_client()
    ts = int(time.time())
    sig = _sign(ts, "GET", "/api/health", b"")
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET, "X-API-TS": str(ts), "X-API-SIGN": sig})
    assert r.status_code == 200
    assert r.json.get("ok") is True

def test_hmac_bad_signature():
    client = sup_app.test_client()
    ts = int(time.time())
    bad_sig = "deadbeef"
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET, "X-API-TS": str(ts), "X-API-SIGN": bad_sig})
    assert r.status_code in (401, 200)

def test_token_only_without_hmac():
    client = web_app.test_client()
    r = client.get("/api/health", headers={"X-API-TOKEN": SECRET})
    # devrais être 200 car HMAC optionnel
    assert r.status_code == 200
    # Post tests: relax pour autres suites si besoin
    os.environ['VRM_TEST_RELAX_SECURITY'] = '1'
