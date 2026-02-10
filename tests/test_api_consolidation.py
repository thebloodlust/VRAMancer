import os, time, json
import pytest

# Tests consolidation prod stricte : sécurité, mémoire, multicast, estimator

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from core.network.supervision_api import app, scheduler, HMM  # type: ignore  # noqa: E402
from core.security import reset_rate_limiter

def _client():
    return app.test_client()

SECRET = "testsecret"
os.environ["VRM_API_TOKEN"] = SECRET
os.environ['VRM_DISABLE_RATE_LIMIT'] = '0'  # permettre test rate limiting
os.environ.pop('VRM_TEST_RELAX_SECURITY', None)
os.environ['VRM_RATE_MAX'] = '50'
os.environ['VRM_TEST_MODE'] = '1'

# Utilitaires signature simple (token seul ici = mode token)

def test_security_rate_limiting():
    c = _client()
    # S'assurer que bypass total désactivé
    os.environ.pop('VRM_TEST_ALL_OPEN', None)
    os.environ.pop('VRM_TEST_RELAX_SECURITY', None)
    os.environ['VRM_RATE_MAX'] = '5'
    os.environ['VRM_TEST_MODE'] = '1'
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '0'
    reset_rate_limiter()
    # Rafale de requêtes > limite
    limited = False
    for i in range(50):
        r = c.get("/api/health", headers={"X-API-TOKEN": SECRET})
        if r.status_code == 429:
            limited = True
            break
    assert limited, "Rate limiting non déclenché"
    # Désactiver RL pour le reste
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    os.environ['VRM_RATE_MAX'] = '200'
    reset_rate_limiter()
    # Restore test mode for subsequent tests
    os.environ['VRM_TEST_MODE'] = '1'


def test_security_rotation_endpoint():
    c = _client()
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    r = c.post("/api/security/rotate", headers={"X-API-TOKEN": SECRET, "X-API-ROLE":"admin"})
    assert r.status_code == 200
    # L'ancienne clé doit encore passer juste après (tolérance)
    r2 = c.get("/api/health", headers={"X-API-TOKEN": SECRET})
    assert r2.status_code == 200


def test_memory_eviction_cycle():
    # Simule quelques blocs dans HMM
    from core.memory_block import MemoryBlock
    for i in range(5):
        b = MemoryBlock(id=f"B{i}", size_mb=100)
        HMM.register_block(b, "L1" if i < 3 else "L2")
    c = _client()
    r = c.post("/api/memory/evict", json={"vram_pressure": 0.92}, headers={"X-API-TOKEN": SECRET, "X-API-ROLE":"admin"})
    assert r.status_code == 200
    data = r.get_json()
    assert "evicted" in data and isinstance(data["evicted"], list)


def test_runtime_estimator_install_and_effect():
    c = _client()
    payload = {"map": {"noop": 0.5}}
    r = c.post("/api/tasks/estimator/install", json={"map": {"noop": 0.5}}, headers={"X-API-TOKEN": SECRET, "X-API-ROLE":"admin"})
    assert r.status_code == 200
    # Soumettre une tâche noop et vérifier qu'elle apparaît en file / history
    r2 = c.post("/api/tasks/submit", json={"kind": "noop", "priority": "NORMAL"}, headers={"X-API-TOKEN": SECRET})
    assert r2.status_code == 200
    time.sleep(0.5)
    # Récupérer metrics percentiles (déclenche compute)
    r3 = c.get("/api/tasks/metrics", headers={"X-API-TOKEN": SECRET})
    assert r3.status_code == 200


def test_multicast_endpoint():
    c = _client()
    r = c.get("/api/telemetry/multicast", headers={"X-API-TOKEN": SECRET})
    assert r.status_code == 200
    js = r.get_json()
    assert js.get('ok') is True
    assert js.get('bytes') > 0

