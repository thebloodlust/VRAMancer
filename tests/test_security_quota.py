import os
import contextlib
from core.api.unified_api import app, API_QUOTA_EXCEEDED  # type: ignore
from core.metrics import counter_value


@contextlib.contextmanager
def temp_env(var: str, value: str | None):
    old = os.environ.get(var)
    if value is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def test_private_endpoint_requires_auth():
    # Forcer désactivation mode test + quota off
    with temp_env('VRM_TEST_MODE', '0'), temp_env('VRM_UNIFIED_API_QUOTA', '0'):
        c = app.test_client()
        # /api/info n'est pas dans public_paths -> doit renvoyer 401 sans Bearer
        r = c.get('/api/info')
        assert r.status_code == 401, f"Attendu 401, reçu {r.status_code} corps={r.data!r}"


def test_quota_enforcement_simple():
    with temp_env('VRM_TEST_MODE', '0'), temp_env('VRM_UNIFIED_API_QUOTA', '1'):
        c = app.test_client()
        before = counter_value(API_QUOTA_EXCEEDED)
        # Première requête (public path) -> OK
        r1 = c.get('/api/version')
        assert r1.status_code == 200
        # Deuxième requête même token implicite 'anonymous' -> quota dépassé
        r2 = c.get('/api/version')
        assert r2.status_code == 429, f"Attendu 429 quota exceeded, reçu {r2.status_code}"
        after = counter_value(API_QUOTA_EXCEEDED)
        assert after == before + 1, f"Compteur quota non incrémenté: before={before} after={after}"
