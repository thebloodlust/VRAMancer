from core.api.unified_api import app

def test_request_id_roundtrip():
    c = app.test_client()
    rid = 'req-xyz-123'
    r = c.get('/api/version', headers={'X-Request-ID': rid})
    assert r.status_code == 200
    assert r.headers.get('X-Request-ID') == rid
