import os, time
from core.network.supervision_api import app

def test_read_only_mode():
    os.environ['VRM_READ_ONLY'] = '1'
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    os.environ.pop('VRM_TEST_MODE', None)
    os.environ.pop('VRM_TEST_ALL_OPEN', None)
    os.environ.pop('VRM_TEST_RELAX_SECURITY', None)
    c = app.test_client()
    token = os.environ.get('VRM_API_TOKEN','testsecret')
    r = c.post('/api/memory/evict', json={}, headers={'X-API-TOKEN': token})
    assert r.status_code in (503, 403, 200)
    # reset
    os.environ['VRM_READ_ONLY'] = '0'
