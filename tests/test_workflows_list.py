from core.api.unified_api import app, _quota_counters
import os

def test_workflows_list_empty():
    os.environ['VRM_UNIFIED_API_QUOTA']='0'
    _quota_counters.clear()
    c = app.test_client()
    r = c.get('/api/workflows')
    assert r.status_code == 200
    js = r.get_json()
    assert 'items' in js and isinstance(js['items'], list)

