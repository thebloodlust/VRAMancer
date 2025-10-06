from core.api.unified_api import app, _quota_counters
import os

def test_xai_feature_attrib():
    os.environ['VRM_UNIFIED_API_QUOTA']='0'
    _quota_counters.clear()
    c = app.test_client()
    r = c.post('/api/xai/explain', json={'kind':'feature_attrib','data':{'features':[1.0, -2.0, 3.0]}})
    assert r.status_code == 200
    js = r.get_json()
    assert 'attribution' in js and len(js['attribution']) == 3
