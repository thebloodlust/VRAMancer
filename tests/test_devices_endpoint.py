from core.network.supervision_api import app
import os
os.environ.setdefault('VRM_API_TOKEN','testtoken')

def test_devices_endpoint():
    client = app.test_client()
    r = client.get('/api/devices', headers={'X-API-TOKEN': os.environ['VRM_API_TOKEN']})
    assert r.status_code == 200
    assert isinstance(r.json, list)
    assert len(r.json) >= 1
