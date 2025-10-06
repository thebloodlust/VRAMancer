from core.api.unified_api import app, SECURE_AGG
import os, time, hmac, hashlib

def test_secure_aggregation_mask_cycle():
    # S'assurer qu'on est en mode normal (pas de bypass quota)
    os.environ['VRM_TEST_MODE']='0'
    os.environ.setdefault('VRM_API_TOKEN','testsecret')
    c = app.test_client()
    def sign(method: str, path: str, body: bytes):
        ts = str(int(time.time()))
        base = f"{ts}:{method}:{path}".encode() + body
        sig = hmac.new(os.environ['VRM_API_TOKEN'].encode(), base, hashlib.sha256).hexdigest()
        return ts, sig
    # start round
    r = c.post('/api/federated/round/start')
    assert r.status_code == 200
    rid = r.get_json()['round_id']
    # enable secure agg
    body=b'{"enabled": true}'
    ts,sig=sign('POST','/api/federated/secure', body)
    r2 = c.post('/api/federated/secure', data=body, headers={'Content-Type':'application/json','X-API-TS':ts,'X-API-SIGN':sig})
    assert r2.status_code == 200
    # submit updates
    for v in [1.0, 2.0, 3.0]:
        rs = c.post('/api/federated/round/submit', json={'value': v, 'weight': 1.0})
        assert rs.status_code == 200
    # aggregate
    ra = c.get('/api/federated/round/aggregate')
    assert ra.status_code == 200
    data = ra.get_json()
    assert data['count'] == 3
    # Désactiver secure pour éviter impact autres tests
    body=b'{"enabled": false}'
    ts,sig=sign('POST','/api/federated/secure', body)
    c.post('/api/federated/secure', data=body, headers={'Content-Type':'application/json','X-API-TS':ts,'X-API-SIGN':sig})
    assert rid in SECURE_AGG['masks']
