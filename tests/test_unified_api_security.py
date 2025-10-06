import os, time, hmac, hashlib, json
from flask.testing import FlaskClient

from core.api import unified_api as uni

def teardown_function(_fn):
    # reset simple quota counters
    if hasattr(uni, '_quota_counters'):
        try:
            uni._quota_counters.clear()  # type: ignore
        except Exception:
            pass

def _sign(secret: str, method: str, path: str, body: bytes):
    ts = str(int(time.time()))
    base = f"{ts}:{method}:{path}".encode() + body
    sig = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return ts, sig

def setup_module(module):  # test isolation
    os.environ['VRM_API_TOKEN'] = 'testsecret'
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    os.environ['VRM_READ_ONLY'] = '0'
    os.environ['VRM_UNIFIED_API_QUOTA'] = '5'
    os.environ['VRM_TEST_MODE'] = '0'

def test_federated_weighted_and_quota():
    app = uni.app
    client: FlaskClient = app.test_client()
    # start round
    body = b"{}"
    ts, sig = _sign('testsecret','POST','/api/federated/round/start', body)
    r = client.post('/api/federated/round/start', data=body, headers={'X-API-TS':ts,'X-API-SIGN':sig,'Content-Type':'application/json'})
    assert r.status_code == 200
    # submit weighted
    for v,w in [(1,1),(3,2)]:
        data = json.dumps({'value': v, 'weight': w}).encode()
        ts, sig = _sign('testsecret','POST','/api/federated/round/submit', data)
        r = client.post('/api/federated/round/submit', data=data, headers={'X-API-TS':ts,'X-API-SIGN':sig,'Content-Type':'application/json'})
        assert r.status_code == 200
    # aggregate
    ts, sig = _sign('testsecret','GET','/api/federated/round/aggregate', b"")
    r = client.get('/api/federated/round/aggregate', headers={'X-API-TS':ts,'X-API-SIGN':sig})
    assert r.status_code == 200
    agg = r.get_json()['aggregate']
    # weighted mean: (1*1 + 3*2)/(1+2)=7/3
    assert abs(agg - (7/3)) < 1e-6
    # Quota=5 -> déjà 4 consommés. 5e ok, 6e doit être 429
    ts, sig = _sign('testsecret','GET','/api/info', b"")
    r = client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig})  # 5e
    assert r.status_code == 200
    ts, sig = _sign('testsecret','GET','/api/info', b"")
    r = client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig})  # 6e
    assert r.status_code == 429

def test_twin_state_endpoint():
    app = uni.app
    client = app.test_client()
    ts, sig = _sign('testsecret','GET','/api/twin/state', b"")
    r = client.get('/api/twin/state', headers={'X-API-TS':ts,'X-API-SIGN':sig})
    assert r.status_code == 200
    data = r.get_json()
    assert 'cluster_state' in data
    assert 'history_len' in data

def test_read_only_blocks_mutations(monkeypatch):
    monkeypatch.setenv('VRM_READ_ONLY','1')
    app = uni.app
    client = app.test_client()
    payload = b'{"tasks": []}'
    ts, sig = _sign('testsecret','POST','/api/workflows', payload)
    r = client.post('/api/workflows', data=payload, headers={'X-API-TS':ts,'X-API-SIGN':sig,'Content-Type':'application/json'})
    assert r.status_code == 503
    monkeypatch.setenv('VRM_READ_ONLY','0')

def test_quota_reset():
    from core.api import unified_api as uni
    app = uni.app
    client = app.test_client()
    os.environ['VRM_UNIFIED_API_QUOTA']='2'
    # 1ère requête ok
    ts,sig=_sign('testsecret','GET','/api/info',b'')
    assert client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig}).status_code==200
    # 2e ok
    ts,sig=_sign('testsecret','GET','/api/info',b'')
    assert client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig}).status_code==200
    # 3e dépasse quota
    ts,sig=_sign('testsecret','GET','/api/info',b'')
    assert client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig}).status_code==429
    # reset
    ts,sig=_sign('testsecret','POST','/api/quota/reset',b'')
    assert client.post('/api/quota/reset', headers={'X-API-TS':ts,'X-API-SIGN':sig}).status_code==200
    # quota reparti
    ts,sig=_sign('testsecret','GET','/api/info',b'')
    assert client.get('/api/info', headers={'X-API-TS':ts,'X-API-SIGN':sig}).status_code==200