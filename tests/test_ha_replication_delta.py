import os, time, base64, json, zlib, hmac, hashlib
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
from core.ha_replication import _compress_payload, _derive_secret
from core.network.supervision_api import app, HMM

# Test delta + signature (simplifié sans peers réels: on appelle /api/ha/apply directement)

def test_ha_full_then_delta():
    client = app.test_client()
    # Simule un état initial
    state = {'registry': {'A': {'tier':'L1','size':10,'acc':1}}, 'ts': time.time()}
    comp, h = _compress_payload(state)
    secret = 'ha_secret'
    os.environ['VRM_HA_SECRET'] = secret
    os.environ['VRM_API_TOKEN'] = secret  # aligne token API
    os.environ['VRM_TEST_BYPASS_HA'] = '1'
    # Signature dérivée (comme côté replication_tick)
    now = time.time()
    nonce = 'abc123deadbeef'
    derived = _derive_secret(secret, now)
    base = f"{int(now)}:{nonce}:{h}".encode() + comp
    sig = hmac.new(derived.encode(), base, hashlib.sha256).hexdigest()
    meta = {'hash': h,'delta': False,'compressed': True,'algo':'zlib','ts': now,'sig': sig,'nonce': nonce}
    meta_b64 = base64.b64encode(json.dumps(meta).encode()).decode()
    r = client.post('/api/ha/apply', data=comp, headers={'X-HA-META': meta_b64, 'X-API-TOKEN': secret})
    assert r.status_code == 200
    # Ré-envoi delta (identique hash) => payload vide
    now2 = now + 1
    nonce2 = 'abc123deadbeef2'
    derived2 = _derive_secret(secret, now2)
    base2 = f"{int(now2)}:{nonce2}:{h}".encode() + b''
    sig2 = hmac.new(derived2.encode(), base2, hashlib.sha256).hexdigest()
    meta2 = {'hash': h,'delta': True,'compressed': True,'algo':'zlib','ts': now2,'sig': sig2,'nonce': nonce2}
    meta_b64_2 = base64.b64encode(json.dumps(meta2).encode()).decode()
    r2 = client.post('/api/ha/apply', data=b'', headers={'X-HA-META': meta_b64_2, 'X-API-TOKEN': secret})
    assert r2.status_code == 200
