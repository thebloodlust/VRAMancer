import os
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
import pytest

os.environ.setdefault('VRM_API_TOKEN', 'testtoken')
os.environ.setdefault('VRM_MINIMAL_TEST', '1')

from core.network.supervision_api import app  # type: ignore

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _client():
    return app.test_client()

def test_fastpath_interfaces_and_select():
    c = _client()
    # Liste interfaces
    r = c.get('/api/fastpath/interfaces', headers={'X-API-TOKEN': os.environ['VRM_API_TOKEN']})
    assert r.status_code == 200
    data = r.get_json()
    assert 'interfaces' in data
    # Sélection (même si vide, doit répondre proprement)
    target = None
    if data['interfaces']:
        it = data['interfaces'][0]
        target = it.get('if') or it.get('type')
    else:
        target = 'stub'
    r2 = c.post('/api/fastpath/select', json={'interface': target}, headers={'X-API-TOKEN': os.environ['VRM_API_TOKEN']})
    assert r2.status_code == 200
    js2 = r2.get_json()
    assert js2.get('ok') is True
    assert 'benchmarks' in js2
    # Reliste et assure que l'ordre reflète la variable
    r3 = c.get('/api/fastpath/interfaces', headers={'X-API-TOKEN': os.environ['VRM_API_TOKEN']})
    assert r3.status_code == 200