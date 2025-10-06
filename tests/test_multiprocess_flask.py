import multiprocessing, time, requests, os
from core.network.supervision_api import app

# Test multiprocess: démarrage dans un process isolé et requêtes HTTP réelles

def run():  # pragma: no cover
    port = int(os.environ.get('VRM_TEST_PORT','5123'))
    app.run(port=port)

def test_multiprocess_server():
    port = 5123
    p = multiprocessing.Process(target=run, daemon=True)
    p.start()
    time.sleep(1.5)
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    base = f'http://127.0.0.1:{port}'
    r = requests.get(base + '/api/health')
    assert r.status_code == 200
    r2 = requests.get(base + '/api/fastpath/capabilities')
    assert r2.status_code == 200
    p.terminate()
    p.join(timeout=2)
