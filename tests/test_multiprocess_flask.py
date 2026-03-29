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
    os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'
    base = f'http://127.0.0.1:{port}'
    # Wait for server to be ready with retries
    for _ in range(20):
        try:
            requests.get(base + '/api/health', timeout=0.5)
            break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    r = requests.get(base + '/api/health')
    assert r.status_code == 200
    r2 = requests.get(base + '/api/fastpath/capabilities')
    assert r2.status_code == 200
    p.terminate()
    p.join(timeout=2)
    if p.is_alive():
        p.kill()
        p.join(timeout=1)
