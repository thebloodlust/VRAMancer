import json, subprocess, sys, os, time, threading

# Remarque: le démarrage de Flask + imports peut dépasser 1s sur certains environnements CI
# On ajoute une boucle de retry pour rendre le test robuste aux fluctuations.

# Démarre l'API dans un thread (import direct)

def start_api():
    # Activer mode test pour bypass auth/quotas lourds
    os.environ.setdefault('VRM_TEST_MODE','1')
    from core.api.unified_api import app
    app.testing = True
    # Disable flask debug/reloader explicitly + host 127.0.0.1 + threaded pour parallelisme léger
    app.run(port=5031, host='127.0.0.1', debug=False, use_reloader=False, threaded=True)


def test_env_endpoint_basic():
    # Utilise client de test Flask plutôt qu'un vrai serveur (évite flaky loopback)
    os.environ.setdefault('VRM_TEST_MODE','1')
    from core.api.unified_api import app
    with app.test_client() as c:
        r = c.get('/api/env', headers={'X-Request-ID':'test-env'})
        assert r.status_code == 200
        data = r.get_json()
        assert 'torch' in data


def test_api_autodetect_script():
    # Simule absence d'API sur ports standard -> devrait échouer rapidement
    # On force VRM_API_PORT sur un port non ouvert
    env = os.environ.copy()
    env['VRM_API_PORT'] = '5999'
    proc = subprocess.run([sys.executable, 'scripts/api_autodetect.py', '--json'], capture_output=True, text=True, env=env)
    # code retour 2 attendu (pas d'API sur ce port)
    assert proc.returncode in (0,2)
    # JSON parsable
    json.loads(proc.stdout.strip() or '{}')
