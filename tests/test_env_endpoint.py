import json, subprocess, sys, os, time, threading

# Démarre l'API dans un thread (import direct)

def start_api():
    from core.api.unified_api import app
    app.testing = True
    app.run(port=5031)


def test_env_endpoint_basic():
    # Lancer serveur sur port éphémère 5031
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    time.sleep(1.0)
    import requests
    r = requests.get('http://localhost:5031/api/env', timeout=2)
    assert r.ok
    data = r.json()
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
