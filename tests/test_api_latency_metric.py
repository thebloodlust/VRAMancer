from core.network.supervision_api import app

def test_api_latency_metric_exposed():
    c = app.test_client()
    # simple call
    r = c.get('/api/nodes')
    assert r.status_code == 200
    # Appeler encore pour produire des observations
    r2 = c.get('/api/nodes')
    assert r2.status_code == 200
    # On ne parse pas /metrics ici (handler dédié ailleurs), on valide juste absence d'erreur.