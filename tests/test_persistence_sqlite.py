import os, tempfile, json
from core.api import unified_api as uni

def test_persistence_workflow_and_round(tmp_path):
    db = tmp_path / 'state.db'
    os.environ['VRM_SQLITE_PATH'] = str(db)
    # Re-import persistence to pick up env (unified_api already imported; simulate enabling)
    from importlib import reload
    from core import persistence
    reload(persistence)
    # Créer workflow
    app = uni.app
    c = app.test_client()
    r = c.post('/api/workflows', json={'tasks':[{'type':'inference'}]})
    assert r.status_code == 200
    wid = r.get_json()['id']
    # Round fédéré
    r = c.post('/api/federated/round/start', json={})
    rid = r.get_json()['round_id']
    r = c.post('/api/federated/round/submit', json={'value':1,'weight':1})
    assert r.status_code==200
    # Recharger depuis disque (simulé)
    assert db.exists()
    # Vérifier workflow list inclut wid
    r = c.get('/api/workflows')
    assert any(w['id']==wid for w in r.get_json()['items'])