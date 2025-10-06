from core.api.unified_api import app

def test_version_and_marketplace():
    c = app.test_client()
    rv = c.get('/api/version')
    assert rv.status_code == 200
    assert 'version' in rv.get_json()
    mp = c.get('/api/marketplace/plugins').get_json()
    assert 'plugins' in mp and len(mp['plugins']) >= 2
    assert 'signature' in mp['plugins'][0]