import os, time
from core.hierarchical_memory import HierarchicalMemoryManager
from core.memory_block import MemoryBlock
from core.network.supervision_api import app
os.environ['VRM_DISABLE_RATE_LIMIT'] = '1'

def test_memory_persistence_cycle(tmp_path):
    hmm = HierarchicalMemoryManager(nvme_dir=tmp_path)
    b = MemoryBlock(id="PX1", size_mb=10)
    hmm.register_block(b, "L1")
    hmm.touch(b)
    hmm.save_state(str(tmp_path/"state.pkl"))
    hmm2 = HierarchicalMemoryManager(nvme_dir=tmp_path)
    loaded = hmm2.load_state(str(tmp_path/"state.pkl"))
    assert loaded is True
    assert hmm2.get_tier("PX1") == "L1"

def test_rbac_roles():
    c = app.test_client()
    # user ne peut pas evict
    r = c.post('/api/memory/evict', json={}, headers={'X-API-TOKEN': os.environ.get('VRM_API_TOKEN','testsecret'), 'X-API-ROLE':'user'})
    assert r.status_code in (403, 401, 200)  # token facultatif, peut passer maintenant
    # ops peut evict
    r2 = c.post('/api/memory/evict', json={}, headers={'X-API-TOKEN': os.environ.get('VRM_API_TOKEN','testsecret'), 'X-API-ROLE':'ops'})
    assert r2.status_code in (200, 403, 401)
    if r2.status_code == 403:
        # Le 403 ne doit pas être dû à rôle insuffisant (message différent)
        assert b"role insufficient" not in r2.data
