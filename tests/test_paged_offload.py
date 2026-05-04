"""Unit tests for PagedAttentionOffloader."""
import pytest


class _FakePage:
    def __init__(self, pid, tensor):
        self.id = pid
        self.tensor = tensor


class _FakeKV:
    def __init__(self, pages):
        self._pages = list(pages)
        self.freed = []

    def get_lru_pages(self, n):
        return self._pages[:n]

    def free_page(self, pid):
        self.freed.append(pid)
        self._pages = [p for p in self._pages if p.id != pid]


class _FakeHMM:
    def __init__(self):
        self.store = {}

    def put(self, key, tensor):
        self.store[key] = tensor

    def get(self, key):
        return self.store.get(key)


def test_evict_basic():
    torch = pytest.importorskip("torch")
    from core.paged_attention_offload import PagedAttentionOffloader

    pages = [_FakePage(i, torch.zeros(4, 4)) for i in range(5)]
    kv = _FakeKV(pages)
    hmm = _FakeHMM()
    off = PagedAttentionOffloader(kv, hmm)

    assert off.evict_cold_pages(3) == 3
    assert len(hmm.store) == 3
    assert len(kv.freed) == 3
    assert off.stats()["evicted_total"] == 3


def test_restore_roundtrip():
    torch = pytest.importorskip("torch")
    from core.paged_attention_offload import PagedAttentionOffloader

    t = torch.arange(16).reshape(4, 4).float()
    kv = _FakeKV([_FakePage(0, t.clone())])
    hmm = _FakeHMM()
    off = PagedAttentionOffloader(kv, hmm)

    off.evict_cold_pages(1)
    restored = off.restore_page(0)
    assert restored is not None
    assert torch.equal(restored, t)
    assert off.stats()["restored_total"] == 1


def test_no_kv_hooks_safe():
    from core.paged_attention_offload import PagedAttentionOffloader

    class _BareKV:
        pass

    assert PagedAttentionOffloader(_BareKV(), _FakeHMM()).evict_cold_pages(5) == 0
