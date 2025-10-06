def test_fastpath_stub_channel():
    from core.network.fibre_fastpath import open_low_latency_channel
    ch = open_low_latency_channel()
    assert ch is not None
    caps = ch.capabilities()
    assert 'kind' in caps and 'latency_us' in caps
    sent = ch.send(b"hello")
    assert sent == 5
    data = ch.recv()
    assert data == b"hello"
from core.network.fibre_fastpath import open_low_latency_channel

def test_fastpath_echo():
    ch = open_low_latency_channel()
    payload = b"unit-fastpath-payload"
    sent = ch.send(payload)
    assert sent == len(payload)
    recv = ch.recv()
    assert recv == payload