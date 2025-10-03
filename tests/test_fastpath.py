from core.network.fibre_fastpath import open_low_latency_channel

def test_fastpath_echo():
    ch = open_low_latency_channel()
    payload = b"unit-fastpath-payload"
    sent = ch.send(payload)
    assert sent == len(payload)
    recv = ch.recv()
    assert recv == payload