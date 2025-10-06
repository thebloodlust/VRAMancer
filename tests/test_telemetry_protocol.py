from core.telemetry import encode_packet, decode_stream

def test_encode_decode_roundtrip():
    node = {"id":"edge1","cpu_load_pct":12.34,"free_cores":4,"vram_used_mb":512,"vram_total_mb":8192}
    blob = encode_packet(node)
    decoded = list(decode_stream(blob))
    assert len(decoded) == 1
    d = decoded[0]
    assert d['id'] == 'edge1'
    assert abs(d['cpu_load_pct'] - 12.34) < 0.01
    assert d['free_cores'] == 4
    assert d['vram_used_mb'] == 512
    assert d['vram_total_mb'] == 8192