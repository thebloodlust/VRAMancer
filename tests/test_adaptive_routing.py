import pytest
import zlib
from core.orchestrator.adaptive_routing import (
    route_layers,
    smart_preload,
    compress_weights,
    async_pipeline,
    dynamic_profiling
)

def test_route_layers():
    node_caps = [
        {"host": "nodeA", "vram": 8192},
        {"host": "nodeB", "vram": 16384},
    ]
    layers = [
        {"name": "layer1", "size": 512},
        {"name": "layer2", "size": 1024},
    ]
    
    routed = route_layers(layers, node_caps)
    assert len(routed) == 2
    # nodeB has more vram, so both layers go there
    assert routed[0][1] == "nodeB"
    assert routed[1][1] == "nodeB"
    # Both layers routed to nodeB: 512 + 1024 = 1536
    assert node_caps[1]["used_vram"] == 1536

def test_smart_preload():
    node_caps = [{"host": "nodeA", "vram": 8192}]
    layers = [{"name": "layer1", "size": 512}]
    routed = smart_preload(layers, node_caps)
    assert len(routed) == 1
    assert routed[0][1] == "nodeA"

def test_compress_weights():
    w = b"dummy_weights_that_repeat_dummy_weights_that_repeat"
    c = compress_weights(w)
    assert isinstance(c, bytes)
    assert zlib.decompress(c) == w

def test_async_pipeline():
    node_caps = [
        {"host": "nodeA", "vram": 8192},
    ]
    tasks = [
        {"name": "inference", "duration": 0.1},
        {"name": "transfer", "duration": 0.1}
    ]
    # Should not crash and finish relatively quickly
    async_pipeline(tasks, node_caps)

def test_dynamic_profiling():
    node_caps = [{"host": "nodeA", "vram": 8192}]
    layers = [{"name": "layer1", "size": 512}]
    # Should not crash
    dynamic_profiling(layers, node_caps)
