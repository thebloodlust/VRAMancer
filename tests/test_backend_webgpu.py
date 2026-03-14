import pytest
import asyncio
from unittest.mock import MagicMock

@pytest.fixture
def mock_node_manager(monkeypatch):
    from core.network.webgpu_node import WebGPUNodeManager
    manager = WebGPUNodeManager(port=0)  
    manager.clients = {"test_client": {"busy": False}}
    
    # Mock future handling returning an INT8 dummy encoded answer for JS
    future = asyncio.Future()
    future.set_result(b"\x00\x00\x00\x00" + b"A") 
    manager.submit_tensor = MagicMock(return_value=future)
    return manager

@pytest.mark.asyncio
async def test_webgpu_backend_serialization():
    from core.backends_webgpu import WebGPUBackend
    import threading
    threading.Thread.start = MagicMock()
    
    backend = WebGPUBackend()
    backend._loop = asyncio.get_event_loop()
    
    # Test fallback dummy tensor serialization (no torch in test env)
    b, scale = backend._serialize_tensor(None)
    assert scale == 1.0
    assert b == b"dummy_tensor_data"

@pytest.mark.asyncio
async def test_webgpu_generate_redundancy(mock_node_manager):
    from core.backends_webgpu import WebGPUBackend
    import threading
    threading.Thread.start = MagicMock()
    
    backend = WebGPUBackend()
    backend._loop = asyncio.get_event_loop()
    backend.node_manager = mock_node_manager
    
    # Trigger generation
    res = backend.generate("Hello WebGPU", max_new_tokens=2)
    assert res is not None
    
    # Verify the fallback redundancy submits tasks 
    assert mock_node_manager.submit_tensor.call_count >= 2
