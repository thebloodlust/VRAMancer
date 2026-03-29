import pytest
import struct
from unittest.mock import MagicMock, patch


def test_webgpu_backend_init_stub():
    """WebGPU backend initializes in stub mode without websockets."""
    from _deprecated.backends_webgpu import WebGPUBackend

    backend = WebGPUBackend.__new__(WebGPUBackend)
    backend.log = MagicMock()
    backend.model_name = None
    backend.tokenizer = None
    backend._clients = []
    backend._clients_lock = __import__("threading").Lock()
    backend._rr = 0
    backend._loop = None
    backend._thread = None
    backend._server = None

    # Verify stub state
    assert backend.num_workers == 0
    assert backend._server is None


def test_webgpu_backend_load_model():
    """load_model stores name and returns dict."""
    from _deprecated.backends_webgpu import WebGPUBackend
    import threading

    backend = WebGPUBackend.__new__(WebGPUBackend)
    backend.log = MagicMock()
    backend.model_name = None
    backend.tokenizer = None
    backend._clients = []
    backend._clients_lock = __import__("threading").Lock()
    backend._rr = 0
    backend._loop = None
    backend._thread = None
    backend._server = None

    with patch.object(threading.Thread, 'start', MagicMock()):
        result = backend.load_model("test-model")
    assert result["name"] == "test-model"
    assert result["type"] == "webgpu_distributed"
    assert backend.model_name == "test-model"


def test_webgpu_backend_no_workers_error():
    """generate() raises when no workers connected."""
    from _deprecated.backends_webgpu import WebGPUBackend

    backend = WebGPUBackend.__new__(WebGPUBackend)
    backend.log = MagicMock()
    backend.model_name = "test"
    backend.tokenizer = None
    backend._clients = []
    backend._clients_lock = __import__("threading").Lock()
    backend._rr = 0
    backend._loop = None
    backend._thread = None
    backend._server = None

    with pytest.raises(RuntimeError, match="No browser workers"):
        backend.generate("Hello", max_new_tokens=1)
