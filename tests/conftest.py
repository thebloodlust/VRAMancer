"""VRAMancer test configuration — shared fixtures and markers.

Environment variables set for all tests:
  VRM_API_TOKEN=testtoken
  VRM_MINIMAL_TEST=1
  VRM_DISABLE_RATE_LIMIT=1
  VRM_TEST_MODE=1
"""
import sys
import os
import types
import pytest

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Test environment variables
os.environ.setdefault('VRM_API_TOKEN', 'testtoken')
os.environ.setdefault('VRM_MINIMAL_TEST', '1')
os.environ.setdefault('VRM_DISABLE_RATE_LIMIT', '1')
os.environ.setdefault('VRM_TEST_MODE', '1')

# Mock torch if not installed (lightweight CI)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            Module=object,
            Identity=lambda: (lambda x: x),
            Sequential=lambda *a: list(a),
            ModuleList=list,
            Linear=lambda *a, **k: type('Linear', (object,), {'parameters': lambda self: []})(),
            functional=types.SimpleNamespace(relu=lambda x: x),
        ),
        device=lambda *a, **k: 'cpu',
        randn=lambda *a, **k: __import__('random').random(),
        tensor=lambda *a, **k: a[0] if a else 0,
        zeros=lambda *a, **k: 0,
        ones=lambda *a, **k: 1,
        onnx=types.SimpleNamespace(export=lambda *a, **k: None),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            memory_allocated=lambda *a: 0,
            memory_reserved=lambda *a: 0,
        ),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        ),
        version=types.SimpleNamespace(hip=None),
        compile=lambda fn, **k: fn,
        save=lambda *a, **k: None,
        load=lambda *a, **k: {},
        quantization=types.SimpleNamespace(
            quantize_dynamic=lambda *a, **k: a[0] if a else None,
        ),
    )
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.cuda'] = torch.cuda


# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: quick smoke tests (< 5s)")
    config.addinivalue_line("markers", "integration: integration tests (may need GPU/network)")
    config.addinivalue_line("markers", "slow: slow tests (> 30s, may download models)")
    config.addinivalue_line("markers", "heavy: heavyweight tests (need real GPU + model)")
    config.addinivalue_line("markers", "network: tests requiring network access")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gpu_monitor():
    """Create a GPUMonitor instance (stub-safe)."""
    from core.monitor import GPUMonitor
    return GPUMonitor()


@pytest.fixture
def scheduler():
    """Create a SimpleScheduler with no blocks (stub-safe)."""
    from core.scheduler import SimpleScheduler
    return SimpleScheduler(blocks=[])


@pytest.fixture
def block_router():
    """Create a BlockRouter instance (stub-safe)."""
    from core.block_router import BlockRouter
    return BlockRouter(verbose=False)


@pytest.fixture
def compressor():
    """Create a Compressor instance."""
    from core.compressor import Compressor
    return Compressor(strategy="adaptive", verbose=False)


@pytest.fixture
def config():
    """Get VRAMancer configuration."""
    from core.config import get_config, reload_config
    # Force reload to pick up test env vars
    return reload_config()


@pytest.fixture
def stream_manager(scheduler, gpu_monitor):
    """Create a StreamManager with scheduler and monitor."""
    from core.stream_manager import StreamManager
    return StreamManager(
        scheduler=scheduler,
        monitor=gpu_monitor,
        verbose=False,
    )


@pytest.fixture
def flask_test_client():
    """Create a Flask test client for the production API."""
    try:
        from core.production_api import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    except ImportError:
        pytest.skip("production_api not available")


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary cache directory for NVMe tests."""
    cache = tmp_path / "vramancer_cache"
    cache.mkdir()
    return cache
