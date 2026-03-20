# core/__init__.py
import os as _os
__version__ = "1.5.0"
_STRICT = _os.environ.get('VRM_STRICT_IMPORT','0') in {'1','true','TRUE'}


# -- Lightweight stubs (used when real modules cannot be loaded) -----------
def _stub_get_device_type(idx):
    return f"cpu:{idx}"

def _stub_assign_block(block, idx):
    return block

def _stub_get_tokenizer(name):
    return None

class _StubGPUMonitor:
    pass

class _StubScheduler:
    def __init__(self, *a, **k): pass
    def forward(self, x): return x
    def predict(self, x): return x


# -- Resolve real implementations or fall back to stubs --------------------
_use_stubs = bool(_os.environ.get('VRM_MINIMAL_TEST')
                  or _os.environ.get('VRM_DASHBOARD_MINIMAL', '0') == '1')

if _use_stubs:
    get_device_type = _stub_get_device_type
    assign_block_to_device = _stub_assign_block
    get_tokenizer = _stub_get_tokenizer
    GPUMonitor = _StubGPUMonitor
    SimpleScheduler = _StubScheduler
else:
    try:
        from .utils import get_device_type, assign_block_to_device, get_tokenizer  # type: ignore
        from .monitor import GPUMonitor  # type: ignore
        from .scheduler import SimpleScheduler  # type: ignore
    except Exception:
        if _STRICT:
            raise
        get_device_type = _stub_get_device_type
        assign_block_to_device = _stub_assign_block
        get_tokenizer = _stub_get_tokenizer
        GPUMonitor = _StubGPUMonitor
        SimpleScheduler = _StubScheduler

__all__ = [
    "get_device_type",
    "assign_block_to_device",
    "get_tokenizer",
    "GPUMonitor",
    "SimpleScheduler",
    "__version__",
]
