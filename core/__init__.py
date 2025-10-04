# core/__init__.py
import os as _os
__version__ = "0.2.0"

if not _os.environ.get('VRM_MINIMAL_TEST'):
    # Mode normal ou dashboard minimal
    if _os.environ.get('VRM_DASHBOARD_MINIMAL','0') == '1':
        # Mode ultra léger: éviter d'importer torch/transformers via utils
        def get_device_type(idx):  # type: ignore
            class _D: pass
            return _D()
        def assign_block_to_device(block, idx): return block
        def get_tokenizer(name): return None
        class GPUMonitor:  # stub réduit
            pass
        class SimpleScheduler:  # stub réduit
            def __init__(self,*a,**k): pass
            def forward(self,x): return x
    else:
        from .utils import get_device_type, assign_block_to_device, get_tokenizer  # type: ignore
        from .monitor import GPUMonitor  # type: ignore
        from .scheduler import SimpleScheduler  # type: ignore
else:
    # mode minimal tests
    def get_device_type(idx):  # pragma: no cover
        return f"cpu:{idx}"
    def assign_block_to_device(block, idx):  # pragma: no cover
        return block
    def get_tokenizer(name):  # pragma: no cover
        return None
    class GPUMonitor:  # pragma: no cover
        pass
    class SimpleScheduler:  # pragma: no cover
        def __init__(self,*a,**k): pass
        def forward(self,x): return x

__all__ = [
    "get_device_type",
    "assign_block_to_device",
    "get_tokenizer",
    "GPUMonitor",
    "SimpleScheduler",
    "__version__",
]
