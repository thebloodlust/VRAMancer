# core/monitor.py
"""
Simple GPU‑monitor.
"""

import torch
from typing import Dict, List, Any
from core.utils import enumerate_devices, detect_backend

class GPUMonitor:
    """
    Gather a short‑summary of the GPU state.
    """

    def __init__(self) -> None:
        self.gpus: List[Dict[str, Any]] = []
        for d in enumerate_devices():
            self.gpus.append({
                'index': d['index'] if d['backend'] != 'mps' else 'mps',
                'name': d['name'],
                'type': d['backend'],
                'total_memory': d['total_memory'],
            })

    # ----------------------------------------------------------------------
    # 1️⃣  Memory helpers
    # ----------------------------------------------------------------------
    def memory_allocated(self, idx: int | str) -> int:
        try:
            if isinstance(idx, int) and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.cuda.memory_allocated(idx)
            if idx == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
                return torch.mps.memory_allocated()
        except Exception:
            return 0
        return 0

    def memory_reserved(self, idx: int | str) -> int:
        if isinstance(idx, int) and idx < torch.cuda.device_count():
            return torch.cuda.memory_reserved(idx)
        return 0

    def total_memory(self, idx: int | str) -> int | None:
        try:
            if isinstance(idx, int) and torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.cuda.get_device_properties(idx).total_memory
            if idx == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
                return torch.mps.get_total_memory()
        except Exception:
            return None
        return None

    # ----------------------------------------------------------------------
    # 2️⃣  Pretty print
    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        lines = ["GPUMonitor:"]
        for gpu in self.gpus:
            lines.append(
                f"  [{gpu['index']}] {gpu['name']} ({gpu['type']}) – "
                f"Total: {gpu['total_memory'] or 'N/A'} bytes, "
                f"Allocated: {self.memory_allocated(gpu['index'])} bytes"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()
