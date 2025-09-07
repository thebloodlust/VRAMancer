# core/monitor.py
"""
Simple GPU‑monitor.
"""

import torch
from typing import Dict, List, Any

class GPUMonitor:
    """
    Gather a short‑summary of the GPU state.
    """

    def __init__(self) -> None:
        self.gpus: List[Dict[str, Any]] = []
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "type": "cuda",
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            self.gpus.append(gpu_info)

        # ROCm/hip
        if getattr(torch, "hip", None) is not None:
            for i in range(torch.cuda.device_count()):
                if torch.cuda.get_device_properties(i).name.startswith("AMD"):
                    self.gpus[i]["type"] = "rocm"

        # MPS
        if hasattr(torch, "mps") and torch.mps.is_available():
            self.gpus.append({
                "index": "mps",
                "name": "Apple Silicon MPS",
                "type": "mps",
                "total_memory": None,
            })

    # ----------------------------------------------------------------------
    # 1️⃣  Memory helpers
    # ----------------------------------------------------------------------
    def memory_allocated(self, idx: int | str) -> int:
        if isinstance(idx, int) and idx < torch.cuda.device_count():
            return torch.cuda.memory_allocated(idx)
        if idx == "mps" and hasattr(torch, "mps"):
            return torch.mps.memory_allocated()
        return 0

    def memory_reserved(self, idx: int | str) -> int:
        if isinstance(idx, int) and idx < torch.cuda.device_count():
            return torch.cuda.memory_reserved(idx)
        return 0

    def total_memory(self, idx: int | str) -> int | None:
        if isinstance(idx, int) and idx < torch.cuda.device_count():
            return torch.cuda.get_device_properties(idx).total_memory
        if idx == "mps" and hasattr(torch, "mps"):
            return torch.mps.get_total_memory()
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
