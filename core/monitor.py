# core/monitor.py
"""
Petit monitor d’état GPU (CUDA / ROCm / MPS).
"""

import torch
import torch.backends.mps


class GPUMonitor:
    """Récupère les statistiques de mémoire GPU."""
    def __init__(self):
        self.gpus = []

        # CUDA / ROCm
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_type = "rocm" if torch.version.hip else "cuda"
            self.gpus.append({
                "index": i,
                "name": props.name,
                "type": device_type,
                "total": props.total_memory,
            })

        # MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            self.gpus.append({
                "index": "mps",
                "name": "Apple Silicon MPS",
                "type": "mps",
                "total": None,
            })

    def memory_allocated(self, device_index: int) -> int:
        """Mémoire allouée (bytes) pour le GPU."""
        if isinstance(device_index, int) and device_index < torch.cuda.device_count():
            return torch.cuda.memory_allocated(device_index)
        # MPS n’expose pas cette API → 0
        return 0

    def memory_reserved(self, device_index: int) -> int:
        """Mémoire réservée par le runtime (CUDA/ROCm)."""
        if isinstance(device_index, int) and device_index < torch.cuda.device_count():
            return torch.cuda.memory_reserved(device_index)
        return 0
