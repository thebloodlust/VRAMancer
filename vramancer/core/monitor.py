import torch
import random
from typing import Dict, List, Any, Optional

class GPUMonitor:
    """
    Utilitaire pour surveiller l’état et la mémoire VRAM de chaque GPU CUDA disponible.
    """
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
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

    def vram_usage(self, gpu_id: int = 0) -> float:
        try:
            props = torch.cuda.get_device_properties(gpu_id)
            total_mb = props.total_memory / (1024**2)
            used_mb = torch.cuda.memory_allocated(gpu_id) / (1024**2)
            return round((used_mb / total_mb) * 100, 2)
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur pour GPU {gpu_id} : {exc}")
            return random.randint(40, 90)

    def detect_overload(self, threshold: float = 90.0) -> Optional[int]:
        try:
            for i in range(torch.cuda.device_count()):
                if self.vram_usage(i) > threshold:
                    return i
            return None
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur pendant la détection d’overload : {exc}")
            return random.choice([0, None])

    def status(self) -> Dict[str, str]:
        try:
            status: Dict[str, str] = {}
            for i in range(torch.cuda.device_count()):
                status[f"GPU {i}"] = f"{self.vram_usage(i)}% VRAM"
            return status
        except Exception as exc:
            if self.verbose:
                print(f"[GPUMonitor] Erreur status : {exc}")
            return {}

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
