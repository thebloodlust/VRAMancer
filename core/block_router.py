# core/block_router.py

from core.compute_engine import ComputeEngine
from memory_monitor import get_ram_status
from storage_manager import load_block_from_disk
from core.network.remote_executor import RemoteBlock

class BlockRouter:
    def __init__(self, verbose=True):
        self.engine = ComputeEngine(verbose=verbose)
        self.verbose = verbose

    def route(self, block, input_tensor, index=0, importance="normal", estimated_size_mb=100):
        backend = self.engine.backend
        ram_available, _ = self.engine.get_ram_status()

        if estimated_size_mb > 1000 and ram_available < 2 * 1024**3:
            if self._nvme_available():
                if self.verbose:
                    print(f"📦 Bloc {index} → NVMe (poids élevé)")
                block = load_block_from_disk(f"blocks/block_{index}.pt")
                return block(input_tensor)

        if importance == "critical" and backend in ["cuda", "rocm", "mps"]:
            device = self.engine._get_device(index)
            if self.verbose:
                print(f"📦 Bloc {index} → {device} (critique)")
            return block.to(device)(input_tensor)

        if importance == "low":
            if self.verbose:
                print(f"📦 Bloc {index} → Réseau (faible priorité)")
            remote = RemoteBlock("192.168.1.42", 9000)
            return remote.forward(input_tensor)

        if ram_available > 2 * 1024**3:
            if self.verbose:
                print(f"📦 Bloc {index} → CPU (RAM disponible)")
            return block.to("cpu")(input_tensor)

        if self.verbose:
            print(f"📦 Bloc {index} → Réseau (fallback)")
        remote = RemoteBlock("192.168.1.42", 9000)
        return remote.forward(input_tensor)

    def _nvme_available(self):
        return True  # À remplacer par un vrai check disque
