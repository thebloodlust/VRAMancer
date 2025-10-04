# core/block_router.py

try:
    from core.compute_engine import ComputeEngine
except Exception:  # pragma: no cover - extrême fallback si import casse
    class ComputeEngine:  # minimal stub
        def __init__(self, *a, **k):
            self.backend='cpu'
        def _get_device(self, device_id=0):
            return 'cpu'
        def get_ram_status(self):
            return 0, 0
try:
    from core.memory_monitor import get_ram_status  # type: ignore
except Exception:  # pragma: no cover
    def get_ram_status():
        return 0, 0
try:
    from core.storage_manager import load_block_from_disk  # type: ignore
except Exception:  # pragma: no cover
    def load_block_from_disk(path):
        import torch
        return torch.nn.Identity()
try:
    from core.network.remote_executor import RemoteBlock  # type: ignore
except Exception:  # pragma: no cover
    class RemoteBlock:
        def __init__(self, host, port):
            self.host = host; self.port = port
        def forward(self, x):
            return x

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
            print(f"📦 Bloc {index} → CPU (fallback neutre)")
        # On évite appels réseau pour tests – simple exécution locale
        return block.to('cpu')(input_tensor)

    def _nvme_available(self):
        return True  # À remplacer par un vrai check disque
