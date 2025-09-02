from core.memory_block import MemoryBlock

class Scheduler:
    def __init__(self, gpu_list, strategy="balanced"):
        self.gpu_list = gpu_list
        self.strategy = strategy
        self.allocations = {gpu["id"]: [] for gpu in gpu_list}

    def allocate_block(self, size_mb):
        """
        Alloue un bloc de mémoire selon la stratégie définie.
        """
        target_gpu = self._select_gpu(size_mb)
        if target_gpu is None:
            raise RuntimeError("Aucun GPU ne peut accueillir ce bloc.")

        block = MemoryBlock(size_mb=size_mb, gpu_id=target_gpu["id"])
        block.reserve()
        block.allocate()
        self.allocations[target_gpu["id"]].append(block)
        return block

    def _select_gpu(self, size_mb):
        """
        Sélectionne le GPU cible selon la stratégie.
        """
        if self.strategy == "balanced":
            # Choisir le GPU avec le moins de blocs alloués
            sorted_gpus = sorted(self.gpu_list, key=lambda g: len(self.allocations[g["id"]]))
            for gpu in sorted_gpus:
                if gpu["total_vram_mb"] >= size_mb:
                    return gpu
        elif self.strategy == "priority":
            # Choisir le GPU avec le plus de VRAM
            sorted_gpus = sorted(self.gpu_list, key=lambda g: g["total_vram_mb"], reverse=True)
            for gpu in sorted_gpus:
                if gpu["total_vram_mb"] >= size_mb:
                    return gpu
        return None

    def show_allocations(self):
        print("📦 Allocations mémoire :")
        for gpu_id, blocks in self.allocations.items():
            print(f"GPU {gpu_id} :")
            for block in blocks:
                print(f"  - {block}")
