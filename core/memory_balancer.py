import uuid
import random

class MemoryBlock:
    def __init__(self, size_mb, gpu_id, status="free"):
        self.id = str(uuid.uuid4())
        self.size_mb = size_mb
        self.gpu_id = gpu_id
        self.status = status  # "free", "reserved", "allocated"

    def reserve(self):
        if self.status == "free":
            self.status = "reserved"
        else:
            raise RuntimeError(f"Block {self.id} is not free.")

    def allocate(self):
        if self.status == "reserved":
            self.status = "allocated"
        else:
            raise RuntimeError(f"Block {self.id} must be reserved before allocation.")

    def release(self):
        self.status = "free"

    def __repr__(self):
        return f"<Block {self.id[:8]} | {self.size_mb}MB | GPU {self.gpu_id} | {self.status}>"

class MemoryBalancer:
    def __init__(self, gpu_profiles, verbose=True):
        self.verbose = verbose
        self.blocks = []
        for gpu in gpu_profiles:
            for _ in range(gpu["block_count"]):
                self.blocks.append(MemoryBlock(size_mb=gpu["block_size_mb"], gpu_id=gpu["id"]))

    def allocate_for_layer(self, layer):
        for block in self.blocks:
            if block.status == "free" and block.size_mb >= layer["size_mb"]:
                block.reserve()
                block.allocate()
                if self.verbose:
                    print(f"✅ Layer {layer['name']} → Block {block.id[:8]} (GPU {block.gpu_id})")
                return block
        if self.verbose:
            print(f"❌ Aucun bloc disponible pour {layer['name']} ({layer['size_mb']}MB)")
        return None

    def visualize_blocks(self):
        print("📊 État des blocs mémoire :")
        for block in self.blocks:
            print(block)

    def simulate_overload(self, threshold=90):
        print("⚠️ Simulation de surcharge VRAM...")
        usage_by_gpu = {}
        for block in self.blocks:
            usage_by_gpu.setdefault(block.gpu_id, 0)
            if block.status == "allocated":
                usage_by_gpu[block.gpu_id] += block.size_mb

        for gpu_id, usage in usage_by_gpu.items():
            total = sum(b.size_mb for b in self.blocks if b.gpu_id == gpu_id)
            percent = round((usage / total) * 100, 2)
            print(f"GPU {gpu_id} → {percent}% utilisé")
            if percent > threshold:
                print(f"🔥 GPU {gpu_id} dépasse le seuil ({percent}%)")

    def release_all(self):
        for block in self.blocks:
            block.release()
        print("🧹 Tous les blocs ont été libérés.")
