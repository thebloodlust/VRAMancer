import uuid
import random
import shutil

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
        gpu_totals = {}

        for block in self.blocks:
            gpu_id = block.gpu_id
            gpu_totals.setdefault(gpu_id, 0)
            usage_by_gpu.setdefault(gpu_id, 0)
            gpu_totals[gpu_id] += block.size_mb
            if block.status == "allocated":
                usage_by_gpu[gpu_id] += block.size_mb

        for gpu_id in sorted(gpu_totals.keys()):
            used = usage_by_gpu[gpu_id]
            total = gpu_totals[gpu_id]
            percent = round((used / total) * 100, 2)
            print(f"GPU {gpu_id} → {percent}% utilisé")
            if percent > threshold:
                print(f"🔥 GPU {gpu_id} dépasse le seuil ({percent}%)")

    def dashboard(self):
        term_width = shutil.get_terminal_size().columns
        print("\n🖥️ Dashboard VRAM")
        usage_by_gpu = {}
        gpu_total = {}

        for block in self.blocks:
            gpu_id = block.gpu_id
            usage_by_gpu.setdefault(gpu_id, 0)
            gpu_total.setdefault(gpu_id, 0)
            gpu_total[gpu_id] += block.size_mb
            if block.status == "allocated":
                usage_by_gpu[gpu_id] += block.size_mb

        for gpu_id in sorted(gpu_total.keys()):
            used = usage_by_gpu[gpu_id]
            total = gpu_total[gpu_id]
            percent = int((used / total) * 100)
            bar_length = int((percent / 100) * (term_width - 30))
            bar = "█" * bar_length + "-" * (term_width - 30 - bar_length)
            print(f"GPU {gpu_id} [{percent}%] |{bar}| {used}/{total} MB")

    def release_all(self):
        for block in self.blocks:
            block.release()
        print("🧹 Tous les blocs ont été libérés.")
