import torch

class VRAMancer:
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        self.gpus = self._get_gpu_info()

    def _get_gpu_info(self):
        gpu_info = []
        for i in range(self.gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / (1024 ** 2)  # Convert to MB
            name = props.name
            gpu_info.append({
                "id": i,
                "name": name,
                "total_vram_mb": round(total_vram, 2)
            })
        return gpu_info

    def show_summary(self):
        print("🔍 GPU Summary:")
        for gpu in self.gpus:
            print(f"GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

# Exemple d'utilisation
if __name__ == "__main__":
    v = VRAMancer()
    v.show_summary()
