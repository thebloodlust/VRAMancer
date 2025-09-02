from core.gpu_interface import get_available_gpus

class VRAMancer:
    def __init__(self):
        self.gpus = get_available_gpus()

    def show_summary(self):
        print("🔍 GPU Summary:")
        for gpu in self.gpus:
            status = "✅" if gpu["is_available"] else "❌"
            print(f"{status} GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

# Exemple d'utilisation
if __name__ == "__main__":
    v = VRAMancer()
    v.show_summary()

