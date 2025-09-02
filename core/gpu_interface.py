import torch

def get_available_gpus():
    """
    Retourne une liste des GPU disponibles avec leurs propriétés.
    """
    gpus = []
    gpu_count = torch.cuda.device_count()

    for i in range(gpu_count):
        try:
            props = torch.cuda.get_device_properties(i)
            total_vram_mb = round(props.total_memory / (1024 ** 2), 2)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_vram_mb": total_vram_mb,
                "is_available": torch.cuda.get_device_capability(i) is not None
            })
        except Exception as e:
            gpus.append({
                "id": i,
                "name": "Unknown",
                "total_vram_mb": 0,
                "is_available": False,
                "error": str(e)
            })

    return gpus

def print_gpu_summary():
    """
    Affiche un résumé des GPU détectés.
    """
    gpus = get_available_gpus()
    print("🔍 GPU Summary:")
    for gpu in gpus:
        status = "✅" if gpu["is_available"] else "❌"
        print(f"{status} GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

# Exemple d'utilisation
if __name__ == "__main__":
    print_gpu_summary()
