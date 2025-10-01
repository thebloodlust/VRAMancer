def get_unused_gpus(used_gpu_ids=None):
    """
    Retourne la liste des GPU non utilisés (par défaut, tous sauf GPU0).
    used_gpu_ids : liste d’ID GPU déjà utilisés pour l’inférence principale.
    """
    gpus = get_available_gpus()
    if used_gpu_ids is None:
        used_gpu_ids = [0]  # convention : GPU0 principal
    return [gpu for gpu in gpus if gpu["id"] not in used_gpu_ids and gpu["is_available"]]
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

def use_secondary_gpus(task_fn, exclude_gpu0=True):
    """
    Exécute une fonction sur tous les GPU secondaires disponibles (monitoring, offload, orchestration, worker réseau).
    task_fn : fonction à exécuter, reçoit l’id GPU en argument.
    exclude_gpu0 : si True, ignore le GPU principal (0).
    """
    gpus = get_available_gpus()
    for gpu in gpus:
        if not gpu["is_available"]:
            continue
        if exclude_gpu0 and gpu["id"] == 0:
            continue
        print(f"[Secondary GPU] Exécution sur GPU{gpu['id']} ({gpu['name']})")
        try:
            task_fn(gpu["id"])
        except Exception as e:
            print(f"Erreur sur GPU{gpu['id']}: {e}")

# Exemple d’utilisation : monitoring VRAM
if __name__ == "__main__":
    print_gpu_summary()
    def monitor_vram(gpu_id):
        import torch
        torch.cuda.set_device(gpu_id)
        vram = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
        print(f"GPU{gpu_id} VRAM utilisée : {vram:.2f} MB")
    use_secondary_gpus(monitor_vram)
