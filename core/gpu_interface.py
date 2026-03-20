import logging
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
from core.utils import enumerate_devices, detect_backend

def get_available_gpus():
    """Retourne une liste normalisée des accélérateurs (CUDA/ROCm/MPS)."""
    devices = enumerate_devices()
    out = []
    for d in devices:
        if d['backend'] in ('cuda', 'rocm'):
            # Convertir total_memory en MB
            tm_mb = round(d['total_memory'] / (1024**2), 2) if d['total_memory'] else None
            out.append({
                'id': d['index'],
                'backend': d['backend'],
                'name': d['name'],
                'total_vram_mb': tm_mb or 0,
                'is_available': True,
            })
        elif d['backend'] == 'mps':
            out.append({
                'id': 'mps',
                'backend': 'mps',
                'name': d['name'],
                'total_vram_mb': 0,
                'is_available': True,
            })
    return out or [{'id': 'cpu', 'backend': 'cpu', 'name': 'CPU', 'total_vram_mb': 0, 'is_available': True}]

def print_gpu_summary():
    """
    Affiche un résumé des GPU détectés.
    """
    gpus = get_available_gpus()
    logging.info(" GPU Summary:")
    for gpu in gpus:
        status = "" if gpu["is_available"] else ""
        logging.info(f"{status} GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

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
        logging.info(f"[Secondary GPU] Exécution sur GPU{gpu['id']} ({gpu['name']})")
        try:
            task_fn(gpu["id"])
        except Exception as e:
            logging.info(f"Erreur sur GPU{gpu['id']}: {e}")

# Exemple d’utilisation : monitoring VRAM
if __name__ == "__main__":
    print_gpu_summary()
    def monitor_vram(gpu_id):
        import torch
        torch.cuda.set_device(gpu_id)
        vram = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
        logging.info(f"GPU{gpu_id} VRAM utilisée : {vram:.2f} MB")
    use_secondary_gpus(monitor_vram)
