import torch
from pynvml import *
import time

# ------------------------------------------------------------------
# 1️⃣  Balancer la VRAM entre les GPU
# ------------------------------------------------------------------
def get_vram_usage(device_index):
    """Renvoie la VRAM utilisée (GB) et la mémoire totale (GB)."""
    handle = nvmlDeviceGetHandleByIndex(device_index)
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    used_gb  = mem_info.used / 1024**3
    total_gb = mem_info.total / 1024**3
    return used_gb, total_gb

def balance_memory(gpu_blocks, target_utilization=0.7):
    """
    Si un GPU dépasse `target_utilization` de sa VRAM,
    on déplace un bloc de celui‑ci vers un autre GPU qui est moins saturé.
    """
    # 1️⃣ Récupérer l’état actuel
    stats = []
    for i in range(len(gpu_blocks)):
        used, total = get_vram_usage(i)
        stats.append((i, used, total, used/total))

    # 2️⃣ Trouver le GPU le plus saturé et le moins saturé
    stats.sort(key=lambda x: x[3])  # tri par utilisation
    most_saturated = stats[-1]
    least_saturated = stats[0]

    # 3️⃣ Si le plus saturé dépasse le seuil, déplace‑on un bloc
    if most_saturated[3] > target_utilization:
        src_gpu = most_saturated[0]
        dst_gpu = least_saturated[0]
        block_to_move = gpu_blocks[src_gpu].pop(-1)   # dernier bloc
        block_to_move.to(f"cuda:{dst_gpu}")
        gpu_blocks[dst_gpu].append(block_to_move)
        print(f"[Balancer] Déplacé bloc du GPU {src_gpu} → GPU {dst_gpu}")

    return gpu_blocks

# ------------------------------------------------------------------
# 2️⃣  Surveillance simple
# ------------------------------------------------------------------
def monitor_vram(interval=5):
    """Affiche en temps réel la consommation de VRAM de chaque GPU."""
    while True:
        for i in range(torch.cuda.device_count()):
            used, total = get_vram_usage(i)
            print(f"GPU {i}: {used:.1f}/{total:.1f} GB ({used/total:.0%})")
        time.sleep(interval)
