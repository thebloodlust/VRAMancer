import os
import json
import numpy as np
import torch

# ------------------------------------------------------------------
# 1️⃣  Logique utilitaire générique
# ------------------------------------------------------------------
def get_available_gpus():
    """Retourne la liste des GPUs disponibles sous forme de tuples
       (device_index, vram_gb, name)."""
    gpus = []
    if not torch.cuda.is_available():
        return gpus
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        vram_gb = prop.total_memory // (1024 ** 3)
        gpus.append((i, vram_gb, prop.name))
    return gpus

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ------------------------------------------------------------------
# 2️⃣  Enregistrement simple de statistiques
# ------------------------------------------------------------------
def record_stats(stats, out_dir="stats"):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{stats['timestamp']}.json")
    save_json(stats, filename)

# ------------------------------------------------------------------
# 3️⃣  Fonctions de sérialisation / désérialisation
# ------------------------------------------------------------------
def serialize_tensors(tensors):
    """Serialize une liste de tensors en bytes."""
    return b"".join([t.cpu().numpy().tobytes() for t in tensors])

def deserialize_tensors(data, shapes, dtypes):
    """Inverse de serialize_tensors."""
    tensors = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = np.prod(shape)
        arr = np.frombuffer(data[offset:offset+size*dtype.itemsize], dtype=dtype)
        tensors.append(torch.from_numpy(arr).reshape(shape))
        offset += size * dtype.itemsize
    return tensors
