
import torch
from transformers import AutoModel, AutoConfig


# ------------------------------------------------------------------
# 1️⃣  Découpage générique d’un modèle HuggingFace en « blocs » (multi-LLM)
# ------------------------------------------------------------------
def split_model_into_blocks(model_name, num_gpus, model_path=None):
    """
    Charge n’importe quel modèle HuggingFace et le découpe en blocs de couches consécutives.
    Compatible GPT, Llama, Mistral, Falcon, etc.
    """
    model = AutoModel.from_pretrained(model_name)
    if model_path:
        torch.save(model, model_path)

    # Détection automatique des couches (transformer blocks)
    # On cherche les attributs courants :
    candidates = [
        ["transformer", "h"],           # GPT-2, GPT-J
        ["model", "layers"],            # Llama, Mistral, Falcon
        ["layers"],                      # Certains modèles
        ["encoder", "layer"],           # BERT, T5
        ["block"],                       # T5
    ]
    layers = None
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                layers = list(obj)
                break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError(f"Impossible de détecter les couches du modèle {model_name}. Veuillez adapter le splitter.")

    # Partitionner les couches en blocs adaptatifs selon la VRAM réelle
    blocks = []
    n = len(layers)
    if num_gpus == 1 or not hasattr(torch, 'cuda') or not torch.cuda.is_available():
        # Fallback : un seul bloc
        blocks.append(torch.nn.Sequential(*layers))
        return blocks

    # Si vram_per_gpu est fourni, découpe adaptatif
    import numpy as np
    def split_by_vram(layers, vram_per_gpu):
        total_vram = sum(vram_per_gpu)
        n_layers = len(layers)
        # Nombre de couches par GPU proportionnel à la VRAM
        ratios = [v/total_vram for v in vram_per_gpu]
        counts = [int(round(r * n_layers)) for r in ratios]
        # Ajustement pour que la somme == n_layers
        while sum(counts) < n_layers:
            counts[np.argmax(ratios)] += 1
        while sum(counts) > n_layers:
            counts[np.argmax(counts)] -= 1
        idx = 0
        blocks = []
        for c in counts:
            block_layers = layers[idx:idx+c]
            blocks.append(torch.nn.Sequential(*block_layers))
            idx += c
        return blocks

    # Récupère la VRAM réelle si possible
    vram_per_gpu = None
    try:
        import pynvml
        pynvml.nvmlInit()
        vram_per_gpu = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total // (1024**2) for i in range(num_gpus)]
        pynvml.nvmlShutdown()
    except Exception:
        vram_per_gpu = [1]*num_gpus  # fallback égal

    blocks = split_by_vram(layers, vram_per_gpu)
    return blocks

# ------------------------------------------------------------------
# 2️⃣  Exemple d’utilisation : assigner chaque bloc à un GPU
# ------------------------------------------------------------------
def assign_blocks_to_gpus(blocks):
    gpus = [f"cuda:{i}" for i in range(len(blocks)) if torch.cuda.is_available()]
    for block, device in zip(blocks, gpus):
        block.to(device)
    return blocks
