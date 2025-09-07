import torch
from transformers import GPT2Model, GPT2Config

# ------------------------------------------------------------------
# 1️⃣  Découpage d’un modèle GPT‑2 en « blocs » (peuvent être répartis sur GPU)
# ------------------------------------------------------------------
def split_gpt2_into_blocks(num_gpus, model_path="models/model.pkl"):
    """
    Charge un modèle GPT‑2 et le divise en `num_gpus` blocs de couches
    consécutives. Chaque bloc sera chargé sur un GPU distinct.
    """
    # 1️⃣ Charger le modèle (on charge l’intégralité sur CPU)
    model = GPT2Model.from_pretrained("gpt2")
    torch.save(model, model_path)

    # 2️⃣ Récupérer la liste des couches transformer
    layers = list(model.transformer.h)

    # 3️⃣ Partitionner les couches en blocs égaux
    blocks = []
    n = len(layers)
    for i in range(num_gpus):
        start = i * n // num_gpus
        end   = (i + 1) * n // num_gpus
        block_layers = layers[start:end]
        # Crée un sous‑module contenant uniquement ces couches
        block = torch.nn.Sequential(*block_layers)
        blocks.append(block)

    return blocks

# ------------------------------------------------------------------
# 2️⃣  Exemple d’utilisation : assigner chaque bloc à un GPU
# ------------------------------------------------------------------
def assign_blocks_to_gpus(blocks):
    gpus = [f"cuda:{i}" for i in range(len(blocks)) if torch.cuda.is_available()]
    for block, device in zip(blocks, gpus):
        block.to(device)
    return blocks
