# examples/backend_demo.py
"""
Démo d’utilisation de l’abstraction backend LLM (auto, HuggingFace, vLLM, Ollama).
"""
from core.backends import select_backend

# Choix du backend : "auto", "huggingface", "vllm", "ollama"
backend = select_backend("auto")  # ou "huggingface", etc.

# Exemple : charger un modèle et découper selon la VRAM
model_name = "gpt2"  # ou tout autre modèle compatible
num_gpus = 2

print(f"[Backend] Utilisé : {backend.__class__.__name__}")

try:
    model = backend.load_model(model_name)
    print("Modèle chargé.")
    blocks = backend.split_model(num_gpus)
    print(f"Découpage : {len(blocks)} blocs.")
    # Dummy input pour test
    import torch
    x = torch.randint(0, 50257, (1, 10))
    out = backend.infer(x)
    print("Sortie :", out.shape if hasattr(out, 'shape') else type(out))
except NotImplementedError as e:
    print("[Non implémenté]", e)
except Exception as e:
    print("[Erreur]", e)
