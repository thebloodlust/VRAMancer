import time
import torch

# ------------------------------------------------------------------
# 1️⃣  Planification adaptative – simple “Round‑Robin”
# ------------------------------------------------------------------
class SimpleScheduler:
    """
    Le Scheduler est le cœur de l’orchestration :
    - Charge un modèle sur un GPU
    - Envoie les données vers le prochain GPU
    - Récupère la sortie en temps réel
    """

    def __init__(self, blocks, schedule_interval=0.1):
        self.blocks = blocks
        self.schedule_interval = schedule_interval

    def forward(self, input_ids):
        """
        Passe l’entrée à travers tous les blocs (GPU‑par‑GPU).
        """
        x = input_ids
        for i, block in enumerate(self.blocks):
            device = f"cuda:{i}"
            x = x.to(device)
            with torch.no_grad():
                x = block(x)
        # Fusionner les sorties (simple concaténation)
        return x.cpu()

    def run_loop(self, prompt="Bonjour", max_tokens=50):
        """Boucle d’inférence simple (un prompt → texte)."""
        tokenizer = self.blocks[0][0].transformer.tokenizer
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        for _ in range(max_tokens):
            output = self.forward(input_ids)
            # Prendre le token de sortie le plus probable
            next_token_id = output[0, -1, :].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])])
            token = tokenizer.decode(next_token_id)
            print(token, end="", flush=True)
        print()
