import time
import torch
from typing import List, Iterable, Any


# --------------------------------------------------------------------
# 1️⃣  Planification adaptative – simple “Round‑Robin”
# --------------------------------------------------------------------
class SimpleScheduler:
    """
    Le Scheduler est le cœur de l’orchestration :
    - Charge un modèle sur un GPU
    - Envoie les données vers le prochain GPU
    - Récupère la sortie en temps réel
    """

    def __init__(self, blocks: List[torch.nn.Module], schedule_interval: float = 0.1):
        """
        :param blocks: liste de blocs (un `torch.nn.Module` par GPU) ;
                       ils sont supposés déjà chargés sur le GPU correspondant.
        :param schedule_interval: délai (en s) entre deux passes “Round‑Robin”.
        """
        self.blocks = blocks
        self.schedule_interval = schedule_interval

    # --------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Passe l’entrée à travers tous les blocs (GPU‑par‑GPU).
        :param input_ids: Tensor de forme `[batch, seq_len]` sur `cpu`.
        :return: Tensor de sortie placé sur `cpu`.
        """
        x = input_ids
        for i, block in enumerate(self.blocks):
            device = f"cuda:{i}"
            x = x.to(device)

            # On ne garde pas le graphe de calcul (no‑grad) pour l’inférence
            with torch.no_grad():
                x = block(x)

        return x.cpu()

    # --------------------------------------------------------------------
    def run_loop(
        self,
        prompt: str = "Bonjour",
        max_tokens: int = 50,
        tokenizer: Any = None,
    ) -> None:
        """
        Boucle d’inférence simple (un prompt → texte).

        Vous pouvez soit passer un `tokenizer` explicite (ex. de HuggingFace),
        soit laisser le scheduler récupérer le tokenizer depuis le premier
        bloc.
        """
        # On récupère le tokenizer
        if tokenizer is None:
            try:
                # Le premier bloc doit exposer un attribut `tokenizer`
                tokenizer = self.blocks[0].tokenizer
            except AttributeError:
                raise RuntimeError(
                    "Pas de tokenizer fourni ! "
                    "Passez un objet `tokenizer` ou assurez‑vous que "
                    "le premier bloc possède un attribut `tokenizer`."
                )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        for _ in range(max_tokens):
            # On attend le temps du scheduler (Round‑Robin)
            time.sleep(self.schedule_interval)

            output = self.forward(input_ids)

            # Prendre le token de sortie le plus probable
            next_token_id = output[0, -1, :].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])])

            token = tokenizer.decode(next_token_id)
            print(token, end="", flush=True)

        print()  # fin de la ligne

    # --------------------------------------------------------------------
    # Méthodes d’extension pour le dashboard / MemoryBalancer
    # --------------------------------------------------------------------
    def get_available_gpus(self) -> list[int]:
        """Renvoie la liste des IDs de GPU disponibles (0…N‑1)."""
        return list(range(len(self.blocks)))

    def get_gpu_usage(self, gpu_id: int) -> dict[str, int]:
        """
        Renvoie un dictionnaire `{ "used": X, "total": Y }` pour le GPU
        donné. Ici on retourne les chiffres **statistiques** calculés
        par la classe `MemoryBalancer`. Si vous voulez les vrais chiffres
        temps réel, utilisez `torch.cuda.memory_allocated` / `memory_reserved`.
        """
        if gpu_id < 0 or gpu_id >= len(self.blocks):
            raise ValueError(f"gpu_id {gpu_id} hors limites")

        # Exemple simple : on retourne 0 MB utilisés, 8 GB total
        # à remplacer par vos métriques réelles
        return {"used": 0, "total": 8192}
