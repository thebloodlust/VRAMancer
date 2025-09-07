# core/scheduler.py
"""
Module `scheduler` – gestion simple de l’inférence GPU‑par‑GPU.

Classes
-------
SimpleScheduler
    Orchestration Round‑Robin sur les blocs (GPU) d’un modèle.
"""

from __future__ import annotations

import time
from typing import List, Iterable, Dict, Any

import torch

# Import local GPUMonitor (si vous l’avez dans `core/monitor.py`).
# Vous pouvez commenter cette ligne si vous ne l’utilisez pas.
try:
    from .monitor import GPUMonitor
except Exception:   # pragma: no cover
    GPUMonitor = None  # type: ignore[assignment]


class SimpleScheduler:
    """
    Scheduler Round‑Robin.

    Chaque bloc (`torch.nn.Module`) est chargé sur un GPU différent.
    L’entrée est transmise de manière séquentielle d’un GPU vers le
    suivant, sans duplication de données en mémoire.

    Parameters
    ----------
    blocks : List[torch.nn.Module]
        Liste de modules, un module par GPU. Les modules doivent
        être pré‑chargés sur le GPU correspondant.
    schedule_interval : float, optional
        Pause (en s) entre deux passes Round‑Robin.
        Utilisé uniquement dans `run_loop` pour simuler un
        traitement temps réel.
    verbose : bool, optional
        Si `True`, affiche des logs de debug (surcharge GPU, etc.).
    """

    def __init__(
        self,
        blocks: List[torch.nn.Module],
        schedule_interval: float = 0.1,
        verbose: bool = False,
    ) -> None:
        self.blocks = blocks
        self.schedule_interval = schedule_interval
        self.verbose = verbose
        self.monitor = GPUMonitor() if GPUMonitor else None

    # ------------------------------------------------------------------
    # Méthodes principales
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Passe `input_ids` à travers chaque bloc en séquence.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor de forme `[batch, seq_len]` initialisé sur `cpu`.

        Returns
        -------
        torch.Tensor
            Tensor de sortie placé sur `cpu`.
        """
        x = input_ids
        for i, block in enumerate(self.blocks):
            device = f"cuda:{i}"
            x = x.to(device)

            # Inference en mode « no‑grad » pour libérer la mémoire GPU
            with torch.no_grad():
                x = block(x)

        return x.cpu()

    def run_loop(
        self,
        prompt: str = "Bonjour",
        max_tokens: int = 50,
        tokenizer: Any = None,
    ) -> None:
        """
        Boucle d’inférence « prompt‑to‑text ».

        Cette fonction est surtout utile pour tester le scheduler
        en mode autonome. Elle :

        1. Encode le prompt en `input_ids`.
        2. Génère `max_tokens` en appelant `forward` de manière
           séquentielle (Round‑Robin).
        3. Affiche chaque token généré en temps réel.
        4. Vérifie (via `GPUMonitor`) si un GPU est surchargé
           avant chaque itération.

        Parameters
        ----------
        prompt : str, optional
            Texte de départ.
        max_tokens : int, optional
            Nombre de tokens à générer.
        tokenizer : Any, optional
            Tokenizer (ex. `transformers.BertTokenizer`).
            Si `None`, on extrait le tokenizer du premier bloc
            (ex. `block.tokenizer`).
        """
        # ------------------------------------------------------------------
        # Pré‑remontage du tokenizer
        # ------------------------------------------------------------------
        if tokenizer is None:
            if hasattr(self.blocks[0], "tokenizer"):
                tokenizer = self.blocks[0].tokenizer
            else:
                raise RuntimeError(
                    "Un tokenizer doit être fourni ou "
                    "existant dans le premier bloc."
                )

        # ------------------------------------------------------------------
        # Encoder le prompt
        # ------------------------------------------------------------------
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)

        for _ in range(max_tokens):
            # Petite pause pour laisser le temps aux GPU de se
            # synchroniser dans un vrai scénario
            time.sleep(self.schedule_interval)

            # Vérification de surcharge GPU (si GPUMonitor est présent)
            if self.monitor:
                if (gpu_id := self.monitor.detect_overload(threshold=90.0)) is not None:
                    if self.verbose:
                        print(
                            f"⚠️  GPU {gpu_id} surchargé ({self.monitor.vram_usage(gpu_id)} %) – "
                            "libération de mémoire recommandée."
                        )

            # Inference Round‑Robin
            output = self.forward(input_ids)

            # Décodage du token le plus probable
            next_token_id = torch.argmax(output, dim=-1)[-1].item()
            next_token = tokenizer.convert_ids_to_tokens(next_token_id)

            # Affichage en temps réel
            print(next_token, end="", flush=True)

            # Mise à jour de l’entrée pour la prochaine itération
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_token_id]], dtype=input_ids.dtype))
            )

        print()  # saut de ligne final

    # ------------------------------------------------------------------
    # Méthodes auxiliaires (exemple d’utilisation du monitor)
    # ------------------------------------------------------------------
    def get_gpu_status(self) -> Dict[str, str]:
        """
        Retourne un résumé de l’état des GPUs via `GPUMonitor`.

        Returns
        -------
        dict
            `{ "GPU i" : "xx% VRAM" }` ou `{ "GPU i" : "Simulé" }`
            si aucun GPU est disponible.
        """
        if self.monitor:
            return self.monitor.status()
        return {"GPU 0": "GPUMonitor non disponible"}

    def get_overloaded_gpu(self, threshold: float = 90.0) -> Any:
        """
        Renvoie l’ID du GPU dépassant le seuil `threshold`.

        Returns
        -------
        int or None
            ID du GPU en surcharge, ou `None` s’il n’y en a pas.
        """
        if self.monitor:
            return self.monitor.detect_overload(threshold)
        return None

    # ------------------------------------------------------------------
    # Méthodes de diagnostics
    # ------------------------------------------------------------------
    def print_status(self) -> None:
        """Affiche le status actuel des GPUs (utile en debug)."""
        if self.monitor:
            status = self.monitor.status()
            for gpu, txt in status.items():
                print(f"{gpu}: {txt}")
        else:
            print("GPUMonitor non installé – status simulé.")
