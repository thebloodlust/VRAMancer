"""Petit module de config — sujet de la tâche agent (C1)."""
from benchmarks.agent_proof.utils import to_int


def parse_config(raw: dict) -> dict:
    """Parse une config brute en dict typé.

    (Tâche C1 pour l'agent : ajouter la validation des entrées + tests.)
    """
    return {
        "host": raw["host"],
        "port": to_int(raw["port"]),
        "workers": to_int(raw.get("workers", 1)),
    }
