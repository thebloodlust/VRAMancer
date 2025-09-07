"""
Protocole ultra‑léger (VRAMancer Link) – version de démonstration.
Il s’agit d’un wrapper autour de Socket‑IO mais vous pourrez
remplacer toute la couche de transport par votre firmware fibre SFP+.
"""

# Exports
from .transmission import start_client, send_block
