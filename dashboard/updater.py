"""
Helper qui envoie l’état de la mémoire du `MemoryBalancer`
au widget/dashboard actif (CLI / Tk / Qt).
"""
from typing import Any

def update_dashboard(balancer: "MemoryBalancer", dashboard_module: Any) -> None:
    """
    - `balancer`   : instance de votre `MemoryBalancer`
    - `dashboard_module` : le wrapper/instance du dashboard
    """
    # 1️⃣  Récupération de l’état mémoire
    memory_state = balancer.get_memory_state()

    # 2️⃣  Si c’est la CLI, on ne fait rien (le wrapper console le lit)
    #     Si c’est Tk ou Qt, on appelle la méthode `update(...)`
    if hasattr(dashboard_module, "update"):
        dashboard_module.update(memory_state)
