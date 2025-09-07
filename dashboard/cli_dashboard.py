# dashboard/cli_dashboard.py
import os
import time
from core.memory_balancer import MemoryBalancer
from dashboard.visualizer import render_bar  # import de la fonction d’affichage

def clear_console() -> None:
    """Fonction multiplateforme."""
    os.system("cls" if os.name == "nt" else "clear")

def launch():
    """
    Version console « plain‑text » du dashboard.
    """
    gpus = [
        {"id": 0, "block_count": 4, "block_size_mb": 1024},
        {"id": 1, "block_count": 4, "block_size_mb": 1024},
    ]
    balancer = MemoryBalancer(gpus)

    # Simulation d’allocation initiale
    layers = [
        {"name": "encoder.layer.0.attention", "size_mb": 512},
        {"name": "encoder.layer.1.output", "size_mb": 768},
        {"name": "decoder.layer.0", "size_mb": 1024},
        {"name": "decoder.layer.1", "size_mb": 256},
    ]
    for layer in layers:
        balancer.allocate_for_layer(layer)

    try:
        while True:
            clear_console()
            print("\n🧠 VRAMancer CLI Dashboard\n")
            print(balancer.dashboard())
            print("\nCtrl+C pour quitter.")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard arrêté.")
