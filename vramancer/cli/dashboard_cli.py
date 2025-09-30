import time
from core.memory_balancer import MemoryBalancer

def render_bar(used, total, width=30):
    ratio = used / total
    filled = int(ratio * width)
    bar = "█" * filled + "-" * (width - filled)
    return f"[{bar}] {used}MB / {total}MB"

def launch_cli_dashboard():
    gpus = [
        {"id": 0, "block_count": 4, "block_size_mb": 1024},
        {"id": 1, "block_count": 4, "block_size_mb": 1024}
    ]
    balancer = MemoryBalancer(gpus)

    # Simulation d’allocation initiale
    layers = [
        {"name": "encoder.layer.0.attention", "size_mb": 512},
        {"name": "encoder.layer.1.output", "size_mb": 768},
        {"name": "decoder.layer.0", "size_mb": 1024},
        {"name": "decoder.layer.1", "size_mb": 256}
    ]
    for layer in layers:
        balancer.allocate_for_layer(layer)

    try:
        while True:
            print("\033c")  # Clear terminal
            print("🧠 VRAMancer CLI Dashboard\n")
            balancer.dashboard()
            print("\nCtrl+C pour quitter.")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard arrêté.")
