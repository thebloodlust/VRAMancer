import torch
from core.model_splitter import ModelSplitter
from core.stream_manager import StreamManager
from core.compute_engine import ComputeEngine
from core.transfer_manager import TransferManager
from core.memory_balancer import MemoryBalancer
from core.scheduler import Scheduler
from core.logger import Logger

def main():
    print("🚀 VRAMancer Deluxe California Ripper — Initialisation")

    # Initialisation des modules
    scheduler = Scheduler()
    logger = Logger()
    splitter = ModelSplitter("bert-base-uncased")
    streamer = StreamManager(scheduler, logger)
    engine = ComputeEngine(backend="auto")
    transfer = TransferManager(protocol="vramancer", secure=True)
    balancer = MemoryBalancer(scheduler, logger)

    # Détection des GPUs
    gpus = scheduler.get_available_gpus()
    print(f"🧠 GPUs détectés : {[gpu['id'] for gpu in gpus]}")

    # Découpage du modèle
    allocation = splitter.split_by_gpu(gpus)
    print("📦 Répartition des couches :")
    for gpu_id, layers in allocation.items():
        print(f"  GPU {gpu_id} → {len(layers)} couches")

    # Simulation d’un batch d’entrée
    batch = torch.randn(32, 768)

    # Historique fictif pour prédiction
    usage_history = {gpu["id"]: [random.randint(60, 95) for _ in range(5)] for gpu in gpus}

    # Boucle d’exécution
    for gpu_id, layers in allocation.items():
        print(f"\n🎯 Traitement sur GPU {gpu_id}")
        streamer.prefetch_layers(layers, lookahead=3)

        for layer in layers:
            # Chargement
            block = streamer.preload_layer(layer)

            # Exécution
            output = engine.execute_layer(layer, batch, device_id=gpu_id, track_gradients=True, use_compile=True, profile_gpu=True)

            # Synchronisation
            activations_map = {layer["name"]: (gpu_id, (gpu_id + 1) % len(gpus), output)}
            transfer.sync_activations(activations_map)

            # Rééquilibrage mémoire
            balancer.balance([layer])
            balancer.predictive_balance(usage_history)

            # Libération
            streamer.release_layer(layer["name"])

        # Monitoring live
        streamer.live_monitoring()
        balancer.emergency_release()

    print("\n✅ Exécution terminée — VRAMancer Deluxe a tout donné 💥")

if __name__ == "__main__":
    main()
