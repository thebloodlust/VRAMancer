#!/usr/bin/env python3
"""
VRAMancer Deluxe – Point d’entrée principal.
"""
import random          # <‑‑ Ajout
import torch

from core.model_splitter   import ModelSplitter
from core.stream_manager   import StreamManager
from core.compute_engine   import ComputeEngine
from core.transfer_manager import TransferManager
from core.memory_balancer  import MemoryBalancer
from core.scheduler        import Scheduler
from core.logger           import Logger

# Dashboards
from dashboard.updater import update_dashboard
from dashboard.app import launch_tk_dashboard  # ou launch() pour la CLI

def main():
    print("🚀 VRAMancer Deluxe California Ripper — Initialisation")

    # 1️⃣  Instanciation des modules métier
    scheduler = Scheduler()
    logger    = Logger()
    splitter  = ModelSplitter("bert-base-uncased")
    streamer  = StreamManager(scheduler, logger)
    engine    = ComputeEngine(backend="auto")
    transfer  = TransferManager(protocol="vramancer", secure=True)
    balancer  = MemoryBalancer(scheduler, logger)

    gpus = scheduler.get_available_gpus()
    print(f"🧠 GPUs détectés : {[gpu['id'] for gpu in gpus]}")

    # 2️⃣  Découpage du modèle
    allocation = splitter.split_by_gpu(gpus)
    print("📦 Répartition des couches :")
    for gpu_id, layers in allocation.items():
        print(f"  GPU {gpu_id} → {len(layers)} couches")

    # 3️⃣  Lancement du dashboard (en parallèle)
    import threading
    t_dashboard = threading.Thread(target=launch_tk_dashboard, daemon=True)
    t_dashboard.start()

    # 4️⃣  Simulation d’un batch d’entrée
    batch = torch.randn(32, 768)

    # Historique fictif pour la prédiction de rééquilibrage
    usage_history = {
        gpu["id"]: [random.randint(60, 95) for _ in range(5)] for gpu in gpus
    }

    # 5️⃣  Boucle d’exécution
    for gpu_id, layers in allocation.items():
        print(f"\n🎯 Traitement sur GPU {gpu_id}")
        streamer.prefetch_layers(layers, lookahead=3)

        for layer in layers:
            # ── 5.1  Chargement + exécution
            block  = streamer.preload_layer(layer)
            output = engine.execute_layer(
                layer, batch,
                device_id=gpu_id,
                track_gradients=True,
                use_compile=True,
                profile_gpu=True
            )

            # ── 5.2  Synchronisation activations
            activations_map = {layer["name"]: (gpu_id, (gpu_id + 1) % len(gpus), output)}
            transfer.sync_activations(activations_map)

            # ── 5.3  Rééquilibrage mémoire
            balancer.balance([layer])
            balancer.predictive_balance(usage_history)

            # ── 5.4  Libération couche
            streamer.release_layer(layer["name"])

            # ── 5.5  Mise à jour du dashboard (mémoire a changé)
            update_dashboard(balancer, dashboard_module=balancer)

        # Monitoring live (si vous l’avez implémenté)
        streamer.live_monitoring()
        balancer.emergency_release()

    print("\n✅ Exécution terminée — VRAMancer Deluxe a tout donné 💥")


if __name__ == "__main__":
    main()
