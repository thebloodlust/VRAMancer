#!/usr/bin/env python3
"""
VRAMancer – Point d’entrée principal unifié.
"""

import random
import torch
import argparse
from core.backends import select_backend
from core.stream_manager   import StreamManager
from core.compute_engine   import ComputeEngine
from core.transfer_manager import TransferManager
from core.memory_balancer  import MemoryBalancer
from core.scheduler        import SimpleScheduler as Scheduler
from core.logger           import Logger
from dashboard.updater import update_dashboard
from dashboard.dashboard_qt import launch_dashboard as launch_qt_dashboard

def main():

    import os
    import yaml

    parser = argparse.ArgumentParser(description="VRAMancer — LLM Orchestrator")
    parser.add_argument("--backend", type=str, default=None, help="Backend LLM : auto, huggingface, vllm, ollama")
    parser.add_argument("--model", type=str, default=None, help="Nom du modèle LLM à charger")
    parser.add_argument("--gpus", type=int, default=None, help="Nombre de GPUs à utiliser (détection auto si non spécifié)")
    parser.add_argument("--net-mode", type=str, default=None, help="Mode réseau : auto ou manual")
    parser.add_argument("--net-iface", type=str, default=None, help="Nom de l’interface réseau à forcer (manual)")
    args = parser.parse_args()

    # Lecture config.yaml si présent
    config = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

    backend_name = args.backend or config.get("backend", "auto")
    model_name = args.model or config.get("model", "gpt2")
    num_gpus = args.gpus if args.gpus is not None else config.get("num_gpus")
    net_mode = args.net_mode or config.get("net_mode", "auto")
    net_iface = args.net_iface or config.get("net_iface")

    print(f"🚀 VRAMancer — Initialisation (backend={backend_name}, modèle={model_name})")

    backend = select_backend(backend_name)
    print(f"[Backend] Utilisé : {backend.__class__.__name__}")

    # 1️⃣  Instanciation des modules métier
    scheduler = Scheduler()
    logger    = Logger()
    streamer  = StreamManager(scheduler, logger)
    engine    = ComputeEngine(backend="auto")
    transfer  = TransferManager(protocol="vramancer", secure=True)
    balancer  = MemoryBalancer(scheduler, logger)

    gpus = scheduler.get_available_gpus()
    if num_gpus is None:
        num_gpus = len(gpus)
    print(f"🧠 GPUs détectés : {[gpu['id'] for gpu in gpus]}")

    # 2️⃣ bis — Exploitation des GPU secondaires pour tâches annexes
    from core.gpu_interface import get_unused_gpus
    used_gpu_ids = list(range(num_gpus))
    secondary_gpus = get_unused_gpus(used_gpu_ids)
    if secondary_gpus:
        print(f"🔄 GPU secondaires disponibles pour tâches annexes : {[g['id'] for g in secondary_gpus]}")
        # Exemple concret : monitoring GPU secondaire en thread
        import threading, time
        from core.monitor import GPUMonitor
        def monitor_secondary(gpu_id):
            mon = GPUMonitor()
            while True:
                usage = mon.memory_allocated(gpu_id)
                print(f"[MONITOR] GPU secondaire {gpu_id} : {usage/1024**2:.1f} MB alloués")
                time.sleep(5)
        for g in secondary_gpus:
            t = threading.Thread(target=monitor_secondary, args=(g['id'],), daemon=True)
            t.start()
        # Exemple offload (simulation)
        print("[OFFLOAD] Vous pouvez utiliser ces GPU pour des tâches de swap, worker réseau, etc.")
    else:
        print("ℹ️ Aucun GPU secondaire libre pour tâches annexes.")

    # 2️⃣ ter — Sélection réseau auto/manuel
    from core.network.interface_selector import select_network_interface
    if net_mode == "manual" or net_iface:
        iface = net_iface or select_network_interface("manual")
    else:
        iface = select_network_interface("auto")
    print(f"🌐 Interface réseau utilisée : {iface}")

    # 2️⃣  Chargement et découpage du modèle via backend unifié
    try:
        model = backend.load_model(model_name)
        blocks = backend.split_model(num_gpus)
        print(f"📦 Modèle découpé en {len(blocks)} blocs.")
    except NotImplementedError as e:
        print("[Non implémenté]", e)
        return
    except Exception as e:
        print("[Erreur]", e)
        return

    # 3️⃣  Lancement du dashboard (en parallèle)
    import threading
    t_dashboard = threading.Thread(target=launch_qt_dashboard, daemon=True)
    t_dashboard.start()

    # 4️⃣  Simulation d’un batch d’entrée
    batch = torch.randint(0, 50257, (1, 10))

    # Historique fictif pour la prédiction de rééquilibrage
    usage_history = {
        gpu["id"]: [random.randint(60, 95) for _ in range(5)] for gpu in gpus
    }

    # 5️⃣  Inférence via backend unifié (sur tous les blocs)
    try:
        out = backend.infer(batch)
        print("Sortie backend :", out.shape if hasattr(out, 'shape') else type(out))
    except NotImplementedError as e:
        print("[Non implémenté]", e)
    except Exception as e:
        print("[Erreur inférence]", e)

    print("\n✅ Exécution terminée — VRAMancer a tout donné 💥")

if __name__ == "__main__":
    main()
