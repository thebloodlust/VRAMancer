#!/usr/bin/env python3
"""
VRAMancer – Point d’entrée principal unifié.
"""

import random
import torch
import argparse
from core.backends import select_backend
from core.metrics import (
    metrics_server_start,
    INFER_REQUESTS,
    INFER_ERRORS,
    INFER_LATENCY,
    GPU_MEMORY_USED,
)
from core.stream_manager   import StreamManager
from core.compute_engine   import ComputeEngine
from core.transfer_manager import TransferManager
from core.memory_balancer  import MemoryBalancer
from core.hierarchical_memory import HierarchicalMemoryManager
from core.scheduler        import SimpleScheduler as Scheduler
from core.logger           import LoggerAdapter, get_logger
from core.config           import get_config
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

    merged = get_config()
    backend_name = args.backend or merged.get("backend")
    model_name = args.model or merged.get("model")
    num_gpus = args.gpus if args.gpus is not None else merged.get("num_gpus")
    net_mode = args.net_mode or merged.get("net_mode")
    net_iface = args.net_iface or merged.get("net_iface")

    log = get_logger("main")

    log.info(f"Démarrage VRAMancer (backend={backend_name}, model={model_name})")

    # Metrics server
    metrics_server_start()
    backend = select_backend(backend_name)
    log.info(f"Backend sélectionné : {backend.__class__.__name__}")

    # 1️⃣  Instanciation des modules métier
    scheduler = Scheduler()
    logger    = LoggerAdapter("runtime")
    streamer  = StreamManager(scheduler, logger)
    engine    = ComputeEngine(backend="auto")
    transfer  = TransferManager(protocol="vramancer", secure=True)
    balancer  = MemoryBalancer(scheduler, logger)

    gpus = scheduler.get_available_gpus()
    if num_gpus is None:
        num_gpus = len(gpus)
    log.info(f"GPUs détectés : {[gpu['id'] for gpu in gpus]}")

    # 2️⃣ bis — Exploitation des GPU secondaires pour tâches annexes
    from core.gpu_interface import get_unused_gpus
    used_gpu_ids = list(range(num_gpus))
    secondary_gpus = get_unused_gpus(used_gpu_ids)
    if secondary_gpus:
        log.info(f"GPU secondaires disponibles : {[g['id'] for g in secondary_gpus]}")
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
        log.info("Aucun GPU secondaire libre.")

    # 2️⃣ ter — Sélection réseau auto/manuel
    from core.network.interface_selector import select_network_interface
    if net_mode == "manual" or net_iface:
        iface = net_iface or select_network_interface("manual")
    else:
        iface = select_network_interface("auto")
    log.info(f"Interface réseau utilisée : {iface}")

    # 2️⃣  Chargement et découpage du modèle via backend unifié
    try:
        model = backend.load_model(model_name)
        blocks = backend.split_model(num_gpus)
        log.info(f"Modèle découpé en {len(blocks)} blocs")
        hmem = HierarchicalMemoryManager()
        # Enregistrer blocs VRAM primaire (L1)
        from core.memory_block import MemoryBlock
        for i, b in enumerate(blocks):
            mb = MemoryBlock(size_mb=getattr(b, 'size_mb', 128), gpu_id=0, status="allocated")
            hmem.register_block(mb, "L1")
        # Watcher de pression VRAM (simulation simple)
        import threading, time
        def vram_watcher():
            while True:
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated(0)
                    total = torch.cuda.get_device_properties(0).total_memory
                    pct = used / total * 100 if total else 0
                    # Politique de démotion simple
                    for blk_id in list(hmem.registry.keys()):
                        dummy = type("_B", (), {"id": blk_id})()
                        try:
                            # Recréation d'un objet MemoryBlock factice juste pour policy
                            from core.memory_block import MemoryBlock as MB
                            mb_fake = MB(size_mb=128, gpu_id=0, status="allocated")
                            mb_fake.id = blk_id
                            hmem.policy_demote_if_needed(mb_fake, pct)
                        except Exception:
                            pass
                time.sleep(5)
        threading.Thread(target=vram_watcher, daemon=True).start()
    except NotImplementedError as e:
        log.warning(f"Non implémenté: {e}")
        return
    except Exception as e:
        log.error(f"Erreur chargement modèle: {e}")
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
        INFER_REQUESTS.inc()
        with INFER_LATENCY.time():
            out = backend.infer(batch)
        # Mise à jour gauges GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                GPU_MEMORY_USED.labels(gpu=str(i)).set(torch.cuda.memory_allocated(i))
        log.info(f"Sortie backend: {getattr(out,'shape', type(out))}")
    except NotImplementedError as e:
        INFER_ERRORS.inc()
        log.warning(f"Non implémenté: {e}")
    except Exception as e:
        INFER_ERRORS.inc()
        log.error(f"Erreur inférence: {e}")

    log.info("Exécution terminée ✅")

if __name__ == "__main__":
    main()
