"""Bootstrap production stricte.

Lance :
 - Tracing (si VRM_TRACING=1)
 - Démarrage serveur metrics
 - Cycle périodique d'éviction adaptative basé sur pression VRAM réelle (si torch dispo)

Usage :
    python -m scripts.prod_bootstrap &
"""
from __future__ import annotations
import threading, time, os
from core.tracing import start_tracing
from core.metrics import metrics_server_start
from core.hierarchical_memory import HierarchicalMemoryManager

try:  # torch optionnel
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

start_tracing()
metrics_server_start()
HMM = HierarchicalMemoryManager()

def vram_pressure() -> float | None:
    if torch and torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            used = torch.cuda.memory_allocated(0)
            return used / total
        except Exception:
            return None
    return None

if os.environ.get('VRM_ENABLE_EVICTION','1') == '1':
    def eviction_daemon():  # pragma: no cover - thread logique
        while True:
            pr = vram_pressure()
            if pr and pr > float(os.environ.get('VRM_EVICT_PRESSURE','0.85')):
                HMM.eviction_cycle(vram_pressure=pr)
            time.sleep(int(os.environ.get('VRM_EVICT_INTERVAL','5')))
    t = threading.Thread(target=eviction_daemon, daemon=True)
    t.start()

if __name__ == "__main__":  # Simple attente passive
    print("[prod_bootstrap] Started (tracing=%s)" % bool(os.environ.get("VRM_TRACING") == "1"))
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping bootstrap.")
