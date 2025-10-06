"""Agent edge minimal.

Fonctions:
 - Collecte charge CPU, estimation cœurs libres
 - (Optionnel) collecte load GPU si PyTorch disponible
 - Envoie périodiquement /api/edge/report (JSON)
 - Peut aussi pousser un paquet binaire local (pour un futur concentrateur) – ici on log seulement.

Usage basique:
    python3 edge_agent.py --id myedge --api http://localhost:5010 --interval 5
"""
from __future__ import annotations
import time, argparse, json, requests, psutil
try:
    import torch
except Exception:
    torch = None
from core.telemetry import encode_packet, decode_stream


def collect_state(node_id: str):
    load = psutil.cpu_percent(interval=0.0)
    total = psutil.cpu_count(logical=True)
    phys = psutil.cpu_count(logical=False) or total
    free_cores = max(0, total - phys)  # approximation libre
    vram_used = 0; vram_total = 0
    if torch and torch.cuda.is_available():
        try:
            vram_used = int(torch.cuda.memory_allocated(0) / (1024**2))
            props = torch.cuda.get_device_properties(0)
            vram_total = int(props.total_memory / (1024**2))
        except Exception:
            pass
    return {
        'id': node_id,
        'cpu_load': load,
        'free_cores': free_cores,
        'vram_used_mb': vram_used,
        'vram_total_mb': vram_total,
        'type': 'edge'
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', required=True)
    ap.add_argument('--api', default='http://localhost:5010')
    ap.add_argument('--interval', type=int, default=5)
    args = ap.parse_args()

    while True:
        state = collect_state(args.id)
        try:
            requests.post(f"{args.api}/api/edge/report", json=state, timeout=2)
        except Exception as e:
            print('[edge-agent] erreur report:', e)
        # Paquet binaire local (démonstration)
        pkt = encode_packet({
            'id': state['id'],
            'cpu_load_pct': state['cpu_load'],
            'free_cores': state['free_cores'],
            'vram_used_mb': state['vram_used_mb'],
            'vram_total_mb': state['vram_total_mb'],
        })
        # Ici on pourrait l'envoyer sur un socket UDP vers un collecteur.
        # Pour l'instant on log la taille.
        print(f"[edge-agent] sent report; binary_packet={len(pkt)} bytes")
        time.sleep(args.interval)

if __name__ == '__main__':
    main()
