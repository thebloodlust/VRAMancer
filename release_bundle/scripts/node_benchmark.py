"""
Script universel de benchmark pour chaque nœud du cluster VRAMancer.
- Mesure CPU, VRAM, GPU, bande passante réseau
- Envoie les résultats au master
- Permet d’adapter la répartition des poids LLM selon la perf de chaque couche
"""
import platform
import socket
import json
import time
try:
    import torch
except ImportError:
    torch = None


def get_cpu_info():
    return {
        "cpu": platform.processor(),
        "arch": platform.machine(),
        "cores": getattr(platform, "cpu_count", lambda: 4)(),
    }

def get_gpu_info():
    if torch and torch.cuda.is_available():
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram": torch.cuda.get_device_properties(0).total_memory // (1024**2),
        }
    return {"gpu": "None", "vram": 0}

def get_network_bandwidth():
    # Test simple : ping Google
    import subprocess
    try:
        out = subprocess.check_output(["ping", "-c", "2", "8.8.8.8"]).decode()
        avg = [float(x.split("/")[4]) for x in out.splitlines() if "rtt" in x][0]
        return avg
    except Exception:
        return -1

def main(master_ip="127.0.0.1", master_port=55555):
    node = {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        **get_cpu_info(),
        **get_gpu_info(),
        "network": get_network_bandwidth(),
        "timestamp": time.time(),
    }
    print(f"[Benchmark] Résultat : {node}")
    # Envoi au master
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(json.dumps(node).encode(), (master_ip, master_port))
    sock.close()

if __name__ == "__main__":
    main()
