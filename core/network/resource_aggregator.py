import json
from core.network.cluster_discovery import create_local_cluster

# --- Agrégation des ressources (VRAM, CPU) ---

def aggregate_resources():
    nodes, usb4_mounts = create_local_cluster()
    total_vram = 0
    total_cpu = 0
    node_caps = []
    for node in nodes:
        # Simule la détection VRAM/CPU (à compléter avec vrai monitoring)
        vram = node.get("vram", 8192)  # 8 Go par défaut
        cpu = 4  # 4 cœurs par défaut
        total_vram += vram
        total_cpu += cpu
        node_caps.append({"host": node["hostname"], "vram": vram, "cpu": cpu})
    print(f"VRAM totale combinée : {total_vram} MB")
    print(f"CPU total mutualisé : {total_cpu} cœurs")
    print("Capacités par nœud :")
    for cap in node_caps:
        print(f"- {cap['host']} | VRAM: {cap['vram']} MB | CPU: {cap['cpu']} cœurs")
    return node_caps, total_vram, total_cpu, usb4_mounts

# --- Routage intelligent des blocs ---

def route_block(block, node_caps):
    """
    Routage intelligent : envoie le bloc au nœud le plus adapté (VRAM/CPU).
    """
    # Simule le choix du nœud avec le plus de VRAM disponible
    best_node = max(node_caps, key=lambda n: n["vram"])
    print(f"Bloc routé vers {best_node['host']} (VRAM: {best_node['vram']} MB)")
    # Ici, on pourrait appeler send_block(..., target_device=best_node['host'])
    return best_node

if __name__ == "__main__":
    caps, vram, cpu, mounts = aggregate_resources()
    # Simule un bloc à router
    route_block("dummy_block", caps)
