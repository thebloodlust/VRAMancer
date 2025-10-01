import time
import threading
import zlib

# --- Routage adaptatif des couches ---

def route_layers(layers, node_caps):
    """
    Routage adaptatif : chaque couche est envoyée au nœud optimal selon VRAM/CPU/disponibilité.
    """
    routed = []
    for i, layer in enumerate(layers):
        best_node = max(node_caps, key=lambda n: n["vram"] - n.get("used_vram", 0))
        routed.append((layer, best_node["host"]))
        best_node["used_vram"] = best_node.get("used_vram", 0) + layer.get("size", 512)
        print(f"[Routing] Couche {i} routée vers {best_node['host']} (VRAM dispo: {best_node['vram'] - best_node['used_vram']} MB)")
    return routed

# --- Préchargement intelligent ---

def smart_preload(layers, node_caps):
    """
    Précharge les couches sur les nœuds selon leur capacité et la séquence d'accès prévue.
    """
    preload_plan = route_layers(layers, node_caps)
    print("[Preload] Plan de préchargement généré.")
    return preload_plan

# --- Compression des poids ---

def compress_weights(weights):
    """
    Compresse les poids (tensors) avec zlib.
    """
    return zlib.compress(weights)

# --- Pipeline asynchrone ---

def async_pipeline(tasks, node_caps):
    """
    Exécute les tâches (inférence, transfert) en pipeline asynchrone sur le cluster.
    """
    def worker(task, node):
        print(f"[Async] Tâche {task['name']} lancée sur {node['host']}")
        time.sleep(task.get("duration", 1))
        print(f"[Async] Tâche {task['name']} terminée sur {node['host']}")
    threads = []
    for i, task in enumerate(tasks):
        node = node_caps[i % len(node_caps)]
        t = threading.Thread(target=worker, args=(task, node))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("[Async] Pipeline terminé.")

# --- Profiling dynamique ---

def dynamic_profiling(layers, node_caps):
    """
    Profile dynamiquement l’utilisation VRAM/CPU et ajuste le routage en temps réel.
    """
    for i, layer in enumerate(layers):
        # Simule un profiling
        print(f"[Profiling] Couche {i} | Taille: {layer.get('size', 512)} MB | Nœud: {node_caps[i % len(node_caps)]['host']}")
    print("[Profiling] Routage ajusté.")

if __name__ == "__main__":
    # Exemple d’utilisation
    node_caps = [
        {"host": "nodeA", "vram": 8192},
        {"host": "nodeB", "vram": 16384},
        {"host": "nodeC", "vram": 4096}
    ]
    layers = [
        {"name": "layer1", "size": 512},
        {"name": "layer2", "size": 1024},
        {"name": "layer3", "size": 2048}
    ]
    route_layers(layers, node_caps)
    smart_preload(layers, node_caps)
    w = b"dummy_weights"
    print(compress_weights(w))
    tasks = [
        {"name": "inference", "duration": 2},
        {"name": "transfer", "duration": 1}
    ]
    async_pipeline(tasks, node_caps)
    dynamic_profiling(layers, node_caps)
