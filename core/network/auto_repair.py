import time
import threading

class NodeAutoRepair:
    def __init__(self, cluster_state):
        self.cluster_state = cluster_state
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self.monitor)
        t.start()
        print("[AutoRepair] Module d’auto-réparation démarré.")

    def stop(self):
        self.running = False
        print("[AutoRepair] Arrêt du module.")

    def monitor(self):
        while self.running:
            down_nodes = []
            for node in self.cluster_state["nodes"]:
                if node["status"] == "down":
                    print(f"[AutoRepair] Nœud {node['host']} en panne, tentative de réparation...")
                    self.repair_node(node)
                    down_nodes.append(node)
            if down_nodes:
                self.high_availability(down_nodes)
            time.sleep(5)

    def high_availability(self, down_nodes):
        # Répartition automatique des tâches/ressources sur les nœuds restants
        active_nodes = [n for n in self.cluster_state["nodes"] if n["status"] == "active"]
        if not active_nodes:
            print("[HA] Aucun nœud actif, cluster indisponible !")
            return
        for node in down_nodes:
            print(f"[HA] Répartition des tâches du nœud {node['host']} sur les nœuds restants...")
            # Stub : ici, on répartit les tâches (à compléter selon logique métier)
            for task in node.get("tasks", ["dummy_task"]):
                target = active_nodes[0]  # Simple round-robin
                print(f"[HA] Tâche '{task}' transférée vers {target['host']}")

    def repair_node(self, node):
        # Stub : logique de redémarrage ou remplacement
        node["status"] = "active"
        print(f"[AutoRepair] Nœud {node['host']} réparé et réintégré au cluster.")

if __name__ == "__main__":
    # Exemple d’utilisation avec cluster_state simulé
    cluster_state = {
        "nodes": [
            {"host": "nodeA", "status": "active"},
            {"host": "nodeB", "status": "down"},
            {"host": "nodeC", "status": "active"}
        ]
    }
    auto_repair = NodeAutoRepair(cluster_state)
    auto_repair.start()
    time.sleep(10)
    auto_repair.stop()
