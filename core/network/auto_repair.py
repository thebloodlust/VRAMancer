import time
import threading

class NodeAutoRepair:
    """Surveillance / auto-réparation basique de nœuds.

    Production minimal:
      - Thread daemon
      - Callbacks extensibles (on_repair, on_ha_redispatch)
      - Protection contre modifications concurrentes
    """
    def __init__(self, cluster_state, interval=5):
        self.cluster_state = cluster_state
        self.interval = interval
        self.running = False
        self._thread: threading.Thread | None = None
        self.on_repair = []  # liste de callbacks(node)
        self.on_ha_redispatch = []  # callbacks(node, target, task)
        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[AutoRepair] Module d’auto-réparation démarré (daemon).")

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        print("[AutoRepair] Arrêt du module.")

    def _monitor_loop(self):
        while self.running:
            try:
                down_nodes = []
                with self._lock:
                    nodes_snapshot = list(self.cluster_state.get("nodes", []))
                for node in nodes_snapshot:
                    if node.get("status") == "down":
                        print(f"[AutoRepair] Nœud {node.get('host')} en panne, tentative de réparation...")
                        self._repair_node(node)
                        down_nodes.append(node)
                if down_nodes:
                    self._ha_redispatch(down_nodes)
            except Exception as e:  # pragma: no cover
                print("[AutoRepair] Erreur monitor:", e)
            time.sleep(self.interval)

    def _ha_redispatch(self, down_nodes):
        with self._lock:
            active_nodes = [n for n in self.cluster_state.get("nodes", []) if n.get("status") == "active"]
        if not active_nodes:
            print("[HA] Aucun nœud actif, cluster indisponible !")
            return
        rr_index = 0
        for node in down_nodes:
            tasks = node.get("tasks", ["dummy_task"])
            for task in tasks:
                target = active_nodes[rr_index % len(active_nodes)]
                rr_index += 1
                print(f"[HA] Tâche '{task}' transférée {node.get('host')} -> {target.get('host')}")
                for cb in self.on_ha_redispatch:
                    try:
                        cb(node, target, task)
                    except Exception:
                        pass

    def _repair_node(self, node):
        node["status"] = "active"
        print(f"[AutoRepair] Nœud {node.get('host')} réparé.")
        for cb in self.on_repair:
            try:
                cb(node)
            except Exception:
                pass

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
