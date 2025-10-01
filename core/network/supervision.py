import socket
import time
import threading

class NodeSupervisor:
    def __init__(self, nodes=None):
        self.nodes = nodes or []
        self.status = {}
        self.lock = threading.Lock()

    def register_node(self, node):
        with self.lock:
            self.nodes.append(node)
            self.status[node.host] = "active"
            print(f"[Supervision] Noeud {node.host} enregistré.")

    def update_status(self, node, status):
        with self.lock:
            self.status[node.host] = status
            print(f"[Supervision] Statut de {node.host} : {status}")

    def monitor(self):
        print("[Supervision] Démarrage de la supervision...")
        while True:
            with self.lock:
                for node in self.nodes:
                    # Simuler un check de santé
                    self.status[node.host] = "active" if node.status == "active" else "inactive"
            time.sleep(5)
            print(f"[Supervision] Statuts: {self.status}")

if __name__ == "__main__":
    from edge_iot import EdgeNode
    sup = NodeSupervisor()
    n1 = EdgeNode(host="raspberrypi")
    n2 = EdgeNode(host="jetson")
    sup.register_node(n1)
    sup.register_node(n2)
    threading.Thread(target=sup.monitor, daemon=True).start()
    n1.run_task("inference")
    n2.run_task("monitoring")
    time.sleep(10)
