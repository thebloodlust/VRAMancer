import platform
import socket
import time

class EdgeNode:
    def __init__(self, host=None, capabilities=None):
        self.host = host or socket.gethostname()
        self.capabilities = capabilities or self.detect_capabilities()
        self.status = "active"

    def detect_capabilities(self):
        # Détection simplifiée pour edge/IoT
        return {
            "cpu": platform.processor(),
            "arch": platform.machine(),
            "ram": 1024,  # Simulé, à adapter
            "os": platform.system(),
            "type": "edge" if self.is_edge() else "standard"
        }

    def is_edge(self):
        # Détection simple : Raspberry Pi, Jetson, etc.
        return any(x in platform.platform().lower() for x in ["raspberry", "jetson", "arm"])

    def run_task(self, task):
        print(f"[EdgeNode] Tâche '{task}' lancée sur {self.host} ({self.capabilities['type']})")
        time.sleep(1)
        print(f"[EdgeNode] Tâche '{task}' terminée sur {self.host}")

if __name__ == "__main__":
    node = EdgeNode()
    node.run_task("inference")
