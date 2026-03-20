import logging
import platform
import socket
import json
import threading

class ClusterMaster:
    def __init__(self, port=55555):
        self.port = port
        self.nodes = []
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self.listen)
        t.start()
        logging.info(f"[Master] Cluster master démarré sur port {self.port}")

    def listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", self.port))
        sock.settimeout(1)
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                node = json.loads(data.decode())
                if node not in self.nodes:
                    self.nodes.append(node)
                    logging.info(f"[Master] Nouveau nœud ajouté : {node['hostname']} ({node['os']})")
            except socket.timeout:
                continue
        sock.close()

    def stop(self):
        self.running = False
        logging.info("[Master] Cluster master arrêté.")

    def show_nodes(self):
        logging.info(f"[Master] Nœuds connectés : {len(self.nodes)}")
        for node in self.nodes:
            logging.info(f"- {node['hostname']} | OS: {node['os']} | CPU: {node['cpu']} | Arch: {node['arch']}")
        self.select_master_slave()

    def select_master_slave(self, override=None):
        """
        Sélectionne automatiquement le master selon benchmark (VRAM/CPU), override manuel possible.
        """
        if not self.nodes:
            logging.info("Aucun nœud détecté.")
            return
        if override:
            master = next((n for n in self.nodes if n['hostname'] == override), None)
            if master:
                logging.info(f"[Master] Override manuel : {master['hostname']} est master.")
            else:
                logging.info(f"[Master] Override : nœud non trouvé.")
            return
        # Benchmark simple : le nœud avec le plus de VRAM/CPU
        def score(n):
            vram = int(n.get('vram', 8192))
            cpu = int(n.get('cpu', 4))
            return vram + cpu * 1000
        master = max(self.nodes, key=score)
        logging.info(f"[Master] Sélection automatique : {master['hostname']} est master (score={score(master)})")
        for node in self.nodes:
            if node != master:
                logging.info(f"[Slave] {node['hostname']} est slave.")

if __name__ == "__main__":
    master = ClusterMaster()
    master.start()
    try:
        while True:
            cmd = input("[Master] Commande (show/stop): ").strip()
            if cmd == "show":
                master.show_nodes()
            elif cmd == "stop":
                master.stop()
                break
    except KeyboardInterrupt:
        master.stop()
