"""
VRAMancer Swarm Inference (P2P Mesh Network)
--------------------------------------------
Ce module implémente le concept de "Swarm Inference" (BitTorrent de l'IA).
Au lieu d'avoir un Master centralisé, les nœuds se découvrent via mDNS
et forment un réseau maillé (Mesh). Chaque nœud stocke une partie du modèle
et participe à la génération des tokens de manière décentralisée.
"""

import os
import time
import threading
import json
import socket
from typing import Dict, Any

try:
    from core.network.cluster_discovery import ClusterDiscovery
except ImportError:
    ClusterDiscovery = None

class SwarmNode:
    def __init__(self, node_id: str, port: int = 5050):
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, Any] = {}
        self.is_running = False
        self.discovery = ClusterDiscovery(port=port) if ClusterDiscovery else None
        self.model_chunks = [] # Les couches du modèle que ce nœud héberge
        
    def start(self):
        self.is_running = True
        if self.discovery:
            self.discovery.start()
            # S'abonner aux événements de découverte
            self.discovery.on_node_joined = self._handle_peer_joined
            self.discovery.on_node_left = self._handle_peer_left
            
        # Démarrer le serveur d'écoute P2P
        threading.Thread(target=self._listen_for_tensors, daemon=True).start()
        print(f"[Swarm] Nœud {self.node_id} démarré sur le port {self.port}. En attente de pairs...")

    def _handle_peer_joined(self, peer_info):
        peer_id = peer_info.get("node_id")
        if peer_id and peer_id != self.node_id:
            self.peers[peer_id] = peer_info
            print(f"[Swarm] Nouveau pair détecté : {peer_id} ({peer_info.get('ip')})")
            self._rebalance_model()

    def _handle_peer_left(self, peer_id):
        if peer_id in self.peers:
            del self.peers[peer_id]
            print(f"[Swarm] Pair perdu : {peer_id}")
            self._rebalance_model()

    def _rebalance_model(self):
        """Répartit les couches du modèle équitablement entre tous les pairs connus."""
        total_nodes = len(self.peers) + 1
        print(f"[Swarm] Rééquilibrage du modèle sur {total_nodes} nœuds...")
        # Logique de redistribution (ex: Nœud 1 prend couches 0-10, Nœud 2 prend 11-20, etc.)

    def _listen_for_tensors(self):
        """Écoute les tenseurs (activations) provenant du nœud précédent dans le pipeline."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", self.port + 1)) # Port de données
        server.listen(5)
        
        while self.is_running:
            try:
                client, addr = server.accept()
                # Recevoir le tenseur, faire le calcul (forward pass), et l'envoyer au nœud suivant
                # (Simulation pour l'instant)
                client.close()
            except Exception as e:
                if self.is_running:
                    print(f"[Swarm] Erreur réseau : {e}")

    def stop(self):
        self.is_running = False
        if self.discovery:
            self.discovery.stop()
        print(f"[Swarm] Nœud {self.node_id} arrêté.")

if __name__ == "__main__":
    import uuid
    node = SwarmNode(node_id=f"node-{uuid.uuid4().hex[:6]}")
    node.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()
