"""
VRAMancer Connectome (Neuroplasticity Engine)
==================================================
Implémente la Loi de Hebb pour les réseaux distribués : 
"Des neurones qui s'excitent ensemble se lient entre eux".

Ce module surveille en permanence (en tâche de fond) la qualité des "synapses" (liens réseau ou PCIe) 
vers les noeuds distants/locaux.
Si une connexion subit des latences ou des erreurs, son "poids synaptique" (Synaptic Strength) diminue.
Le PlacementEngine utilisera ce score pour délaisser organiquement les noeuds malades,
imitant la plasticité cérébrale sans nécessiter d'intervention humaine.
"""

import time
import math
import socket
import threading
import logging
from typing import Dict, List, Any

class Synapse:
    """Représente une connexion physique/logique vers un noeud."""
    def __init__(self, target_id: str, ip_address: str, port: int = 5000):
        self.target_id = target_id
        self.ip_address = ip_address
        self.port = port
        
        # Heuristiques
        self.latency_ms = 1.0  # par défaut, très court
        self.error_rate = 0.0
        self.total_transfers = 0
        self.failed_transfers = 0
        
        # Poids Synaptique (1.0 = parfait, 0.0 = mort)
        self.strength = 1.0
        self.last_ping = time.time()

    def update_latency(self, new_latency_ms: float):
        """Met à jour via une Moyenne Mobile Exponentielle (EMA) = adaptation douce."""
        alpha = 0.3
        self.latency_ms = (alpha * new_latency_ms) + ((1 - alpha) * self.latency_ms)
        self._recalculate_strength()

    def record_transfer(self, success: bool):
        self.total_transfers += 1
        if not success:
            self.failed_transfers += 1
        self.error_rate = self.failed_transfers / self.total_transfers
        self._recalculate_strength()

    def _recalculate_strength(self):
        """
        Formule de dégradation synaptique locale :
        On pénalise exponentiellement avec la latence.
        On écrase violemment si le taux d'erreur monte.
        """
        # La force diminue si la latence dépasse 50ms (0.05s)
        # S = exp(-lambda * latency) * (1 - error_rate)^2
        decay_factor = math.exp(-0.01 * max(0, self.latency_ms - 20)) 
        reliability = max(0.0, 1.0 - self.error_rate) ** 2
        self.strength = decay_factor * reliability
        
        # Plancher
        if self.strength < 0.01:
            self.strength = 0.01


class Connectome:
    """La Matrice Globale de toutes les synapses (cerveau distribué)."""
    def __init__(self, interval_s: float = 5.0):
        self.log = logging.getLogger("vramancer.connectome")
        self.synapses: Dict[str, Synapse] = {}
        self.interval_s = interval_s
        self._running = False
        self._thread = None
        
    def add_node(self, node_id: str, ip: str, port: int = 5000):
        if node_id not in self.synapses:
            self.synapses[node_id] = Synapse(node_id, ip, port)
            self.log.info(f"🌱 [Neuroplasticité] Nouvelle synapse formée vers {node_id} ({ip}).")

    def get_synaptic_weight(self, node_id: str) -> float:
        """Récupère la force du lien. 1.0 = local/parfait."""
        # 1.0 pour le localhost ou noeud inconnu (optimisme)
        if node_id == "local" or node_id not in self.synapses:
            return 1.0
        return self.synapses[node_id].strength

    def record_transfer_result(self, node_id: str, success: bool):
        if node_id in self.synapses:
            self.synapses[node_id].record_transfer(success)
            if not success:
                self.log.warning(f"⚠️ [Neuroplasticité] Perte de paquet vers {node_id}. Atrophie synaptique déclenchée (Force: {self.synapses[node_id].strength:.2f}).")

    def start_heartbeat(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="ConnectomePulse")
        self._thread.start()
        self.log.info("🧠 [Connectome] Battement de coeur synaptique initié.")

    def stop(self):
        self._running = False

    def _heartbeat_loop(self):
        """Tente de pinger les noeuds périodiquement pour adapter le réseau nerveu vivant."""
        while self._running:
            for node_id, synapse in list(self.synapses.items()):
                self._ping_synapse(synapse)
            time.sleep(self.interval_s)

    def _ping_synapse(self, synapse: Synapse):
        start_time = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            # Simple check TCP pour évaluer la conductivité nerveuse
            sock.connect((synapse.ip_address, synapse.port))
            latency = (time.perf_counter() - start_time) * 1000
            synapse.update_latency(latency)
        except OSError:
            # Echec du ping = sclérose de la synapse
            synapse.update_latency(5000)  # Puni avec une latence factice de 5s
            synapse.record_transfer(False) 
        finally:
            sock.close()

# Singleton global (La Matrice Commune)
global_connectome = Connectome()
