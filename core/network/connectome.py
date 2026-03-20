"""
VRAMancer Connectome (Neuroplasticity Engine)
==================================================
Implémente la Loi de Hebb pour les réseaux distribués : 
"Des neurones qui s'excitent ensemble se lient entre eux".

Ce module surveille en permanence (en tâche de fond) la qualité des "synapses" (liens réseau ou PCIe) 
vers les noeuds distants/locaux.
Si une connexion subit des latences ou des erreurs, son "poids synaptique" (Synaptic Strength) diminue.
Le PlacementEngine utilise ce score pour délaisser organiquement les noeuds malades,
imitant la plasticité cérébrale sans nécessiter d'intervention humaine.
"""

import time
import math
import socket
import threading
import logging
from typing import Dict, List, Any, Optional


class Synapse:
    """Représente une connexion physique/logique vers un noeud."""
    # Number of initial pings used to establish latency baseline
    CALIBRATION_SAMPLES = 10

    def __init__(self, target_id: str, ip_address: str, port: int = 5000):
        self.target_id = target_id
        self.ip_address = ip_address
        self.port = port
        self._lock = threading.Lock()

        # Heuristiques
        self.latency_ms = 1.0
        self.error_rate = 0.0
        self.total_transfers = 0
        self.failed_transfers = 0

        # Poids Synaptique (1.0 = parfait, 0.0 = mort)
        self.strength = 1.0
        self.last_ping = time.time()

        # Auto-calibration: baseline latency per synapse
        self._calibration_pings: list = []
        self._baseline_ms: float | None = None  # set after CALIBRATION_SAMPLES pings

    def update_latency(self, new_latency_ms: float):
        """Met à jour via une Moyenne Mobile Exponentielle (EMA) = adaptation douce."""
        with self._lock:
            alpha = 0.3
            self.latency_ms = (alpha * new_latency_ms) + ((1 - alpha) * self.latency_ms)
            # Auto-calibrate: collect first N pings to establish baseline
            if self._baseline_ms is None:
                self._calibration_pings.append(new_latency_ms)
                if len(self._calibration_pings) >= self.CALIBRATION_SAMPLES:
                    # Use median as robust baseline
                    sorted_pings = sorted(self._calibration_pings)
                    mid = len(sorted_pings) // 2
                    self._baseline_ms = sorted_pings[mid]
            self._recalculate_strength()
            self.last_ping = time.time()

    def record_transfer(self, success: bool):
        with self._lock:
            self.total_transfers += 1
            if not success:
                self.failed_transfers += 1
            self.error_rate = self.failed_transfers / max(1, self.total_transfers)
            self._recalculate_strength()

    def _recalculate_strength(self):
        """Formule de dégradation synaptique : exponentielle latence × fiabilité².

        Uses auto-calibrated baseline instead of hardcoded 20ms onset.
        The onset is set to 2× the measured baseline (tolerates normal jitter).
        Decay rate adapts to the baseline: faster decay for LAN, slower for WAN.
        """
        # Adaptive onset: 2× baseline, minimum 5ms, default 20ms during calibration
        onset = 20.0
        decay = 0.01
        if self._baseline_ms is not None:
            onset = max(5.0, self._baseline_ms * 2.0)
            # Decay: 0.05 for LAN (<5ms), 0.005 for WAN (>100ms), linear interpolation
            decay = max(0.005, min(0.05, 0.05 - 0.00045 * self._baseline_ms))
        decay_factor = math.exp(-decay * max(0, self.latency_ms - onset))
        reliability = max(0.0, 1.0 - self.error_rate) ** 2
        self.strength = max(0.01, decay_factor * reliability)

    def to_dict(self) -> Dict[str, Any]:
        """Export synapse state for monitoring/placement."""
        with self._lock:
            return {
                "target_id": self.target_id,
                "ip": self.ip_address,
                "port": self.port,
                "latency_ms": round(self.latency_ms, 2),
                "error_rate": round(self.error_rate, 4),
                "strength": round(self.strength, 4),
                "total_transfers": self.total_transfers,
                "last_ping": self.last_ping,
            }


class Connectome:
    """La Matrice Globale de toutes les synapses (cerveau distribué).

    Thread-safe: all synapse mutations are guarded by locks.
    """
    def __init__(self, interval_s: float = 5.0):
        self.log = logging.getLogger("vramancer.connectome")
        self._lock = threading.Lock()
        self.synapses: Dict[str, Synapse] = {}
        self.interval_s = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_node(self, node_id: str, ip: str, port: int = 5000):
        with self._lock:
            if node_id not in self.synapses:
                self.synapses[node_id] = Synapse(node_id, ip, port)
                self.log.info("[Neuroplasticité] Nouvelle synapse vers %s (%s)", node_id, ip)

    def remove_node(self, node_id: str):
        with self._lock:
            self.synapses.pop(node_id, None)

    def get_synaptic_weight(self, node_id: str) -> float:
        """Récupère la force du lien. 1.0 = local/parfait."""
        if node_id == "local":
            return 1.0
        with self._lock:
            syn = self.synapses.get(node_id)
        if syn is None:
            return 1.0
        return syn.strength

    def get_all_weights(self) -> Dict[str, float]:
        """Return {node_id: strength} for all known nodes (for PlacementEngine)."""
        with self._lock:
            return {nid: s.strength for nid, s in self.synapses.items()}

    def get_ranked_nodes(self) -> List[str]:
        """Return node_ids sorted by descending synaptic strength."""
        weights = self.get_all_weights()
        return sorted(weights, key=weights.get, reverse=True)

    def record_transfer_result(self, node_id: str, success: bool):
        with self._lock:
            syn = self.synapses.get(node_id)
        if syn is not None:
            syn.record_transfer(success)
            if not success:
                self.log.warning("[Neuroplasticité] Perte vers %s — force: %.2f",
                                 node_id, syn.strength)

    def snapshot(self) -> List[Dict[str, Any]]:
        """Full synapse state for monitoring dashboards."""
        with self._lock:
            return [s.to_dict() for s in self.synapses.values()]

    def start_heartbeat(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="ConnectomePulse"
        )
        self._thread.start()
        self.log.info("[Connectome] Battement de coeur synaptique initié.")

    def stop(self):
        self._running = False

    def _heartbeat_loop(self):
        """Ping les noeuds périodiquement pour adapter le réseau nerveux vivant."""
        while self._running:
            with self._lock:
                targets = list(self.synapses.values())
            for synapse in targets:
                self._ping_synapse(synapse)
            time.sleep(self.interval_s)

    def _ping_synapse(self, synapse: Synapse):
        start_time = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((synapse.ip_address, synapse.port))
            latency = (time.perf_counter() - start_time) * 1000
            synapse.update_latency(latency)
        except OSError:
            synapse.update_latency(5000)
            synapse.record_transfer(False)
        finally:
            sock.close()


# Singleton global (La Matrice Commune)
global_connectome = Connectome()
