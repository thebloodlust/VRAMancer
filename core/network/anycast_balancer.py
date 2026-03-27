"""
Anycast Load Balancer — IPv6 health-aware tensor routing.
==========================================================

Routes tensor transfers to the best available node in the cluster
using Connectome synapse weights (Hebbian health scores) and
real-time liveness from AITP Sensing.

Strategies:
  - ``weighted``: Weighted random selection proportional to synapse strength.
  - ``least_latency``: Always pick the node with lowest current latency.
  - ``round_robin``: Cycle through healthy nodes (weighted order).

Integrates with:
  - ``core.network.connectome.global_connectome`` for health scores
  - ``core.network.aitp_sensing.AITPSensor`` for peer liveness
  - ``core.network.aitp_protocol.AITPProtocol`` for actual transport

Environment:
  - ``VRM_ANYCAST_STRATEGY``: ``weighted`` (default), ``least_latency``, ``round_robin``
  - ``VRM_ANYCAST_MIN_STRENGTH``: Minimum synapse strength to consider (default 0.1)
  - ``VRM_ANYCAST_GROUP``: IPv6 multicast group for anycast (default ``ff02::vrm``)
"""

import os
import time
import random
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("vramancer.anycast_balancer")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

# ── Configuration ──────────────────────────────────────────────────────
ANYCAST_STRATEGY = os.environ.get("VRM_ANYCAST_STRATEGY", "weighted")
ANYCAST_MIN_STRENGTH = float(os.environ.get("VRM_ANYCAST_MIN_STRENGTH", "0.1"))
ANYCAST_GROUP = os.environ.get("VRM_ANYCAST_GROUP", "ff02::vrm")

# ── Prometheus metrics (lazy) ──────────────────────────────────────────
_LB_ROUTES = None
_LB_FAILOVERS = None
_LB_NO_HEALTHY = None


def _init_lb_metrics():
    global _LB_ROUTES, _LB_FAILOVERS, _LB_NO_HEALTHY
    if _LB_ROUTES is not None:
        return
    try:
        from prometheus_client import Counter
        _LB_ROUTES = Counter(
            "vramancer_anycast_routes_total",
            "Tensor routing decisions",
            ["strategy", "node_id"],
        )
        _LB_FAILOVERS = Counter(
            "vramancer_anycast_failovers_total",
            "Failover events (primary unavailable)",
        )
        _LB_NO_HEALTHY = Counter(
            "vramancer_anycast_no_healthy_total",
            "Routing failures (no healthy node available)",
        )
    except Exception:
        pass


class AnycastNode:
    """Represents a routable node in the anycast group."""

    __slots__ = (
        "node_id", "ipv6", "port", "strength", "latency_ms",
        "last_seen", "vram_free", "active_transfers",
    )

    def __init__(
        self,
        node_id: str,
        ipv6: str,
        port: int = 9100,
        vram_free: int = 0,
    ):
        self.node_id = node_id
        self.ipv6 = ipv6
        self.port = port
        self.strength = 1.0
        self.latency_ms = 0.0
        self.last_seen = time.time()
        self.vram_free = vram_free
        self.active_transfers = 0

    def is_healthy(self, min_strength: float = ANYCAST_MIN_STRENGTH) -> bool:
        """Node is healthy if strength above threshold and seen recently."""
        stale = (time.time() - self.last_seen) > 30.0
        return self.strength >= min_strength and not stale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "ipv6": self.ipv6,
            "port": self.port,
            "strength": round(self.strength, 4),
            "latency_ms": round(self.latency_ms, 2),
            "vram_free": self.vram_free,
            "active_transfers": self.active_transfers,
            "healthy": self.is_healthy(),
        }


class AnycastLoadBalancer:
    """IPv6 anycast load balancer with health-aware routing.

    Maintains a registry of cluster nodes and selects the best target
    for each tensor transfer based on the chosen strategy and live
    health data from the Connectome.

    Usage::

        lb = AnycastLoadBalancer(strategy="weighted")
        lb.register_node("node-1", "fe80::1", port=9100)
        lb.register_node("node-2", "fe80::2", port=9100)

        # Sync weights from Connectome
        lb.sync_from_connectome()

        # Pick best node
        target = lb.select_target()
        if target:
            aitp.send_anycast(target.ipv6, layer_id, tensor_bytes)
            lb.record_result(target.node_id, success=True)
    """

    def __init__(self, strategy: str = None):
        self.strategy = strategy or ANYCAST_STRATEGY
        if self.strategy not in ("weighted", "least_latency", "round_robin"):
            logger.warning(
                f"Unknown anycast strategy '{self.strategy}', falling back to 'weighted'"
            )
            self.strategy = "weighted"

        self._lock = threading.Lock()
        self._nodes: Dict[str, AnycastNode] = {}
        self._rr_index = 0  # round-robin cursor

        _init_lb_metrics()
        logger.info(f"AnycastLoadBalancer initialized: strategy={self.strategy}")

    # ── Node management ────────────────────────────────────────────────

    def register_node(
        self,
        node_id: str,
        ipv6: str,
        port: int = 9100,
        vram_free: int = 0,
    ):
        """Add or update a node in the anycast group."""
        with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.ipv6 = ipv6
                node.port = port
                node.vram_free = vram_free
                node.last_seen = time.time()
            else:
                self._nodes[node_id] = AnycastNode(
                    node_id=node_id, ipv6=ipv6, port=port, vram_free=vram_free,
                )
                logger.info(f"Anycast: registered node {node_id} @ [{ipv6}]:{port}")

    def unregister_node(self, node_id: str):
        """Remove a node from the anycast group."""
        with self._lock:
            removed = self._nodes.pop(node_id, None)
        if removed:
            logger.info(f"Anycast: unregistered node {node_id}")

    def get_nodes(self) -> List[AnycastNode]:
        """Return all registered nodes."""
        with self._lock:
            return list(self._nodes.values())

    def get_healthy_nodes(self) -> List[AnycastNode]:
        """Return only nodes passing health check."""
        with self._lock:
            return [n for n in self._nodes.values() if n.is_healthy()]

    # ── Connectome integration ─────────────────────────────────────────

    def sync_from_connectome(self, connectome=None):
        """Pull synapse weights from the global Connectome into node health.

        This bridges the Hebbian learning (live latency/error tracking)
        into the load balancer's routing decisions.
        """
        if connectome is None:
            try:
                from core.network.connectome import global_connectome
                connectome = global_connectome
            except ImportError:
                return

        weights = connectome.get_all_weights()
        with self._lock:
            for node_id, node in self._nodes.items():
                if node_id in weights:
                    node.strength = weights[node_id]
                # Also pull latency if synapse exists
                syn = connectome.synapses.get(node_id)
                if syn is not None:
                    node.latency_ms = syn.latency_ms
                    node.last_seen = syn.last_ping

    def sync_from_sensing(self, sensor=None):
        """Pull peer liveness from AITP Sensing into node registry.

        Auto-registers new peers discovered via multicast and updates
        last_seen timestamps.
        """
        if sensor is None:
            try:
                from core.network.aitp_sensing import AITPSensor
                # No global singleton, caller must provide
                return
            except ImportError:
                return

        for uid, peer_info in sensor.peers.items():
            ipv6 = peer_info.get("ipv6", "")
            if not ipv6:
                continue
            hw = peer_info.get("hw", {})
            vram_free = hw.get("vram", 0)
            self.register_node(uid, ipv6, vram_free=vram_free)
            # Update last_seen from sensing
            with self._lock:
                if uid in self._nodes:
                    self._nodes[uid].last_seen = peer_info.get("last_seen", time.time())

    # ── Selection strategies ───────────────────────────────────────────

    def select_target(
        self,
        exclude: Optional[List[str]] = None,
    ) -> Optional[AnycastNode]:
        """Select the best node for the next transfer.

        Args:
            exclude: Node IDs to skip (e.g., already-tried nodes for retry).

        Returns:
            The selected AnycastNode, or None if no healthy node available.
        """
        healthy = self.get_healthy_nodes()
        if exclude:
            exclude_set = set(exclude)
            healthy = [n for n in healthy if n.node_id not in exclude_set]

        if not healthy:
            if _LB_NO_HEALTHY:
                _LB_NO_HEALTHY.inc()
            logger.warning("Anycast: no healthy node available")
            return None

        if self.strategy == "weighted":
            target = self._select_weighted(healthy)
        elif self.strategy == "least_latency":
            target = self._select_least_latency(healthy)
        elif self.strategy == "round_robin":
            target = self._select_round_robin(healthy)
        else:
            target = self._select_weighted(healthy)

        if target and _LB_ROUTES:
            _LB_ROUTES.labels(strategy=self.strategy, node_id=target.node_id).inc()

        return target

    def select_targets(
        self,
        count: int,
        exclude: Optional[List[str]] = None,
    ) -> List[AnycastNode]:
        """Select multiple distinct targets (for striped/parallel sends).

        Returns up to *count* healthy nodes, ordered by descending strength.
        """
        healthy = self.get_healthy_nodes()
        if exclude:
            exclude_set = set(exclude)
            healthy = [n for n in healthy if n.node_id not in exclude_set]

        # Sort by strength descending
        healthy.sort(key=lambda n: n.strength, reverse=True)
        return healthy[:count]

    def _select_weighted(self, nodes: List[AnycastNode]) -> AnycastNode:
        """Weighted random: probability proportional to synapse strength."""
        weights = [n.strength for n in nodes]
        total = sum(weights)
        if total <= 0:
            return random.choice(nodes)
        # random.choices does weighted selection
        return random.choices(nodes, weights=weights, k=1)[0]

    def _select_least_latency(self, nodes: List[AnycastNode]) -> AnycastNode:
        """Pick the node with lowest current latency."""
        return min(nodes, key=lambda n: n.latency_ms)

    def _select_round_robin(self, nodes: List[AnycastNode]) -> AnycastNode:
        """Cycle through nodes in strength-sorted order."""
        nodes.sort(key=lambda n: n.strength, reverse=True)
        with self._lock:
            idx = self._rr_index % len(nodes)
            self._rr_index += 1
        return nodes[idx]

    # ── Feedback ───────────────────────────────────────────────────────

    def record_result(self, node_id: str, success: bool):
        """Record transfer outcome for a node — feeds back into Connectome.

        This creates a Hebbian learning loop:
        success → strength increases → node gets more traffic
        failure → strength decreases → node gets less traffic
        """
        # Update local counter
        with self._lock:
            node = self._nodes.get(node_id)
            if node:
                if success:
                    node.active_transfers = max(0, node.active_transfers - 1)
                else:
                    node.active_transfers = max(0, node.active_transfers - 1)
                    if _LB_FAILOVERS:
                        _LB_FAILOVERS.inc()

        # Feed back to Connectome (Hebbian: reinforce or weaken)
        try:
            from core.network.connectome import global_connectome
            global_connectome.record_transfer_result(node_id, success)
        except ImportError:
            pass

    def select_and_send(
        self,
        aitp_protocol,
        layer_id: int,
        tensor_bytes: bytes,
        retries: int = 2,
    ) -> bool:
        """Select best node and send tensor, with automatic failover.

        On failure, retries with the next-best node (excluding failed ones).

        Args:
            aitp_protocol: AITPProtocol instance for transport.
            layer_id: Layer ID for the AITP packet.
            tensor_bytes: Raw tensor bytes to send.
            retries: Number of failover attempts.

        Returns:
            True if send succeeded, False if all retries exhausted.
        """
        excluded: List[str] = []

        for attempt in range(1 + retries):
            target = self.select_target(exclude=excluded)
            if target is None:
                logger.error(
                    f"Anycast: no healthy target (attempt {attempt + 1}/{1 + retries})"
                )
                return False

            try:
                aitp_protocol.send_anycast(
                    target.ipv6, layer_id, tensor_bytes,
                )
                self.record_result(target.node_id, success=True)
                logger.debug(
                    f"Anycast: sent to {target.node_id} [{target.ipv6}] "
                    f"(attempt {attempt + 1}, strategy={self.strategy})"
                )
                return True
            except Exception as e:
                logger.warning(
                    f"Anycast: send to {target.node_id} failed: {e} — "
                    f"trying failover ({attempt + 1}/{1 + retries})"
                )
                self.record_result(target.node_id, success=False)
                excluded.append(target.node_id)

        return False

    # ── Diagnostics ────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return full balancer state for monitoring/debugging."""
        nodes = self.get_nodes()
        healthy = [n for n in nodes if n.is_healthy()]
        return {
            "strategy": self.strategy,
            "total_nodes": len(nodes),
            "healthy_nodes": len(healthy),
            "anycast_group": ANYCAST_GROUP,
            "min_strength": ANYCAST_MIN_STRENGTH,
            "nodes": [n.to_dict() for n in nodes],
        }


# ── Singleton ──────────────────────────────────────────────────────────
_global_balancer: Optional[AnycastLoadBalancer] = None


def get_anycast_balancer() -> AnycastLoadBalancer:
    """Get or create the global anycast load balancer."""
    global _global_balancer
    if _global_balancer is None:
        _global_balancer = AnycastLoadBalancer()
    return _global_balancer
