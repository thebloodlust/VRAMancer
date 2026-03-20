"""VRAMancer Swarm Inference (P2P Mesh Network).

Implements decentralised pipeline-parallel inference across a mesh of
nodes discovered via mDNS/ClusterDiscovery. Each SwarmNode owns a
contiguous range of model layers and forwards intermediate activations
to the next node in the logical pipeline.

Protocol (TCP, length-prefixed JSON + binary):
  HEADER  = 8-byte big-endian payload length
  MESSAGE = JSON envelope: {"shape": [B, S, D], "dtype": "float32",
                             "src": "node-abc", "layer_range": [10, 20]}
  PAYLOAD = raw tensor bytes (numpy → bytes)

Supports VRM_MINIMAL_TEST=1 for stub-only mode in CI.
"""

from __future__ import annotations

import io
import json
import os
import socket
import struct
import threading
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger("vramancer.swarm")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore
    _NP = False

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False

try:
    from core.network.cluster_discovery import ClusterDiscovery
except ImportError:
    ClusterDiscovery = None


def _tensor_to_bytes(tensor: Any) -> Tuple[bytes, list, str]:
    """Serialize a tensor (torch or numpy) to raw bytes + metadata."""
    if _TORCH and isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    elif _NP and isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    raw = arr.tobytes()
    return raw, shape, dtype


def _bytes_to_tensor(raw: bytes, shape: list, dtype: str, use_torch: bool = False) -> Any:
    """Deserialize raw bytes back to a tensor."""
    if not _NP:
        raise RuntimeError("numpy required for tensor deserialization")
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
    if use_torch and _TORCH:
        return torch.from_numpy(arr.copy())
    return arr


class SwarmNode:
    """A node in the decentralised inference mesh.

    Parameters
    ----------
    node_id : str
        Unique name for this node.
    port : int
        Base port. Tensor data socket uses port+1.
    compute_fn : callable, optional
        Function ``(tensor, layer_start, layer_end) -> tensor`` that runs
        the local forward pass on the assigned layers. If None, a pass-through
        is used (useful for testing).
    """

    def __init__(
        self,
        node_id: str,
        port: int = 5050,
        compute_fn: Optional[Callable] = None,
    ):
        self.node_id = node_id
        self.port = port
        self.peers: Dict[str, dict] = {}
        self.is_running = False
        self._lock = threading.Lock()
        self._server_sock: Optional[socket.socket] = None
        self._compute_fn = compute_fn or self._passthrough
        self.discovery = ClusterDiscovery(port=port) if ClusterDiscovery else None
        self.layer_range: Tuple[int, int] = (0, 0)

        # Pipeline ordering — ring topology
        self._next_peer: Optional[dict] = None
        self._results: Dict[str, Any] = {}
        self._result_event = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        self.is_running = True
        if self.discovery:
            self.discovery.start()
            self.discovery.on_node_joined = self._handle_peer_joined
            self.discovery.on_node_left = self._handle_peer_left

        threading.Thread(
            target=self._listen_for_tensors, daemon=True, name=f"swarm-{self.node_id}"
        ).start()
        _logger.info("Swarm node %s started on port %d", self.node_id, self.port)

    def stop(self) -> None:
        self.is_running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        if self.discovery:
            self.discovery.stop()
        _logger.info("Swarm node %s stopped", self.node_id)

    # ── peer management ───────────────────────────────────────

    def _handle_peer_joined(self, peer_info: dict) -> None:
        peer_id = peer_info.get("node_id")
        if peer_id and peer_id != self.node_id:
            with self._lock:
                self.peers[peer_id] = peer_info
            _logger.info("Peer joined: %s (%s)", peer_id, peer_info.get("ip"))
            self._rebalance_model()

    def _handle_peer_left(self, peer_id: str) -> None:
        with self._lock:
            self.peers.pop(peer_id, None)
        _logger.info("Peer left: %s", peer_id)
        self._rebalance_model()

    def add_peer(self, peer_id: str, ip: str, port: int) -> None:
        """Manually register a peer (useful when not using mDNS)."""
        with self._lock:
            self.peers[peer_id] = {"node_id": peer_id, "ip": ip, "port": port}
        self._rebalance_model()

    def _rebalance_model(self) -> None:
        """Recompute layer range assignments across known peers."""
        with self._lock:
            all_ids = sorted([self.node_id] + list(self.peers.keys()))
        total_nodes = len(all_ids)
        my_index = all_ids.index(self.node_id)

        # Simple equal partitioning; total layers unknown → assign index range
        # Callers should set total_layers on the node before using.
        total_layers = getattr(self, 'total_layers', 0)
        if total_layers > 0:
            per_node = total_layers // total_nodes
            start = my_index * per_node
            end = start + per_node if my_index < total_nodes - 1 else total_layers
            self.layer_range = (start, end)
        _logger.info(
            "Rebalanced: %d nodes, %s owns layers %s",
            total_nodes, self.node_id, self.layer_range,
        )

        # Determine next peer in the ring
        if total_nodes > 1:
            next_idx = (my_index + 1) % total_nodes
            next_id = all_ids[next_idx]
            with self._lock:
                self._next_peer = self.peers.get(next_id)
        else:
            self._next_peer = None

    # ── network I/O ───────────────────────────────────────────

    def _listen_for_tensors(self) -> None:
        """Accept incoming tensor data from the previous node."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind(("0.0.0.0", self.port + 1))
        srv.listen(8)
        self._server_sock = srv

        while self.is_running:
            try:
                client, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handle_connection,
                args=(client,),
                daemon=True,
            ).start()

    def _handle_connection(self, sock: socket.socket) -> None:
        """Receive tensor, compute, forward to next peer or store result."""
        try:
            header_json, payload = self._recv_message(sock)
            if header_json is None:
                return

            shape = header_json["shape"]
            dtype = header_json["dtype"]
            src = header_json.get("src", "unknown")
            request_id = header_json.get("request_id", "")

            tensor = _bytes_to_tensor(payload, shape, dtype, use_torch=_TORCH)
            _logger.debug("Received tensor %s from %s", shape, src)

            # Local forward pass on assigned layers
            result = self._compute_fn(tensor, self.layer_range[0], self.layer_range[1])

            # Forward to next peer or store as final result
            if self._next_peer and self._next_peer.get("node_id") != src:
                self.send_tensor(
                    result,
                    self._next_peer["ip"],
                    self._next_peer["port"] + 1,
                    request_id=request_id,
                )
            else:
                # Completed the ring — store result
                self._results[request_id] = result
                self._result_event.set()
        except Exception as e:
            _logger.error("Error handling connection: %s", e)
        finally:
            sock.close()

    def send_tensor(
        self,
        tensor: Any,
        target_ip: str,
        target_port: int,
        request_id: str = "",
    ) -> None:
        """Send a tensor to a remote peer over TCP."""
        raw, shape, dtype = _tensor_to_bytes(tensor)

        header = json.dumps({
            "shape": shape,
            "dtype": dtype,
            "src": self.node_id,
            "request_id": request_id,
        }).encode("utf-8")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        try:
            sock.connect((target_ip, target_port))
            # Length-prefixed: header_len (4B) + header + payload_len (8B) + payload
            sock.sendall(struct.pack("!I", len(header)))
            sock.sendall(header)
            sock.sendall(struct.pack("!Q", len(raw)))
            sock.sendall(raw)
        finally:
            sock.close()

    @staticmethod
    def _recv_message(sock: socket.socket) -> Tuple[Optional[dict], Optional[bytes]]:
        """Receive a length-prefixed message from a socket."""
        try:
            # Header length (4 bytes)
            hdr_len_bytes = SwarmNode._recv_exact(sock, 4)
            if not hdr_len_bytes:
                return None, None
            hdr_len = struct.unpack("!I", hdr_len_bytes)[0]

            # Header (JSON)
            hdr_bytes = SwarmNode._recv_exact(sock, hdr_len)
            if not hdr_bytes:
                return None, None
            header = json.loads(hdr_bytes.decode("utf-8"))

            # Payload length (8 bytes)
            pay_len_bytes = SwarmNode._recv_exact(sock, 8)
            if not pay_len_bytes:
                return header, None
            pay_len = struct.unpack("!Q", pay_len_bytes)[0]

            # Payload
            payload = SwarmNode._recv_exact(sock, pay_len)
            return header, payload
        except Exception as e:
            _logger.error("Failed to receive message: %s", e)
            return None, None

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> Optional[bytes]:
        """Read exactly nbytes from a socket."""
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = sock.recv(min(nbytes - len(buf), 65536))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    # ── inference entry point ─────────────────────────────────

    def infer(self, tensor: Any, timeout: float = 30.0) -> Any:
        """Run a forward pass across the swarm starting from this node.

        Sends the tensor through the ring of peers and waits for
        the result to come back.
        """
        import uuid
        request_id = uuid.uuid4().hex[:8]
        self._result_event.clear()

        # Local compute first
        result = self._compute_fn(tensor, self.layer_range[0], self.layer_range[1])

        # If no peers, return local result directly
        if not self._next_peer:
            return result

        # Send to next peer
        self.send_tensor(
            result,
            self._next_peer["ip"],
            self._next_peer["port"] + 1,
            request_id=request_id,
        )

        # Wait for result
        if self._result_event.wait(timeout=timeout):
            return self._results.pop(request_id, result)
        _logger.warning("Swarm inference timed out after %.1fs", timeout)
        return result

    @staticmethod
    def _passthrough(tensor: Any, layer_start: int, layer_end: int) -> Any:
        """Default no-op compute function."""
        return tensor

