"""Block-level tensor transmission across machines.

Supports multiple transport protocols:
  - socketio: WebSocket-based (default, requires python-socketio)
  - tcp: Raw TCP socket (reliable, moderate latency)
  - udp: UDP datagram (low latency, unreliable for large payloads)
  - file: Shared filesystem/NVMe/USB4 (for large blocks)

For high-performance GPU-to-GPU transfers, prefer:
  - ``TransferManager`` (same-node, P2P/CPU-staged)
  - ``fibre_fastpath`` (cross-node RDMA/zero-copy TCP)
"""
import os
import zlib
import socket
import logging

_log = logging.getLogger("vramancer.transmission")

try:
    if os.environ.get("VRM_DISABLE_SOCKETIO", "0") in {"1", "true", "TRUE"}:
        raise ImportError("socketio disabled by env")
    import socketio  # type: ignore
except ImportError:
    socketio = None  # noqa: N816

from .packets import Packet

try:
    from utils.helpers import serialize_tensors, deserialize_tensors
except ImportError:
    def serialize_tensors(tensors):
        """Fallback serializer when utils.helpers is unavailable."""
        import pickle
        return pickle.dumps(tensors)

    def deserialize_tensors(data):
        """Fallback deserializer when utils.helpers is unavailable."""
        import pickle
        return pickle.loads(data)

sio = socketio.Client() if socketio else None

if sio:
    @sio.event  # type: ignore[misc]
    def connect():  # pragma: no cover
        _log.info("SocketIO connection established")

    @sio.event  # type: ignore[misc]
    def disconnect():  # pragma: no cover
        _log.info("SocketIO disconnected")


def send_block(tensors, shapes=None, dtypes=None, target_device="localhost",
               storage_path=None, compress=True, protocol="socketio",
               usb4_path=None, tcp_port=12345):
    """Send a block of weights or batch to a target machine."""
    payload = serialize_tensors(tensors)
    packet = Packet(payload)
    data = packet.pack()
    if compress:
        data = zlib.compress(data)
    if usb4_path:
        fname = f"block_{target_device}_usb4.bin"
        full_path = os.path.join(usb4_path, fname)
        with open(full_path, "wb") as f:
            f.write(data)
        _log.info("Block via USB4: %s (%d bytes)", full_path, len(data))
        return
    if storage_path:
        fname = f"block_{target_device}.bin"
        full_path = os.path.join(storage_path, fname)
        with open(full_path, "wb") as f:
            f.write(data)
        _log.info("Block to storage: %s (%d bytes)", full_path, len(data))
        return
    if protocol == "socketio":
        if sio is None or not sio.connected:
            raise ConnectionError("SocketIO not connected")
        sio.emit("vramancer_packet", data, namespace="/vram")
        _log.info("Block via SocketIO (%d bytes)", len(data))
    elif protocol == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        try:
            sock.connect((target_device, tcp_port))
            sock.sendall(data)
        finally:
            sock.close()
    elif protocol == "udp":
        if len(data) > 65000:
            raise ValueError(f"UDP payload too large ({len(data)} bytes)")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.sendto(data, (target_device, tcp_port))
        finally:
            sock.close()
    else:
        raise ValueError(f"Unknown protocol '{protocol}'")

if sio:
    @sio.on("vramancer_packet", namespace="/vram")  # type: ignore[misc]
    def on_packet(data):  # pragma: no cover
        _log.info("Received block packet (%d bytes)", len(data) if data else 0)


def start_client(server_url: str = "http://localhost:5000") -> bool:
    """Connect SocketIO client to a VRAMancer server.

    Args:
        server_url: Server URL (e.g., http://192.168.1.10:5030)

    Returns:
        True if connected successfully, False otherwise.
    """
    if not sio:
        _log.warning("SocketIO not available (install python-socketio)")
        return False
    try:
        sio.connect(server_url)
        return True
    except Exception as e:
        _log.error("SocketIO connection to %s failed: %s", server_url, e)
        return False
