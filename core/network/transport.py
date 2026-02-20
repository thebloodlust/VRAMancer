"""Network transport abstraction for inter-node GPU block transfer.

Provides a TCP-based transport with optional TLS encryption.
For high-performance transfers, use ``fibre_fastpath.py`` (RDMA/zero-copy)
or ``TransferManager`` (intra-node P2P/CPU-staged).

This module handles:
  - TCP connections with configurable timeout
  - Optional TLS encryption via ssl module
  - Binary payload send/receive with length-prefixed framing
  - Connection pooling (basic, for multi-node setups)

Usage:
    transport = Transport(host="192.168.1.10", port=9000, secure=True)
    if transport.connect():
        transport.send(payload_bytes)
        response = transport.receive()
        transport.close()
"""
from dataclasses import dataclass, field
from typing import Optional
import socket
import struct
import logging
import os

_log = logging.getLogger("vramancer.transport")

# Header: 4-byte big-endian uint32 payload length
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_MAX_PAYLOAD = 256 * 1024 * 1024  # 256 MB safety limit
_DEFAULT_TIMEOUT = int(os.environ.get("VRM_TRANSPORT_TIMEOUT", "30"))


@dataclass
class Transport:
    """TCP transport with optional TLS and length-prefixed framing."""
    host: str = "localhost"
    port: int = 9000
    secure: bool = False
    timeout: int = _DEFAULT_TIMEOUT
    _sock: Optional[socket.socket] = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False, repr=False)

    def connect(self) -> bool:
        """Establish TCP connection (with optional TLS)."""
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.settimeout(self.timeout)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            if self.secure:
                import ssl
                ctx = ssl.create_default_context()
                # Allow self-signed certs in dev (controlled by env var)
                if os.environ.get("VRM_TRANSPORT_INSECURE", "0") == "1":
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                self._sock = ctx.wrap_socket(raw, server_hostname=self.host)
            else:
                self._sock = raw

            self._sock.connect((self.host, self.port))
            self._connected = True
            _log.debug("Connected to %s:%d (secure=%s)", self.host, self.port, self.secure)
            return True
        except Exception as e:
            _log.error("Connection to %s:%d failed: %s", self.host, self.port, e)
            self._connected = False
            return False

    def send(self, payload: bytes) -> int:
        """Send binary payload with length-prefixed framing.

        Returns number of payload bytes sent (excluding header).
        """
        if not self._connected or self._sock is None:
            raise ConnectionError("Not connected — call connect() first")
        if len(payload) > _MAX_PAYLOAD:
            raise ValueError(f"Payload too large: {len(payload)} bytes (max {_MAX_PAYLOAD})")

        header = struct.pack(_HEADER_FMT, len(payload))
        self._sock.sendall(header + payload)
        _log.debug("Sent %d bytes to %s:%d", len(payload), self.host, self.port)
        return len(payload)

    def receive(self) -> Optional[bytes]:
        """Receive a length-prefixed binary payload.

        Returns payload bytes, or None on connection close/error.
        """
        if not self._connected or self._sock is None:
            raise ConnectionError("Not connected — call connect() first")

        # Read header
        header_buf = self._recv_exact(_HEADER_SIZE)
        if header_buf is None:
            return None

        (length,) = struct.unpack(_HEADER_FMT, header_buf)
        if length > _MAX_PAYLOAD:
            _log.error("Incoming payload too large: %d bytes", length)
            return None
        if length == 0:
            return b""

        payload = self._recv_exact(length)
        if payload is not None:
            _log.debug("Received %d bytes from %s:%d", len(payload),
                        self.host, self.port)
        return payload

    def _recv_exact(self, nbytes: int) -> Optional[bytes]:
        """Receive exactly nbytes from socket."""
        data = bytearray()
        while len(data) < nbytes:
            try:
                chunk = self._sock.recv(nbytes - len(data))
                if not chunk:
                    _log.warning("Connection closed by peer")
                    return None
                data.extend(chunk)
            except socket.timeout:
                _log.error("Receive timeout after %ds", self.timeout)
                return None
            except Exception as e:
                _log.error("Receive error: %s", e)
                return None
        return bytes(data)

    def close(self) -> None:
        """Close the TCP connection."""
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        self._connected = False
        _log.debug("Connection closed to %s:%d", self.host, self.port)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


__all__ = ["Transport"]
