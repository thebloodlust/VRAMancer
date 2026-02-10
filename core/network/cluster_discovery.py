# core/network/cluster_discovery.py
"""Production cluster discovery — multi-OS auto-discovery with heartbeat.

Discovery methods (priority order):
  1. mDNS / ZeroConf (if zeroconf library available)
  2. UDP broadcast (fallback, works on any LAN)
  3. Static node list (from config)

Features:
  - GPU enumeration in node advertisements
  - Heartbeat thread with configurable interval
  - Node membership tracking (join/leave events)
  - USB4/Thunderbolt hot-plug detection (Linux pyudev, macOS IOKit, Windows stub)
  - Multi-OS: Linux, macOS, Windows
  - Thread-safe node registry
"""

from __future__ import annotations

import os
import socket
import platform
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any

try:
    from core.logger import LoggerAdapter
    _logger = LoggerAdapter("discovery")
except Exception:  # pragma: no cover
    import logging
    _logger = logging.getLogger("vramancer.discovery")

try:
    from core.metrics import (
        ORCH_PLACEMENTS as _ORCH_PLACEMENTS,
    )
except Exception:  # pragma: no cover
    _ORCH_PLACEMENTS = None

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    from core.utils import enumerate_devices, detect_backend
except ImportError:  # pragma: no cover
    def enumerate_devices():
        return []
    def detect_backend():
        return "cpu"


# =========================================================================
# Node info
# =========================================================================

def get_local_info() -> Dict[str, Any]:
    """Gather comprehensive local node information."""
    gpu_list = []
    if not _MINIMAL:
        try:
            for d in enumerate_devices():
                gpu_list.append({
                    "id": d.get("id", "unknown"),
                    "backend": d.get("backend", "cpu"),
                    "name": d.get("name", "unknown"),
                    "total_memory": d.get("total_memory"),
                })
        except Exception:
            pass

    ram_bytes = _get_total_ram()

    return {
        "hostname": socket.gethostname(),
        "ip": _get_local_ip(),
        "cpu": platform.processor() or platform.machine(),
        "arch": platform.machine(),
        "os": platform.system(),
        "platform_type": detect_platform_type(),
        "python_version": platform.python_version(),
        "gpus": gpu_list,
        "gpu_count": len(gpu_list),
        "ram_bytes": ram_bytes,
        "vramancer_port": int(os.environ.get("VRM_API_PORT", 5000)),
        "timestamp": time.time(),
    }


def detect_platform_type() -> str:
    """Detect platform type with GPU-awareness."""
    system = platform.system().lower()
    arch = platform.machine().lower()

    if system == "darwin":
        if "arm" in arch or "aarch64" in arch:
            return "Apple Silicon"
        return "Apple Intel"
    elif system == "windows":
        return "Windows"
    elif system == "linux":
        # Check for GPU type
        backend = "cpu"
        try:
            backend = detect_backend()
        except Exception:
            pass
        if backend == "rocm":
            return "Linux AMD ROCm"
        elif backend == "cuda":
            return "Linux NVIDIA CUDA"
        elif backend == "mps":
            return "Linux MPS"  # shouldn't happen but handle it

        # CPU type
        cpu = platform.processor().lower()
        if "amd" in cpu or "epyc" in cpu or "ryzen" in cpu or "threadripper" in cpu:
            return "Linux AMD CPU"
        elif "intel" in cpu or "xeon" in cpu or "core" in cpu:
            return "Linux Intel CPU"
        elif "arm" in arch or "aarch64" in arch:
            return "Linux ARM"
        return "Linux Generic"
    return system


def _get_local_ip() -> str:
    """Get local IP address reliably."""
    try:
        # Connect to external address to determine local IP (no data sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _get_total_ram() -> int:
    """Get total system RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    # Fallback for Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    # macOS fallback
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


# =========================================================================
# Cluster discovery (UDP broadcast)
# =========================================================================

class ClusterDiscovery:
    """Production cluster discovery with heartbeat and membership tracking.

    Usage:
        disco = ClusterDiscovery(port=55555)
        disco.start()               # start discovery + heartbeat
        nodes = disco.get_nodes()    # get all known nodes
        disco.on_join(callback)      # register join callback
        disco.stop()
    """

    PROTOCOL_VERSION = 2
    SERVICE_TYPE = "_vramancer._tcp.local."

    def __init__(self, port: int = 55555, heartbeat_interval: float = 10.0,
                 node_timeout: float = 30.0):
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout

        self._lock = threading.Lock()
        self._nodes: Dict[str, Dict[str, Any]] = {}  # hostname -> info
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None

        self._on_join_callbacks: List[Callable] = []
        self._on_leave_callbacks: List[Callable] = []
        self._shutdown = threading.Event()

        self._local_info = get_local_info()
        self._mdns_browser = None

        # Stats
        self._stats = {
            "nodes_joined": 0,
            "nodes_left": 0,
            "heartbeats_sent": 0,
            "heartbeats_failed": 0,
            "udp_errors": 0,
            "mdns_active": False,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start discovery, heartbeat, and cleanup threads."""
        if self._running:
            return
        self._running = True

        # Register self
        self._register_node(self._local_info)

        # Try mDNS first
        if self._start_mdns():
            _logger.info("Discovery started via mDNS/ZeroConf (port=%d)", self.port)
        else:
            _logger.info("Discovery started via UDP broadcast (port=%d)", self.port)

        # UDP listener (always active as fallback)
        self._listener_thread = threading.Thread(
            target=self._udp_listener, daemon=True, name="cluster-udp-listener"
        )
        self._listener_thread.start()

        # Heartbeat sender
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="cluster-heartbeat"
        )
        self._heartbeat_thread.start()

        # Node cleanup (remove stale nodes)
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="cluster-cleanup"
        )
        self._cleanup_thread.start()

    def stop(self) -> None:
        """Stop all discovery threads."""
        self._running = False
        self._shutdown.set()
        self._stop_mdns()
        _logger.info("Discovery stopped")

    @property
    def stats(self) -> Dict[str, Any]:
        """Return discovery statistics."""
        with self._lock:
            s = dict(self._stats)
            s["active_nodes"] = len(self._nodes)
            s["running"] = self._running
        return s

    def __repr__(self) -> str:
        return (f"ClusterDiscovery(port={self.port}, "
                f"nodes={self.node_count()}, running={self._running})")

    def get_nodes(self) -> List[Dict[str, Any]]:
        """Return all known nodes (thread-safe copy)."""
        with self._lock:
            return list(self._nodes.values())

    def get_node(self, hostname: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific node."""
        with self._lock:
            return self._nodes.get(hostname)

    def on_join(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for node join events."""
        self._on_join_callbacks.append(callback)

    def on_leave(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for node leave events."""
        self._on_leave_callbacks.append(callback)

    def node_count(self) -> int:
        """Number of known nodes."""
        with self._lock:
            return len(self._nodes)

    # ------------------------------------------------------------------
    # mDNS / ZeroConf (preferred discovery method)
    # ------------------------------------------------------------------

    def _start_mdns(self) -> bool:
        """Try to start mDNS discovery using zeroconf library."""
        try:
            from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo  # type: ignore

            self._zeroconf = Zeroconf()
            info = ServiceInfo(
                self.SERVICE_TYPE,
                f"vramancer-{self._local_info['hostname']}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(self._local_info["ip"])],
                port=self._local_info.get("vramancer_port", 5000),
                properties={
                    b"version": str(self.PROTOCOL_VERSION).encode(),
                    b"hostname": self._local_info["hostname"].encode(),
                    b"gpu_count": str(self._local_info["gpu_count"]).encode(),
                    b"platform": self._local_info["platform_type"].encode(),
                },
            )
            self._zeroconf.register_service(info)
            self._mdns_info = info

            class _Listener:
                def __init__(self, parent):
                    self.parent = parent
                def add_service(self, zc, stype, name):
                    info = zc.get_service_info(stype, name)
                    if info:
                        self.parent._handle_mdns_service(info)
                def remove_service(self, zc, stype, name):
                    pass
                def update_service(self, zc, stype, name):
                    info = zc.get_service_info(stype, name)
                    if info:
                        self.parent._handle_mdns_service(info)

            self._mdns_browser = ServiceBrowser(
                self._zeroconf, self.SERVICE_TYPE, _Listener(self)
            )
            self._stats["mdns_active"] = True
            return True
        except ImportError:
            _logger.debug("zeroconf not available, falling back to UDP")
            return False
        except Exception as exc:
            _logger.debug("mDNS startup failed: %s", exc)
            return False

    def _stop_mdns(self) -> None:
        """Cleanup mDNS resources."""
        try:
            if hasattr(self, "_zeroconf"):
                if hasattr(self, "_mdns_info"):
                    self._zeroconf.unregister_service(self._mdns_info)
                self._zeroconf.close()
        except Exception:
            pass

    def _handle_mdns_service(self, info: Any) -> None:
        """Process a discovered mDNS service."""
        try:
            props = info.properties or {}
            hostname = props.get(b"hostname", b"unknown").decode()
            ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else "unknown"
            node_info = {
                "hostname": hostname,
                "ip": ip,
                "platform_type": props.get(b"platform", b"unknown").decode(),
                "gpu_count": int(props.get(b"gpu_count", b"0")),
                "vramancer_port": info.port,
                "timestamp": time.time(),
                "discovery": "mdns",
            }
            self._register_node(node_info)
        except Exception as exc:
            _logger.debug("mDNS service parse error: %s", exc)

    # ------------------------------------------------------------------
    # UDP broadcast discovery
    # ------------------------------------------------------------------

    def _udp_listener(self) -> None:
        """Listen for UDP discovery broadcasts."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass  # SO_REUSEPORT not available on some platforms
            sock.bind(("", self.port))
            sock.settimeout(2.0)
        except Exception as exc:
            _logger.error("UDP listener bind failed: %s", exc)
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(8192)
                node = json.loads(data.decode("utf-8"))
                node["timestamp"] = time.time()
                node["discovery"] = "udp"
                self._register_node(node)
            except socket.timeout:
                continue
            except Exception as exc:
                self._stats["udp_errors"] += 1
                _logger.debug("UDP recv error: %s", exc)
        sock.close()

    def _heartbeat_loop(self) -> None:
        """Periodically broadcast local info via UDP."""
        while self._running:
            try:
                self._local_info["timestamp"] = time.time()
                msg = json.dumps(self._local_info).encode("utf-8")
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(2.0)
                sock.sendto(msg, ("<broadcast>", self.port))
                sock.close()
                self._stats["heartbeats_sent"] += 1
            except Exception as exc:
                _logger.debug("Heartbeat send error: %s", exc)
                self._stats["heartbeats_failed"] += 1
            self._shutdown.wait(timeout=self.heartbeat_interval)

    # ------------------------------------------------------------------
    # Node membership
    # ------------------------------------------------------------------

    def _register_node(self, info: Dict[str, Any]) -> None:
        """Register or update a node in the registry."""
        hostname = info.get("hostname", "unknown")
        with self._lock:
            is_new = hostname not in self._nodes
            self._nodes[hostname] = info
        if is_new:
            self._stats["nodes_joined"] += 1
            if _ORCH_PLACEMENTS is not None:
                try:
                    _ORCH_PLACEMENTS.labels(level="node_join").inc()
                except Exception:
                    pass
            _logger.info("Node joined: %s (%s) at %s [GPUs: %d]",
                         hostname, info.get("platform_type", "?"),
                         info.get("ip", "?"), info.get("gpu_count", 0))
            for cb in self._on_join_callbacks:
                try:
                    cb(info)
                except Exception:
                    pass

    def _cleanup_loop(self) -> None:
        """Remove nodes that haven't sent a heartbeat within timeout."""
        while self._running:
            now = time.time()
            stale = []
            with self._lock:
                for hostname, info in list(self._nodes.items()):
                    # Don't remove self
                    if hostname == self._local_info.get("hostname"):
                        continue
                    age = now - info.get("timestamp", 0)
                    if age > self.node_timeout:
                        stale.append(hostname)
                for hostname in stale:
                    info = self._nodes.pop(hostname, {})
                    self._stats["nodes_left"] += 1
                    _logger.info("Node left (timeout): %s", hostname)
                    for cb in self._on_leave_callbacks:
                        try:
                            cb(info)
                        except Exception:
                            pass
            self._shutdown.wait(timeout=self.node_timeout / 3)


# =========================================================================
# USB4 / Thunderbolt hot-plug detection (multi-OS)
# =========================================================================

class USB4HotPlug:
    """Detect USB4/Thunderbolt device connections for plug-and-play eGPU/storage.

    Linux:  uses pyudev for real-time monitoring
    macOS:  polls /Library/Preferences for Thunderbolt devices
    Windows: WMI-based detection (stub)
    """

    def __init__(self, on_connect: Optional[Callable] = None,
                 on_disconnect: Optional[Callable] = None):
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._system = platform.system().lower()

    def start(self) -> None:
        """Start hot-plug monitoring."""
        if self._running:
            return
        self._running = True

        if self._system == "linux":
            self._thread = threading.Thread(
                target=self._monitor_linux, daemon=True, name="usb4-hotplug"
            )
        elif self._system == "darwin":
            self._thread = threading.Thread(
                target=self._monitor_macos, daemon=True, name="usb4-hotplug"
            )
        else:
            self._thread = threading.Thread(
                target=self._monitor_generic, daemon=True, name="usb4-hotplug"
            )

        self._thread.start()
        _logger.info("USB4/Thunderbolt hot-plug monitoring started (%s)", self._system)

    def stop(self) -> None:
        self._running = False

    def _monitor_linux(self) -> None:
        """Linux: use pyudev for Thunderbolt/USB4 device monitoring."""
        try:
            import pyudev  # type: ignore
            context = pyudev.Context()
            monitor = pyudev.Monitor.from_netlink(context)
            # Monitor Thunderbolt and USB subsystems
            monitor.filter_by(subsystem="thunderbolt")
            monitor.start()

            while self._running:
                device = monitor.poll(timeout=2)
                if device is None:
                    continue
                event = {
                    "action": device.action,
                    "subsystem": device.subsystem,
                    "device_path": device.device_path,
                    "device_type": device.device_type,
                    "properties": dict(device.properties),
                }
                if device.action == "add":
                    _logger.info("USB4/TB device connected: %s", device.device_path)
                    if self.on_connect:
                        self.on_connect(event)
                elif device.action == "remove":
                    _logger.info("USB4/TB device disconnected: %s", device.device_path)
                    if self.on_disconnect:
                        self.on_disconnect(event)
        except ImportError:
            _logger.debug("pyudev not available, falling back to sysfs polling")
            self._poll_sysfs()
        except Exception as exc:
            _logger.error("USB4 monitor error: %s", exc)

    def _poll_sysfs(self) -> None:
        """Fallback: poll /sys/bus/thunderbolt/devices/ on Linux."""
        import glob
        known: set = set()
        while self._running:
            try:
                current = set(glob.glob("/sys/bus/thunderbolt/devices/*"))
                new = current - known
                removed = known - current
                for dev in new:
                    _logger.info("TB device detected: %s", dev)
                    if self.on_connect:
                        self.on_connect({"device_path": dev, "action": "add"})
                for dev in removed:
                    _logger.info("TB device removed: %s", dev)
                    if self.on_disconnect:
                        self.on_disconnect({"device_path": dev, "action": "remove"})
                known = current
            except Exception:
                pass
            time.sleep(3)

    def _monitor_macos(self) -> None:
        """macOS: poll system_profiler for Thunderbolt devices."""
        import subprocess
        known_serials: set = set()
        while self._running:
            try:
                result = subprocess.run(
                    ["system_profiler", "SPThunderboltDataType", "-json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    current_serials: set = set()
                    items = data.get("SPThunderboltDataType", [])
                    for item in items:
                        serial = item.get("device_name_key", str(item))
                        current_serials.add(serial)
                    new = current_serials - known_serials
                    removed = known_serials - current_serials
                    for s in new:
                        _logger.info("TB device connected: %s", s)
                        if self.on_connect:
                            self.on_connect({"device": s, "action": "add"})
                    for s in removed:
                        _logger.info("TB device disconnected: %s", s)
                        if self.on_disconnect:
                            self.on_disconnect({"device": s, "action": "remove"})
                    known_serials = current_serials
            except Exception:
                pass
            time.sleep(5)

    def _monitor_generic(self) -> None:
        """Generic fallback: just log that monitoring is not supported."""
        _logger.warning("USB4/TB hot-plug not supported on %s", self._system)
        while self._running:
            time.sleep(30)


# =========================================================================
# Legacy API (backward compatible)
# =========================================================================

def discover_nodes(port: int = 55555, timeout: float = 2.0) -> List[Dict[str, Any]]:
    """Discover nodes via UDP broadcast (legacy function).

    For production use, prefer ClusterDiscovery class with heartbeat.
    """
    info = get_local_info()
    msg = json.dumps(info).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)

    try:
        sock.sendto(msg, ("<broadcast>", port))
    except Exception as exc:
        _logger.warning("Broadcast send failed: %s", exc)

    nodes = [info]

    def listen():
        try:
            while True:
                data, addr = sock.recvfrom(8192)
                node = json.loads(data.decode("utf-8"))
                if node.get("hostname") != info["hostname"]:
                    nodes.append(node)
        except socket.timeout:
            pass
        except Exception:
            pass

    t = threading.Thread(target=listen)
    t.start()
    t.join(timeout + 0.5)
    sock.close()
    return nodes


def plug_and_play_usb4(mount_base: str = "/mnt/usb4_share") -> List[str]:
    """Legacy USB4 plug-and-play function.

    For production use, prefer USB4HotPlug class for real hot-plug detection.
    """
    _logger.info("USB4 plug-and-play scan (legacy)...")
    mounts = []
    for i in range(1, 5):
        mount_path = f"{mount_base}_{i}"
        if not os.path.exists(mount_path):
            try:
                os.makedirs(mount_path, exist_ok=True)
                _logger.info("USB4 mount point created: %s", mount_path)
            except PermissionError:
                _logger.warning("Cannot create %s (permission denied)", mount_path)
                continue
        mounts.append(mount_path)
    return mounts


def create_local_cluster(port: int = 55555) -> tuple:
    """Legacy cluster creation function."""
    nodes = discover_nodes(port=port)
    _logger.info("Cluster: %d nodes discovered", len(nodes))
    for node in nodes:
        _logger.info("  - %s (%s) | %s | GPUs: %d",
                     node.get("hostname", "?"),
                     node.get("platform_type", "?"),
                     node.get("ip", "?"),
                     len(node.get("gpus", [])))
    usb4_mounts = plug_and_play_usb4()
    return nodes, usb4_mounts


if __name__ == "__main__":
    create_local_cluster()
