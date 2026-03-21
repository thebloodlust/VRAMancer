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
except Exception as e:  # pragma: no cover
    import logging
    _logger = logging.getLogger("vramancer.discovery")

try:
    from core.metrics import (
        ORCH_PLACEMENTS as _ORCH_PLACEMENTS,
    )
except Exception as e:  # pragma: no cover
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

def _get_mac_address() -> str:
    """Get the primary MAC address formatted nicely."""
    try:
        import uuid
        mac = uuid.getnode()
        # Convert to hex and format with colon
        mac_hex = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) 
                           for elements in range(0, 8*6, 8)][::-1])
        return mac_hex
    except Exception as e:
        return "00:00:00:00:00:00"

def _is_edge_device(gpu_list: list, ram_bytes: int) -> bool:
    """Heuristic: classify as edge if ARM with low RAM and no discrete GPU."""
    arch = platform.machine().lower()
    is_arm = "arm" in arch or "aarch64" in arch
    has_gpu = any(g.get("backend") in ("cuda", "rocm") for g in gpu_list)
    low_ram = ram_bytes < 8 * 1024 ** 3  # < 8 GB
    return is_arm and not has_gpu and low_ram


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
        except Exception as e:
            pass

    # Apple Silicon: report MPS as 1 GPU even if torch isn't available
    if not gpu_list and platform.system() == "Darwin" and _is_apple_silicon():
        gpu_list.append({
            "id": "mps:0",
            "backend": "mps",
            "name": "Apple MPS (unified memory)",
            "total_memory": _get_total_ram(),
        })

    ram_bytes = _get_total_ram()

    return {
        "hostname": socket.gethostname(),
        "ip": _get_local_ip(),
        "mac": _get_mac_address(),
        "cpu": platform.processor() or platform.machine(),
        "arch": platform.machine(),
        "os": platform.system(),
        "platform_type": detect_platform_type(),
        "python_version": platform.python_version(),
        "gpus": gpu_list,
        "gpu_count": len(gpu_list),
        "ram_bytes": ram_bytes,
        "is_edge": _is_edge_device(gpu_list, ram_bytes),
        "vramancer_port": int(os.environ.get("VRM_API_PORT", 5000)),
        "timestamp": time.time(),
    }


def _is_apple_silicon() -> bool:
    """Detect Apple Silicon even under Rosetta 2 (x86_64 emulation)."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            return True
    except Exception:
        pass
    # Fallback: check arch directly (works when not under Rosetta)
    arch = platform.machine().lower()
    return "arm" in arch or "aarch64" in arch


def detect_platform_type() -> str:
    """Detect platform type with GPU-awareness."""
    system = platform.system().lower()
    arch = platform.machine().lower()

    if system == "darwin":
        if _is_apple_silicon():
            return "Apple Silicon"
        return "Apple Intel"
    elif system == "windows":
        return "Windows"
    elif system == "linux":
        # Check for GPU type
        backend = "cpu"
        try:
            backend = detect_backend()
        except Exception as e:
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
    except Exception as e:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception as e:
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
    except Exception as e:
        pass
    # macOS fallback
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception as e:
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

        # Leader election (Bully algorithm: highest gpu_count wins, hostname tiebreaker)
        self._leader: Optional[str] = None
        self._leader_lock = threading.Lock()

        # Persistent membership journal (JSON-lines)
        self._membership_log_path = os.environ.get(
            "VRM_MEMBERSHIP_LOG",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "membership.jsonl"),
        )

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
        self._elect_leader()

        # Auto-wire connectome: register new nodes as synapses
        try:
            from core.network.connectome import global_connectome
            self._connectome = global_connectome

            def _wire_synapse(info):
                hostname = info.get("hostname", "")
                ip = info.get("ip", "")
                port = info.get("vramancer_port", 5000)
                if ip and hostname != self._local_info.get("hostname"):
                    self._connectome.add_node(hostname, ip, port)

            def _unwire_synapse(info):
                hostname = info.get("hostname", "")
                self._connectome.remove_node(hostname)

            self.on_join(_wire_synapse)
            self.on_leave(_unwire_synapse)
            _logger.info("Connectome wired to cluster discovery")
        except Exception:
            self._connectome = None

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
    # Leader election (Bully algorithm)
    # ------------------------------------------------------------------

    def get_leader(self) -> Optional[str]:
        """Return the current cluster leader hostname (or None)."""
        with self._leader_lock:
            return self._leader

    def _elect_leader(self) -> None:
        """Bully election: node with most GPUs wins, hostname tiebreaker."""
        with self._lock:
            candidates = list(self._nodes.values())
        if not candidates:
            with self._leader_lock:
                self._leader = None
            return
        # Sort by (-gpu_count, hostname) — most GPUs first, then alphabetical
        best = max(candidates, key=lambda n: (n.get("gpu_count", 0), n.get("hostname", "")))
        new_leader = best.get("hostname")
        with self._leader_lock:
            old = self._leader
            self._leader = new_leader
        if old != new_leader:
            _logger.info("Leader elected: %s (gpus=%d)", new_leader, best.get("gpu_count", 0))

    def is_leader(self) -> bool:
        """Return True if this node is the current cluster leader."""
        with self._leader_lock:
            return self._leader == self._local_info.get("hostname")

    # ------------------------------------------------------------------
    # Persistent membership journal
    # ------------------------------------------------------------------

    def _log_membership_event(self, event_type: str, info: Dict[str, Any]) -> None:
        """Append a join/leave event to the JSONL membership log."""
        entry = {
            "ts": time.time(),
            "event": event_type,
            "hostname": info.get("hostname", "?"),
            "ip": info.get("ip", "?"),
            "gpu_count": info.get("gpu_count", 0),
        }
        try:
            with open(self._membership_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            _logger.debug("Membership log write failed: %s", exc)

    def load_membership_log(self) -> List[Dict[str, Any]]:
        """Read back all membership events from the persistent log."""
        entries: List[Dict[str, Any]] = []
        try:
            with open(self._membership_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as exc:
            _logger.debug("Membership log read failed: %s", exc)
        return entries

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
        except Exception as e:
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
                try:
                    node = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
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
                
                # Envoi direct vers IPs spécifiques (contournement blocage réseaux locaux)
                peer_ips = os.environ.get("VRM_PEER_IPS", "")
                if peer_ips:
                    for peer_ip in peer_ips.split(","):
                        peer_ip = peer_ip.strip()
                        if peer_ip:
                            # 1. UDP Unicast
                            try:
                                sock.sendto(msg, (peer_ip, self.port))
                            except Exception as e:
                                pass
                                
                            # 2. HTTP POST fallback (si UDP est bloqué au niveau firewall)
                            try:
                                import requests
                                target_port = os.environ.get("VRM_API_PORT", "8000")
                                requests.post(f"http://{peer_ip}:{target_port}/api/nodes", 
                                             json=self._local_info, 
                                             timeout=1.0)
                            except Exception as e:
                                pass

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
            prev = self._nodes.get(hostname, {})
            info["_state"] = "ACTIVE"
            info["_missed_heartbeats"] = 0
            self._nodes[hostname] = info
        if is_new:
            self._stats["nodes_joined"] += 1
            self._log_membership_event("join", info)
            if _ORCH_PLACEMENTS is not None:
                try:
                    _ORCH_PLACEMENTS.labels(level="node_join").inc()
                except Exception as e:
                    pass
            _logger.info("Node joined: %s (%s) at %s [GPUs: %d]",
                         hostname, info.get("platform_type", "?"),
                         info.get("ip", "?"), info.get("gpu_count", 0))
            for cb in self._on_join_callbacks:
                try:
                    cb(info)
                except Exception as exc:
                    _logger.debug("Join callback error: %s", exc)
            self._elect_leader()
        elif prev.get("_state") == "STALE":
            _logger.info("Node recovered: %s", hostname)

    def _probe_node(self, info: Dict[str, Any]) -> bool:
        """Send a TCP probe to confirm if a node is truly unreachable."""
        ip = info.get("ip", "")
        port = info.get("vramancer_port", 5000)
        if not ip or ip == "127.0.0.1":
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((ip, port))
            sock.close()
            return True
        except Exception:
            return False

    def _cleanup_loop(self) -> None:
        """Remove stale nodes with confirmation probes before declaring dead.

        State transitions: ACTIVE → STALE (missed heartbeat) → LEFT (3 missed + probe fails)
        """
        while self._running:
            now = time.time()
            to_probe: List[tuple] = []
            to_remove: List[str] = []

            with self._lock:
                for hostname, info in list(self._nodes.items()):
                    if hostname == self._local_info.get("hostname"):
                        continue
                    age = now - info.get("timestamp", 0)
                    if age > self.node_timeout:
                        missed = info.get("_missed_heartbeats", 0) + 1
                        info["_missed_heartbeats"] = missed
                        if missed == 1:
                            info["_state"] = "STALE"
                            _logger.debug("Node stale: %s (missed 1)", hostname)
                        elif missed >= 3:
                            to_probe.append((hostname, dict(info)))

            # Probe outside the lock to avoid blocking
            for hostname, info in to_probe:
                if self._probe_node(info):
                    _logger.info("Node %s responded to probe, keeping", hostname)
                    with self._lock:
                        if hostname in self._nodes:
                            self._nodes[hostname]["_missed_heartbeats"] = 0
                            self._nodes[hostname]["_state"] = "ACTIVE"
                else:
                    to_remove.append(hostname)

            # Remove confirmed dead nodes
            if to_remove:
                with self._lock:
                    for hostname in to_remove:
                        info = self._nodes.pop(hostname, {})
                        if info:
                            self._stats["nodes_left"] += 1
                            self._log_membership_event("leave", info)
                            _logger.info("Node left (confirmed dead): %s", hostname)
                            for cb in self._on_leave_callbacks:
                                try:
                                    cb(info)
                                except Exception as exc:
                                    _logger.debug("Leave callback error: %s", exc)
                if to_remove:
                    self._elect_leader()

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
            except Exception as e:
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
            except Exception as e:
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
                try:
                    node = json.loads(data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if node.get("hostname") != info["hostname"]:
                    nodes.append(node)
        except socket.timeout:
            pass
        except Exception as e:
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
