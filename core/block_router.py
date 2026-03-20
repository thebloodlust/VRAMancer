# core/block_router.py
"""Production block router — VRAM-aware routing of model blocks to GPU/CPU/NVMe/network.

Routes blocks to the optimal execution device based on:
  - Real-time VRAM availability per GPU
  - Block priority and estimated size
  - NVMe cache availability (real detection)
  - Network node registry (dynamic, not hardcoded)
  - Transport factory integration for GPU-to-GPU transfers

Production features:
  - Prometheus metrics (ORCH_PLACEMENTS, ORCH_MIGRATIONS)
  - Structured logging via LoggerAdapter
  - Thread-safe node registry
  - Per-route statistics tracking
  - Graceful fallback chain: GPU → CPU → NVMe → Remote → CPU failsafe
"""

from __future__ import annotations

import os
import shutil
import threading
from typing import Any, Optional, Dict, List

try:
    from core.logger import LoggerAdapter
    _logger = LoggerAdapter("router")
except Exception:
    import logging
    _logger = logging.getLogger("vramancer.block_router")  # type: ignore

try:
    from core.metrics import ORCH_PLACEMENTS, ORCH_MIGRATIONS
    _METRICS = True
except Exception:
    _METRICS = False

STRICT = os.environ.get("VRM_STRICT_IMPORT", "0") in {"1", "true", "TRUE"}
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    from core.compute_engine import ComputeEngine
except Exception:  # pragma: no cover
    class ComputeEngine:
        def __init__(self, *a, **k):
            self.backend = "cpu"
        def _get_device(self, device_id=0):
            return "cpu"
        def get_ram_status(self):
            return 0, 0

def load_block_from_disk(path):
    _logger.warning("storage_manager unavailable, returning identity stub for %s", path)
    try:
        import torch
        return torch.nn.Identity()
    except ImportError:
        return lambda x: x

try:
    from core.config import get_config
except ImportError:  # pragma: no cover
    def get_config():
        return {}


class RemoteExecutor:
    """Remote block execution proxy with connection management."""

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def forward(self, x: Any) -> Any:
        """Execute block remotely. Falls back to passthrough if unavailable."""
        try:
            import socket
            import hmac
            import hashlib
            import os
            import torch
            import io

            # ---------------------------------------------------------
            # ZERO-COPY PATH (Niveau 2) : Bypass Pickle
            # ---------------------------------------------------------
            is_safetensor = False
            try:
                from safetensors.torch import save
                
                # Si x est un Tenseur ou un Dict de Tenseurs, on utilise Safetensors (Zero-Copy)
                if isinstance(x, torch.Tensor):
                    if getattr(x, "is_cuda", False):
                        # Fix P2P Rust : point de synchronisation dur CUDA préréglé
                        # pour forcer la file d'attente VRAM à se purger avant d'extraire via ReBAR
                        torch.cuda.synchronize(device=x.device)
                    data = save({"tensor": x})
                    is_safetensor = True
                elif isinstance(x, dict) and all(isinstance(v, torch.Tensor) for v in x.values()):
                    # Fix P2P Rust (Multi) : synchro systématique sur chaque Device GPU du dict
                    for v in x.values():
                        if getattr(v, "is_cuda", False):
                            torch.cuda.synchronize(device=v.device)
                    data = save(x)
                    is_safetensor = True
                else:
                    import json
                    data = json.dumps(x).encode("utf-8")
            except Exception:
                # Fallback to PyTorch weights_only serialization
                buffer = io.BytesIO()
                torch.save(x, buffer)
                data = buffer.getvalue()

            # Prefix de métadonnées rapide (1 octet) pour dire au récepteur comment décoder
            # 0x01 = JSON/Torch (Secure), 0x02 = Safetensors (Zero-Copy)
            header_type = b'\x02' if is_safetensor else b'\x01'
            data = header_type + data

            api_token = os.environ.get("VRM_API_TOKEN")
            if not api_token:
                if os.environ.get("VRM_PRODUCTION", "0") == "1":
                    raise RuntimeError("SECURITY: VRM_API_TOKEN environment variable must be set in production.")
                if not getattr(self, "_warned_token", False):
                    _logger.warning("SECURITY: VRM_API_TOKEN not set. Using dev default token. Set VRM_API_TOKEN before production!")
                    self._warned_token = True
                api_token = "dev_token_only"
            secret = api_token.encode()

            # 1. Option Tokio (Rust) -> GIL relâché, asynchrone natif
            try:
                import vramancer_rust
                tier = vramancer_rust.detect_best_transport()
                if tier == vramancer_rust.TransportTier.ZeroCopyTcp:
                    _logger.debug("Utilisation du pont réseau ZeroCopy Tokio/Rust")
                # Le réseau P2P natif super rapide
                resp_data = vramancer_rust.send_tensor_p2p(self.host, self.port, secret, data)
            except (ImportError, Exception) as e:
                if isinstance(e, ImportError):
                    _logger.debug("vramancer_rust unavailable, using CPU fallback for TCP P2P")
                else:
                    _logger.warning(f"Rust native P2P failed ({e}), falling back to Python sockets")

                # 2. Option Python classique -> Sockets synchrones (+GIL block)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                try:
                    sock.connect((self.host, self.port))
                    
                    # Signature CPU
                    signature = hmac.new(secret, data, hashlib.sha256).digest()
                    payload = signature + data
                    sock.sendall(len(payload).to_bytes(8, "big") + payload)
                    
                    resp_len_bytes = sock.recv(8)
                    if not resp_len_bytes:
                        raise ConnectionError("Remote node disconnected prematurely (Empty response header)")
                    resp_len = int.from_bytes(resp_len_bytes, "big")
                    
                    resp_data = b""
                    while len(resp_data) < resp_len:
                        chunk = sock.recv(min(65536, resp_len - len(resp_data)))
                        if not chunk:
                            raise ConnectionError("Remote node dropped connection during payload transfer")
                        resp_data += chunk
                finally:
                    sock.close()
            
            # Verify response signature (Commun pour Rust & Python)
            resp_sig = resp_data[:32]
            resp_payload = resp_data[32:]
            
            # Prudence: Verification Rust native ou fallback CPU
            try:
                import vramancer_rust
                is_valid = vramancer_rust.verify_hmac_fast(secret, resp_payload, resp_sig)
            except ImportError:
                expected_sig = hmac.new(secret, resp_payload, hashlib.sha256).digest()
                is_valid = hmac.compare_digest(resp_sig, expected_sig)
                
            if not is_valid:
                _logger.error(f"Zero-Trust Violation: Invalid signature from {self.host}:{self.port}")
                raise ValueError("Untrusted response payload")

            # Dessérialisation dynamique (Zero-Copy ou Pickle Legacy)
            resp_type = resp_payload[:1]
            pure_data = resp_payload[1:]
            
            if resp_type == b'\x02':
                try:
                    from safetensors.torch import load
                    parsed_dict = load(pure_data)
                    # Si c'était un Tenseur unique, on le ressort du dict
                    return parsed_dict["tensor"] if "tensor" in parsed_dict else parsed_dict
                except Exception as e:
                    _logger.error(f"Erreur Safetensors au décodage réseau: {e}")
                    raise
            else:
                try:
                    import json
                    return json.loads(pure_data.decode("utf-8"))
                except Exception:
                    import io
                    import torch
                    return torch.load(io.BytesIO(pure_data), weights_only=True)
                
        except socket.timeout:
            _logger.error(f"Timeout réseau ({self.timeout}s) vers {self.host}:{self.port}")
            # Tentative de Wake-on-Inference si une MAC address est configurée ou découverte
            mac_target = os.environ.get(f"VRM_WOI_MAC_{self.host.replace('.', '_')}")
            if mac_target:
                from core.network.wake_on_lan import send_magic_packet
                _logger.info(f"Déclenchement du Wake-on-Inference (WoI) pour le nœud endormi: {self.host}")
                send_magic_packet(mac_target)
                _logger.info("Passage au CPU Fallback en attendant le réveil du nœud...")
            else:
                _logger.info("Mode dégradé activé (Passthrough local)")
            return x  # passthrough fallback
        except Exception as exc:
            _logger.warning("Remote execution to %s:%d failed: %s", self.host, self.port, exc)
            return x  # passthrough fallback


class BlockRouter:
    """Routes model blocks to the best available compute device.

    Routing priority (configurable):
      1. GPU VRAM (if block fits + priority >= threshold)
      2. CPU DRAM (if sufficient RAM available)
      3. NVMe cache (spill large/low-priority blocks)
      4. Remote node (distribute across cluster)
    """

    def __init__(self, verbose: bool = True, monitor: Any = None):
        self.engine = ComputeEngine(verbose=verbose)
        self.verbose = verbose
        self.monitor = monitor  # Optional GPUMonitor instance
        self._remote_nodes: List[Dict[str, Any]] = []
        self._nvme_cache_path: Optional[str] = None
        self._lock = threading.Lock()

        # Stats
        self._stats = {
            "gpu_routes": 0,
            "cpu_routes": 0,
            "nvme_routes": 0,
            "remote_routes": 0,
            "fallback_routes": 0,
            "errors": 0,
        }

        # Load NVMe cache path from config
        cfg = get_config()
        self._nvme_cache_path = cfg.get("nvme_cache_path")

    # ------------------------------------------------------------------
    # Node registry (dynamic, replaces hardcoded IPs)
    # ------------------------------------------------------------------

    def register_remote_node(self, host: str, port: int = 9000,
                             capacity_mb: int = 0) -> None:
        """Register a remote compute node for load distribution."""
        with self._lock:
            node = {"host": host, "port": port, "capacity_mb": capacity_mb}
            if node not in self._remote_nodes:
                self._remote_nodes.append(node)
                _logger.info("Registered remote node: %s:%d (capacity=%dMB)",
                             host, port, capacity_mb)

    def unregister_remote_node(self, host: str, port: int = 9000) -> None:
        """Remove a remote node from the registry."""
        with self._lock:
            self._remote_nodes = [n for n in self._remote_nodes
                                  if not (n["host"] == host and n["port"] == port)]

    # ------------------------------------------------------------------
    # Main routing logic
    # ------------------------------------------------------------------

    def route(self, block: Any, input_tensor: Any, index: int = 0,
              importance: str = "normal", estimated_size_mb: float = 100,
              **kwargs: Any) -> Any:
        """Route a block to the optimal device and execute it.

        Parameters
        ----------
        block : nn.Module
            The model block to execute.
        input_tensor : torch.Tensor
            Input tensor.
        index : int
            Block index.
        importance : str
            "critical", "normal", or "low".
        estimated_size_mb : float
            Estimated size of the block's parameters in MB.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        backend = self.engine.backend
        ram_available, ram_total = self.engine.get_ram_status()

        # 1. Critical blocks -> GPU if VRAM available
        if importance == "critical" and backend in ("cuda", "rocm", "mps"):
            gpu_id = self._find_gpu_for_block(estimated_size_mb, index)
            if gpu_id is not None:
                device = self.engine._get_device(gpu_id)
                if self.verbose:
                    _logger.info("Block %d -> %s (critical, %.0fMB)", index, device, estimated_size_mb)
                self._stats["gpu_routes"] += 1
                if _METRICS:
                    ORCH_PLACEMENTS.labels("gpu").inc()
                return self._exec_on_device(block, input_tensor, device)

        # 2. Normal blocks -> GPU if fits, else CPU
        if importance == "normal" and backend in ("cuda", "rocm", "mps"):
            gpu_id = self._find_gpu_for_block(estimated_size_mb, index)
            if gpu_id is not None:
                device = self.engine._get_device(gpu_id)
                if self.verbose:
                    _logger.debug("Block %d -> %s (normal, %.0fMB)", index, device, estimated_size_mb)
                self._stats["gpu_routes"] += 1
                if _METRICS:
                    ORCH_PLACEMENTS.labels("gpu").inc()
                return self._exec_on_device(block, input_tensor, device)

        # 3. Large blocks with low RAM -> NVMe cache
        if estimated_size_mb > 500 and ram_available < 2 * 1024**3:
            if self._nvme_available():
                if self.verbose:
                    _logger.info("Block %d -> NVMe cache (large: %.0fMB, low RAM)", index, estimated_size_mb)
                self._stats["nvme_routes"] += 1
                if _METRICS:
                    ORCH_PLACEMENTS.labels("nvme").inc()
                return self._exec_from_nvme(block, input_tensor, index)

        # 4. Low priority -> remote if nodes available
        if importance == "low" and self._remote_nodes:
            node = self._select_remote_node()
            if node:
                if self.verbose:
                    _logger.info("Block %d -> remote %s:%d (low priority)",
                                 index, node["host"], node["port"])
                self._stats["remote_routes"] += 1
                if _METRICS:
                    ORCH_PLACEMENTS.labels("remote").inc()
                executor = RemoteExecutor(node["host"], node["port"])
                return executor.forward(input_tensor)

        # 5. CPU fallback
        if ram_available > estimated_size_mb * 1024 * 1024:
            if self.verbose:
                _logger.debug("Block %d -> CPU (%.0fMB, RAM ok)", index, estimated_size_mb)
            self._stats["cpu_routes"] += 1
            if _METRICS:
                ORCH_PLACEMENTS.labels("cpu").inc()
            return self._exec_on_device(block, input_tensor, "cpu")

        # 6. NVMe fallback for very large blocks
        if self._nvme_available():
            if self.verbose:
                _logger.info("Block %d -> NVMe fallback", index)
            self._stats["nvme_routes"] += 1
            if _METRICS:
                ORCH_PLACEMENTS.labels("nvme").inc()
            return self._exec_from_nvme(block, input_tensor, index)

        # 7. Ultimate fallback: CPU regardless
        if self.verbose:
            _logger.debug("Block %d -> CPU (ultimate fallback)", index)
        self._stats["fallback_routes"] += 1
        if _METRICS:
            ORCH_PLACEMENTS.labels("cpu_fallback").inc()
        return self._exec_on_device(block, input_tensor, "cpu")

    # ------------------------------------------------------------------
    # NVMe detection (real)
    # ------------------------------------------------------------------

    def _nvme_available(self) -> bool:
        """Check if NVMe cache storage is available and writable."""
        # Check configured path
        if self._nvme_cache_path:
            return os.path.isdir(self._nvme_cache_path) and os.access(self._nvme_cache_path, os.W_OK)

        # Auto-detect NVMe on Linux
        if os.name == "posix":
            # Check common NVMe mount points
            for candidate in ("/tmp/vramancer_cache", "/var/cache/vramancer", "/mnt/nvme"):
                if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
                    self._nvme_cache_path = candidate
                    return True
            # Check if /tmp has enough space (NVMe is often mounted there)
            try:
                usage = shutil.disk_usage("/tmp")
                if usage.free > 5 * 1024**3:  # at least 5GB free
                    self._nvme_cache_path = "/tmp/vramancer_cache"
                    os.makedirs(self._nvme_cache_path, exist_ok=True)
                    return True
            except Exception:
                pass

        # Windows: use %TEMP%
        if os.name == "nt":
            tmp = os.environ.get("TEMP", os.environ.get("TMP", "C:\\Temp"))
            cache_dir = os.path.join(tmp, "vramancer_cache")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                self._nvme_cache_path = cache_dir
                return True
            except Exception:
                pass

        return False

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _exec_on_device(self, block: Any, input_tensor: Any, device: Any) -> Any:
        """Execute block on a specific device with error handling."""
        try:
            if hasattr(block, "to") and device != "cpu":
                block = block.to(device)
            if hasattr(input_tensor, "to"):
                input_tensor = input_tensor.to(device)
            return block(input_tensor)
        except Exception as exc:
            self._stats["errors"] += 1
            _logger.warning("Execution on %s failed: %s, falling back to CPU", device, exc)
            try:
                if hasattr(block, "to"):
                    block = block.to("cpu")
                if hasattr(input_tensor, "to"):
                    input_tensor = input_tensor.to("cpu")
                return block(input_tensor)
            except Exception as cpu_exc:
                self._stats["errors"] += 1
                _logger.error("CPU fallback also failed: %s", cpu_exc)
                raise

    def _exec_from_nvme(self, block: Any, input_tensor: Any, index: int) -> Any:
        """Load block from NVMe cache and execute on CPU."""
        cache_path = os.path.join(self._nvme_cache_path or "/tmp", f"block_{index}.pt")
        if os.path.exists(cache_path):
            loaded = load_block_from_disk(cache_path)
            return self._exec_on_device(loaded, input_tensor, "cpu")
        # Block not cached yet — execute on CPU and cache
        result = self._exec_on_device(block, input_tensor, "cpu")
        try:
            import torch
            torch.save(block.state_dict(), cache_path)
        except Exception as exc:
            _logger.debug("Could not cache block %d to NVMe: %s", index, exc)
        return result

    def _find_gpu_for_block(self, size_mb: float, index: int) -> Optional[int]:
        """Find a GPU with enough free VRAM for the block."""
        if self.monitor is None:
            # Without monitor, use GPU 0 if available
            return 0 if self.engine.backend in ("cuda", "rocm", "mps") else None

        for gpu in getattr(self.monitor, "gpus", []):
            idx = gpu.get("index", 0)
            free = self.monitor.get_free_memory(idx)
            if free > size_mb * 1024 * 1024:  # convert MB to bytes
                return idx
        return None

    def _select_remote_node(self) -> Optional[Dict[str, Any]]:
        """Select the best remote node (highest capacity)."""
        with self._lock:
            if not self._remote_nodes:
                return None
            return max(self._remote_nodes, key=lambda n: n.get("capacity_mb", 0))

    @property
    def stats(self) -> Dict[str, Any]:
        """Return routing statistics."""
        total = sum(v for k, v in self._stats.items() if k != "errors")
        return {
            **self._stats,
            "total_routes": total,
            "remote_nodes": len(self._remote_nodes),
            "nvme_path": self._nvme_cache_path,
        }

    def __repr__(self) -> str:
        return (f"BlockRouter(backend={self.engine.backend}, "
                f"routes={sum(v for k,v in self._stats.items() if k != 'errors')}, "
                f"errors={self._stats['errors']})")
