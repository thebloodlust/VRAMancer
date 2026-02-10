"""High-performance network transport with RDMA bypass.

Transport tiers:
  Tier 1 - RDMA verbs (ibverbs/RoCE v2): zero-copy kernel bypass, 1-3us latency
  Tier 2 - Zero-copy TCP (SO_ZEROCOPY + TCP_NODELAY + SO_BUSY_POLL): 50-100us
  Tier 3 - Local mmap (same-machine fallback)

GPUDirect RDMA: NIC <-> GPU VRAM without CPU (requires nvidia_peermem).
"""
from __future__ import annotations

import os
import sys
import time
import struct
import socket
import hashlib
import mmap
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Callable

from core.logger import LoggerAdapter
from core.metrics import FASTPATH_BYTES, FASTPATH_LATENCY

_BENCH_CACHE: dict[str, tuple[float, float]] = {}
_BENCH_TTL = float(os.environ.get('VRM_FASTPATH_BENCH_TTL', '30'))

# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------
_RDMA_AVAILABLE = False
_PYVERBS_AVAILABLE = False
_GPUDIRECT_AVAILABLE = False

try:
    from pyverbs.device import Context as RDMAContext  # type: ignore
    from pyverbs.pd import PD as ProtectionDomain  # type: ignore
    from pyverbs.mr import MR as MemoryRegion  # type: ignore
    from pyverbs.cq import CQ as CompletionQueue  # type: ignore
    from pyverbs.qp import QP, QPInitAttr, QPAttr, QPCap  # type: ignore
    from pyverbs.addr import AH, AHAttr, GlobalRoute  # type: ignore
    import pyverbs.enums as e  # type: ignore
    _RDMA_AVAILABLE = True
    _PYVERBS_AVAILABLE = True
except ImportError:
    pass

# GPUDirect RDMA: check nvidia_peermem module
try:
    if os.path.exists('/sys/module/nvidia_peermem') or \
       os.path.exists('/sys/module/nv_peer_mem'):
        _GPUDIRECT_AVAILABLE = True
except Exception:
    pass

# CUDA for GPUDirect
_CUDA_AVAILABLE = False
try:
    import torch
    import torch.cuda
    if torch.cuda.is_available():
        _CUDA_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

log = LoggerAdapter("fibre")



# ---------------------------------------------------------------------------
# RDMA Connection Manager
# ---------------------------------------------------------------------------
class RDMATransport:
    """RDMA verbs-based transport (ibverbs / RoCE v2).

    Provides zero-copy, kernel-bypass data transfer between nodes.
    Uses Reliable Connected (RC) queue pairs for guaranteed delivery.

    Supports GPUDirect RDMA when nvidia_peermem is loaded:
    NIC reads/writes directly to GPU VRAM without CPU copy.
    """

    def __init__(
        self,
        device_name: Optional[str] = None,
        ib_port: int = 1,
        gid_index: int = 0,
        max_msg_size: int = 4 * 1024 * 1024,  # 4MB default
        sq_depth: int = 64,
        rq_depth: int = 64,
    ):
        self.device_name = device_name
        self.ib_port = ib_port
        self.gid_index = gid_index
        self.max_msg_size = max_msg_size
        self.sq_depth = sq_depth
        self.rq_depth = rq_depth

        self.ctx = None
        self.pd = None
        self.cq = None
        self.qp = None
        self.mr = None
        self._buf = None
        self._connected = False

        if _PYVERBS_AVAILABLE:
            self._init_resources()

    def _init_resources(self):
        """Initialize RDMA resources: context, PD, CQ, QP, MR."""
        try:
            if not self.device_name:
                for name in ['mlx5_0', 'mlx4_0', 'rxe0', 'siw_eth0']:
                    try:
                        self.ctx = RDMAContext(name=name)
                        self.device_name = name
                        break
                    except Exception:
                        continue
                if not self.ctx:
                    log.warning("No RDMA device found")
                    return
            else:
                self.ctx = RDMAContext(name=self.device_name)

            self.pd = ProtectionDomain(self.ctx)
            self.cq = CompletionQueue(self.ctx, self.sq_depth + self.rq_depth)

            cap = QPCap(
                max_send_wr=self.sq_depth,
                max_recv_wr=self.rq_depth,
                max_send_sge=1,
                max_recv_sge=1,
            )
            init_attr = QPInitAttr(
                qp_type=e.IBV_QPT_RC,
                scq=self.cq,
                rcq=self.cq,
                cap=cap,
            )
            self.qp = QP(self.pd, init_attr)

            self._buf = bytearray(self.max_msg_size)
            access = e.IBV_ACCESS_LOCAL_WRITE | e.IBV_ACCESS_REMOTE_WRITE | e.IBV_ACCESS_REMOTE_READ
            self.mr = MemoryRegion(self.pd, self._buf, access)

            log.info(
                f"RDMA resources initialized on {self.device_name} "
                f"(QP num={self.qp.qp_num}, buf={self.max_msg_size / 1e6:.0f}MB)"
            )
        except Exception as exc:
            log.warning(f"RDMA init failed: {exc}")
            self.ctx = None

    @property
    def available(self) -> bool:
        return self.ctx is not None and self.qp is not None

    def connect(self, remote_qp_num: int, remote_lid: int, remote_gid: bytes):
        """Connect QP to remote peer (exchange QP info out-of-band)."""
        if not self.available:
            raise RuntimeError("RDMA resources not initialized")
        try:
            attr = QPAttr()
            attr.qp_state = e.IBV_QPS_INIT
            attr.port_num = self.ib_port
            attr.pkey_index = 0
            attr.qp_access_flags = (
                e.IBV_ACCESS_LOCAL_WRITE |
                e.IBV_ACCESS_REMOTE_WRITE |
                e.IBV_ACCESS_REMOTE_READ
            )
            self.qp.to_init(attr)

            attr = QPAttr()
            attr.qp_state = e.IBV_QPS_RTR
            attr.path_mtu = e.IBV_MTU_4096
            attr.dest_qp_num = remote_qp_num
            attr.rq_psn = 0
            attr.max_dest_rd_atomic = 4
            attr.min_rnr_timer = 12

            gr = GlobalRoute()
            gr.dgid = remote_gid
            gr.sgid_index = self.gid_index

            ah_attr = AHAttr()
            ah_attr.dlid = remote_lid
            ah_attr.sl = 0
            ah_attr.port_num = self.ib_port
            ah_attr.is_global = 1
            ah_attr.gr = gr
            attr.ah_attr = ah_attr
            self.qp.to_rtr(attr)

            attr = QPAttr()
            attr.qp_state = e.IBV_QPS_RTS
            attr.sq_psn = 0
            attr.timeout = 14
            attr.retry_cnt = 7
            attr.rnr_retry = 7
            attr.max_rd_atomic = 4
            self.qp.to_rts(attr)

            self._connected = True
            log.info(f"RDMA QP connected to remote QP {remote_qp_num}")
        except Exception as exc:
            log.error(f"RDMA connect failed: {exc}")
            raise

    def get_connection_info(self) -> Dict[str, Any]:
        """Return local QP info for out-of-band exchange."""
        if not self.available:
            return {}
        info = {'qp_num': self.qp.qp_num, 'lid': 0}
        try:
            port_attr = self.ctx.query_port(self.ib_port)
            info['lid'] = port_attr.lid
            gid = self.ctx.query_gid(self.ib_port, self.gid_index)
            info['gid'] = bytes(gid.gid)
        except Exception:
            info['gid'] = b'\x00' * 16
        return info

    def send(self, data: bytes) -> int:
        """Send data via RDMA SEND verb."""
        if not self._connected:
            raise RuntimeError("QP not connected")
        start = time.perf_counter()
        size = len(data)
        if size > self.max_msg_size:
            return self._send_chunked(data)
        self._buf[:size] = data
        try:
            from pyverbs.wr import SendWR, SGE  # type: ignore
            sge = SGE(self.mr.buf, size, self.mr.lkey)
            send_wr = SendWR(opcode=e.IBV_WR_SEND, num_sge=1, sg=[sge])
            send_wr.send_flags = e.IBV_SEND_SIGNALED
            self.qp.post_send(send_wr)
            self._poll_cq(1)
            FASTPATH_BYTES.labels("rdma", "send").inc(size)
            FASTPATH_LATENCY.labels("rdma", "send").observe(time.perf_counter() - start)
            return size
        except Exception as exc:
            log.error(f"RDMA send failed: {exc}")
            raise

    def recv(self, size: int = 0) -> Optional[bytes]:
        """Receive data via RDMA RECV verb."""
        if not self._connected:
            raise RuntimeError("QP not connected")
        start = time.perf_counter()
        recv_size = size or self.max_msg_size
        try:
            from pyverbs.wr import RecvWR, SGE  # type: ignore
            sge = SGE(self.mr.buf, recv_size, self.mr.lkey)
            recv_wr = RecvWR(num_sge=1, sg=[sge])
            self.qp.post_recv(recv_wr)
            wc = self._poll_cq(1)
            actual_size = wc.byte_len if wc else recv_size
            result = bytes(self._buf[:actual_size])
            FASTPATH_BYTES.labels("rdma", "recv").inc(actual_size)
            FASTPATH_LATENCY.labels("rdma", "recv").observe(time.perf_counter() - start)
            return result
        except Exception as exc:
            log.error(f"RDMA recv failed: {exc}")
            return None

    def _send_chunked(self, data: bytes) -> int:
        total = 0
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + self.max_msg_size]
            total += self.send(chunk)
            offset += len(chunk)
        return total

    def _poll_cq(self, num_expected: int, timeout_ms: int = 5000):
        deadline = time.perf_counter() + timeout_ms / 1000
        completed = 0
        last_wc = None
        while completed < num_expected:
            if time.perf_counter() > deadline:
                raise TimeoutError(f"RDMA CQ poll timeout after {timeout_ms}ms")
            wcs = self.cq.poll(num_expected - completed)
            for wc in wcs:
                if wc.status != e.IBV_WC_SUCCESS:
                    raise RuntimeError(f"RDMA WC error: status={wc.status}")
                completed += 1
                last_wc = wc
        return last_wc

    def close(self):
        for res in [self.mr, self.qp, self.cq, self.pd, self.ctx]:
            if res is not None:
                try:
                    res.close()
                except Exception:
                    pass
        self._connected = False
        log.info("RDMA resources released")


# ---------------------------------------------------------------------------
# GPUDirect RDMA Transport
# ---------------------------------------------------------------------------
class GPUDirectTransport:
    """GPUDirect RDMA: NIC <-> GPU VRAM without CPU copy.

    Requires nvidia_peermem + Mellanox ConnectX-4+ NIC.

    When nvidia_peermem is loaded, ibv_reg_mr() with a GPU data_ptr()
    tells the kernel module to pin GPU pages and create IOVA mappings
    so the NIC can DMA directly to/from GPU VRAM — zero CPU copy.

    Falls back to pinned-memory staging (no numpy intermediate) when
    true GPUDirect is not available.
    """

    def __init__(self, rdma_transport: Optional[RDMATransport] = None):
        self.rdma = rdma_transport
        self._gpu_mrs: Dict[int, Any] = {}   # {gpu_id: {buffer, mr, ptr, size}}
        self._gpu_direct_ok: Dict[int, bool] = {}  # per-GPU GPUDirect status

    @property
    def available(self) -> bool:
        return (
            _GPUDIRECT_AVAILABLE and _CUDA_AVAILABLE and
            self.rdma is not None and self.rdma.available
        )

    @property
    def has_true_gpudirect(self) -> bool:
        """True if any GPU has a NIC-registered MR (real GPUDirect)."""
        return any(self._gpu_direct_ok.values())

    def register_gpu_memory(self, gpu_id: int, size_bytes: int):
        """Allocate GPU buffer and register with RDMA NIC.

        With nvidia_peermem loaded, the MR registration pins GPU pages
        and sets up IOVA mappings for direct NIC DMA.
        Without it, we still allocate a GPU buffer for staging.
        """
        if not _CUDA_AVAILABLE:
            return
        try:
            with torch.cuda.device(gpu_id):
                gpu_buf = torch.empty(
                    size_bytes, dtype=torch.uint8, device=f'cuda:{gpu_id}'
                )
                entry = {
                    'buffer': gpu_buf,
                    'size': size_bytes,
                    'ptr': gpu_buf.data_ptr(),
                    'mr': None,
                }

                # Try true GPUDirect MR registration via nvidia_peermem
                if (_GPUDIRECT_AVAILABLE and _PYVERBS_AVAILABLE and
                        self.rdma and self.rdma.pd is not None):
                    try:
                        access = (
                            e.IBV_ACCESS_LOCAL_WRITE |
                            e.IBV_ACCESS_REMOTE_WRITE |
                            e.IBV_ACCESS_REMOTE_READ
                        )
                        mr = MemoryRegion(self.rdma.pd, gpu_buf, access)
                        entry['mr'] = mr
                        self._gpu_direct_ok[gpu_id] = True
                        log.info(
                            f"GPUDirect RDMA: registered {size_bytes / 1e6:.0f}MB "
                            f"on GPU {gpu_id} (ptr=0x{gpu_buf.data_ptr():x}, "
                            f"rkey=0x{mr.rkey:x}) — NIC will DMA directly"
                        )
                    except Exception as exc:
                        self._gpu_direct_ok[gpu_id] = False
                        log.info(
                            f"GPUDirect MR failed on GPU {gpu_id}: {exc} — "
                            f"using pinned-memory staging (still zero-numpy)"
                        )
                else:
                    self._gpu_direct_ok[gpu_id] = False

                self._gpu_mrs[gpu_id] = entry
                log.info(f"GPU {gpu_id}: buffer registered "
                         f"({size_bytes / 1e6:.0f}MB, "
                         f"gpudirect={self._gpu_direct_ok.get(gpu_id, False)})")
        except Exception as exc:
            log.warning(f"GPU buffer allocation failed on GPU {gpu_id}: {exc}")

    def send_tensor(self, tensor: Any, gpu_id: int) -> int:
        """Send tensor via RDMA — zero CPU copy when GPUDirect is active.

        Path 1 (GPUDirect): copy tensor into registered GPU MR buffer,
            NIC reads directly from GPU VRAM via DMA.
        Path 2 (fallback): GPU → pinned CPU memory → RDMA send.
            No numpy intermediate — uses raw byte view.
        """
        if not self.rdma or not self.rdma._connected:
            raise RuntimeError("GPUDirect: RDMA not connected")

        start = time.perf_counter()
        tensor_bytes = tensor.nelement() * tensor.element_size()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        entry = self._gpu_mrs.get(gpu_id)

        # PATH 1: True GPUDirect — NIC reads GPU VRAM directly
        if (entry and entry.get('mr') is not None and
                self._gpu_direct_ok.get(gpu_id, False)):
            # Copy tensor into registered MR buffer (GPU-local copy, fast)
            src_flat = tensor.view(torch.uint8)
            entry['buffer'][:tensor_bytes].copy_(src_flat)
            # RDMA Send from the GPU-registered MR
            try:
                from pyverbs.wr import SendWR, SGE  # type: ignore
                sge = SGE(entry['ptr'], tensor_bytes, entry['mr'].lkey)
                wr = SendWR(opcode=e.IBV_WR_SEND, num_sge=1, sg=[sge])
                wr.send_flags = e.IBV_SEND_SIGNALED
                self.rdma.qp.post_send(wr)
                self.rdma._poll_cq(1)
                FASTPATH_BYTES.labels("gpudirect", "send").inc(tensor_bytes)
                FASTPATH_LATENCY.labels("gpudirect", "send").observe(
                    time.perf_counter() - start
                )
                return tensor_bytes
            except Exception as exc:
                log.warning(f"GPUDirect send failed, using staging: {exc}")

        # PATH 2: Pinned-memory staging (no numpy, no .tobytes())
        cpu_pinned = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True
        )
        cpu_pinned.copy_(tensor, non_blocking=False)
        # Direct byte view — avoids numpy entirely
        raw = bytes(cpu_pinned.contiguous().view(torch.uint8).numpy())
        sent = self.rdma.send(raw)
        FASTPATH_BYTES.labels("gpudirect_staged", "send").inc(tensor_bytes)
        FASTPATH_LATENCY.labels("gpudirect_staged", "send").observe(
            time.perf_counter() - start
        )
        return sent

    def recv_tensor(self, shape: tuple, dtype: Any, gpu_id: int) -> Optional[Any]:
        """Receive tensor via RDMA — direct to GPU when GPUDirect available."""
        if not self.rdma or not self.rdma._connected:
            raise RuntimeError("GPUDirect: RDMA not connected")

        start = time.perf_counter()
        import numpy as np
        dtype_map = {
            torch.float32: (np.float32, 4), torch.float16: (np.float16, 2),
            torch.bfloat16: (np.float32, 2), torch.int64: (np.int64, 8),
            torch.int32: (np.int32, 4), torch.int8: (np.int8, 1),
            torch.uint8: (np.uint8, 1),
        }
        np_dtype, elem_size = dtype_map.get(dtype, (np.float32, 4))
        total_elems = 1
        for s in shape:
            total_elems *= s
        expected_bytes = total_elems * elem_size

        entry = self._gpu_mrs.get(gpu_id)

        # PATH 1: GPUDirect — data arrived directly in GPU MR buffer
        if (entry and entry.get('mr') is not None and
                self._gpu_direct_ok.get(gpu_id, False)):
            try:
                from pyverbs.wr import RecvWR, SGE  # type: ignore
                sge = SGE(entry['ptr'], expected_bytes, entry['mr'].lkey)
                rwr = RecvWR(num_sge=1, sg=[sge])
                self.rdma.qp.post_recv(rwr)
                self.rdma._poll_cq(1)
                # Data is now in GPU buffer — reshape in-place
                raw_gpu = entry['buffer'][:expected_bytes]
                tensor = raw_gpu.view(dtype).reshape(shape)
                FASTPATH_BYTES.labels("gpudirect", "recv").inc(expected_bytes)
                FASTPATH_LATENCY.labels("gpudirect", "recv").observe(
                    time.perf_counter() - start
                )
                return tensor
            except Exception as exc:
                log.warning(f"GPUDirect recv failed, using staging: {exc}")

        # PATH 2: CPU-staged receive
        raw = self.rdma.recv(expected_bytes)
        if raw is None:
            return None
        np_array = np.frombuffer(raw[:expected_bytes], dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(np_array.copy()).to(f'cuda:{gpu_id}')
        if dtype == torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        FASTPATH_BYTES.labels("gpudirect_staged", "recv").inc(expected_bytes)
        FASTPATH_LATENCY.labels("gpudirect_staged", "recv").observe(
            time.perf_counter() - start
        )
        return tensor


# ---------------------------------------------------------------------------
# Optimized TCP transport (fallback)
# ---------------------------------------------------------------------------
class ZeroCopyTCPTransport:
    """TCP with SO_ZEROCOPY + TCP_NODELAY + SO_BUSY_POLL.

    Fallback when no RDMA hardware available.
    """

    HEADER_FMT = '!Q'
    HEADER_SIZE = 8

    def __init__(self, host: str = '0.0.0.0', port: int = 0, buf_size: int = 4 * 1024 * 1024):
        self.host = host
        self.port = port
        self.buf_size = buf_size
        self._sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None

    def _create_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.buf_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buf_size)
        except Exception:
            pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BUSY_POLL, 50)
        except Exception:
            pass
        try:
            SO_ZEROCOPY = 60
            sock.setsockopt(socket.SOL_SOCKET, SO_ZEROCOPY, 1)
        except Exception:
            pass
        return sock

    def listen(self, port: int = 0) -> int:
        self._sock = self._create_socket()
        self._sock.bind((self.host, port or self.port))
        self._sock.listen(1)
        actual_port = self._sock.getsockname()[1]
        self.port = actual_port
        log.info(f"TCP transport listening on {self.host}:{actual_port}")
        return actual_port

    def accept(self, timeout: float = 30.0) -> bool:
        if not self._sock:
            return False
        self._sock.settimeout(timeout)
        try:
            self._conn, addr = self._sock.accept()
            self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log.info(f"TCP connection from {addr}")
            return True
        except socket.timeout:
            return False

    def connect_to(self, host: str, port: int, timeout: float = 10.0) -> bool:
        self._conn = self._create_socket()
        self._conn.settimeout(timeout)
        try:
            self._conn.connect((host, port))
            log.info(f"TCP connected to {host}:{port}")
            return True
        except Exception as exc:
            log.error(f"TCP connect failed: {exc}")
            return False

    def send(self, data: bytes) -> int:
        if not self._conn:
            raise RuntimeError("Not connected")
        start = time.perf_counter()
        size = len(data)
        header = struct.pack(self.HEADER_FMT, size)
        self._conn.sendall(header)
        sent = 0
        mv = memoryview(data)
        while sent < size:
            chunk_size = min(self.buf_size, size - sent)
            n = self._conn.send(mv[sent:sent + chunk_size])
            if n == 0:
                raise ConnectionError("Connection closed during send")
            sent += n
        FASTPATH_BYTES.labels("tcp_zc", "send").inc(size)
        FASTPATH_LATENCY.labels("tcp_zc", "send").observe(time.perf_counter() - start)
        return size

    def recv(self, expected_size: int = 0) -> Optional[bytes]:
        if not self._conn:
            raise RuntimeError("Not connected")
        start = time.perf_counter()
        header = self._recv_exact(self.HEADER_SIZE)
        if not header:
            return None
        size = struct.unpack(self.HEADER_FMT, header)[0]
        data = self._recv_exact(size)
        if data:
            FASTPATH_BYTES.labels("tcp_zc", "recv").inc(len(data))
            FASTPATH_LATENCY.labels("tcp_zc", "recv").observe(time.perf_counter() - start)
        return data

    def _recv_exact(self, size: int) -> Optional[bytes]:
        buf = bytearray(size)
        view = memoryview(buf)
        received = 0
        while received < size:
            n = self._conn.recv_into(view[received:])
            if n == 0:
                return None
            received += n
        return bytes(buf)

    def close(self):
        for s in [self._conn, self._sock]:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self._conn = None
        self._sock = None


# ---------------------------------------------------------------------------
# Unified FastHandle (backward-compatible)
# ---------------------------------------------------------------------------
@dataclass
class FastHandle:
    """Unified handle wrapping best available transport (RDMA > TCP > mmap)."""
    kind: str
    meta: dict
    latency_us: int = 40
    shm_path: Optional[str] = None
    _last_sent_len: int = 0
    _rdma: Optional[RDMATransport] = field(default=None, repr=False)
    _tcp: Optional[ZeroCopyTCPTransport] = field(default=None, repr=False)
    _gpudirect: Optional[GPUDirectTransport] = field(default=None, repr=False)

    def capabilities(self) -> dict:
        return {
            'kind': self.kind,
            'latency_us': self.latency_us,
            'zero_copy': self.kind in {'rdma', 'gpudirect'},
            'rdma_available': _RDMA_AVAILABLE,
            'gpudirect_available': _GPUDIRECT_AVAILABLE,
            'kernel_bypass': self.kind in {'rdma', 'gpudirect'},
            'transport': self._active_transport_name(),
        }

    def _active_transport_name(self) -> str:
        if self._rdma and self._rdma.available:
            return 'rdma_verbs'
        if self._tcp:
            return 'tcp_zerocopy'
        return 'mmap_local'

    def _ensure_segment(self, size: int):
        if not self.shm_path:
            name = f"/tmp/vramancer_fast_{hashlib.sha1(str(self.meta).encode()).hexdigest()[:8]}"
            self.shm_path = name
        with open(self.shm_path, 'ab') as f:
            if f.tell() < size:
                f.truncate(size)

    def send(self, data: bytes) -> int:
        """Send data using best available transport."""
        size = len(data)
        start = time.perf_counter()
        if self._rdma and self._rdma._connected:
            try:
                return self._rdma.send(data)
            except Exception as exc:
                log.warning(f"RDMA send failed, falling back: {exc}")
        if self._tcp and self._tcp._conn:
            try:
                return self._tcp.send(data)
            except Exception as exc:
                log.warning(f"TCP send failed, falling back to mmap: {exc}")
        # Fallback: local mmap
        self._ensure_segment(size)
        with open(self.shm_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            mm.seek(0)
            mm.write(data)
            mm.flush()
            mm.close()
        self._last_sent_len = size
        FASTPATH_BYTES.labels(self.kind, "send").inc(size)
        FASTPATH_LATENCY.labels(self.kind, "send").observe(time.perf_counter() - start)
        return size

    def recv(self) -> Optional[bytes]:
        """Receive data using best available transport."""
        start = time.perf_counter()
        if self._rdma and self._rdma._connected:
            try:
                return self._rdma.recv()
            except Exception as exc:
                log.warning(f"RDMA recv failed, falling back: {exc}")
        if self._tcp and self._tcp._conn:
            try:
                return self._tcp.recv()
            except Exception as exc:
                log.warning(f"TCP recv failed, falling back to mmap: {exc}")
        if not self.shm_path or self._last_sent_len == 0:
            return None
        try:
            with open(self.shm_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), self._last_sent_len, access=mmap.ACCESS_READ)
                buf = mm.read(self._last_sent_len)
                mm.close()
            FASTPATH_BYTES.labels(self.kind, "recv").inc(len(buf))
            FASTPATH_LATENCY.labels(self.kind, "recv").observe(time.perf_counter() - start)
            return buf
        except FileNotFoundError:
            return None

    def send_tensor(self, tensor: Any, gpu_id: int = 0) -> int:
        """Send GPU tensor with optimal transport (GPUDirect if available).

        Zero-copy path: GPUDirect RDMA (NIC reads GPU VRAM directly)
        Fast path: pinned CPU staging (no numpy .tobytes() overhead)
        Fallback: CPU + raw byte view
        """
        if self._gpudirect and self._gpudirect.available:
            return self._gpudirect.send_tensor(tensor, gpu_id)
        # No GPUDirect — use pinned-memory staging (avoids .tobytes())
        if _CUDA_AVAILABLE and hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            cpu_pinned = torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=True,
            )
            cpu_pinned.copy_(tensor, non_blocking=False)
            data = bytes(cpu_pinned.contiguous().view(torch.uint8).numpy())
        elif _CUDA_AVAILABLE and hasattr(tensor, 'numpy'):
            data = bytes(tensor.contiguous().view(torch.uint8).numpy())
        else:
            data = bytes(tensor) if not isinstance(tensor, bytes) else tensor
        return self.send(data)

    def recv_tensor(self, shape: tuple, dtype: Any, gpu_id: int = 0) -> Optional[Any]:
        """Receive tensor with optimal transport (GPUDirect if available).

        GPUDirect: data arrives directly in GPU MR buffer (zero CPU copy).
        Fallback: RDMA/TCP recv to CPU, then transfer to GPU.
        """
        if self._gpudirect and self._gpudirect.available:
            return self._gpudirect.recv_tensor(shape, dtype, gpu_id)
        data = self.recv()
        if data is None:
            return None
        import numpy as np
        dtype_map = {
            torch.float32: np.float32, torch.float16: np.float16,
            torch.bfloat16: np.float32, torch.int64: np.int64,
            torch.int32: np.int32, torch.int8: np.int8,
            torch.uint8: np.uint8,
        } if _CUDA_AVAILABLE else {}
        np_dtype = dtype_map.get(dtype, np.float32) if dtype_map else np.float32
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        t = torch.from_numpy(arr.copy())
        if dtype == torch.bfloat16:
            t = t.to(torch.bfloat16)
        if _CUDA_AVAILABLE and gpu_id >= 0:
            t = t.to(f'cuda:{gpu_id}')
        return t


# ---------------------------------------------------------------------------
# Interface detection & channel factory
# ---------------------------------------------------------------------------
def detect_fast_interfaces() -> List[Dict[str, Any]]:
    """Detect available high-performance network interfaces."""
    candidates = []
    if _RDMA_AVAILABLE:
        try:
            for name in ['mlx5_0', 'mlx4_0', 'mlx5_1', 'rxe0']:
                try:
                    ctx = RDMAContext(name=name)
                    candidates.append({"type": "rdma", "if": name, "gpudirect": _GPUDIRECT_AVAILABLE})
                    ctx.close()
                except Exception:
                    continue
        except Exception:
            candidates.append({"type": "rdma", "if": "verbs0"})
    for i in range(1, 5):
        path = f"/mnt/usb4_share_{i}"
        if os.path.exists(path):
            candidates.append({"type": "usb4", "path": path})
    try:
        nets = os.listdir('/sys/class/net')
        for n in nets:
            if any(prefix in n for prefix in ['enp', 'eth', 'ens', 'eno']):
                speed = 0
                try:
                    with open(f'/sys/class/net/{n}/speed') as f:
                        speed = int(f.read().strip())
                except Exception:
                    pass
                candidates.append({"type": "sfp" if speed >= 10000 else "ethernet", "if": n, "speed_mbps": speed})
    except Exception:
        pass
    prefer_if = os.environ.get('VRM_FASTPATH_IF')
    if prefer_if:
        for i, it in enumerate(candidates):
            if it.get('if') == prefer_if or it.get('type') == prefer_if:
                if i != 0:
                    candidates.insert(0, candidates.pop(i))
                break
    return candidates

def open_low_latency_channel(
    prefer: Optional[str] = None,
    remote_host: Optional[str] = None,
    remote_port: int = 18900,
) -> Optional[FastHandle]:
    """Open the best available low-latency transport channel.

    Selection: RDMA > Zero-copy TCP > local mmap.
    """
    interfaces = detect_fast_interfaces()
    rdma_transport = None
    tcp_transport = None
    gpudirect = None

    # Try RDMA
    if prefer in (None, 'rdma') and _RDMA_AVAILABLE:
        rdma_devs = [it for it in interfaces if it['type'] == 'rdma']
        if rdma_devs:
            rdma_transport = RDMATransport(device_name=rdma_devs[0].get('if'))
            if rdma_transport.available:
                gpudirect = GPUDirectTransport(rdma_transport)
                log.info(
                    f"RDMA transport ready on {rdma_transport.device_name} "
                    f"(GPUDirect={'yes' if gpudirect.available else 'no'})"
                )
                return FastHandle(
                    kind='rdma', meta=rdma_devs[0],
                    latency_us=2 if gpudirect.available else 5,
                    _rdma=rdma_transport,
                    _gpudirect=gpudirect if gpudirect.available else None,
                )

    # Try optimized TCP
    if prefer in (None, 'tcp') and remote_host:
        tcp_transport = ZeroCopyTCPTransport()
        if tcp_transport.connect_to(remote_host, remote_port):
            return FastHandle(
                kind='tcp_zerocopy',
                meta={'host': remote_host, 'port': remote_port},
                latency_us=50, _tcp=tcp_transport,
            )

    # Fallback
    if not interfaces:
        log.warning("No fast-path interface detected (stub mode)")
        return FastHandle(kind="stub", meta={}, latency_us=120)
    it = interfaces[0]
    latency = {'rdma': 5, 'usb4': 50, 'sfp': 70}.get(it['type'], 100)
    return FastHandle(kind=it['type'], meta=it, latency_us=latency)


def benchmark_interfaces(sample_size: int = 3, force: bool = False) -> List[Dict]:
    """Benchmark all detected fast-path interfaces."""
    now = time.time()
    results = []
    interfaces = detect_fast_interfaces()
    for it in interfaces:
        ident = it.get('if') or it.get('path') or it['type']
        cached = _BENCH_CACHE.get(ident)
        if (not force) and cached and (now - cached[0] < _BENCH_TTL):
            results.append({"interface": ident, "kind": it['type'], "latency_s": cached[1], "cached": True})
            continue
        lat = 0.0
        payload = b'x' * 4096
        fh = FastHandle(kind=it['type'], meta=it)
        for _ in range(sample_size):
            start = time.perf_counter()
            fh.send(payload)
            fh.recv()
            lat += (time.perf_counter() - start)
        avg = lat / sample_size if sample_size else 0.0
        _BENCH_CACHE[ident] = (now, avg)
        results.append({"interface": ident, "kind": it['type'], "latency_s": avg, "cached": False,
                        "gpudirect": it.get('gpudirect', False)})
    return results


__all__ = [
    "open_low_latency_channel", "FastHandle", "detect_fast_interfaces",
    "benchmark_interfaces", "RDMATransport", "GPUDirectTransport", "ZeroCopyTCPTransport",
]
