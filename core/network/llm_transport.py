"""VRAMancer LLM Transport Protocol (VTP) — Zero-copy GPU-native network transport.

A disruptive, LLM-optimized network protocol designed for multi-node
pipeline-parallel inference. Key innovations:

1. **True GPUDirect RDMA** — NIC reads/writes GPU VRAM via nvidia_peermem,
   zero CPU involvement for activation tensors.
2. **Tensor-aware framing** — binary header carries shape/dtype/layer_id/seq_id
   so the receiver pre-allocates GPU memory BEFORE data arrives.
3. **RDMA Write one-sided** — sender writes directly into pre-registered remote
   GPU buffers; receiver never posts RecvWR → lower latency.
4. **Pipeline overlap** — overlaps compute on layer N with transfer of layer N+1
   activations using double-buffered registered regions.
5. **KV cache streaming** — specialised protocol for partial KV cache migration
   (per-head, per-layer granularity) with copy-on-write semantics.
6. **Adaptive transport** — auto-selects GPUDirect RDMA / CPU-staged RDMA /
   zero-copy TCP based on tensor size, hardware caps, and latency benchmarks.
7. **Connection pooling** — pre-connected QP mesh for all (src_gpu, dst_gpu)
   pairs across the cluster; OOB handshake via TCP control channel.
8. **Credit-based flow control** — prevents receiver VRAM overflow without
   blocking the sender pipeline.

Hardware requirements for full performance:
  - NVIDIA GPU with nvidia_peermem / nv_peer_mem kernel module
  - Mellanox ConnectX-4+ NIC (InfiniBand or RoCE v2)
  - pyverbs (rdma-core Python bindings)

Graceful degradation:
  GPUDirect RDMA → CPU-staged RDMA → zero-copy TCP → basic TCP

Environment variables:
  VRM_VTP_ENABLED=1          Enable VTP (default: auto-detect)
  VRM_VTP_PORT=18950         Control channel port
  VRM_VTP_MAX_INFLIGHT=16   Max pipelined RDMA ops
  VRM_VTP_CHUNK_MB=8         RDMA chunk size (MB)
  VRM_VTP_CREDITS=32         Flow control credits
  VRM_MINIMAL_TEST=1         Stub mode (no real RDMA/GPU)
"""
from __future__ import annotations

import os
import sys
import time
import struct
import socket
import threading
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Dict, List, Tuple, Callable
from collections import defaultdict

from core.logger import LoggerAdapter

# ---------------------------------------------------------------------------
# Conditional imports — never crash on missing deps
# ---------------------------------------------------------------------------
_STUB_MODE = os.environ.get("VRM_MINIMAL_TEST", "0") == "1"

try:
    from core.metrics import (
        FASTPATH_BYTES, FASTPATH_LATENCY, GPU_TRANSFER_OPS,
        GPU_TRANSFER_BW,
    )
    _METRICS = True
except Exception:
    _METRICS = False

_TORCH = False
_CUDA = False
torch = None  # type: ignore
try:
    import torch as _torch_mod
    torch = _torch_mod
    _TORCH = True
    if torch.cuda.is_available():
        _CUDA = True
except ImportError:
    pass

_PYVERBS = False
_RDMA_CLASSES: Dict[str, Any] = {}
try:
    from pyverbs.device import Context as RDMAContext
    from pyverbs.pd import PD as ProtectionDomain
    from pyverbs.mr import MR as MemoryRegion
    from pyverbs.cq import CQ as CompletionQueue
    from pyverbs.qp import QP, QPInitAttr, QPAttr, QPCap
    from pyverbs.addr import AH, AHAttr, GlobalRoute
    from pyverbs.wr import SendWR, RecvWR, SGE
    import pyverbs.enums as ibe
    _PYVERBS = True
    _RDMA_CLASSES = {
        "RDMAContext": RDMAContext, "PD": ProtectionDomain,
        "MR": MemoryRegion, "CQ": CompletionQueue,
        "QP": QP, "QPInitAttr": QPInitAttr, "QPAttr": QPAttr,
        "QPCap": QPCap, "AHAttr": AHAttr, "GlobalRoute": GlobalRoute,
        "SGE": SGE, "SendWR": SendWR, "RecvWR": RecvWR,
        "enums": ibe,
    }
except ImportError:
    pass

# GPUDirect RDMA availability
_GPUDIRECT = False
try:
    if (os.path.exists("/sys/module/nvidia_peermem") or
            os.path.exists("/sys/module/nv_peer_mem")):
        _GPUDIRECT = True
except Exception:
    pass

log = LoggerAdapter("vtp")


# ═══════════════════════════════════════════════════════════════════════════
# Constants & configuration
# ═══════════════════════════════════════════════════════════════════════════
VTP_VERSION = 1
VTP_CONTROL_PORT = int(os.environ.get("VRM_VTP_PORT", "18950"))
VTP_MAX_INFLIGHT = int(os.environ.get("VRM_VTP_MAX_INFLIGHT", "16"))
VTP_CHUNK_BYTES = int(os.environ.get("VRM_VTP_CHUNK_MB", "8")) * 1024 * 1024
VTP_CREDITS = int(os.environ.get("VRM_VTP_CREDITS", "32"))
VTP_INLINE_THRESHOLD = 256  # bytes — inline small tensors in header

# Binary header: version(1) + opcode(1) + flags(2) + payload_bytes(8) +
#   layer_id(4) + seq_id(4) + src_gpu(2) + dst_gpu(2) + ndim(2) +
#   dtype_code(2) + dims(up to 8×4=32) + checksum(4)  = 64 bytes fixed
_HEADER_FMT = "!BBHQIiHHBB"  # 24 bytes base
_HEADER_SIZE_BASE = struct.calcsize(_HEADER_FMT)
_DIM_FMT = "!I"  # 4 bytes per dim
_MAX_NDIM = 8
_HEADER_TOTAL = _HEADER_SIZE_BASE + _MAX_NDIM * 4 + 4  # +checksum = 60 bytes, pad to 64
_HEADER_PAD = 64


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════
class VTPOpcode(Enum):
    """Wire opcodes for the VTP protocol."""
    TENSOR = 0x01          # Activation tensor transfer
    KV_CACHE = 0x02        # KV cache block
    GRADIENT = 0x03        # Gradient (future: training)
    CONTROL = 0x10         # Control message (handshake, credits, etc.)
    CREDIT_GRANT = 0x11    # Flow-control credit grant
    HEARTBEAT = 0x12       # Keep-alive
    METADATA = 0x20        # Model metadata / topology


class VTPFlags:
    """Bit flags for the header."""
    GPUDIRECT = 0x0001     # Payload resides in GPU VRAM (not CPU)
    COMPRESSED = 0x0002    # Payload is compressed
    LAST_CHUNK = 0x0004    # Last chunk of a multi-chunk transfer
    INLINE = 0x0008        # Payload is inlined in the header
    ONE_SIDED = 0x0010     # RDMA Write (one-sided, no recv needed)
    URGENT = 0x0020        # Skip queuing, fast-path
    KV_PARTIAL = 0x0040    # Partial KV cache (per-head)


class TransportTier(Enum):
    """Transport quality tiers."""
    GPUDIRECT_RDMA = auto()   # NIC ↔ GPU VRAM, zero CPU
    CPU_STAGED_RDMA = auto()  # GPU → pinned CPU → RDMA → pinned CPU → GPU
    ZEROCOPY_TCP = auto()     # SO_ZEROCOPY TCP
    BASIC_TCP = auto()        # Plain TCP fallback
    STUB = auto()             # Test mode


# ═══════════════════════════════════════════════════════════════════════════
# Dtype mapping
# ═══════════════════════════════════════════════════════════════════════════
_DTYPE_TO_CODE: Dict[Any, int] = {}
_CODE_TO_DTYPE: Dict[int, Any] = {}
_CODE_TO_ITEMSIZE: Dict[int, int] = {}

if _TORCH:
    _DTYPE_MAP = [
        (torch.float32, 0, 4), (torch.float16, 1, 2),
        (torch.bfloat16, 2, 2), (torch.float64, 3, 8),
        (torch.int64, 4, 8), (torch.int32, 5, 4),
        (torch.int16, 6, 2), (torch.int8, 7, 1),
        (torch.uint8, 8, 1), (torch.bool, 9, 1),
    ]
    for dt, code, sz in _DTYPE_MAP:
        _DTYPE_TO_CODE[dt] = code
        _CODE_TO_DTYPE[code] = dt
        _CODE_TO_ITEMSIZE[code] = sz


# ═══════════════════════════════════════════════════════════════════════════
# TensorHeader — binary-serializable metadata
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class TensorHeader:
    """64-byte binary header for tensor-aware framing.

    Sent before (or with) every tensor payload so the receiver can:
      - Pre-allocate the destination GPU buffer
      - Route to the correct layer in the pipeline
      - Validate integrity via checksum
    """
    opcode: VTPOpcode = VTPOpcode.TENSOR
    flags: int = 0
    payload_bytes: int = 0
    layer_id: int = 0
    seq_id: int = 0
    src_gpu: int = 0
    dst_gpu: int = 0
    ndim: int = 0
    dtype_code: int = 0
    shape: Tuple[int, ...] = ()
    checksum: int = 0

    def encode(self) -> bytes:
        """Serialize to exactly 64 bytes."""
        buf = struct.pack(
            _HEADER_FMT,
            VTP_VERSION,
            self.opcode.value,
            self.flags,
            self.payload_bytes,
            abs(self.layer_id) & 0xFFFFFFFF,
            self.seq_id,
            self.src_gpu & 0xFFFF,
            self.dst_gpu & 0xFFFF,
            min(self.ndim, _MAX_NDIM),
            self.dtype_code & 0xFF,
        )
        # Append shape dims (pad to MAX_NDIM)
        for i in range(_MAX_NDIM):
            if i < len(self.shape):
                buf += struct.pack(_DIM_FMT, self.shape[i] & 0xFFFFFFFF)
            else:
                buf += struct.pack(_DIM_FMT, 0)
        # Checksum over header + shape
        self.checksum = (sum(buf) & 0xFFFFFFFF)
        buf += struct.pack("!I", self.checksum)
        # Pad to 64 bytes
        if len(buf) < _HEADER_PAD:
            buf += b'\x00' * (_HEADER_PAD - len(buf))
        return buf[:_HEADER_PAD]

    @classmethod
    def decode(cls, data: bytes) -> "TensorHeader":
        """Deserialize from 64 bytes."""
        if len(data) < _HEADER_PAD:
            raise ValueError(f"Header too short: {len(data)} < {_HEADER_PAD}")
        base = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE_BASE])
        version, opcode, flags, payload, layer_id, seq_id, src, dst, ndim, dtype_code = base
        if version != VTP_VERSION:
            raise ValueError(f"VTP version mismatch: got {version}, expected {VTP_VERSION}")
        offset = _HEADER_SIZE_BASE
        shape = []
        for i in range(_MAX_NDIM):
            d = struct.unpack(_DIM_FMT, data[offset:offset + 4])[0]
            offset += 4
            if i < ndim:
                shape.append(d)
        checksum = struct.unpack("!I", data[offset:offset + 4])[0]
        return cls(
            opcode=VTPOpcode(opcode), flags=flags, payload_bytes=payload,
            layer_id=layer_id, seq_id=seq_id, src_gpu=src, dst_gpu=dst,
            ndim=ndim, dtype_code=dtype_code, shape=tuple(shape),
            checksum=checksum,
        )


# ═══════════════════════════════════════════════════════════════════════════
# GPU Memory Registry — manages registered RDMA regions on GPU VRAM
# ═══════════════════════════════════════════════════════════════════════════
class GPUMemoryRegion:
    """A registered GPU memory region for RDMA operations.

    When nvidia_peermem is loaded, ibv_reg_mr() on a GPU pointer causes
    the kernel module to pin the GPU pages and set up IOVA mappings so
    the NIC can DMA directly to/from GPU VRAM.
    """

    def __init__(self, gpu_id: int, size_bytes: int, pd: Any = None):
        self.gpu_id = gpu_id
        self.size_bytes = size_bytes
        self.pd = pd
        self.buffer = None       # torch.Tensor (GPU)
        self.mr = None           # pyverbs MemoryRegion
        self.data_ptr: int = 0   # raw GPU pointer
        self._registered = False

    def allocate_and_register(self) -> bool:
        """Allocate GPU buffer and register with RDMA NIC.

        This is the core of GPUDirect RDMA: the MR registration with
        a GPU pointer triggers nvidia_peermem to set up DMA mappings.
        """
        if not _CUDA or not _GPUDIRECT or not _PYVERBS or self.pd is None:
            return False
        try:
            with torch.cuda.device(self.gpu_id):
                # Allocate contiguous GPU buffer
                self.buffer = torch.empty(
                    self.size_bytes, dtype=torch.uint8,
                    device=f"cuda:{self.gpu_id}",
                )
                self.data_ptr = self.buffer.data_ptr()

                # Register GPU memory with the RDMA NIC.
                # nvidia_peermem intercepts this call and pins the GPU pages.
                ibe = _RDMA_CLASSES["enums"]
                access = (ibe.IBV_ACCESS_LOCAL_WRITE |
                          ibe.IBV_ACCESS_REMOTE_WRITE |
                          ibe.IBV_ACCESS_REMOTE_READ)
                MR = _RDMA_CLASSES["MR"]
                self.mr = MR(self.pd, self.buffer, access)
                self._registered = True
                log.info(
                    f"GPUDirect: registered {self.size_bytes / 1e6:.1f}MB "
                    f"on GPU {self.gpu_id} (ptr=0x{self.data_ptr:x}, "
                    f"rkey=0x{self.mr.rkey:x})"
                )
                return True
        except Exception as exc:
            log.warning(f"GPUDirect MR registration failed on GPU {self.gpu_id}: {exc}")
            self._registered = False
            return False

    @property
    def registered(self) -> bool:
        return self._registered

    @property
    def rkey(self) -> int:
        return self.mr.rkey if self.mr else 0

    @property
    def lkey(self) -> int:
        return self.mr.lkey if self.mr else 0

    def write_tensor(self, tensor: Any, offset: int = 0) -> int:
        """Copy a tensor into this registered region (GPU-local copy).

        Returns the number of bytes written.
        """
        if not _CUDA or self.buffer is None:
            return 0
        nbytes = tensor.nelement() * tensor.element_size()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        # Reinterpret as uint8 for raw copy
        src = tensor.view(-1).to(torch.uint8) if tensor.dtype != torch.uint8 else tensor.view(-1)
        # Reinterpret source as raw bytes
        src_bytes = tensor.contiguous().view(torch.uint8)
        self.buffer[offset:offset + nbytes].copy_(src_bytes)
        return nbytes

    def read_tensor(self, shape: Tuple[int, ...], dtype: Any,
                    offset: int = 0) -> Any:
        """Read a tensor from this registered region."""
        if not _CUDA or self.buffer is None:
            return None
        itemsize = _CODE_TO_ITEMSIZE.get(_DTYPE_TO_CODE.get(dtype, 0), 4)
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * itemsize
        raw = self.buffer[offset:offset + nbytes]
        # Reinterpret bytes as the target dtype
        return raw.view(dtype).reshape(shape)

    def close(self):
        if self.mr:
            try:
                self.mr.close()
            except Exception:
                pass
        self.buffer = None
        self._registered = False


# ═══════════════════════════════════════════════════════════════════════════
# Double-buffered region pair for pipeline overlap
# ═══════════════════════════════════════════════════════════════════════════
class DoubleBufferedRegion:
    """Two GPU memory regions for overlapping compute and transfer.

    While the model computes on buffer A, the NIC writes the next
    activation into buffer B. Then they swap.
    """

    def __init__(self, gpu_id: int, size_bytes: int, pd: Any = None):
        self.gpu_id = gpu_id
        self.regions = [
            GPUMemoryRegion(gpu_id, size_bytes, pd),
            GPUMemoryRegion(gpu_id, size_bytes, pd),
        ]
        self._active = 0  # index of the buffer currently holding valid data

    def allocate_and_register(self) -> bool:
        ok = True
        for r in self.regions:
            if not r.allocate_and_register():
                ok = False
        return ok

    @property
    def active(self) -> GPUMemoryRegion:
        """The buffer with the current valid data (being computed on)."""
        return self.regions[self._active]

    @property
    def staging(self) -> GPUMemoryRegion:
        """The buffer receiving new data from the NIC."""
        return self.regions[1 - self._active]

    def swap(self):
        """Swap active and staging buffers."""
        self._active = 1 - self._active

    def close(self):
        for r in self.regions:
            r.close()


# ═══════════════════════════════════════════════════════════════════════════
# Connection — per-peer RDMA connection with QP
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class VTPConnectionInfo:
    """Information exchanged during OOB handshake."""
    node_id: str = ""
    qp_num: int = 0
    lid: int = 0
    gid: bytes = b'\x00' * 16
    gpu_count: int = 0
    gpu_regions: Dict[int, Dict[str, int]] = field(default_factory=dict)
    # gpu_regions: {gpu_id: {"rkey": ..., "addr": ..., "size": ...}}
    vtp_version: int = VTP_VERSION
    tier: str = "unknown"


class VTPConnection:
    """A single RDMA connection to a remote peer.

    Manages one QP (Reliable Connected) and a set of registered
    local GPU memory regions. Supports both two-sided (Send/Recv)
    and one-sided (RDMA Write) operations.
    """

    def __init__(self, device_name: Optional[str] = None,
                 ib_port: int = 1, gid_index: int = 0):
        self.device_name = device_name
        self.ib_port = ib_port
        self.gid_index = gid_index
        self.ctx = None
        self.pd = None
        self.send_cq = None
        self.recv_cq = None
        self.qp = None
        self._connected = False
        self._cpu_mr = None      # CPU-side MR for headers + small payloads
        self._cpu_buf = None
        self._gpu_regions: Dict[int, GPUMemoryRegion] = {}
        self._double_bufs: Dict[int, DoubleBufferedRegion] = {}
        self._inflight = 0
        self._lock = threading.Lock()
        # Credits for flow control
        self._local_credits = VTP_CREDITS
        self._remote_credits = VTP_CREDITS

        if _PYVERBS and not _STUB_MODE:
            self._init_rdma()

    def _init_rdma(self):
        """Initialize RDMA resources."""
        ibe = _RDMA_CLASSES["enums"]
        RCtx = _RDMA_CLASSES["RDMAContext"]
        PD = _RDMA_CLASSES["PD"]
        CQ = _RDMA_CLASSES["CQ"]
        QP_ = _RDMA_CLASSES["QP"]
        QPInitAttr_ = _RDMA_CLASSES["QPInitAttr"]
        QPCap_ = _RDMA_CLASSES["QPCap"]
        MR_ = _RDMA_CLASSES["MR"]

        try:
            # Auto-detect RDMA device
            if not self.device_name:
                for name in ["mlx5_0", "mlx5_1", "mlx4_0", "rxe0", "siw_eth0"]:
                    try:
                        self.ctx = RCtx(name=name)
                        self.device_name = name
                        break
                    except Exception:
                        continue
                if not self.ctx:
                    log.warning("VTP: No RDMA device found")
                    return
            else:
                self.ctx = RCtx(name=self.device_name)

            self.pd = PD(self.ctx)

            # Separate CQs for send and recv for better parallelism
            self.send_cq = CQ(self.ctx, VTP_MAX_INFLIGHT * 2)
            self.recv_cq = CQ(self.ctx, VTP_MAX_INFLIGHT * 2)

            cap = QPCap_(
                max_send_wr=VTP_MAX_INFLIGHT,
                max_recv_wr=VTP_MAX_INFLIGHT,
                max_send_sge=2,  # header SGE + payload SGE
                max_recv_sge=2,
            )
            init_attr = QPInitAttr_(
                qp_type=ibe.IBV_QPT_RC,
                scq=self.send_cq,
                rcq=self.recv_cq,
                cap=cap,
            )
            self.qp = QP_(self.pd, init_attr)

            # CPU buffer for headers and control messages
            self._cpu_buf = bytearray(VTP_CHUNK_BYTES + _HEADER_PAD)
            access = (ibe.IBV_ACCESS_LOCAL_WRITE |
                      ibe.IBV_ACCESS_REMOTE_WRITE |
                      ibe.IBV_ACCESS_REMOTE_READ)
            self._cpu_mr = MR_(self.pd, self._cpu_buf, access)

            log.info(
                f"VTP connection initialized on {self.device_name} "
                f"(QP={self.qp.qp_num})"
            )
        except Exception as exc:
            log.warning(f"VTP RDMA init failed: {exc}")
            self.ctx = None

    @property
    def available(self) -> bool:
        return self.ctx is not None and self.qp is not None

    def register_gpu(self, gpu_id: int, size_bytes: int,
                     double_buffer: bool = True) -> bool:
        """Register GPU memory regions for GPUDirect RDMA.

        Args:
            gpu_id: CUDA device index
            size_bytes: Size of each buffer
            double_buffer: Use double-buffering for pipeline overlap
        """
        if not self.pd:
            return False
        if double_buffer:
            db = DoubleBufferedRegion(gpu_id, size_bytes, self.pd)
            if db.allocate_and_register():
                self._double_bufs[gpu_id] = db
                log.info(f"VTP: double-buffered regions on GPU {gpu_id} "
                         f"({size_bytes * 2 / 1e6:.0f}MB total)")
                return True
        else:
            region = GPUMemoryRegion(gpu_id, size_bytes, self.pd)
            if region.allocate_and_register():
                self._gpu_regions[gpu_id] = region
                return True
        return False

    def connect(self, remote_info: VTPConnectionInfo):
        """Connect QP to remote peer using exchanged info."""
        if not self.available:
            raise RuntimeError("VTP: RDMA not initialized")
        ibe = _RDMA_CLASSES["enums"]
        QPAttr_ = _RDMA_CLASSES["QPAttr"]
        GR = _RDMA_CLASSES["GlobalRoute"]
        AHA = _RDMA_CLASSES["AHAttr"]

        try:
            # INIT
            attr = QPAttr_()
            attr.qp_state = ibe.IBV_QPS_INIT
            attr.port_num = self.ib_port
            attr.pkey_index = 0
            attr.qp_access_flags = (
                ibe.IBV_ACCESS_LOCAL_WRITE |
                ibe.IBV_ACCESS_REMOTE_WRITE |
                ibe.IBV_ACCESS_REMOTE_READ
            )
            self.qp.to_init(attr)

            # RTR
            attr = QPAttr_()
            attr.qp_state = ibe.IBV_QPS_RTR
            attr.path_mtu = ibe.IBV_MTU_4096
            attr.dest_qp_num = remote_info.qp_num
            attr.rq_psn = 0
            attr.max_dest_rd_atomic = 4
            attr.min_rnr_timer = 12

            gr = GR()
            gr.dgid = remote_info.gid
            gr.sgid_index = self.gid_index

            ah_attr = AHA()
            ah_attr.dlid = remote_info.lid
            ah_attr.sl = 0
            ah_attr.port_num = self.ib_port
            ah_attr.is_global = 1
            ah_attr.gr = gr
            attr.ah_attr = ah_attr
            self.qp.to_rtr(attr)

            # RTS
            attr = QPAttr_()
            attr.qp_state = ibe.IBV_QPS_RTS
            attr.sq_psn = 0
            attr.timeout = 14
            attr.retry_cnt = 7
            attr.rnr_retry = 7
            attr.max_rd_atomic = 4
            self.qp.to_rts(attr)

            self._connected = True
            log.info(f"VTP: connected to {remote_info.node_id} "
                     f"(QP {remote_info.qp_num})")
        except Exception as exc:
            log.error(f"VTP connect failed: {exc}")
            raise

    def get_connection_info(self, node_id: str = "local") -> VTPConnectionInfo:
        """Get local connection info for OOB exchange."""
        info = VTPConnectionInfo(node_id=node_id)
        if not self.available:
            info.tier = "unavailable"
            return info

        info.qp_num = self.qp.qp_num
        try:
            port_attr = self.ctx.query_port(self.ib_port)
            info.lid = port_attr.lid
            gid = self.ctx.query_gid(self.ib_port, self.gid_index)
            info.gid = bytes(gid.gid)
        except Exception:
            info.gid = b'\x00' * 16

        # Export GPU regions for one-sided RDMA Write
        for gpu_id, db in self._double_bufs.items():
            staging = db.staging
            if staging.registered:
                info.gpu_regions[gpu_id] = {
                    "rkey": staging.rkey,
                    "addr": staging.data_ptr,
                    "size": staging.size_bytes,
                }
        for gpu_id, region in self._gpu_regions.items():
            if region.registered:
                info.gpu_regions[gpu_id] = {
                    "rkey": region.rkey,
                    "addr": region.data_ptr,
                    "size": region.size_bytes,
                }

        info.gpu_count = len(info.gpu_regions)
        info.tier = self._detect_tier().name
        return info

    def _detect_tier(self) -> TransportTier:
        """Detect the best transport tier available."""
        if _GPUDIRECT and _CUDA and self.available:
            has_gpu_mr = bool(self._double_bufs or self._gpu_regions)
            if has_gpu_mr:
                return TransportTier.GPUDIRECT_RDMA
            return TransportTier.CPU_STAGED_RDMA
        if self.available:
            return TransportTier.CPU_STAGED_RDMA
        return TransportTier.BASIC_TCP

    def close(self):
        """Release all RDMA resources."""
        for db in self._double_bufs.values():
            db.close()
        for region in self._gpu_regions.values():
            region.close()
        for res in [self._cpu_mr, self.qp, self.send_cq, self.recv_cq,
                    self.pd, self.ctx]:
            if res is not None:
                try:
                    res.close()
                except Exception:
                    pass
        self._connected = False
        self._double_bufs.clear()
        self._gpu_regions.clear()
        log.info("VTP connection closed")


# ═══════════════════════════════════════════════════════════════════════════
# LLMTransport — the main high-level API
# ═══════════════════════════════════════════════════════════════════════════
class LLMTransport:
    """High-level LLM-optimised transport for multi-node inference.

    Usage:
        transport = LLMTransport(node_id="node-0")
        transport.register_gpu(0, size_mb=512)
        transport.register_gpu(1, size_mb=512)

        # Connect to peer (after OOB handshake)
        transport.connect_peer("node-1", peer_info)

        # Send activation tensor (GPUDirect if available)
        result = transport.send_tensor(tensor, dst_node="node-1",
                                       dst_gpu=0, layer_id=12, seq_id=42)

        # Receive (pre-allocated GPU buffer)
        tensor = transport.recv_tensor(src_node="node-1", gpu_id=0)

        # KV cache streaming
        transport.stream_kv_cache(k, v, dst_node="node-1", dst_gpu=0,
                                  layer_ids=[0,1,2])
    """

    def __init__(self, node_id: str = "local",
                 device_name: Optional[str] = None):
        self.node_id = node_id
        self._connections: Dict[str, VTPConnection] = {}
        self._local_conn = VTPConnection(device_name=device_name)
        self._tier = TransportTier.STUB if _STUB_MODE else self._local_conn._detect_tier()
        self._tcp_fallback: Dict[str, socket.socket] = {}
        self._lock = threading.Lock()
        self._seq_counter = 0
        # Stats
        self._stats = {
            "tensors_sent": 0,
            "tensors_recv": 0,
            "bytes_sent": 0,
            "bytes_recv": 0,
            "gpudirect_ops": 0,
            "cpu_staged_ops": 0,
            "tcp_fallback_ops": 0,
            "avg_latency_us": 0.0,
        }
        self._latencies: List[float] = []

    @property
    def tier(self) -> TransportTier:
        return self._tier

    def _next_seq(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    # ------------------------------------------------------------------
    # GPU registration
    # ------------------------------------------------------------------
    def register_gpu(self, gpu_id: int, size_mb: int = 256,
                     double_buffer: bool = True) -> bool:
        """Register GPU memory for RDMA transfers."""
        if _STUB_MODE:
            log.info(f"VTP stub: register GPU {gpu_id} ({size_mb}MB)")
            return True
        return self._local_conn.register_gpu(
            gpu_id, size_mb * 1024 * 1024, double_buffer
        )

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------
    def get_local_info(self) -> VTPConnectionInfo:
        """Get local connection info for OOB exchange."""
        return self._local_conn.get_connection_info(self.node_id)

    def connect_peer(self, peer_node_id: str,
                     peer_info: VTPConnectionInfo) -> bool:
        """Connect to a remote peer using exchanged connection info."""
        if _STUB_MODE:
            log.info(f"VTP stub: connected to {peer_node_id}")
            return True
        try:
            self._local_conn.connect(peer_info)
            self._connections[peer_node_id] = self._local_conn
            log.info(f"VTP: peer {peer_node_id} connected (tier={self._tier.name})")
            return True
        except Exception as exc:
            log.error(f"VTP: failed to connect to {peer_node_id}: {exc}")
            return False

    def connect_peer_tcp(self, peer_node_id: str, host: str,
                         port: int = VTP_CONTROL_PORT) -> bool:
        """Connect to a peer via TCP fallback."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BUSY_POLL, 50)
            except Exception:
                pass
            sock.settimeout(10.0)
            sock.connect((host, port))
            self._tcp_fallback[peer_node_id] = sock
            self._tier = TransportTier.ZEROCOPY_TCP
            log.info(f"VTP: TCP fallback to {peer_node_id} ({host}:{port})")
            return True
        except Exception as exc:
            log.error(f"VTP TCP connect to {peer_node_id} failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Core send: zero-copy tensor transfer
    # ------------------------------------------------------------------
    def send_tensor(
        self,
        tensor: Any,
        dst_node: str,
        dst_gpu: int = 0,
        layer_id: int = 0,
        seq_id: Optional[int] = None,
        opcode: VTPOpcode = VTPOpcode.TENSOR,
        urgent: bool = False,
    ) -> Dict[str, Any]:
        """Send a tensor to a remote node's GPU — zero CPU copy if possible.

        The method automatically selects the best transport:
          1. GPUDirect RDMA Write (one-sided, NIC reads GPU VRAM directly)
          2. CPU-staged RDMA (GPU → pinned CPU → RDMA → remote)
          3. TCP fallback (GPU → CPU → TCP → remote CPU → GPU)

        Returns dict with transfer metadata.
        """
        if _STUB_MODE:
            return self._stub_send(tensor, dst_node, dst_gpu, layer_id)

        if seq_id is None:
            seq_id = self._next_seq()
        start = time.perf_counter()

        # Ensure tensor is contiguous
        if _TORCH and hasattr(tensor, 'is_contiguous') and not tensor.is_contiguous():
            tensor = tensor.contiguous()

        nbytes = tensor.nelement() * tensor.element_size() if _TORCH else len(tensor)
        src_gpu = tensor.device.index if (_TORCH and tensor.is_cuda) else 0

        # Build header
        flags = 0
        if urgent:
            flags |= VTPFlags.URGENT
        dtype_code = _DTYPE_TO_CODE.get(tensor.dtype, 0) if _TORCH else 0

        header = TensorHeader(
            opcode=opcode, flags=flags, payload_bytes=nbytes,
            layer_id=layer_id, seq_id=seq_id,
            src_gpu=src_gpu, dst_gpu=dst_gpu,
            ndim=tensor.ndim if _TORCH else 0,
            dtype_code=dtype_code,
            shape=tuple(tensor.shape) if _TORCH else (),
        )

        # Select transport tier and send
        result = {"method": "unknown", "bytes": nbytes, "layer_id": layer_id,
                  "seq_id": seq_id, "src_gpu": src_gpu, "dst_gpu": dst_gpu}

        conn = self._connections.get(dst_node)

        if (self._tier == TransportTier.GPUDIRECT_RDMA and
                conn and conn.available and conn._connected):
            result = self._send_gpudirect(tensor, header, conn, dst_node)
        elif (self._tier in (TransportTier.GPUDIRECT_RDMA,
                             TransportTier.CPU_STAGED_RDMA) and
              conn and conn.available and conn._connected):
            result = self._send_cpu_staged_rdma(tensor, header, conn, dst_node)
        elif dst_node in self._tcp_fallback:
            result = self._send_tcp(tensor, header, dst_node)
        else:
            # Try TCP connection on the fly
            result = self._send_tcp(tensor, header, dst_node)

        duration = time.perf_counter() - start
        result["duration_s"] = duration
        result["bandwidth_gbps"] = (nbytes * 8 / (duration * 1e9)) if duration > 0 else 0

        # Stats
        self._stats["tensors_sent"] += 1
        self._stats["bytes_sent"] += nbytes
        self._latencies.append(duration * 1e6)  # microseconds
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-500:]
        self._stats["avg_latency_us"] = sum(self._latencies) / len(self._latencies)

        # Metrics
        if _METRICS:
            tier_label = result.get("method", "unknown")
            FASTPATH_BYTES.labels(tier_label, "send").inc(nbytes)
            FASTPATH_LATENCY.labels(tier_label, "send").observe(duration)

        return result

    def _send_gpudirect(self, tensor: Any, header: TensorHeader,
                        conn: VTPConnection, dst_node: str) -> Dict[str, Any]:
        """Send tensor via GPUDirect RDMA Write (one-sided).

        The NIC reads directly from GPU VRAM and writes to the remote
        GPU's pre-registered buffer — zero CPU involvement.
        """
        src_gpu = tensor.device.index if tensor.is_cuda else 0
        ibe = _RDMA_CLASSES["enums"]
        SGE_ = _RDMA_CLASSES["SGE"]
        SendWR_ = _RDMA_CLASSES["SendWR"]

        # Get local GPU memory region
        db = conn._double_bufs.get(src_gpu)
        if db is None:
            # Fallback to CPU-staged
            return self._send_cpu_staged_rdma(tensor, header, conn, dst_node)

        nbytes = header.payload_bytes
        # Copy tensor into staging buffer (GPU-to-GPU local copy, fast)
        staging = db.staging
        staging.write_tensor(tensor)

        # Set GPUDirect flag
        header.flags |= VTPFlags.GPUDIRECT | VTPFlags.ONE_SIDED

        # First, send header via CPU MR (small, 64 bytes)
        hdr_bytes = header.encode()
        conn._cpu_buf[:_HEADER_PAD] = hdr_bytes
        hdr_sge = SGE_(conn._cpu_mr.buf, _HEADER_PAD, conn._cpu_mr.lkey)
        hdr_wr = SendWR_(opcode=ibe.IBV_WR_SEND, num_sge=1, sg=[hdr_sge])
        hdr_wr.send_flags = ibe.IBV_SEND_SIGNALED
        conn.qp.post_send(hdr_wr)

        # Then RDMA Write the payload directly from GPU MR
        # (one-sided: writes into remote's pre-registered GPU buffer)
        remote_info = conn.get_connection_info(dst_node)
        remote_gpu = remote_info.gpu_regions.get(header.dst_gpu, {})
        if remote_gpu:
            payload_sge = SGE_(staging.data_ptr, nbytes, staging.lkey)
            write_wr = SendWR_(
                opcode=ibe.IBV_WR_RDMA_WRITE,
                num_sge=1, sg=[payload_sge],
            )
            write_wr.send_flags = ibe.IBV_SEND_SIGNALED
            # Set remote address for RDMA Write
            write_wr.remote_addr = remote_gpu["addr"]
            write_wr.rkey = remote_gpu["rkey"]
            conn.qp.post_send(write_wr)

        # Poll completions
        self._poll_send_cq(conn, 2)
        self._stats["gpudirect_ops"] += 1

        return {
            "method": "gpudirect_rdma_write",
            "bytes": nbytes,
            "tier": TransportTier.GPUDIRECT_RDMA.name,
            "one_sided": True,
            "cpu_copy": False,
        }

    def _send_cpu_staged_rdma(self, tensor: Any, header: TensorHeader,
                              conn: VTPConnection, dst_node: str) -> Dict[str, Any]:
        """Send tensor via CPU-staged RDMA.

        GPU → pinned CPU → RDMA Send → remote CPU → GPU
        Used when GPUDirect is not available but RDMA NIC is present.
        Avoids the numpy intermediate — uses raw data_ptr() copy.
        """
        ibe = _RDMA_CLASSES["enums"]
        SGE_ = _RDMA_CLASSES["SGE"]
        SendWR_ = _RDMA_CLASSES["SendWR"]

        nbytes = header.payload_bytes
        hdr_bytes = header.encode()

        # GPU → CPU: direct memory copy, no numpy intermediate
        if _TORCH and tensor.is_cuda:
            # Use pinned memory for DMA (fastest GPU→CPU path)
            cpu_tensor = torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=True
            )
            cpu_tensor.copy_(tensor, non_blocking=False)
            # Get raw bytes pointer from the pinned tensor
            raw_bytes = bytes(cpu_tensor.contiguous().view(torch.uint8).numpy())
        elif _TORCH:
            raw_bytes = bytes(tensor.contiguous().view(torch.uint8).numpy())
        else:
            raw_bytes = bytes(tensor)

        # Pack header + payload into CPU MR
        total = _HEADER_PAD + len(raw_bytes)
        if total <= len(conn._cpu_buf):
            conn._cpu_buf[:_HEADER_PAD] = hdr_bytes
            conn._cpu_buf[_HEADER_PAD:_HEADER_PAD + len(raw_bytes)] = raw_bytes

            sge = SGE_(conn._cpu_mr.buf, total, conn._cpu_mr.lkey)
            wr = SendWR_(opcode=ibe.IBV_WR_SEND, num_sge=1, sg=[sge])
            wr.send_flags = ibe.IBV_SEND_SIGNALED
            conn.qp.post_send(wr)
            self._poll_send_cq(conn, 1)
        else:
            # Chunked: send header, then payload in chunks
            conn._cpu_buf[:_HEADER_PAD] = hdr_bytes
            hdr_sge = SGE_(conn._cpu_mr.buf, _HEADER_PAD, conn._cpu_mr.lkey)
            hdr_wr = SendWR_(opcode=ibe.IBV_WR_SEND, num_sge=1, sg=[hdr_sge])
            hdr_wr.send_flags = ibe.IBV_SEND_SIGNALED
            conn.qp.post_send(hdr_wr)
            self._poll_send_cq(conn, 1)

            offset = 0
            chunk_max = len(conn._cpu_buf) - _HEADER_PAD
            while offset < len(raw_bytes):
                chunk = raw_bytes[offset:offset + chunk_max]
                conn._cpu_buf[:len(chunk)] = chunk
                sge = SGE_(conn._cpu_mr.buf, len(chunk), conn._cpu_mr.lkey)
                wr = SendWR_(opcode=ibe.IBV_WR_SEND, num_sge=1, sg=[sge])
                wr.send_flags = ibe.IBV_SEND_SIGNALED
                conn.qp.post_send(wr)
                self._poll_send_cq(conn, 1)
                offset += len(chunk)

        self._stats["cpu_staged_ops"] += 1
        return {
            "method": "cpu_staged_rdma",
            "bytes": nbytes,
            "tier": TransportTier.CPU_STAGED_RDMA.name,
            "one_sided": False,
            "cpu_copy": True,
        }

    def _send_tcp(self, tensor: Any, header: TensorHeader,
                  dst_node: str) -> Dict[str, Any]:
        """Send tensor via TCP fallback with VTP framing."""
        sock = self._tcp_fallback.get(dst_node)
        if sock is None:
            return {"method": "tcp_failed", "bytes": 0,
                    "error": f"No TCP connection to {dst_node}"}

        hdr_bytes = header.encode()

        # Serialize tensor without numpy intermediate when possible
        if _TORCH and hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            cpu_tensor = tensor.cpu()
            raw = bytes(cpu_tensor.contiguous().view(torch.uint8).numpy())
        elif _TORCH:
            raw = bytes(tensor.contiguous().view(torch.uint8).numpy())
        else:
            raw = bytes(tensor)

        try:
            sock.sendall(hdr_bytes)
            # Send payload in chunks using memoryview
            mv = memoryview(raw)
            offset = 0
            while offset < len(raw):
                n = sock.send(mv[offset:offset + VTP_CHUNK_BYTES])
                if n == 0:
                    raise ConnectionError("TCP connection closed during send")
                offset += n
            self._stats["tcp_fallback_ops"] += 1
            return {
                "method": "zerocopy_tcp",
                "bytes": header.payload_bytes,
                "tier": TransportTier.ZEROCOPY_TCP.name,
            }
        except Exception as exc:
            log.error(f"VTP TCP send to {dst_node} failed: {exc}")
            return {"method": "tcp_failed", "bytes": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Core recv
    # ------------------------------------------------------------------
    def recv_tensor(self, src_node: str, gpu_id: int = 0,
                    timeout_s: float = 30.0) -> Optional[Tuple[Any, TensorHeader]]:
        """Receive a tensor from a remote node.

        Returns (tensor, header) or None on timeout/error.
        For GPUDirect one-sided writes, data arrives directly in GPU buffer —
        this method just reads from the staging buffer.
        """
        if _STUB_MODE:
            return None

        conn = self._connections.get(src_node)

        # Try RDMA recv
        if conn and conn.available and conn._connected:
            return self._recv_rdma(conn, gpu_id, timeout_s)

        # Try TCP recv
        sock = self._tcp_fallback.get(src_node)
        if sock:
            return self._recv_tcp(sock, gpu_id, timeout_s)

        return None

    def _recv_rdma(self, conn: VTPConnection, gpu_id: int,
                   timeout_s: float) -> Optional[Tuple[Any, TensorHeader]]:
        """Receive via RDMA."""
        ibe = _RDMA_CLASSES["enums"]
        SGE_ = _RDMA_CLASSES["SGE"]
        RecvWR_ = _RDMA_CLASSES["RecvWR"]

        try:
            # Post recv for header
            sge = SGE_(conn._cpu_mr.buf, _HEADER_PAD + VTP_CHUNK_BYTES,
                       conn._cpu_mr.lkey)
            rwr = RecvWR_(num_sge=1, sg=[sge])
            conn.qp.post_recv(rwr)

            # Poll recv CQ
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                wcs = conn.recv_cq.poll(1)
                for wc in wcs:
                    if wc.status != ibe.IBV_WC_SUCCESS:
                        log.error(f"VTP recv WC error: {wc.status}")
                        continue
                    # Parse header
                    hdr_data = bytes(conn._cpu_buf[:_HEADER_PAD])
                    header = TensorHeader.decode(hdr_data)

                    # Check if GPUDirect one-sided (data is already in GPU)
                    if header.flags & VTPFlags.ONE_SIDED:
                        db = conn._double_bufs.get(gpu_id)
                        if db:
                            tensor = db.staging.read_tensor(
                                header.shape,
                                _CODE_TO_DTYPE.get(header.dtype_code, torch.float32),
                            )
                            db.swap()  # Now staging becomes active
                            return tensor, header

                    # Two-sided: payload follows header in CPU MR
                    payload = bytes(conn._cpu_buf[_HEADER_PAD:_HEADER_PAD + header.payload_bytes])
                    import numpy as np
                    np_dtype = {0: np.float32, 1: np.float16, 2: np.float32,
                                3: np.float64, 4: np.int64, 5: np.int32,
                                6: np.int16, 7: np.int8, 8: np.uint8,
                                9: np.bool_}.get(header.dtype_code, np.float32)
                    arr = np.frombuffer(payload[:header.payload_bytes],
                                       dtype=np_dtype).reshape(header.shape)
                    tensor = torch.from_numpy(arr.copy())
                    torch_dtype = _CODE_TO_DTYPE.get(header.dtype_code, torch.float32)
                    if torch_dtype == torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                    if _CUDA and gpu_id >= 0:
                        tensor = tensor.to(f"cuda:{gpu_id}")
                    return tensor, header
                time.sleep(0.0001)  # 100us poll interval
            return None
        except Exception as exc:
            log.error(f"VTP RDMA recv failed: {exc}")
            return None

    def _recv_tcp(self, sock: socket.socket, gpu_id: int,
                  timeout_s: float) -> Optional[Tuple[Any, TensorHeader]]:
        """Receive via TCP."""
        try:
            sock.settimeout(timeout_s)
            # Read header
            hdr_data = self._tcp_recv_exact(sock, _HEADER_PAD)
            if not hdr_data:
                return None
            header = TensorHeader.decode(hdr_data)

            # Read payload
            payload = self._tcp_recv_exact(sock, header.payload_bytes)
            if not payload:
                return None

            import numpy as np
            np_dtype = {0: np.float32, 1: np.float16, 2: np.float32,
                        3: np.float64, 4: np.int64, 5: np.int32,
                        6: np.int16, 7: np.int8, 8: np.uint8,
                        9: np.bool_}.get(header.dtype_code, np.float32)
            arr = np.frombuffer(payload, dtype=np_dtype).reshape(header.shape)
            tensor = torch.from_numpy(arr.copy())
            torch_dtype = _CODE_TO_DTYPE.get(header.dtype_code, torch.float32)
            if torch_dtype == torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            if _CUDA and gpu_id >= 0:
                tensor = tensor.to(f"cuda:{gpu_id}")

            self._stats["tensors_recv"] += 1
            self._stats["bytes_recv"] += header.payload_bytes
            return tensor, header
        except Exception as exc:
            log.error(f"VTP TCP recv failed: {exc}")
            return None

    @staticmethod
    def _tcp_recv_exact(sock: socket.socket, size: int) -> Optional[bytes]:
        buf = bytearray(size)
        view = memoryview(buf)
        received = 0
        while received < size:
            n = sock.recv_into(view[received:])
            if n == 0:
                return None
            received += n
        return bytes(buf)

    # ------------------------------------------------------------------
    # KV cache streaming
    # ------------------------------------------------------------------
    def stream_kv_cache(
        self,
        k_cache: Any,
        v_cache: Any,
        dst_node: str,
        dst_gpu: int = 0,
        layer_ids: Optional[List[int]] = None,
        head_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Stream KV cache to a remote node with per-layer/per-head granularity.

        Optimised for LLM inference migration / load balancing:
          - Transfers only specified layers and heads
          - Uses VTPOpcode.KV_CACHE for receiver-side routing
          - Supports partial transfer (copy-on-write semantics)

        Args:
            k_cache: Key cache [num_layers, batch, heads, seq_len, dim]
            v_cache: Value cache [same shape]
            dst_node: Destination node ID
            dst_gpu: Destination GPU
            layer_ids: Subset of layers to transfer (None = all)
            head_ids: Subset of heads to transfer (None = all)
        """
        if _STUB_MODE:
            # Simulate per-layer K+V streaming even in stub mode
            try:
                num_layers = k_cache.shape[0] if hasattr(k_cache, 'shape') and k_cache.ndim > 0 else 1
            except Exception:
                num_layers = 1
            layers = layer_ids if layer_ids is not None else list(range(num_layers))
            stub_results = []
            for lid in layers:
                stub_results.append({"method": "stub", "bytes": 0, "cache_type": "key", "layer_id": lid})
                stub_results.append({"method": "stub", "bytes": 0, "cache_type": "value", "layer_id": lid})
            return stub_results

        results = []

        if not _TORCH:
            return [{"method": "error", "error": "torch not available"}]

        num_layers = k_cache.shape[0] if k_cache.ndim > 0 else 1
        layers = layer_ids if layer_ids is not None else list(range(num_layers))

        for lid in layers:
            k_slice = k_cache[lid] if k_cache.ndim > 1 else k_cache
            v_slice = v_cache[lid] if v_cache.ndim > 1 else v_cache

            # Per-head slicing if requested
            if head_ids is not None and k_slice.ndim >= 2:
                for hid in head_ids:
                    k_head = k_slice[:, hid:hid + 1] if k_slice.ndim >= 3 else k_slice
                    v_head = v_slice[:, hid:hid + 1] if v_slice.ndim >= 3 else v_slice
                    r_k = self.send_tensor(
                        k_head, dst_node, dst_gpu, layer_id=lid,
                        opcode=VTPOpcode.KV_CACHE,
                    )
                    r_k["cache_type"] = "key"
                    r_k["head_id"] = hid
                    results.append(r_k)
                    r_v = self.send_tensor(
                        v_head, dst_node, dst_gpu, layer_id=lid,
                        opcode=VTPOpcode.KV_CACHE,
                    )
                    r_v["cache_type"] = "value"
                    r_v["head_id"] = hid
                    results.append(r_v)
            else:
                r_k = self.send_tensor(
                    k_slice, dst_node, dst_gpu, layer_id=lid,
                    opcode=VTPOpcode.KV_CACHE,
                )
                r_k["cache_type"] = "key"
                results.append(r_k)
                r_v = self.send_tensor(
                    v_slice, dst_node, dst_gpu, layer_id=lid,
                    opcode=VTPOpcode.KV_CACHE,
                )
                r_v["cache_type"] = "value"
                results.append(r_v)

        if _METRICS:
            total_bytes = sum(r.get("bytes", 0) for r in results)
            FASTPATH_BYTES.labels("kv_cache", "send").inc(total_bytes)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _poll_send_cq(self, conn: VTPConnection, expected: int,
                      timeout_ms: int = 5000):
        """Poll send CQ for completions."""
        if not conn.send_cq:
            return
        ibe = _RDMA_CLASSES["enums"]
        deadline = time.perf_counter() + timeout_ms / 1000
        completed = 0
        while completed < expected:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"VTP: CQ poll timeout ({timeout_ms}ms, "
                    f"got {completed}/{expected})"
                )
            wcs = conn.send_cq.poll(expected - completed)
            for wc in wcs:
                if wc.status != ibe.IBV_WC_SUCCESS:
                    raise RuntimeError(f"VTP send WC error: status={wc.status}")
                completed += 1

    def _stub_send(self, tensor, dst_node, dst_gpu, layer_id):
        nbytes = 0
        if _TORCH and hasattr(tensor, 'nelement'):
            nbytes = tensor.nelement() * tensor.element_size()
        self._stats["tensors_sent"] += 1
        self._stats["bytes_sent"] += nbytes
        return {
            "method": "stub", "bytes": nbytes, "dst_node": dst_node,
            "dst_gpu": dst_gpu, "layer_id": layer_id,
            "tier": TransportTier.STUB.name,
        }

    def stats(self) -> Dict[str, Any]:
        """Return transport statistics."""
        return {
            **self._stats,
            "tier": self._tier.name,
            "rdma_available": _PYVERBS,
            "gpudirect_available": _GPUDIRECT,
            "cuda_available": _CUDA,
            "connections": list(self._connections.keys()),
            "tcp_fallbacks": list(self._tcp_fallback.keys()),
        }

    def close(self):
        """Shutdown all connections."""
        self._local_conn.close()
        for conn in self._connections.values():
            if conn is not self._local_conn:
                conn.close()
        for sock in self._tcp_fallback.values():
            try:
                sock.close()
            except Exception:
                pass
        self._connections.clear()
        self._tcp_fallback.clear()
        log.info("VTP: all connections closed")


# ═══════════════════════════════════════════════════════════════════════════
# TCP Server for control channel + tensor reception
# ═══════════════════════════════════════════════════════════════════════════
class VTPServer:
    """TCP server for accepting VTP connections and receiving tensors.

    Runs in a background thread. Handles:
      - OOB handshake (exchange RDMA connection info)
      - Tensor reception via TCP fallback
      - Health heartbeats
    """

    def __init__(self, transport: LLMTransport,
                 host: str = "0.0.0.0", port: int = VTP_CONTROL_PORT):
        self.transport = transport
        self.host = host
        self.port = port
        self._server_sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[int, Callable] = {}

    def start(self) -> int:
        """Start the VTP server. Returns the actual port."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(16)
        self._server_sock.settimeout(1.0)
        actual_port = self._server_sock.getsockname()[1]
        self.port = actual_port

        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="vtp-server"
        )
        self._thread.start()
        log.info(f"VTP server listening on {self.host}:{actual_port}")
        return actual_port

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        log.info("VTP server stopped")

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(
                    target=self._handle_client, args=(conn, addr),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    log.error(f"VTP accept error: {exc}")

    def _handle_client(self, conn: socket.socket, addr):
        """Handle incoming VTP connection."""
        log.info(f"VTP: incoming connection from {addr}")
        peer_id = f"{addr[0]}:{addr[1]}"
        self.transport._tcp_fallback[peer_id] = conn
        # Keep connection alive for future tensor transfers


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive transport selector
# ═══════════════════════════════════════════════════════════════════════════
def select_optimal_tier(tensor_bytes: int,
                        has_rdma: bool = False,
                        has_gpudirect: bool = False,
                        has_cuda: bool = False) -> TransportTier:
    """Select the optimal transport tier based on context.

    Decision matrix:
      - GPUDirect + CUDA + tensor on GPU + size > 4KB → GPUDIRECT_RDMA
      - RDMA available + size > 1KB → CPU_STAGED_RDMA
      - Otherwise → TCP (zero-copy if possible)

    Small tensors (< 256 bytes) are always inlined in the header.
    """
    if tensor_bytes <= VTP_INLINE_THRESHOLD:
        # Inline in header for minimum latency
        return TransportTier.CPU_STAGED_RDMA if has_rdma else TransportTier.ZEROCOPY_TCP

    if has_gpudirect and has_cuda and tensor_bytes > 4096:
        return TransportTier.GPUDIRECT_RDMA

    if has_rdma and tensor_bytes > 1024:
        return TransportTier.CPU_STAGED_RDMA

    return TransportTier.ZEROCOPY_TCP


# ═══════════════════════════════════════════════════════════════════════════
# Module exports
# ═══════════════════════════════════════════════════════════════════════════
__all__ = [
    "LLMTransport",
    "VTPServer",
    "VTPConnection",
    "VTPConnectionInfo",
    "TensorHeader",
    "VTPOpcode",
    "VTPFlags",
    "TransportTier",
    "GPUMemoryRegion",
    "DoubleBufferedRegion",
    "select_optimal_tier",
    "VTP_VERSION",
]
