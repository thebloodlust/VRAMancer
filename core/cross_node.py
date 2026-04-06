"""
Cross-node distributed inference via VTP (binary TCP) or HTTP layer relay.

Enables pipeline-parallel inference across multiple VRAMancer nodes
on a LAN.  Each node executes a range of transformer layers and relays
hidden states to the next node.

Transport modes:
  1. VTP-lite (default): Persistent TCP + raw binary tensor framing.
     ~10x less overhead than HTTP.  Port 18951.
  2. HTTP fallback: requests-based, via Flask API on port 5030.

Master:  POST /api/distributed/generate  (orchestrates the generation)
Worker:  VTP server on port 18951  (or HTTP /api/worker/forward_layers)
"""

import io
import os
import time
import struct
import socket
import threading
from typing import List, Tuple, Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from core.logger import get_logger

logger = get_logger("cross_node")

# Partially loaded model for worker role
_partial_model = None

# ─── VTP-lite constants ───────────────────────────────────────────
VTP_PORT = int(os.environ.get("VRM_VTP_WORKER_PORT", "18951"))
VTP_MAGIC = b"VTP1"

# Dtype encoding for raw tensor transport (compact, no pickle)
_DTYPE_TO_CODE = {}
_CODE_TO_DTYPE = {}
if _HAS_TORCH:
    _DTYPE_TO_CODE = {
        torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
        torch.float64: 3, torch.int32: 4, torch.int64: 5,
        torch.int8: 6, torch.uint8: 7, torch.int16: 8,
    }
    _CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


# ─── Tensor serialisation (torch.save/load, all dtypes) ──────────
# Used by HTTP fallback path

def tensor_to_bytes(t: "torch.Tensor") -> bytes:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return buf.getvalue()


def bytes_to_tensor(data: bytes, device: str = "cpu") -> "torch.Tensor":
    buf = io.BytesIO(data)
    return torch.load(buf, weights_only=True, map_location="cpu").to(device)


# ─── Fast raw tensor serialisation (VTP path, no pickle) ─────────

def tensor_to_raw(t: "torch.Tensor") -> Tuple[bytes, int, Tuple[int, ...], int]:
    """Serialize tensor to raw bytes + metadata. ~10x faster than torch.save."""
    t = t.detach().contiguous().cpu()
    dtype_code = _DTYPE_TO_CODE.get(t.dtype, 0)
    shape = tuple(t.shape)
    if t.dtype == torch.bfloat16:
        # numpy has no bfloat16 — view as uint16 for raw bytes
        raw = t.view(torch.uint16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    return raw, dtype_code, shape, len(raw)


def raw_to_tensor(data: bytes, dtype_code: int, shape: Tuple[int, ...],
                  device: str = "cpu") -> "torch.Tensor":
    """Deserialize raw bytes to tensor. ~10x faster than torch.load."""
    import numpy as np
    dtype = _CODE_TO_DTYPE.get(dtype_code, torch.float32)
    # numpy dtype mapping
    np_dtype_map = {
        torch.float32: np.float32, torch.float16: np.float16,
        torch.bfloat16: np.float32,  # bfloat16 has no numpy equivalent
        torch.float64: np.float64, torch.int32: np.int32,
        torch.int64: np.int64, torch.int8: np.int8, torch.uint8: np.uint8,
        torch.int16: np.int16,
    }
    np_dt = np_dtype_map.get(dtype, np.float32)

    if dtype == torch.bfloat16:
        # bfloat16: deserialize as uint16, view as bfloat16
        arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    else:
        arr = np.frombuffer(data, dtype=np_dt).reshape(shape)
        t = torch.from_numpy(arr.copy())
    return t.to(device)


# ─── VTP socket helpers ──────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1048576))  # 1MB chunks
        if not chunk:
            raise ConnectionError("Connection closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_all(sock: socket.socket, data: bytes):
    """Send all bytes, handling partial sends."""
    sock.sendall(data)


# ─── VTP Worker Server (runs on worker nodes) ────────────────────

def _make_forward_callback():
    """Create a GPU-pointer forward callback for RustVTPServer.

    Called from Rust with GIL acquired. Input data is already on GPU
    (Rust did TCP→pinned→HtoD). Returns GPU pointer of output tensor
    so Rust can DtoH→TCP without copying bytes through Python.
    """
    # Pre-allocate GPU input buffer — Rust writes incoming data here via HtoD
    _gpu_in_buf = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
    _gpu_in_ptr = _gpu_in_buf.data_ptr()
    # Keep a reference to prevent GC
    _last_result = [None]

    def _forward_gpu(n_bytes, dtype_code, shape, start_layer, end_layer, seq_len):
        model = _partial_model
        if model is None:
            try:
                from core.production_api import _registry
                if _registry.is_loaded:
                    model = _registry._pipeline.backend.model
            except Exception:
                pass
        if model is None:
            # Return a zero-length tensor on GPU
            dummy = torch.empty(0, device='cuda:0')
            return (dummy.data_ptr(), 0, 0, [0])

        # View the GPU input buffer as the correct dtype/shape (zero-copy)
        dtype = _CODE_TO_DTYPE.get(int(dtype_code), torch.float32)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = int(n_bytes) // elem_size
        hidden = _gpu_in_buf[:int(n_bytes)].view(dtype)[:numel].reshape(
            [int(s) for s in shape])

        result = _worker_forward_tensor(
            model, hidden, int(start_layer), int(end_layer), int(seq_len))

        # Ensure CUDA kernels complete before Rust reads the output
        torch.cuda.synchronize()

        # Keep result alive until next call (Rust does DtoH inside GIL,
        # but just in case)
        _last_result[0] = result
        out_bytes = result.nelement() * result.element_size()
        out_dtype = _DTYPE_TO_CODE.get(result.dtype, 0)
        return (result.data_ptr(), out_bytes, out_dtype,
                [int(s) for s in result.shape])

    return _forward_gpu, _gpu_in_ptr


class VTPWorkerServer:
    """Binary TCP server for fast tensor forward on worker nodes.

    Protocol per request:
      Request:  VTP1 | start_layer(H) | end_layer(H) | seq_len(I) |
                ndim(B) | dtype_code(B) | shape(I*ndim) | payload_len(I) | raw_bytes
      Response: VTP1 | ndim(B) | dtype_code(B) | shape(I*ndim) | payload_len(I) | raw_bytes
    """

    def __init__(self, host: str = "0.0.0.0", port: int = VTP_PORT):
        self.host = host
        self.port = port
        self._sock = None
        self._running = False
        self._thread = None
        self._rust_server = None  # RustVTPServer (optional)

    def start(self):
        # ── Try Rust VTP server first (GIL-free network I/O) ──────
        if _HAS_TORCH:
            try:
                import vramancer_rust
                if hasattr(vramancer_rust, "RustVTPServer"):
                    forward_fn, gpu_in_ptr = _make_forward_callback()
                    gpu_id = torch.cuda.current_device()
                    self._rust_server = vramancer_rust.RustVTPServer(
                        gpu_id=gpu_id, buf_size_mb=64, gpu_in_ptr=gpu_in_ptr)
                    self._rust_server.start(self.host, self.port, forward_fn)
                    self._forward_callback = forward_fn  # prevent GC
                    self._running = True
                    logger.info("VTP Rust server listening on %s:%d (GPU %d, zero-copy)",
                                self.host, self.port, gpu_id)
                    return
            except Exception as exc:
                logger.warning("VTP: Rust server unavailable, Python fallback: %s", exc)
                self._rust_server = None

        # ── Python fallback ───────────────────────────────────────
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop,
                                        daemon=True, name="vtp-worker")
        self._thread.start()
        logger.info("VTP worker server listening on %s:%d", self.host, self.port)

    def stop(self):
        self._running = False
        if self._rust_server is not None:
            try:
                self._rust_server.stop()
            except Exception:
                pass
            self._rust_server = None
        if self._thread:
            self._thread.join(timeout=5)
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        logger.info("VTP worker server stopped")

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Large buffers for tensor data
                try:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                    4 * 1024 * 1024)
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                                    4 * 1024 * 1024)
                except Exception:
                    pass
                threading.Thread(target=self._handle_conn,
                                 args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    logger.error("VTP accept error: %s", exc)

    def _handle_conn(self, conn: socket.socket, addr):
        """Handle persistent connection — multiple forward requests."""
        logger.info("VTP: connection from %s", addr)
        try:
            while self._running:
                # Read magic
                try:
                    magic = _recv_exact(conn, 4)
                except ConnectionError:
                    break
                if magic != VTP_MAGIC:
                    logger.warning("VTP: bad magic from %s", addr)
                    break

                # Read header: start(H) end(H) seq_len(I) ndim(B) dtype(B)
                hdr = _recv_exact(conn, 10)
                start_layer, end_layer, seq_len, ndim, dtype_code = \
                    struct.unpack("!HHIBB", hdr)

                # Read shape
                shape_data = _recv_exact(conn, ndim * 4)
                shape = struct.unpack(f"!{ndim}I", shape_data)

                # Read payload
                payload_len_data = _recv_exact(conn, 4)
                payload_len = struct.unpack("!I", payload_len_data)[0]
                payload = _recv_exact(conn, payload_len)

                # Reconstruct tensor and run forward
                model = _partial_model
                if model is None:
                    # Try registry
                    try:
                        from core.production_api import _registry
                        if _registry.is_loaded:
                            model = _registry._pipeline.backend.model
                    except Exception:
                        pass
                if model is None:
                    logger.error("VTP: no model loaded")
                    # Send empty response
                    _send_all(conn, VTP_MAGIC + struct.pack("!BB", 0, 0)
                              + struct.pack("!I", 0))
                    continue

                hidden = raw_to_tensor(payload, dtype_code, shape)
                result_tensor = _worker_forward_tensor(
                    model, hidden, start_layer, end_layer, seq_len)

                # Encode response
                out_raw, out_dtype, out_shape, out_len = \
                    tensor_to_raw(result_tensor)
                out_ndim = len(out_shape)

                resp = VTP_MAGIC
                resp += struct.pack("!BB", out_ndim, out_dtype)
                resp += struct.pack(f"!{out_ndim}I", *out_shape)
                resp += struct.pack("!I", out_len)
                _send_all(conn, resp)
                _send_all(conn, out_raw)

        except Exception as exc:
            logger.error("VTP: connection error from %s: %s", addr, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass
            logger.info("VTP: connection closed from %s", addr)


# Global VTP server instance
_vtp_server = None


def start_vtp_server(port: int = VTP_PORT):
    """Start the VTP worker server (called from production_api)."""
    global _vtp_server
    if _vtp_server is not None:
        return
    _vtp_server = VTPWorkerServer(port=port)
    _vtp_server.start()


def stop_vtp_server():
    """Stop the VTP worker server."""
    global _vtp_server
    if _vtp_server is not None:
        _vtp_server.stop()
        _vtp_server = None


# ─── VTP Remote Worker (master-side, persistent TCP) ─────────────

class VTPRemoteWorker:
    """Persistent TCP connection to a remote VTP worker.

    ~10x less overhead than HTTP RemoteWorker:
    - No HTTP headers/parsing per request
    - No torch.save/load pickle overhead (raw bytes)
    - TCP_NODELAY (no Nagle buffering)
    - Connection reuse across all tokens

    When vramancer_rust.GpuNetBridge is available, the entire
    GPU→pinned→TCP→pinned→GPU data path runs in Rust with GIL released,
    eliminating all Python/numpy overhead from the hot loop.
    """

    def __init__(self, host: str, port: int, start_layer: int, end_layer: int):
        self.host = host
        self.port = port
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._sock = None
        self._bridge = None  # Rust GpuNetBridge (optional)

        # Try to initialize the Rust GPU→Network bridge
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                import vramancer_rust
                if hasattr(vramancer_rust, "GpuNetBridge"):
                    gpu_id = torch.cuda.current_device()
                    self._bridge = vramancer_rust.GpuNetBridge(
                        gpu_id=gpu_id, buf_size_mb=64)
                    self._bridge.connect(host, port)
                    logger.info("VTP: Rust GpuNetBridge active → %s:%d (GPU %d)",
                                host, port, gpu_id)
            except Exception as exc:
                logger.warning("VTP: Rust bridge unavailable, Python fallback: %s", exc)
                self._bridge = None

    def _ensure_connected(self):
        if self._sock is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                  4 * 1024 * 1024)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF,
                                  4 * 1024 * 1024)
        except Exception:
            pass
        self._sock.settimeout(120)
        self._sock.connect((self.host, self.port))
        logger.info("VTP: connected to %s:%d", self.host, self.port)

    def forward(self, hidden_states: "torch.Tensor",
                seq_len: int = 0) -> "torch.Tensor":
        """Send tensor, receive processed tensor back via VTP.

        Prefers the Rust GpuNetBridge path (GPU-direct DMA, GIL released)
        when available, with automatic fallback to the Python path.
        """
        # ── Rust bridge fast path ─────────────────────────────────
        if self._bridge is not None:
            try:
                return self._forward_bridge(hidden_states, seq_len)
            except Exception as exc:
                logger.warning("VTP: Rust bridge error, falling back to Python: %s", exc)
                self._bridge = None

        # ── Python fallback path ──────────────────────────────────
        self._ensure_connected()

        raw, dtype_code, shape, payload_len = tensor_to_raw(hidden_states)
        ndim = len(shape)

        # Build and send request
        req = VTP_MAGIC
        req += struct.pack("!HHIBB", self.start_layer, self.end_layer,
                           seq_len, ndim, dtype_code)
        req += struct.pack(f"!{ndim}I", *shape)
        req += struct.pack("!I", payload_len)
        _send_all(self._sock, req)
        _send_all(self._sock, raw)

        # Read response
        resp_magic = _recv_exact(self._sock, 4)
        if resp_magic != VTP_MAGIC:
            raise ConnectionError(f"Bad VTP response magic: {resp_magic}")

        resp_hdr = _recv_exact(self._sock, 2)
        out_ndim, out_dtype = struct.unpack("!BB", resp_hdr)

        out_shape_data = _recv_exact(self._sock, out_ndim * 4)
        out_shape = struct.unpack(f"!{out_ndim}I", out_shape_data)

        out_len_data = _recv_exact(self._sock, 4)
        out_len = struct.unpack("!I", out_len_data)[0]

        out_raw = _recv_exact(self._sock, out_len)
        return raw_to_tensor(out_raw, out_dtype, out_shape)

    def _forward_bridge(self, hidden_states: "torch.Tensor",
                        seq_len: int) -> "torch.Tensor":
        """Forward via Rust GpuNetBridge — GPU-direct, GIL released."""
        h = hidden_states.contiguous()
        if not h.is_cuda:
            h = h.cuda()

        dtype_code = _DTYPE_TO_CODE.get(h.dtype, 0)
        shape = list(h.shape)
        in_bytes = h.nelement() * h.element_size()
        in_ptr = h.data_ptr()

        # Allocate output buffer on same device (overwritten by bridge HtoD)
        out_buf = torch.empty_like(h)
        out_ptr = out_buf.data_ptr()

        # Entire GPU→pinned→TCP→pinned→GPU in Rust, GIL released
        out_dtype, out_shape, out_bytes = self._bridge.forward(
            in_ptr, in_bytes, out_ptr,
            dtype_code, shape,
            self.start_layer, self.end_layer, seq_len)

        # Reshape output if shape changed (unlikely for hidden states)
        out_torch_dtype = _CODE_TO_DTYPE.get(out_dtype, h.dtype)
        if tuple(out_shape) != tuple(shape) or out_torch_dtype != h.dtype:
            out_buf = out_buf.view(-1)[:out_bytes // h.element_size()]
            out_buf = out_buf.view(out_torch_dtype).reshape(out_shape)

        return out_buf

    def close(self):
        if self._bridge is not None:
            try:
                self._bridge.close()
            except Exception:
                pass
            self._bridge = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ─── Model architecture helpers ──────────────────────────────────

def get_model_layers(model):
    """Return the nn.ModuleList of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # Llama / Qwen / Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # GPT-2
    raise ValueError(f"Unknown architecture: {type(model)}")


def _is_gpt2(model) -> bool:
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")


def get_model_info(model) -> dict:
    layers = get_model_layers(model)
    return {
        "num_layers": len(layers),
        "arch": "gpt2" if _is_gpt2(model) else "llama",
        "hidden_size": getattr(model.config, "hidden_size", 0),
    }


# ─── Partial model loading (worker role) ─────────────────────────

def load_partial_model(model_name: str, start_layer: int, end_layer: int,
                       device: str = "cuda:0",
                       dtype_str: str = "bfloat16") -> dict:
    """Load model with only [start_layer, end_layer) on GPU, rest on CPU.

    Memory-efficient: unused layers stay on CPU with disk offload.
    Only the specified layers consume GPU VRAM.
    """
    global _partial_model
    from transformers import AutoModelForCausalLM, AutoConfig
    import tempfile

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    end_layer = min(end_layer, num_layers)
    model_type = getattr(config, "model_type", "")

    # Worker only runs assigned layers — everything else goes to disk
    # to avoid RAM OOM on machines with limited memory (e.g. 16 GB laptop).
    if model_type in ("llama", "qwen2", "mistral", "gemma", "phi", "phi3"):
        layer_prefix = "model.layers"
        device_map = {
            "model.embed_tokens": "disk",
            "model.norm": "disk",
            "model.rotary_emb": "disk",
            "lm_head": "disk",
        }
    elif model_type == "gpt2":
        layer_prefix = "transformer.h"
        device_map = {
            "transformer.wte": "disk", "transformer.wpe": "disk",
            "transformer.ln_f": "disk", "transformer.drop": "disk",
            "lm_head": "disk",
        }
    else:
        raise ValueError(f"Unsupported model_type for partial load: {model_type}")

    for i in range(num_layers):
        key = f"{layer_prefix}.{i}"
        device_map[key] = device if start_layer <= i < end_layer else "disk"

    offload_dir = os.path.join(tempfile.gettempdir(), "vramancer_offload")
    os.makedirs(offload_dir, exist_ok=True)

    logger.info("Loading partial model %s: layers %d-%d on %s",
                model_name, start_layer, end_layer, device)
    t0 = time.time()
    _partial_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=dtype,
        low_cpu_mem_usage=True, offload_folder=offload_dir,
    )
    _partial_model.eval()
    elapsed = time.time() - t0

    logger.info("Partial model loaded in %.1fs", elapsed)
    return {
        "model_name": model_name,
        "num_layers": num_layers,
        "layers_on_gpu": list(range(start_layer, end_layer)),
        "gpu_device": device,
        "load_seconds": round(elapsed, 1),
    }


# ─── Worker-side: run a range of layers ──────────────────────────

def _worker_forward_tensor(model, hidden: "torch.Tensor",
                           start_layer: int, end_layer: int,
                           seq_len: int = 0) -> "torch.Tensor":
    """Execute layers [start_layer, end_layer) directly on a tensor.

    Fast path used by VTP server — avoids pickle serialization.
    """
    layers = get_model_layers(model)
    device = str(next(layers[start_layer].parameters()).device)
    hidden = hidden.to(device)
    gpt2 = _is_gpt2(model)

    position_ids = None
    position_embeddings = None
    if not gpt2 and seq_len > 0:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            try:
                bufs = list(model.model.rotary_emb.buffers())
                re_dev = bufs[0].device if bufs else device
                pe = model.model.rotary_emb(
                    hidden.to(re_dev), position_ids.to(re_dev))
                position_embeddings = tuple(t.to(device) for t in pe)
            except Exception:
                pass

    with torch.no_grad():
        for i in range(start_layer, end_layer):
            layer_dev = next(layers[i].parameters()).device
            if hidden.device != layer_dev:
                hidden = hidden.to(layer_dev)
            kwargs = {}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids.to(layer_dev)
            if position_embeddings is not None:
                kwargs["position_embeddings"] = tuple(
                    t.to(layer_dev) for t in position_embeddings)
            try:
                out = layers[i](hidden, **kwargs)
            except TypeError:
                out = layers[i](hidden)
            hidden = out[0] if isinstance(out, tuple) else out

    return hidden


def worker_forward(model, hidden_bytes: bytes, start_layer: int, end_layer: int,
                   seq_len: int = 0) -> bytes:
    """Execute layers [start_layer, end_layer) on *hidden_bytes*.

    Called on the worker node.  HTTP fallback path — uses pickle serialization.
    For VTP path, use _worker_forward_tensor() instead (no pickle overhead).
    """
    hidden = bytes_to_tensor(hidden_bytes)
    result = _worker_forward_tensor(model, hidden, start_layer, end_layer, seq_len)
    return tensor_to_bytes(result)


# ─── Master-side: proxy to a remote worker ────────────────────────

class RemoteWorker:
    """HTTP proxy that forwards hidden-state tensors to a remote VRAMancer node."""

    def __init__(self, url: str, token: str, start_layer: int, end_layer: int):
        self.url = url.rstrip("/")
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._sess = _requests.Session()
        self._sess.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        })

    def forward(self, hidden_states: "torch.Tensor",
                seq_len: int = 0) -> "torch.Tensor":
        """POST hidden states to worker, receive processed tensor back."""
        payload = tensor_to_bytes(hidden_states)
        resp = self._sess.post(
            f"{self.url}/api/worker/forward_layers",
            params={
                "start_layer": self.start_layer,
                "end_layer": self.end_layer,
                "seq_len": seq_len,
            },
            data=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return bytes_to_tensor(resp.content)


# ─── Master-side: distributed generation ─────────────────────────

def distributed_generate(
    backend,
    prompt: str,
    remote_workers: List[RemoteWorker],
    local_layer_range: Tuple[int, int],
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """Token-by-token generation with layers split across nodes.

    No KV-cache (full recompute per step).  Suitable for demo
    and moderate-length generations.

    Returns
    -------
    dict with keys: text, tokens, total_seconds, tokens_per_second
    """
    model = backend.model
    tokenizer = backend.tokenizer
    gpt2 = _is_gpt2(model)
    layers = get_model_layers(model)

    # Primary device (embedding location)
    if gpt2:
        device = str(next(model.transformer.wte.parameters()).device)
    else:
        device = str(next(model.model.embed_tokens.parameters()).device)

    # Execution plan — sorted by layer index
    segments = [{"type": "local", "start": local_layer_range[0],
                 "end": local_layer_range[1]}]
    for w in remote_workers:
        segments.append({"type": "remote", "start": w.start_layer,
                         "end": w.end_layer, "worker": w})
    segments.sort(key=lambda s: s["start"])

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generated = input_ids.clone()

    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len = generated.shape[1]

            # ── Embedding ─────────────────────────────────────────
            if gpt2:
                pos = torch.arange(seq_len, device=device).unsqueeze(0)
                hidden = model.transformer.wte(generated) + model.transformer.wpe(pos)
                if getattr(model.transformer, "drop", None) is not None:
                    hidden = model.transformer.drop(hidden)
            else:
                hidden = model.model.embed_tokens(generated.to(device))

            # Position IDs & rotary for local Llama/Qwen layers
            position_ids = None
            position_embeddings = None
            if not gpt2:
                position_ids = torch.arange(
                    seq_len, device=hidden.device).unsqueeze(0)
                if hasattr(model.model, "rotary_emb"):
                    try:
                        bufs = list(model.model.rotary_emb.buffers())
                        re_dev = bufs[0].device if bufs else hidden.device
                        pe = model.model.rotary_emb(
                            hidden.to(re_dev), position_ids.to(re_dev))
                        position_embeddings = tuple(
                            t.to(hidden.device) for t in pe)
                    except Exception:
                        pass

            # ── Layer segments ────────────────────────────────────
            for seg in segments:
                if seg["type"] == "local":
                    for i in range(seg["start"], seg["end"]):
                        layer_dev = next(layers[i].parameters()).device
                        if hidden.device != layer_dev:
                            hidden = hidden.to(layer_dev)
                        kwargs = {}
                        if position_ids is not None:
                            kwargs["position_ids"] = position_ids.to(
                                layer_dev)
                        if position_embeddings is not None:
                            kwargs["position_embeddings"] = tuple(
                                t.to(layer_dev) for t in
                                position_embeddings)
                        try:
                            out = layers[i](hidden, **kwargs)
                        except TypeError:
                            out = layers[i](hidden)
                        hidden = out[0] if isinstance(out, tuple) else out
                else:
                    # ── Remote forward ────────────────────────────
                    worker = seg["worker"]
                    hidden = worker.forward(hidden, seq_len=seq_len)

            # ── Norm + LM head ────────────────────────────────────
            if gpt2:
                hidden = model.transformer.ln_f(hidden)
            else:
                norm_dev = str(next(model.model.norm.parameters()).device)
                hidden = model.model.norm(hidden.to(norm_dev))
            head_dev = str(next(model.lm_head.parameters()).device)
            logits = model.lm_head(hidden.to(head_dev))

            # ── Sampling ──────────────────────────────────────────
            next_logits = logits[:, -1, :].float()
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 0 and top_k < next_logits.size(-1):
                top_vals = torch.topk(next_logits, top_k)[0]
                next_logits[next_logits < top_vals[..., -1:]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cumsum = sorted_p.cumsum(dim=-1)
                sorted_p[(cumsum - sorted_p) >= top_p] = 0.0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_i, sorted_p)
            if temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = probs.argmax(-1, keepdim=True)

            generated = torch.cat([generated, next_token.to(device)], dim=-1)
            if (tokenizer.eos_token_id is not None
                    and next_token.item() == tokenizer.eos_token_id):
                break

    elapsed = time.perf_counter() - t0
    new_tokens = generated[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n = len(new_tokens)

    return {
        "text": text,
        "tokens": n,
        "total_seconds": round(elapsed, 4),
        "tokens_per_second": round(n / elapsed, 2) if elapsed > 0 else 0,
    }
