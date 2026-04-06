"""WebGPU Distributed Backend for VRAMancer.

Dispatches matmul operations to browser WebGPU workers via WebSocket.
Browser runs a WGSL tiled matmul shader (dashboard/worker/).

Binary protocol (matching worker.js):
  Request:  [op:u8] [M:u32le] [N:u32le] [K:u32le] [A:f32*M*K] [B:f32*N*K]
  Response: [op:u8] [M:u32le] [N:u32le] [C:f32*M*N]
  op=0x01: matmul (C = A @ B^T), op=0x02: ping, op=0xFF: shutdown
"""

import asyncio
import ssl
import struct
import subprocess
import tempfile
import threading
import time
import os
from pathlib import Path
from typing import Any, List, Optional

try:
    import websockets
    import websockets.server
except ImportError:
    websockets = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    from core.logger import LoggerAdapter
except ImportError:
    import logging
    LoggerAdapter = lambda name: logging.getLogger(name)

try:
    from core.backends import BaseLLMBackend
except ImportError:
    class BaseLLMBackend:
        pass

OP_MATMUL = 0x01
OP_PING = 0x02
OP_UPLOAD_TENSOR = 0x10
OP_SET_CONFIG = 0x11
OP_GENERATE = 0x30
OP_TOKEN = 0x31
OP_GENERATE_DONE = 0x32
OP_SHUTDOWN = 0xFF

_TIMEOUT_S = float(os.environ.get("VRM_WEBGPU_TIMEOUT", "30.0"))
_WS_PORT = int(os.environ.get("VRM_WEBGPU_WS_PORT", "8765"))
_HTTP_PORT = int(os.environ.get("VRM_WEBGPU_HTTP_PORT", "8766"))

# Path to the dashboard/worker/ directory with WGSL shader + JS + HTML
_WORKER_DIR = Path(__file__).parent.parent / "dashboard" / "worker"


class _WorkerConnection:
    """Wraps a single browser WebGPU worker WebSocket."""

    __slots__ = ("ws", "lock", "addr", "connected_at", "ops_done", "total_gflops")

    def __init__(self, ws, addr):
        self.ws = ws
        self.lock = asyncio.Lock()
        self.addr = addr
        self.connected_at = time.monotonic()
        self.ops_done = 0
        self.total_gflops = 0.0

    async def rpc_matmul(self, M: int, N: int, K: int,
                         a_bytes: bytes, b_bytes: bytes,
                         timeout: float = _TIMEOUT_S) -> bytes:
        """Send matmul request, wait for response. Thread-safe via lock."""
        header = struct.pack("<BIII", OP_MATMUL, M, N, K)
        frame = header + a_bytes + b_bytes

        async with self.lock:
            await self.ws.send(frame)
            resp = await asyncio.wait_for(self.ws.recv(), timeout=timeout)

        if isinstance(resp, str):
            resp = resp.encode()

        resp_op = resp[0]
        if resp_op != OP_MATMUL:
            raise RuntimeError(f"Unexpected response op: 0x{resp_op:02x}")

        resp_M = struct.unpack_from("<I", resp, 1)[0]
        resp_N = struct.unpack_from("<I", resp, 5)[0]
        c_data = resp[9:]
        expected = resp_M * resp_N * 4
        if len(c_data) < expected:
            raise RuntimeError(
                f"Incomplete response: got {len(c_data)} bytes, "
                f"expected {expected} ({resp_M}x{resp_N} f32)"
            )
        self.ops_done += 1
        self.total_gflops += 2.0 * M * N * K / 1e9
        return c_data[:expected]

    async def ping(self, timeout: float = 5.0) -> float:
        """Ping worker, return round-trip time in ms."""
        frame = struct.pack("B", OP_PING)
        t0 = time.monotonic()
        async with self.lock:
            await self.ws.send(frame)
            resp = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        rtt = (time.monotonic() - t0) * 1000
        return rtt

    async def shutdown(self):
        try:
            await self.ws.send(struct.pack("B", OP_SHUTDOWN))
        except Exception:
            pass

    async def rpc_upload_tensor(self, name: str, shape: tuple,
                                data: bytes,
                                timeout: float = _TIMEOUT_S) -> None:
        """Upload a named tensor to the browser worker."""
        name_bytes = name.encode("utf-8")
        ndim = len(shape)
        # Header: [0x10:u8] [name_len:u16le] [name] [ndim:u8] [shapes:u32le*]
        header = struct.pack("<BH", OP_UPLOAD_TENSOR, len(name_bytes))
        header += name_bytes
        header += struct.pack("<B", ndim)
        for s in shape:
            header += struct.pack("<I", s)
        # Align to 4 bytes for float32 data
        pad = (4 - len(header) % 4) % 4
        header += b"\x00" * pad
        frame = header + data

        async with self.lock:
            await self.ws.send(frame)
            resp = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(resp, str):
            resp = resp.encode()
        if resp[0] != OP_UPLOAD_TENSOR or resp[1] != 0x00:
            raise RuntimeError(f"Upload tensor '{name}' failed: {resp!r}")

    async def rpc_set_config(self, config_json: str,
                             timeout: float = _TIMEOUT_S) -> None:
        """Send model config JSON to the browser worker."""
        json_bytes = config_json.encode("utf-8")
        header = struct.pack("<BI", OP_SET_CONFIG, len(json_bytes))
        frame = header + json_bytes

        async with self.lock:
            await self.ws.send(frame)
            resp = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
        if isinstance(resp, str):
            resp = resp.encode()
        if resp[0] != OP_SET_CONFIG or resp[1] != 0x00:
            raise RuntimeError(f"Set config failed: {resp!r}")

    async def rpc_generate(self, max_tokens: int, prompt_ids: list,
                           token_callback=None,
                           timeout: float = 600.0) -> dict:
        """Start autoregressive generation. Calls token_callback(token_id, step, time_ms)
        for each generated token. Returns summary dict."""
        n = len(prompt_ids)
        header = struct.pack("<BII", OP_GENERATE, max_tokens, n)
        for tid in prompt_ids:
            header += struct.pack("<I", tid)

        async with self.lock:
            await self.ws.send(header)
            tokens = []
            while True:
                resp = await asyncio.wait_for(
                    self.ws.recv(), timeout=timeout
                )
                if isinstance(resp, str):
                    resp = resp.encode()
                op = resp[0]
                if op == OP_TOKEN:
                    token_id = struct.unpack_from("<I", resp, 1)[0]
                    step = struct.unpack_from("<I", resp, 5)[0]
                    time_ms = struct.unpack_from("<f", resp, 9)[0]
                    tokens.append(token_id)
                    if token_callback:
                        token_callback(token_id, step, time_ms)
                elif op == OP_GENERATE_DONE:
                    total_tokens = struct.unpack_from("<I", resp, 1)[0]
                    total_ms = struct.unpack_from("<f", resp, 5)[0]
                    return {
                        "tokens": tokens,
                        "total_tokens": total_tokens,
                        "total_ms": total_ms,
                    }
                else:
                    raise RuntimeError(
                        f"Unexpected op during generate: 0x{op:02x}"
                    )


class WebGPUBackend(BaseLLMBackend):
    """WebSocket server dispatching matmul to browser WebGPU workers.

    Quick start:
        backend = WebGPUBackend()
        # Open http://localhost:8766 in a WebGPU browser (Chrome/Edge)
        # Browser auto-connects to ws://localhost:8765
        C = backend.matmul(A, B)  # A @ B^T via browser GPU
    """

    def __init__(self, ws_host: str = "0.0.0.0", ws_port: int = _WS_PORT,
                 http_port: int = _HTTP_PORT, serve_ui: bool = True):
        self.log = LoggerAdapter("backend.webgpu")
        self.model_name = None
        self.tokenizer = None
        self._ws_host = ws_host
        self._ws_port = ws_port
        self._http_port = http_port
        self._serve_ui = serve_ui

        self._workers: list[_WorkerConnection] = []
        self._workers_lock = threading.Lock()
        self._rr = 0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._ws_server = None
        self._http_server = None
        self._ssl_context = None

        if websockets is None:
            self.log.warning(
                "websockets not installed — pip install websockets"
            )
            return

        # Build shared SSL context for WSS + HTTPS
        cert_path, key_path = self._ensure_self_signed_cert()
        if cert_path:
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl_context.load_cert_chain(cert_path, key_path)

        # Single port: WSS + static files on ws_port (avoids cert-per-port issues)
        self._start_ws_server()

    # ------------------------------------------------------------------
    # WebSocket Server
    # ------------------------------------------------------------------

    def _start_ws_server(self):
        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="webgpu-ws"
        )
        self._ws_thread.start()
        for _ in range(300):
            if self._ws_server is not None:
                break
            time.sleep(0.01)
        proto = "wss" if self._ssl_context else "ws"
        self.log.info(
            f"WebGPU server on {proto}://{self._ws_host}:{self._ws_port} "
            f"(WSS + static files on same port)"
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _http_process_request(self, path, request_headers):
        """Serve static files for non-WebSocket HTTP requests.
        Return (status, headers, body) to send HTTP response.
        Return None to proceed with WebSocket upgrade."""
        # WebSocket upgrades pass through
        if request_headers.get("Upgrade", "").lower() == "websocket":
            return None

        if path == "/" or path == "":
            path = "/inference.html"

        # Sanitize to prevent path traversal
        safe = Path(path.lstrip("/")).name
        if not safe:
            return (404, {"Content-Type": "text/plain"}, b"Not Found")

        file_path = _WORKER_DIR / safe
        if not file_path.exists() or not file_path.is_file():
            return (404, {"Content-Type": "text/plain"}, b"Not Found")

        # MIME types
        ext = file_path.suffix.lower()
        mime = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".wgsl": "text/plain",
            ".css": "text/css",
            ".json": "application/json",
        }.get(ext, "application/octet-stream")

        body = file_path.read_bytes()
        return (200, {"Content-Type": mime}, body)

    async def _serve(self):
        async def handler(ws):
            addr = ws.remote_address
            worker = _WorkerConnection(ws, addr)
            with self._workers_lock:
                self._workers.append(worker)
            self.log.info(
                f"WebGPU worker connected: {addr} "
                f"(total: {len(self._workers)})"
            )
            try:
                await ws.wait_closed()
            finally:
                with self._workers_lock:
                    if worker in self._workers:
                        self._workers.remove(worker)
                self.log.info(
                    f"WebGPU worker disconnected: {addr} "
                    f"(ops={worker.ops_done}, "
                    f"gflops={worker.total_gflops:.1f})"
                )

        serve_kwargs = dict(
            max_size=256 * 1024 * 1024,
            ping_interval=None,
            write_limit=2 ** 22,
            ssl=self._ssl_context,
        )
        # Serve static files on the same port (single cert = no browser issues)
        if self._serve_ui and _WORKER_DIR.exists():
            serve_kwargs["process_request"] = self._http_process_request

        self._ws_server = await websockets.server.serve(
            handler, self._ws_host, self._ws_port,
            **serve_kwargs,
        )
        await self._ws_server.wait_closed()

    @staticmethod
    def _ensure_self_signed_cert():
        """Generate a self-signed cert for HTTPS (WebGPU secure context).
        Includes SAN with all local IPs so browsers accept it."""
        cert_dir = Path(tempfile.gettempdir()) / "vramancer_certs"
        cert_path = cert_dir / "cert.pem"
        key_path = cert_dir / "key.pem"
        ext_path = cert_dir / "openssl_ext.cnf"
        # Always regenerate to pick up IP changes
        try:
            cert_dir.mkdir(parents=True, exist_ok=True)
            # Collect all local IPs for SAN
            import socket
            san_entries = ["IP:127.0.0.1", "IP:::1", "DNS:localhost"]
            try:
                for info in socket.getaddrinfo(
                    socket.gethostname(), None, socket.AF_UNSPEC
                ):
                    addr = info[4][0]
                    if ":" in addr:
                        san_entries.append(f"IP:{addr}")
                    else:
                        san_entries.append(f"IP:{addr}")
            except Exception:
                pass
            # Also try common method to get LAN IP
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
                s.close()
                entry = f"IP:{lan_ip}"
                if entry not in san_entries:
                    san_entries.append(entry)
            except Exception:
                pass
            san_str = ",".join(san_entries)
            # Write OpenSSL config file for SAN
            ext_path.write_text(
                "[req]\n"
                "distinguished_name = req_dn\n"
                "x509_extensions = v3_req\n"
                "prompt = no\n"
                "[req_dn]\n"
                "CN = VRAMancer-WebGPU\n"
                "[v3_req]\n"
                "subjectAltName = " + san_str + "\n"
            )
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048",
                    "-keyout", str(key_path), "-out", str(cert_path),
                    "-days", "365", "-nodes",
                    "-config", str(ext_path),
                ],
                check=True, capture_output=True,
            )
            return str(cert_path), str(key_path)
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Worker selection
    # ------------------------------------------------------------------

    @property
    def num_workers(self) -> int:
        with self._workers_lock:
            return len(self._workers)

    def _pick_worker(self) -> Optional[_WorkerConnection]:
        with self._workers_lock:
            if not self._workers:
                return None
            w = self._workers[self._rr % len(self._workers)]
            self._rr += 1
            return w

    def wait_for_workers(self, n: int = 1, timeout: float = 120.0):
        """Block until at least n workers are connected."""
        t0 = time.monotonic()
        while self.num_workers < n:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"Waited {timeout}s for {n} WebGPU worker(s), "
                    f"got {self.num_workers}. "
                    f"Open https://localhost:{self._ws_port}/inference.html "
                    f"in a WebGPU browser."
                )
            time.sleep(0.1)
        self.log.info(f"{self.num_workers} WebGPU worker(s) connected")

    # ------------------------------------------------------------------
    # Public API: matmul
    # ------------------------------------------------------------------

    def matmul(self, A, B, timeout: float = _TIMEOUT_S):
        """Compute A @ B^T via a browser WebGPU worker.

        Args:
            A: [M, K] tensor or numpy array (float32)
            B: [N, K] tensor or numpy array (transposed layout: C = A @ B^T)
            timeout: max seconds to wait for result

        Returns:
            [M, N] tensor (if input was tensor) or numpy array
        """
        if self._loop is None:
            raise RuntimeError("WebSocket server not running")
        if self.num_workers == 0:
            raise RuntimeError(
                "No WebGPU workers connected. "
                f"Open http://localhost:{self._http_port}/index.html"
            )

        return_torch = False
        if torch is not None and isinstance(A, torch.Tensor):
            a_np = A.detach().cpu().float().contiguous().numpy()
            b_np = B.detach().cpu().float().contiguous().numpy()
            return_torch = True
        elif np is not None:
            a_np = np.ascontiguousarray(A, dtype=np.float32)
            b_np = np.ascontiguousarray(B, dtype=np.float32)
        else:
            raise RuntimeError("Neither torch nor numpy available")

        M, K = a_np.shape
        N = b_np.shape[0]
        if b_np.shape[1] != K:
            raise ValueError(f"K mismatch: A={a_np.shape}, B={b_np.shape}")

        worker = self._pick_worker()
        if worker is None:
            raise RuntimeError("No workers available (race)")

        fut = asyncio.run_coroutine_threadsafe(
            worker.rpc_matmul(M, N, K,
                              a_np.tobytes(), b_np.tobytes(),
                              timeout),
            self._loop,
        )
        c_data = fut.result(timeout=timeout + 2.0)
        c_np = np.frombuffer(c_data, dtype=np.float32).reshape(M, N)

        if return_torch:
            return torch.from_numpy(c_np.copy())
        return c_np.copy()

    # ------------------------------------------------------------------
    # Batch matmul (parallel across workers)
    # ------------------------------------------------------------------

    def matmul_batch(self, ops: list, timeout: float = _TIMEOUT_S):
        """Execute multiple matmuls across available workers.

        Args:
            ops: list of (A, B) tuples, each [M,K] @ [N,K]^T
        Returns:
            list of result arrays/tensors
        """
        if self._loop is None or self.num_workers == 0:
            raise RuntimeError("No workers")

        async def _dispatch_all():
            tasks = []
            for A, B in ops:
                return_torch = torch is not None and isinstance(A, torch.Tensor)
                if return_torch:
                    a_np = A.detach().cpu().float().contiguous().numpy()
                    b_np = B.detach().cpu().float().contiguous().numpy()
                else:
                    a_np = np.ascontiguousarray(A, dtype=np.float32)
                    b_np = np.ascontiguousarray(B, dtype=np.float32)
                M, K = a_np.shape
                N = b_np.shape[0]
                worker = self._pick_worker()
                tasks.append((
                    worker.rpc_matmul(M, N, K,
                                      a_np.tobytes(), b_np.tobytes(),
                                      timeout),
                    M, N, return_torch,
                ))
            results = []
            for coro, M, N, rt in tasks:
                c_data = await coro
                c_np = np.frombuffer(c_data, dtype=np.float32).reshape(M, N)
                if rt:
                    results.append(torch.from_numpy(c_np.copy()))
                else:
                    results.append(c_np.copy())
            return results

        fut = asyncio.run_coroutine_threadsafe(
            _dispatch_all(), self._loop
        )
        return fut.result(timeout=timeout * len(ops) + 5.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return backend status summary."""
        with self._workers_lock:
            workers = []
            for w in self._workers:
                workers.append({
                    "addr": str(w.addr),
                    "ops": w.ops_done,
                    "gflops": round(w.total_gflops, 2),
                    "uptime_s": round(time.monotonic() - w.connected_at, 1),
                })
        return {
            "backend": "webgpu",
            "ws_port": self._ws_port,
            "http_port": self._http_port,
            "num_workers": len(workers),
            "workers": workers,
        }

    def ping_workers(self) -> dict:
        """Ping all workers, return RTT in ms."""
        if self._loop is None:
            return {}

        async def _ping_all():
            results = {}
            with self._workers_lock:
                workers = list(self._workers)
            for w in workers:
                try:
                    rtt = await w.ping()
                    results[str(w.addr)] = round(rtt, 2)
                except Exception as e:
                    results[str(w.addr)] = f"error: {e}"
            return results

        fut = asyncio.run_coroutine_threadsafe(_ping_all(), self._loop)
        return fut.result(timeout=10.0)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Send shutdown to all workers and stop servers."""
        if self._loop and self._ws_server:
            async def _do():
                with self._workers_lock:
                    for w in list(self._workers):
                        await w.shutdown()
                self._ws_server.close()
            asyncio.run_coroutine_threadsafe(_do(), self._loop)

        if self._http_server:
            self._http_server.shutdown()

    # ------------------------------------------------------------------
    # Inference: upload tensors, push model, generate
    # ------------------------------------------------------------------

    def upload_tensor(self, name: str, tensor, timeout: float = _TIMEOUT_S):
        """Upload a named tensor to the first connected worker.

        Args:
            name: weight name (e.g. 'h.0.attn.c_attn.weight')
            tensor: torch.Tensor or numpy array (float32)
        """
        if self._loop is None:
            raise RuntimeError("WebSocket server not running")
        worker = self._pick_worker()
        if worker is None:
            raise RuntimeError("No workers connected")

        if torch is not None and isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().contiguous().numpy()
        elif np is not None:
            arr = np.ascontiguousarray(tensor, dtype=np.float32)
        else:
            raise RuntimeError("Neither torch nor numpy available")

        shape = tuple(arr.shape)
        data = arr.tobytes()

        fut = asyncio.run_coroutine_threadsafe(
            worker.rpc_upload_tensor(name, shape, data, timeout),
            self._loop,
        )
        fut.result(timeout=timeout + 5.0)

    def set_config(self, config_dict: dict, timeout: float = _TIMEOUT_S):
        """Send model config to worker."""
        import json
        if self._loop is None:
            raise RuntimeError("WebSocket server not running")
        worker = self._pick_worker()
        if worker is None:
            raise RuntimeError("No workers connected")

        config_json = json.dumps(config_dict)
        fut = asyncio.run_coroutine_threadsafe(
            worker.rpc_set_config(config_json, timeout),
            self._loop,
        )
        fut.result(timeout=timeout + 5.0)

    def push_gpt2_model(self, model, tokenizer_inst=None):
        """Extract GPT-2 weights and upload them all to the browser.

        GPT-2 uses Conv1D (weights are [in, out]) — we transpose to
        [out, in] for the WGSL matmul shader (C = A @ B^T).
        """
        if torch is None:
            raise RuntimeError("torch required for push_gpt2_model")

        sd = model.state_dict()
        gpt2_config = model.config

        # Send model config
        config_dict = {
            "arch": "gpt2",
            "model_name": getattr(gpt2_config, "name_or_path", "gpt2"),
            "hidden_size": gpt2_config.n_embd,
            "num_heads": gpt2_config.n_head,
            "num_layers": gpt2_config.n_layer,
            "vocab_size": gpt2_config.vocab_size,
            "max_position": gpt2_config.n_positions,
            "intermediate_size": gpt2_config.n_embd * 4,
        }
        self.set_config(config_dict)

        # Upload embeddings (CPU-only in browser — no GPU buffer for 1D-ish)
        self.upload_tensor("wte.weight", sd["transformer.wte.weight"])
        self.upload_tensor("wpe.weight", sd["transformer.wpe.weight"])

        # Upload final layer norm
        self.upload_tensor("ln_f.weight", sd["transformer.ln_f.weight"])
        self.upload_tensor("ln_f.bias", sd["transformer.ln_f.bias"])

        # Upload each transformer layer
        for i in range(gpt2_config.n_layer):
            prefix = f"transformer.h.{i}"
            short = f"h.{i}"

            # LayerNorms (1D, CPU-only in browser)
            self.upload_tensor(
                f"{short}.ln_1.weight", sd[f"{prefix}.ln_1.weight"]
            )
            self.upload_tensor(
                f"{short}.ln_1.bias", sd[f"{prefix}.ln_1.bias"]
            )
            self.upload_tensor(
                f"{short}.ln_2.weight", sd[f"{prefix}.ln_2.weight"]
            )
            self.upload_tensor(
                f"{short}.ln_2.bias", sd[f"{prefix}.ln_2.bias"]
            )

            # Attention weights: Conv1D [in, out] → transpose to [out, in]
            self.upload_tensor(
                f"{short}.attn.c_attn.weight",
                sd[f"{prefix}.attn.c_attn.weight"].T.contiguous(),
            )
            self.upload_tensor(
                f"{short}.attn.c_attn.bias",
                sd[f"{prefix}.attn.c_attn.bias"],
            )
            self.upload_tensor(
                f"{short}.attn.c_proj.weight",
                sd[f"{prefix}.attn.c_proj.weight"].T.contiguous(),
            )
            self.upload_tensor(
                f"{short}.attn.c_proj.bias",
                sd[f"{prefix}.attn.c_proj.bias"],
            )

            # MLP weights: Conv1D [in, out] → transpose to [out, in]
            self.upload_tensor(
                f"{short}.mlp.c_fc.weight",
                sd[f"{prefix}.mlp.c_fc.weight"].T.contiguous(),
            )
            self.upload_tensor(
                f"{short}.mlp.c_fc.bias",
                sd[f"{prefix}.mlp.c_fc.bias"],
            )
            self.upload_tensor(
                f"{short}.mlp.c_proj.weight",
                sd[f"{prefix}.mlp.c_proj.weight"].T.contiguous(),
            )
            self.upload_tensor(
                f"{short}.mlp.c_proj.bias",
                sd[f"{prefix}.mlp.c_proj.bias"],
            )

            self.log.info(f"Uploaded layer {i}/{gpt2_config.n_layer}")

        total_params = sum(p.numel() for p in model.parameters())
        total_mb = total_params * 4 / 1024 / 1024
        self.log.info(
            f"GPT-2 model uploaded: {gpt2_config.n_layer} layers, "
            f"{total_params/1e6:.0f}M params, {total_mb:.0f} MB float32"
        )

        if tokenizer_inst:
            self.tokenizer = tokenizer_inst

    def generate_browser(self, prompt: str = None, prompt_ids: list = None,
                         max_tokens: int = 50,
                         token_callback=None,
                         timeout: float = 600.0) -> dict:
        """Run autoregressive generation on the browser WebGPU worker.

        Args:
            prompt: text prompt (requires tokenizer)
            prompt_ids: token IDs (alternative to prompt)
            max_tokens: max tokens to generate
            token_callback: called with (token_id, step, time_ms) per token
            timeout: max seconds to wait

        Returns:
            dict with 'tokens', 'text', 'total_tokens', 'total_ms', 'tok_per_s'
        """
        if self._loop is None:
            raise RuntimeError("WebSocket server not running")
        worker = self._pick_worker()
        if worker is None:
            raise RuntimeError("No workers connected")

        if prompt_ids is None:
            if prompt is None:
                raise ValueError("Provide prompt or prompt_ids")
            if self.tokenizer is None:
                raise RuntimeError("No tokenizer loaded — provide prompt_ids")
            prompt_ids = self.tokenizer.encode(prompt)

        fut = asyncio.run_coroutine_threadsafe(
            worker.rpc_generate(max_tokens, prompt_ids,
                                token_callback, timeout),
            self._loop,
        )
        result = fut.result(timeout=timeout + 5.0)

        # Decode tokens to text if tokenizer available
        if self.tokenizer:
            result["text"] = self.tokenizer.decode(result["tokens"])
            result["prompt_text"] = self.tokenizer.decode(prompt_ids)
        else:
            result["text"] = None

        if result["total_ms"] > 0:
            decode_tokens = result["total_tokens"] - 1
            result["tok_per_s"] = decode_tokens / (result["total_ms"] / 1000)
        else:
            result["tok_per_s"] = 0.0

        return result

    # ------------------------------------------------------------------
    # BaseLLMBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.log.info(
            f"Model {model_name} mapped to WebGPU backend "
            f"({self.num_workers} workers)"
        )
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.log.warning(f"Could not load tokenizer: {e}")
        return {"name": model_name, "type": "webgpu_distributed"}

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        return [self]

    def infer(self, inputs: Any):
        if self.num_workers == 0:
            return inputs
        if torch is not None and isinstance(inputs, torch.Tensor):
            M = inputs.shape[-2] if inputs.ndim >= 2 else 1
            K = inputs.shape[-1]
            eye = torch.eye(K, dtype=torch.float32)
            try:
                return self.matmul(inputs.view(M, K), eye)
            except Exception:
                return inputs
        return inputs

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        if self.num_workers == 0:
            raise RuntimeError(
                "No WebGPU workers connected. "
                f"Open http://localhost:{self._http_port}/index.html"
            )
        return (
            f"[WebGPU backend: {self.num_workers} workers ready, "
            f"use matmul() directly or via InferencePipeline]"
        )

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        yield self.generate(prompt, max_new_tokens, **kwargs)
