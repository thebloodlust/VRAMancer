"""WebGPU Distributed Backend for VRAMancer.

Dispatches matmul operations to browser WebGPU workers via WebSocket.
Browser runs a WGSL tiled matmul shader (dashboard/worker/).

Binary protocol (matching worker.js):
  Request:  [op:u8] [M:u32le] [N:u32le] [K:u32le] [A:f32*M*K] [B:f32*N*K]
  Response: [op:u8] [M:u32le] [N:u32le] [C:f32*M*N]
  op=0x01: matmul (C = A @ B^T), op=0x02: ping, op=0xFF: shutdown
"""

import asyncio
import struct
import threading
import time
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
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

        self._workers: list[_WorkerConnection] = []
        self._workers_lock = threading.Lock()
        self._rr = 0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._ws_server = None
        self._http_server = None

        if websockets is None:
            self.log.warning(
                "websockets not installed — pip install websockets"
            )
            return

        self._start_ws_server()
        if serve_ui and _WORKER_DIR.exists():
            self._start_http_server()

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
        self.log.info(
            f"WebSocket server on ws://{self._ws_host}:{self._ws_port}"
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

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

        self._ws_server = await websockets.server.serve(
            handler, self._ws_host, self._ws_port,
            max_size=256 * 1024 * 1024,  # 256 MB max message
            ping_interval=None,  # disable auto-ping (we do manual ping)
            write_limit=2 ** 22,  # 4 MB write buffer
        )
        await self._ws_server.wait_closed()

    # ------------------------------------------------------------------
    # HTTP Static Server (serves dashboard/worker/ UI)
    # ------------------------------------------------------------------

    def _start_http_server(self):
        worker_dir = str(_WORKER_DIR)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=worker_dir, **kwargs)

            def log_message(self, fmt, *args):
                pass  # silence HTTP logs

        self._http_server = HTTPServer(("0.0.0.0", self._http_port), Handler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            daemon=True, name="webgpu-http",
        )
        self._http_thread.start()
        self.log.info(
            f"Worker UI on http://localhost:{self._http_port}/index.html"
        )

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
                    f"Open http://localhost:{self._http_port}/index.html "
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
