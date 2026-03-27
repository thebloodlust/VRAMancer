"""WebGPU Distributed Backend for VRAMancer.

Bridges Python inference to browser WebGPU workers via WebSocket.
The browser runs a real WGSL tiled matmul shader (dashboard/worker/).

Binary protocol (matching worker.js):
  Request:  [op:u8] [M:u32le] [N:u32le] [K:u32le] [A:f32*M*K] [B:f32*N*K]
  Response: [op:u8] [M:u32le] [N:u32le] [C:f32*M*N]
  op=0x01: matmul, op=0x02: ping, op=0xFF: shutdown
"""

import threading
import struct
import time
import os
from typing import Any, List, Optional

try:
    import asyncio
except ImportError:
    asyncio = None

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

from core.backends import BaseLLMBackend
from core.logger import LoggerAdapter

OP_MATMUL = 0x01
OP_PING = 0x02
OP_SHUTDOWN = 0xFF

# Default timeout for a single matmul RPC (ms)
_TIMEOUT_S = float(os.environ.get("VRM_WEBGPU_TIMEOUT", "10.0"))


class WebGPUBackend(BaseLLMBackend):
    """WebSocket server that dispatches matmul ops to browser WebGPU workers."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.log = LoggerAdapter("backend.webgpu")
        self.model_name = None
        self.tokenizer = None
        self._host = host
        self._port = port

        # Connected worker websockets
        self._clients: list = []
        self._clients_lock = threading.Lock()

        # Round-robin index
        self._rr = 0

        # Event loop runs in a background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None

        if websockets is None:
            self.log.warning(
                "websockets package not installed — WebGPU backend will "
                "operate in stub mode.  pip install websockets"
            )
        else:
            self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _start_server(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="webgpu-ws"
        )
        self._thread.start()
        # Wait for the server to be ready
        for _ in range(200):
            if self._server is not None:
                break
            time.sleep(0.01)
        self.log.info(f"WebSocket server listening on ws://{self._host}:{self._port}")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async def handler(ws):
            addr = ws.remote_address
            self.log.info(f"WebGPU worker connected: {addr}")
            with self._clients_lock:
                self._clients.append(ws)
            try:
                # Keep connection alive — we send requests, worker sends responses
                async for _ in ws:
                    pass  # responses handled by _rpc_matmul
            finally:
                with self._clients_lock:
                    if ws in self._clients:
                        self._clients.remove(ws)
                self.log.info(f"WebGPU worker disconnected: {addr}")

        self._server = await websockets.server.serve(
            handler, self._host, self._port
        )
        await self._server.wait_closed()

    def shutdown(self):
        """Send shutdown to all workers and stop the server."""
        if self._loop and self._server:
            async def _do():
                with self._clients_lock:
                    for ws in list(self._clients):
                        try:
                            await ws.send(struct.pack("B", OP_SHUTDOWN))
                        except Exception:
                            pass
                self._server.close()
            asyncio.run_coroutine_threadsafe(_do(), self._loop)

    # ------------------------------------------------------------------
    # Worker RPC
    # ------------------------------------------------------------------

    def _pick_worker(self):
        """Round-robin worker selection."""
        with self._clients_lock:
            if not self._clients:
                return None
            ws = self._clients[self._rr % len(self._clients)]
            self._rr += 1
            return ws

    @property
    def num_workers(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    async def _rpc_matmul(self, M: int, N: int, K: int,
                          data_a: bytes, data_b: bytes,
                          timeout: float = _TIMEOUT_S):
        """Send a matmul request to a worker, return C as bytes."""
        ws = self._pick_worker()
        if ws is None:
            raise RuntimeError("No WebGPU workers connected")

        header = struct.pack("<BIII", OP_MATMUL, M, N, K)
        frame = header + data_a + data_b

        await ws.send(frame)

        # Wait for response (binary)
        resp = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(resp, str):
            resp = resp.encode()

        # Parse: [op:u8] [M:u32le] [N:u32le] [C:f32*M*N]
        resp_op = resp[0]
        if resp_op != OP_MATMUL:
            raise RuntimeError(f"Unexpected response op: 0x{resp_op:02x}")
        resp_M = struct.unpack_from("<I", resp, 1)[0]
        resp_N = struct.unpack_from("<I", resp, 5)[0]
        c_data = resp[9:]
        expected = resp_M * resp_N * 4
        if len(c_data) < expected:
            raise RuntimeError(
                f"Incomplete matmul response: got {len(c_data)}, "
                f"expected {expected}"
            )
        return c_data[:expected], resp_M, resp_N

    def matmul_sync(self, A, B, timeout: float = _TIMEOUT_S):
        """Synchronous matmul: A @ B^T via WebGPU worker.

        A: [M, K] tensor or numpy array
        B: [N, K] tensor or numpy array (transposed layout)
        Returns: [M, N] tensor or numpy array
        """
        if self._loop is None:
            raise RuntimeError("WebGPU server not running (missing websockets?)")

        # Convert to contiguous float32 bytes
        if torch is not None and isinstance(A, torch.Tensor):
            a_np = A.detach().cpu().float().contiguous().numpy()
            b_np = B.detach().cpu().float().contiguous().numpy()
            return_torch = True
        elif np is not None:
            a_np = np.ascontiguousarray(A, dtype=np.float32)
            b_np = np.ascontiguousarray(B, dtype=np.float32)
            return_torch = False
        else:
            raise RuntimeError("Neither torch nor numpy available")

        M, K = a_np.shape
        N = b_np.shape[0]
        assert b_np.shape[1] == K, f"K mismatch: A={a_np.shape}, B={b_np.shape}"

        fut = asyncio.run_coroutine_threadsafe(
            self._rpc_matmul(M, N, K, a_np.tobytes(), b_np.tobytes(), timeout),
            self._loop,
        )
        c_data, _, _ = fut.result(timeout=timeout + 1.0)

        c_np = np.frombuffer(c_data, dtype=np.float32).reshape(M, N)
        if return_torch:
            return torch.from_numpy(c_np.copy())
        return c_np.copy()

    # ------------------------------------------------------------------
    # BaseLLMBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.log.info(f"Model {model_name} mapped to WebGPU backend "
                      f"({self.num_workers} workers connected)")

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.log.warning(f"Could not load tokenizer: {e}")

        return {"name": model_name, "type": "webgpu_distributed"}

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        return [self]

    def infer(self, inputs: Any):
        """Forward a single tensor through one matmul layer via WebGPU."""
        if self.num_workers == 0:
            return "[Error: No WebGPU workers connected]"

        if torch is not None and isinstance(inputs, torch.Tensor):
            # Treat as activation [1, hidden] * weight [vocab, hidden]
            # For now, just echo through a trivial identity-like matmul
            M, K = inputs.shape[-2], inputs.shape[-1]
            # Identity-ish: B = I padded (this is a stub — real usage
            # sends actual weight slices from the model)
            eye = torch.eye(K, dtype=torch.float32)
            try:
                return self.matmul_sync(inputs.view(M, K), eye)
            except Exception as e:
                self.log.warning(f"WebGPU infer failed: {e}")
                return inputs  # passthrough on failure
        return inputs

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Any:
        if kwargs.get("stream", False):
            return self.generate_stream(prompt, max_new_tokens, **kwargs)

        if self.num_workers == 0:
            raise RuntimeError(
                "WebGPU: No browser workers connected. "
                "Open dashboard/worker/index.html in a WebGPU-capable browser."
            )

        self.log.info(
            f"generate() called with {self.num_workers} workers "
            f"(max_new_tokens={max_new_tokens})"
        )

        # This backend handles individual matmul ops dispatched by the
        # inference pipeline (via infer/matmul_sync). Full autoregressive
        # generation requires the pipeline to drive token-by-token and
        # route each layer's matmul through us.
        #
        # Standalone generate is a stub — the real path is:
        #   InferencePipeline.generate() -> per-layer forward -> backend.infer()
        return f"[WebGPU backend: {self.num_workers} workers ready, use via InferencePipeline]"

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        yield self.generate(prompt, max_new_tokens, **kwargs)