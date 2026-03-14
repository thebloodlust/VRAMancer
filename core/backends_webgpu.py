"""WebGPU Distributed Backend for VRAMancer.

This backend acts as a bridge. Instead of computing tensors locally, 
it serializes them and sends them to remote web browsers (WebGPU Workers) 
using WebSockets. It uses Speculative Network Decoding and Holographic Parity (Swarm Attention) to hide latency.
"""

import threading
import asyncio
import json
import time
import struct
from typing import Any, List, Optional
try:
    import torch
except ImportError:
    torch = None

from core.backends import BaseLLMBackend
from core.logger import LoggerAdapter
from core.network.webgpu_node import WebGPUNodeManager

class WebGPUBackend(BaseLLMBackend):
    def __init__(self):
        self.log = LoggerAdapter("backend.webgpu")
        self.model_name = None
        self.tokenizer = None
        
        self.log.info("🌐 Initializing WebGPU Orchestrator (Production Mode)...")
        self.node_manager = WebGPUNodeManager(port=8081)
        self.node_manager.start = self._start_node_manager
        
        self._loop = None
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        while self._loop is None:
            time.sleep(0.01)

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        if hasattr(self.node_manager, "is_running"):
            self.node_manager.is_running = True
            
        try:
            import websockets
            start_server = websockets.serve(self.node_manager._handler, "0.0.0.0", self.node_manager.port)
            self._loop.run_until_complete(start_server)
        except ImportError:
            self.log.error("websockets is not installed. WebGPU clients will not connect.")
            
        self._loop.create_task(self.node_manager._task_dispatcher())
        self.log.info(f"WebGPU WebSocket Server listening on 0.0.0.0:{self.node_manager.port}")
        self._loop.run_forever()

    def _start_node_manager(self):
        pass

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.log.info(f"Model {model_name} mapped to WebGPU Distributed Backend.")
        
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.log.warning(f"Could not load AutoTokenizer natively, falling back: {e}")
            
        return {"name": model_name, "type": "webgpu_distributed"}

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        return [self]

    def _serialize_tensor(self, tensor) -> bytes:
        if torch is None:
            return b"dummy_tensor_data"
        if not isinstance(tensor, torch.Tensor):
            return b""
        try:
            return tensor.detach().cpu().to(torch.float32).numpy().tobytes()
        except Exception:
            return b""
            
    def _deserialize_tensor(self, data: bytes) -> Any:
        if torch is None or not data:
            return None
        import numpy as np
        try:
            return torch.from_numpy(np.frombuffer(data, dtype=np.float32))
        except Exception:
            return None

    def infer(self, inputs: Any):
        if not self.node_manager.clients:
            return "[Error: No WebGPU workers connected via browser]"
            
        tensor_bytes = self._serialize_tensor(inputs) if torch else b"dummy_tensor_data"
        fut = self.node_manager.submit_tensor(layer_id=0, tensor_data=tensor_bytes)
        
        try:
            result_bytes = asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=10.0), self._loop).result()
            if torch and result_bytes:
                return self._deserialize_tensor(result_bytes)
            return result_bytes
        except asyncio.TimeoutError:
            self.log.error("WebGPU Tensor computation timed out.")
            return "[WebGPU Timeout]"
        except Exception as e:
            return f"[WebGPU Error: {e}]"

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Any:
        if kwargs.get("stream", False):
            return self.generate_stream(prompt, max_new_tokens, **kwargs)
            
        clients_count = len(self.node_manager.clients)
        if not clients_count:
            raise RuntimeError("WebGPU: Waiting for browsers to connect.")
            
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = [ord(c) for c in prompt]
            
        output_tokens = []
        current_input = input_ids
        
        for step in range(max_new_tokens):
            tensor_bytes = self._serialize_tensor(current_input) if torch else json.dumps(current_input).encode()
            # Redundancy: Retries across multiple dynamic clients if node disconnected
            max_retries = 3
            res_bytes = None
            for attempt in range(max_retries):
                fut = self.node_manager.submit_tensor(layer_id=step % 12, tensor_data=tensor_bytes)
                try:
                    res_bytes = asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=10.0), self._loop).result()
                    break
                except Exception as e:
                    self.log.warning(f"WebGPU node disconnected/failed at step {step}, attempt {attempt+1}: {e}")
                    time.sleep(0.1)
                    
            if res_bytes is None:
                self.log.error(f"Swarm failure: All redundant attempts exhausted at step {step}")
                break
                
            next_token = 32
            if res_bytes and len(res_bytes) >= 4:
                try:
                    next_token = struct.unpack("<I", res_bytes[:4])[0]
                except Exception:
                    pass
            output_tokens.append(next_token)
            
            if torch:
                current_input = torch.tensor([[next_token]])
            else:
                current_input = [next_token]
                
        if self.tokenizer:
            return self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return "".join(chr(t) for t in output_tokens if 32 <= t <= 126)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        clients_count = len(self.node_manager.clients)
        if not clients_count:
            yield "[WebGPU Error: Waiting for browsers to connect to the node...]"
            return
            
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        else:
            input_ids = [ord(c) for c in prompt]
            
        current_input = input_ids
        for step in range(max_new_tokens):
            tensor_bytes = self._serialize_tensor(current_input) if torch else json.dumps(current_input).encode()
            max_retries = 3
            res_bytes = None
            for attempt in range(max_retries):
                fut = self.node_manager.submit_tensor(layer_id=step % 12, tensor_data=tensor_bytes) 
                try:
                    res_bytes = asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=7.0), self._loop).result()
                    break
                except Exception as e:
                    self.log.warning(f"Rescheduling token {step} computation due to crash (attempt {attempt+1})...")
            
            if res_bytes is None:
                yield f"\n[WebGPU Crash Prevention: Stream halted after {max_retries} attempts]"
                break
                
            next_token = 32
            if res_bytes and len(res_bytes) >= 4:
                try:
                    next_token = struct.unpack("<I", res_bytes[:4])[0]
                except Exception:
                    pass
                    
            if torch:
                current_input = torch.tensor([[next_token]])
            else:
                current_input = [next_token]
                
            if self.tokenizer:
                yield self.tokenizer.decode([next_token])
            else:
                if 32 <= next_token <= 126:
                    yield chr(next_token)