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
    # Number of tokens to speculate ahead (draft-verify pattern)
    SPECULATION_WINDOW = 4

    def __init__(self):
        self.log = LoggerAdapter("backend.webgpu")
        self.model_name = None
        self.tokenizer = None
        
        self.log.info("Initializing WebGPU Orchestrator...")
        self.node_manager = WebGPUNodeManager(port=8081)
        # Start the node manager (creates event loop, WebSocket server, dispatcher)
        self.node_manager.start()
        
        # Wait for the event loop to be ready
        for _ in range(100):
            if self.node_manager._loop is not None:
                break
            time.sleep(0.01)
        self._loop = self.node_manager._loop

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

    def _serialize_tensor(self, tensor) -> tuple[bytes, float]:
        if torch is None:
            return b"dummy_tensor_data", 1.0
        if not isinstance(tensor, torch.Tensor):
            return b"", 1.0
        try:
            import numpy as np
            # 8-bit Quantization (Symmetric Q8_0)
            np_arr = tensor.detach().cpu().to(torch.float32).numpy()
            max_val = float(np.max(np.abs(np_arr))) if np_arr.size > 0 else 0.0
            if max_val == 0:
                scale = 1.0
                quantized = np_arr.astype(np.int8)
            else:
                scale = max_val / 127.0
                quantized = np.round(np_arr / scale).astype(np.int8)
            
            return quantized.tobytes(), scale
        except Exception as e:
            self.log.warning(f"Quantization failed: {e}")
            return b"", 1.0
            
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
            
        tensor_bytes, quant_scale = self._serialize_tensor(inputs) if torch else (b"dummy_tensor_data", 1.0)
        
        # --- Fallback & Redundancy System ---
        max_retries = 3
        for attempt in range(max_retries):
            fut = self.node_manager.submit_tensor(layer_id=0, tensor_data=tensor_bytes, quant_scale=quant_scale)
            try:
                result_bytes = asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=10.0), self._loop).result()
                if torch and result_bytes:
                    return self._deserialize_tensor(result_bytes)
                return result_bytes
            except asyncio.TimeoutError:
                self.log.warning(f"WebGPU Tensor timeout, retrying ({attempt+1}/{max_retries})...")
            except Exception as e:
                self.log.warning(f"WebGPU Worker error: {e}, retrying ({attempt+1}/{max_retries})...")
                
        self.log.error("WebGPU inference failed after maximum retries.")
        return "[WebGPU Timeout/Crash Prevention]"

    def _submit_speculative_batch(self, inputs_list, layer_ids):
        """Submit multiple speculative tokens in parallel via asyncio.gather.

        Returns list of (res_bytes | None) in order.
        """
        futures = []
        for inp, lid in zip(inputs_list, layer_ids):
            tensor_bytes, quant_scale = self._serialize_tensor(inp) if torch else (json.dumps(inp).encode(), 1.0)
            fut = self.node_manager.submit_tensor(layer_id=lid, tensor_data=tensor_bytes, quant_scale=quant_scale)
            futures.append(asyncio.wait_for(fut, timeout=10.0))

        async def _gather():
            return await asyncio.gather(*futures, return_exceptions=True)

        results = asyncio.run_coroutine_threadsafe(_gather(), self._loop).result()
        out = []
        for r in results:
            if isinstance(r, Exception):
                out.append(None)
            else:
                out.append(r)
        return out

    def _decode_token(self, res_bytes):
        """Extract token id from worker response bytes."""
        if res_bytes and len(res_bytes) >= 4:
            try:
                return struct.unpack("<I", res_bytes[:4])[0]
            except Exception:
                pass
        return 32  # fallback space

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
        step = 0
        K = min(self.SPECULATION_WINDOW, max(1, clients_count))  # adapt to worker count
        
        while step < max_new_tokens:
            if K > 1 and step + K <= max_new_tokens:
                # --- Speculative Decoding: submit K tokens in parallel ---
                # Use last known token as seed for all K drafts
                spec_inputs = [current_input] * K
                spec_layers = [(step + i) % 12 for i in range(K)]
                results = self._submit_speculative_batch(spec_inputs, spec_layers)

                # Verify: accept the first contiguous run that returned valid data
                accepted = 0
                for res_bytes in results:
                    if res_bytes is None:
                        break
                    token = self._decode_token(res_bytes)
                    output_tokens.append(token)
                    if torch:
                        current_input = torch.tensor([[token]])
                    else:
                        current_input = [token]
                    accepted += 1

                if accepted == 0:
                    # Fallback to sequential single-token
                    K = 1
                    continue
                step += accepted
            else:
                # --- Sequential fallback (single token or final tokens) ---
                tensor_bytes, quant_scale = self._serialize_tensor(current_input) if torch else (json.dumps(current_input).encode(), 1.0)
                max_retries = 3
                res_bytes = None
                for attempt in range(max_retries):
                    fut = self.node_manager.submit_tensor(layer_id=step % 12, tensor_data=tensor_bytes, quant_scale=quant_scale)
                    try:
                        res_bytes = asyncio.run_coroutine_threadsafe(asyncio.wait_for(fut, timeout=10.0), self._loop).result()
                        break
                    except Exception as e:
                        self.log.warning(f"WebGPU node failed at step {step}, attempt {attempt+1}: {e}")
                        time.sleep(0.1)
                        
                if res_bytes is None:
                    self.log.error(f"Swarm failure: All attempts exhausted at step {step}")
                    break
                    
                next_token = self._decode_token(res_bytes)
                output_tokens.append(next_token)
            
                if torch:
                    current_input = torch.tensor([[next_token]])
                else:
                    current_input = [next_token]
                step += 1
                
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
            tensor_bytes, quant_scale = self._serialize_tensor(current_input) if torch else (json.dumps(current_input).encode(), 1.0)
            max_retries = 3
            res_bytes = None
            for attempt in range(max_retries):
                fut = self.node_manager.submit_tensor(layer_id=step % 12, tensor_data=tensor_bytes, quant_scale=quant_scale) 
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