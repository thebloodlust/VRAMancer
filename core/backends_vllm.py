"""vLLM backend integration for VRAMancer.

Extracted from core/backends.py for maintainability.
Uses the highly optimized AsyncLLMEngine for continuous batching,
bypassing the Python GIL for heavy concurrent workloads.
"""

import os
import uuid
import asyncio
from typing import Any, List, Optional

from core.backends import BaseLLMBackend, _HAS_TORCH
from core.logger import LoggerAdapter

if _HAS_TORCH:
    import torch as _torch
else:
    _torch = None  # type: ignore


class vLLMBackend(BaseLLMBackend):
    """vLLM backend with true Async continuous batching.

    Uses AsyncLLMEngine to allow the Python event loop to fly while
    the C++/CUDA backend handles real batching in the background.
    """
    def __init__(self, real: bool = True):
        self.engine = None
        self.model_name: Optional[str] = None
        self.log = LoggerAdapter("backend.vllm" + (".stub" if not real else ""))
        self.real = real

    def load_model(self, model_name: str, **kwargs):
        self.model_name = model_name
        if not self.real:
            self.engine = {"name": model_name, "stub": True}
            return self.engine
        try:
            from vllm.engine.arg_utils import EngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            
            # Extract relevant EngineArgs from kwargs to build native engine
            engine_args = EngineArgs(model=model_name, **kwargs)
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            return self.engine
        except Exception as e:
            if os.environ.get("VRM_BACKEND_ALLOW_STUB"):
                self.log.warning(f"Fallback stub vLLM: {e}")
                self.real = False
                self.engine = {"name": model_name, "stub": True}
                return self.engine
            raise

    def split_model(self, num_gpus: int, vram_per_gpu: List[int] = None):
        if self.engine is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        # vLLM handles tensor parallelism internally (via EngineArgs.tensor_parallel_size)
        return [self.engine]

    async def infer_async(self, inputs: Any):
        if not self.real:
            return f"[stub-vllm] {inputs[:50]}" if isinstance(inputs, str) else inputs
            
        if isinstance(inputs, str):
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=64)
            req_id = uuid.uuid4().hex
            
            # Asynchronous unblocked dispatch
            results_generator = self.engine.generate(inputs, params, req_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            if final_output and final_output.outputs:
                return final_output.outputs[0].text
        return inputs

    def infer(self, inputs: Any):
        # Sync wrapper around the async native call
        if getattr(self, "engine", None) is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Cannot block if loop is already running, return awaitable
                return self.infer_async(inputs)
            return loop.run_until_complete(self.infer_async(inputs))
        except Exception as e:
            self.log.warning(f"Infer vLLM échec: {e}")
            return ""

    async def generate_async(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Native async generate utilizing continuous batching."""
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', -1),
        )
        req_id = uuid.uuid4().hex
        results_generator = self.engine.generate(prompt, params, req_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        if final_output and final_output.outputs:
            return final_output.outputs[0].text
        return ""

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        if self.engine is None:
            raise RuntimeError("Modèle vLLM non chargé.")
        if not self.real:
            return f"[stub-vllm] {prompt[:50]}..."
        try:
            loop = asyncio.get_event_loop()
            # If standard sync path
            if not loop.is_running():
                return loop.run_until_complete(self.generate_async(prompt, max_new_tokens, **kwargs))
            else:
                self.log.error("Sync generate() called from within a running asyncio loop. Use async API directly.")
                raise RuntimeError("Blocking call in async loop")
        except Exception as e:
            self.log.error(f"vLLM native generate failed: {e}")
            raise

    async def generate_stream_async(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Native streaming yielding tokens directly from Cuda core via Async Engine."""
        from vllm import SamplingParams
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 1.0),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', -1),
        )
        req_id = uuid.uuid4().hex
        results_generator = self.engine.generate(prompt, params, req_id)
        
        previous_text = ""
        async for request_output in results_generator:
            for out in request_output.outputs:
                text = out.text
                new_text = text[len(previous_text):]
                if new_text:
                    yield new_text
                previous_text = text

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Sync wrapper fallback. In prod, call generate_stream_async natively."""
        if not self.real:
            text = f"[stub-vllm] {prompt[:50]}..."
            for word in text.split(' '):
                yield word + ' '
            return

        # VRAMancer Async bridging via threading
        import threading
        import queue
        q = queue.Queue()
        
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_stream():
                async for token in self.generate_stream_async(prompt, max_new_tokens, **kwargs):
                    q.put(token)
                q.put(None) # EOF
                
            loop.run_until_complete(run_stream())
            loop.close()

        threading.Thread(target=run_loop, daemon=True).start()
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk
