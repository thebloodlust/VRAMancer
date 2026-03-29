import os
import logging
from typing import Any, List, Optional
from core.backends import BaseLLMBackend

logger = logging.getLogger(__name__)

class vLLMBackend(BaseLLMBackend):
    def __init__(self, model_name: str, cache_dir: str = None, 
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90, dtype_str: str = None,
                 target_gpu: int = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype_str = dtype_str  # "bfloat16", "float16", or None (auto)
        self.target_gpu = target_gpu  # Pin to specific GPU for TP=1 hetero setups
        self.engine = None
        self.backend_type = "vllm"
        self.is_loaded = False

    def load_model(self, model_name: str, **kwargs) -> Any:
        self.model_name = model_name
        try:
            from vllm import LLMEngine, EngineArgs
        except ImportError:
            if os.environ.get("VRM_MINIMAL_TEST") == "1":
                self.engine = "STUB_ENGINE"
                self.is_loaded = True
                return self.engine
            logger.error("vLLM n'est pas installé. Lancez: pip install vllm")
            raise ImportError("vLLM not found")

        logger.info(f"Initialisation de vLLM (TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}) pour {self.model_name}")

        # Pin to target GPU when TP=1 on a heterogeneous multi-GPU system
        if self.target_gpu is not None and self.tensor_parallel_size == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.target_gpu)
            logger.info(f"vLLM pinned to GPU {self.target_gpu} via CUDA_VISIBLE_DEVICES")
        
        gpu_utilization = float(kwargs.get("gpu_memory_utilization", self.gpu_memory_utilization))
        max_model_len = int(kwargs.get("max_model_len", 8192))
        dtype = kwargs.get("dtype", self.dtype_str)
        
        logger.info(f"Paramètres vLLM: gpu_memory_utilization={gpu_utilization}, max_model_len={max_model_len}, dtype={dtype}")
        
        engine_kwargs = dict(
            model=self.model_name,
            download_dir=self.cache_dir,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=max_model_len,
            enforce_eager=kwargs.get("enforce_eager", False),
        )
        if dtype:
            engine_kwargs["dtype"] = dtype
        
        engine_args = EngineArgs(**engine_kwargs)
        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM Engine prêt.")
        self.is_loaded = True
        return self.engine

    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None) -> List[Any]:
        # vLLM gère le multi-GPU en interne, on retourne un dummy block pour VRAMancer
        logger.info(f"Découpage VRAMancer contourné, vLLM gère les {num_gpus} GPUs.")
        return [self.engine]

    def infer(self, inputs: Any) -> Any:
        if not self.is_loaded:
            raise RuntimeError("modèle non chargé")
        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            return "vllm_infer_stub"
        raise NotImplementedError("L'inférence par tenseur brut n'est pas supportée par vLLM directement.")

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> Any:
        # Extractions sécurisées (sans détruire l'objet d'origine via pop)
        max_tokens_val = int(kwargs.get('max_tokens', max_new_tokens))
        
        if kwargs.get('stream', False):
            # Passage direct
            return self.generate_stream(prompt, max_new_tokens, **kwargs)

        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            return "vllm_stub_text"
        if not self.is_loaded or self.engine is None:
            raise RuntimeError("Le moteur vLLM n'est pas initialisé.")
            
        from vllm import SamplingParams
        import uuid
        
        request_id = str(uuid.uuid4())
        
        t = kwargs.get('temperature')
        temperature = float(t) if t is not None else 0.7
        
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty'] and v is not None}
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_val,
            **valid_kwargs
        )
        
        logger.debug(f"[vLLM] Ajout de la requête {request_id}")
        self.engine.add_request(request_id, prompt, sampling_params)
        
        final_output = ""
        oom_retried = False
        while self.engine.has_unfinished_requests():
            try:
                step_outputs = self.engine.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and not oom_retried:
                    oom_retried = True
                    logger.warning("[vLLM] OOM detected, clearing cache and retrying with reduced gpu_memory_utilization...")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Reduce GPU memory utilization for the engine (the real OOM cause)
                    new_util = max(0.50, self.gpu_memory_utilization - 0.10)
                    logger.info("[vLLM] Reducing gpu_memory_utilization: %.2f -> %.2f", self.gpu_memory_utilization, new_util)
                    self.gpu_memory_utilization = new_util
                    # Re-submit same request with same params (memory pressure is what matters)
                    request_id_retry = str(uuid.uuid4())
                    retry_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens_val,
                        **valid_kwargs
                    )
                    self.engine.add_request(request_id_retry, prompt, retry_params)
                    request_id = request_id_retry
                    continue
                raise RuntimeError(f"vLLM OOM unrecoverable: {e}") from e
            for output in step_outputs:
                if output.request_id == request_id:
                    final_output = output.outputs[0].text
                    
        return final_output

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            yield "vllm_"
            yield "stream_"
            yield "stub"
            return
        if not self.is_loaded or self.engine is None:
            raise RuntimeError("Le moteur vLLM n'est pas initialisé.")
            
        from vllm import SamplingParams
        import uuid
        
        # Extractions sécurisées (.get) pour protéger la route VRAMancer
        max_tokens_val = int(kwargs.get('max_tokens', max_new_tokens))
        t = kwargs.get('temperature')
        temperature = float(t) if t is not None else 0.7
        
        request_id = str(uuid.uuid4())
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty'] and v is not None}
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_val,
            **valid_kwargs
        )
        
        logger.debug(f"[vLLM Stream] Ajout de la requête {request_id}")
        self.engine.add_request(request_id, prompt, sampling_params)
        
        last_text = ""
        oom_retried = False
        while self.engine.has_unfinished_requests():
            try:
                step_outputs = self.engine.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and not oom_retried:
                    oom_retried = True
                    logger.warning("[vLLM Stream] OOM detected, clearing cache and retrying with reduced gpu_memory_utilization...")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    new_util = max(0.50, self.gpu_memory_utilization - 0.10)
                    logger.info("[vLLM Stream] Reducing gpu_memory_utilization: %.2f -> %.2f", self.gpu_memory_utilization, new_util)
                    self.gpu_memory_utilization = new_util
                    request_id_retry = str(uuid.uuid4())
                    retry_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens_val,
                        **valid_kwargs
                    )
                    self.engine.add_request(request_id_retry, prompt, retry_params)
                    request_id = request_id_retry
                    last_text = ""
                    continue
                raise RuntimeError(f"vLLM Stream OOM unrecoverable: {e}") from e
            for output in step_outputs:
                if output.request_id == request_id:
                    current_text = output.outputs[0].text
                    if current_text:
                        new_text = current_text[len(last_text):]
                        if new_text:
                            yield new_text
                            last_text = current_text

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        """Process multiple prompts via vLLM's native multi-request engine.

        Submits all prompts as separate requests and steps the engine
        until all are complete — leveraging vLLM's continuous batching.
        """
        if os.environ.get("VRM_MINIMAL_TEST") == "1":
            return [f"vllm_batch_stub_{i}" for i in range(len(prompts))]
        if not self.is_loaded or self.engine is None:
            raise RuntimeError("Le moteur vLLM n'est pas initialisé.")

        from vllm import SamplingParams
        import uuid

        max_tokens_val = int(kwargs.get('max_tokens', max_new_tokens))
        t = kwargs.get('temperature')
        temperature = float(t) if t is not None else 0.7
        valid_kwargs = {k: v for k, v in kwargs.items()
                        if k in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty']
                        and v is not None}
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens_val,
            **valid_kwargs,
        )

        # Submit all prompts
        request_ids = []
        for prompt in prompts:
            rid = str(uuid.uuid4())
            request_ids.append(rid)
            self.engine.add_request(rid, prompt, sampling_params)

        # Step until all done
        results: dict[str, str] = {}
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.request_id in request_ids:
                    results[output.request_id] = output.outputs[0].text

        return [results.get(rid, "") for rid in request_ids]