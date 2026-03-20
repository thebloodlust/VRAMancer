import os
import logging
from typing import Any, List, Optional
from core.backends import BaseLLMBackend

logger = logging.getLogger(__name__)

class vLLMBackend(BaseLLMBackend):
    def __init__(self, model_name: str, cache_dir: str = None, 
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
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
        
        gpu_utilization = float(kwargs.get("gpu_memory_utilization", 0.90))
        max_model_len = int(kwargs.get("max_model_len", 8192)) # Force a smaller context length to save KV cache (default would try 32k for Qwen)
        
        logger.info(f"Paramètres vLLM: gpu_memory_utilization={gpu_utilization}, max_model_len={max_model_len}")
        
        engine_args = EngineArgs(
            model=self.model_name,
            download_dir=self.cache_dir,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=max_model_len,
            enforce_eager=kwargs.get("enforce_eager", False)
        )
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
                    logger.warning("[vLLM] OOM detected, clearing cache and retrying...")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    # Re-submit with halved max_tokens
                    request_id_retry = str(uuid.uuid4())
                    retry_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max(1, max_tokens_val // 2),
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
                    logger.warning("[vLLM Stream] OOM detected, clearing cache and retrying...")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    request_id_retry = str(uuid.uuid4())
                    retry_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max(1, max_tokens_val // 2),
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