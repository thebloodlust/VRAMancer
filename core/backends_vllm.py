import logging
from core.backends import BaseLLMBackend

logger = logging.getLogger(__name__)

class vLLMBackend(BaseLLMBackend):
    def __init__(self, model_name: str, cache_dir: str = None, 
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1):
        super().__init__(model_name, cache_dir)
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.engine = None
        self.backend_type = "vllm"

    def load(self, **kwargs):
        try:
            from vllm import LLMEngine, EngineArgs
        except ImportError:
            logger.error("vLLM n'est pas installé. Lancez: pip install vllm")
            raise ImportError("vLLM not found")

        logger.info(f"Initialisation de vLLM (TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}) pour {self.model_name}")
        
        gpu_utilization = kwargs.get("gpu_memory_utilization", 0.90)
        
        engine_args = EngineArgs(
            model=self.model_name,
            download_dir=self.cache_dir,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_utilization,
            enforce_eager=kwargs.get("enforce_eager", False)
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM Engine prêt.")
        self.is_loaded = True

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7, **kwargs):
        if not self.is_loaded or self.engine is None:
            raise RuntimeError("Le moteur vLLM n'est pas initialisé.")
            
        from vllm import SamplingParams
        import uuid
        
        request_id = str(uuid.uuid4())
        
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty']}
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **valid_kwargs
        )
        
        logger.debug(f"[vLLM] Ajout de la requête {request_id}")
        self.engine.add_request(request_id, prompt, sampling_params)
        
        final_output = ""
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.request_id == request_id:
                    final_output = output.outputs[0].text
                    
        return final_output