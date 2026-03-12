import re

try:
    with open('core/backends.py', 'r', encoding='utf-8') as f:
        content = f.read()

    start_token = "# ------------------- HuggingFace Backend -------------------"
    end_token = "# ------------------- vLLM Backend -------------------"

    start_idx = content.find(start_token)
    end_idx = content.find(end_token)

    if start_idx == -1 or end_idx == -1:
        print("Erreur : impossible de trouver les balises dans core/backends.py")
        exit(1)

    new_hf_class = """# ------------------- HuggingFace Backend -------------------
class HuggingFaceBackend(BaseLLMBackend):
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.tokenizer = None
        self.blocks: Optional[List[Any]] = None
        self.block_devices: Optional[List[int]] = None
        self.log = LoggerAdapter("backend.hf")
        self.hmem = None
        self.transfer_manager = None
        self._components: Optional[dict] = None

    def load_model(self, model_name: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.log.info(f"Chargement modèle HuggingFace: {model_name}")
        if model_name.endswith("-AWQ"):
            kwargs["low_cpu_mem_usage"] = True
        else:
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.log.warning(f"Tokenizer load failed: {e}")
        return self.model

    def split_model(self, num_gpus: int, vram_per_gpu: Optional[List[int]] = None):
        # On délègue totalement le multi-GPU à accelerate via device_map="auto"
        if self.model is None:
            raise RuntimeError("Modèle non chargé — appeler load_model() d'abord")
        self.blocks = [self.model]
        self.block_devices = [0]
        self.log.info("HuggingFace accelerate device_map gère le multi-GPU. Pas de découpage manuel.")
        return self.blocks

    def infer(self, inputs: Any, **kwargs) -> Any:
        if self.model is None:
            raise RuntimeError("Modèle non chargé")
        
        # S'assurer que les inputs sont sur le même device que la première couche
        first_device = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(first_device)
            
        out = self.model(inputs, **kwargs)
        if hasattr(out, "logits"):
            return out.logits
        return out

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modèle ou Tokenizer non chargé")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        first_device = next(self.model.parameters()).device
        input_ids = input_ids.to(first_device)

        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        
        # Isoler les nouveaux tokens générés
        new_tokens = out_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_stream(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modèle ou Tokenizer non chargé")

        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt")
        first_device = next(self.model.parameters()).device
        inputs = inputs.to(first_device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 128, **kwargs) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modèle ou Tokenizer non chargé")
        if not prompts:
            return []

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        first_device = next(self.model.parameters()).device
        inputs = inputs.to(first_device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        results = []
        for i in range(len(prompts)):
            new_tokens = out_ids[i][inputs["input_ids"].shape[1]:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results

"""

    new_content = content[:start_idx] + new_hf_class + "\n" + content[end_idx:]

    with open('core/backends.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("Patch appliqué avec succès à core/backends.py ! Le backend HuggingFace utilise désormais nativement accelerate.")
except Exception as e:
    print(f"Erreur : {e}")
