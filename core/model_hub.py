"""
Interface pour rechercher les modèles et leurs quantizations sur HuggingFace Hub.
"""
from typing import Dict, List, Any
import logging
from core.logger import get_logger

log = get_logger("model_hub")

def search_huggingface_model(repo_id: str) -> Dict[str, Any]:
    """Recherche un modèle sur HuggingFace Hub et détermine les types de pondération disponibles.
    
    Args:
        repo_id: Identifiant du repository (ex: 'meta-llama/Llama-2-7b-hf')
        
    Returns:
        Dict retournant les informations du modèle et les formats trouvés.
    """
    try:
        from huggingface_hub import HfApi, ModelInfo
        api = HfApi()
        
        info: ModelInfo = api.model_info(repo_id, files_metadata=True)
        
        # Analyser les fichiers pour déterminer les formats disponibles
        formats = set()
        precision = "fp16/bf16 (default)" # Assumption pour transformers/vLLM par defaut
        
        files = [f.rfilename for f in info.siblings] if info.siblings else []
        
        for fname in files:
            fname_lower = fname.lower()
            if ".gguf" in fname_lower:
                formats.add("GGUF")
            if "awq" in fname_lower or "awq.json" in fname_lower:
                formats.add("AWQ")
            if "gptq" in fname_lower or "quantize_config.json" in fname_lower:
                formats.add("GPTQ")
            if "int8" in fname_lower:
                formats.add("INT8")
            if "fp4" in fname_lower or "nvfp4" in fname_lower:
                formats.add("NVFP4")
            if "bnb" in fname_lower:
                formats.add("BitsAndBytes")
            if "safetensors" in fname_lower:
                formats.add("SafeTensors")
        
        tags = info.tags if getattr(info, "tags", None) else []
        for tag in tags:
            tag_l = tag.lower()
            if "awq" in tag_l: formats.add("AWQ")
            if "gptq" in tag_l: formats.add("GPTQ")
            if "gguf" in tag_l: formats.add("GGUF")
            if "marlin" in tag_l: formats.add("Marlin")
            if "nvfp4" in tag_l: formats.add("NVFP4")
            
        return {
            "id": info.id,
            "pipeline_tag": getattr(info, "pipeline_tag", "text-generation"),
            "downloads": getattr(info, "downloads", 0),
            "formats": sorted(list(formats)) if formats else ["PyTorch/SafeTensors (Default)"],
            "base_precision": precision
        }
    except Exception as e:
        log.error(f"Cannot fetch from HuggingFace Hub: {e}")
        return {
            "id": repo_id,
            "error": str(e)
        }
