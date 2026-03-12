"""Backend TensorRT pour VRAMancer.

Fournit l'integration avec TensorRT / TensorRT-LLM pour l'inference
GPU optimisee avec compilation de graphes.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Optional

try:
    import tensorrt as trt  # type: ignore
    _HAS_TENSORRT = True
except ImportError:
    trt = None  # type: ignore
    _HAS_TENSORRT = False

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

_log = logging.getLogger("vramancer.backends_tensorrt")


def is_tensorrt_available() -> bool:
    return _HAS_TENSORRT


def build_engine(onnx_path: str, fp16: bool = True,
                  max_batch_size: int = 8,
                  workspace_mb: int = 2048) -> Any:
    """Construit un engine TensorRT a partir d'un modele ONNX.

    Parameters
    ----------
    onnx_path : str
        Chemin vers le fichier ONNX.
    fp16 : bool
        Activer l'inference FP16.
    max_batch_size : int
        Taille max du batch.
    workspace_mb : int
        Memoire workspace en MB.

    Returns
    -------
    trt.ICudaEngine
    """
    if not _HAS_TENSORRT:
        raise ImportError("tensorrt requis : pip install tensorrt")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Fichier ONNX introuvable : {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                _log.error("TensorRT parse error: %s", parser.get_error(i))
            raise RuntimeError("Echec parsing ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    _log.info("Engine TensorRT construit (%s, fp16=%s)", onnx_path, fp16)
    return engine


def run_tensorrt_inference(engine: Any, inputs: Any, output_shape: tuple = None) -> Any:
    """Intégration Zero-Copy C++ TensorRT via pointeurs CUDA (torch.data_ptr)."""
    if not _HAS_TENSORRT or torch is None:
        raise ImportError("tensorrt et pytorch requis pour l'inference native.")
    
    # Creation d'un contexte d'execution CUDA pur
    context = engine.create_execution_context()
    
    # Convert input to contiguous CUDA tensor to extract Raw Data Pointers
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, device='cuda', dtype=torch.float16)
    elif not inputs.is_cuda:
        inputs = inputs.cuda()
    inputs = inputs.contiguous()

    # Pre-allocation Tensor Out pour Zero-Copy (suppose FP16 output)
    if output_shape is None:
        output_shape = inputs.shape  # heuristic for stub
        
    outputs = torch.empty(output_shape, device='cuda', dtype=inputs.dtype)

    # Resolution des bindings natifs GPU
    bindings = [None] * engine.num_bindings
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            bindings[i] = inputs.data_ptr()
        else:
            bindings[i] = outputs.data_ptr()
            
    # Lancement asynchrone VRAM vers VRAM sans passer par la RAM CPU (GIL Bypassed)
    stream = torch.cuda.current_stream()
    try:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    except Exception as e:
        _log.error(f"Erreur fatale TensorRT asynchrone: {e}")
        raise RuntimeError(f"Echec execute_async_v2: {e}")
    finally:
        # Hardening: Toujours synchroniser le stream pour éviter un Deadlock/Segfault
        # si un nœud plante avant la lecture de la VRAM
        stream.synchronize()
        del context  # Nettoyage immédiat du pointeur C++
    
    _log.debug("Inference TensorRT (VRAM Zero-Copy) terminée.")
    return outputs
