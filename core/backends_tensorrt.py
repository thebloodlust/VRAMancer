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


def run_tensorrt_inference(engine: Any, inputs: Any) -> Any:
    """Execute une inference via un engine TensorRT."""
    if not _HAS_TENSORRT:
        raise ImportError("tensorrt requis : pip install tensorrt")
    # Simplified — real impl uses ExecutionContext + buffer allocation
    _log.info("Inference TensorRT en cours...")
    return engine
