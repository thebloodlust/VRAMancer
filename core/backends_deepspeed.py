"""Backend DeepSpeed pour VRAMancer.

Fournit l'integration avec DeepSpeed-Inference pour l'inference
optimisee (tensor/pipeline parallelism, quantization automatique).
"""
from __future__ import annotations
import logging
from typing import Any

try:
    import deepspeed  # type: ignore
    _HAS_DEEPSPEED = True
except ImportError:
    deepspeed = None  # type: ignore
    _HAS_DEEPSPEED = False

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

_log = logging.getLogger("vramancer.backends_deepspeed")


def is_deepspeed_available() -> bool:
    return _HAS_DEEPSPEED


def init_deepspeed_inference(model: Any, mp_size: int = 1,
                              dtype: Any = None, replace_with_kernel_inject: bool = True,
                              **kwargs) -> Any:
    """Initialise un modele avec DeepSpeed-Inference.

    Parameters
    ----------
    model : torch.nn.Module
        Modele HuggingFace ou PyTorch a optimiser.
    mp_size : int
        Degre de model parallelism (nombre de GPUs).
    dtype : torch.dtype | None
        Type de donnees (torch.float16, torch.bfloat16, etc.). None = auto.
    replace_with_kernel_inject : bool
        Utiliser les kernels optimises DeepSpeed.

    Returns
    -------
    Le modele wrappe par DeepSpeed.
    """
    if not _HAS_DEEPSPEED:
        raise ImportError("deepspeed requis : pip install deepspeed")
    if dtype is None and torch is not None:
        dtype = torch.float16

    ds_config = {
        "tensor_parallel": {"tp_size": mp_size},
        "dtype": dtype,
        "replace_with_kernel_inject": replace_with_kernel_inject,
    }
    ds_config.update(kwargs)

    engine = deepspeed.init_inference(model, **ds_config)
    _log.info("DeepSpeed-Inference initialise (mp_size=%d, dtype=%s)", mp_size, dtype)
    return engine


def run_deepspeed_inference(model: Any, inputs: Any, **kwargs) -> Any:
    """Execute une inference via un modele DeepSpeed."""
    if not _HAS_DEEPSPEED:
        raise ImportError("deepspeed requis : pip install deepspeed")
    return model(inputs, **kwargs)
