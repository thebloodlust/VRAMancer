# core/model_splitter.py
"""Production model splitter — split HuggingFace models across GPUs.

Supports:
  - Any HuggingFace model (GPT, Llama, Mistral, Falcon, BERT, T5, etc.)
  - VRAM-proportional splitting based on FREE memory (not total)
  - CUDA, ROCm (via torch.cuda API), MPS, CPU
  - Defensive imports (works without torch/transformers installed)
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Any

_logger = logging.getLogger("vramancer.model_splitter")

STRICT = os.environ.get("VRM_STRICT_IMPORT", "0") in {"1", "true", "TRUE"}
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch  # type: ignore
    import torch.nn as nn
except ImportError:
    if STRICT:
        raise
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from transformers import AutoModel, AutoConfig  # type: ignore
except ImportError:
    if STRICT:
        raise
    AutoModel = None  # type: ignore
    AutoConfig = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None

try:
    from core.utils import detect_backend, get_device_type
except ImportError:  # pragma: no cover
    def detect_backend():
        return "cpu"
    def get_device_type(idx):
        return "cpu"

try:
    from core.layer_profiler import (
        LayerProfiler,
        compute_optimal_placement,
    )
    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False
    LayerProfiler = None


# ------------------------------------------------------------------
# Layer detection patterns for different model architectures
# ------------------------------------------------------------------
_LAYER_CANDIDATES = [
    ["transformer", "h"],           # GPT-2, GPT-J
    ["model", "layers"],            # Llama, Mistral, Falcon, Mixtral
    ["model", "decoder", "layers"], # Llama variants
    ["layers"],                     # Generic
    ["encoder", "layer"],           # BERT, T5 encoder
    ["decoder", "block"],           # T5 decoder
    ["block"],                      # T5
    ["h"],                          # GPT-Neo
]


def _extract_layers(model: Any) -> Optional[list]:
    """Extract transformer layers from a model using known patterns."""
    for path in _LAYER_CANDIDATES:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
                return list(obj)
        except (AttributeError, TypeError):
            continue
    return None


# ------------------------------------------------------------------
# VRAM detection (free memory, not total)
# ------------------------------------------------------------------

def _get_free_vram_per_gpu(num_gpus: int) -> List[int]:
    """Get free VRAM in MB for each GPU."""
    backend = detect_backend()

    # MPS: unified memory
    if backend == "mps":
        try:
            import psutil
            avail_mb = psutil.virtual_memory().available // (1024 * 1024)
            return [avail_mb]
        except Exception:
            return [8192]

    # pynvml for NVIDIA / some ROCm
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            result = []
            for i in range(num_gpus):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                result.append(int(mem.free // (1024 * 1024)))
            pynvml.nvmlShutdown()
            return result
        except Exception:
            pass

    # torch.cuda fallback
    if torch is not None and torch.cuda.is_available():
        result = []
        for i in range(num_gpus):
            try:
                total = torch.cuda.get_device_properties(i).total_mem
                allocated = torch.cuda.memory_allocated(i)
                free = total - allocated
                result.append(int(free // (1024 * 1024)))
            except Exception:
                result.append(8192)
        return result

    return [8192] * num_gpus


def _get_total_vram_per_gpu(num_gpus: int) -> List[int]:
    """Get total VRAM in MB for each GPU."""
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            result = []
            for i in range(num_gpus):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                result.append(int(mem.total // (1024 * 1024)))
            pynvml.nvmlShutdown()
            return result
        except Exception:
            pass

    if torch is not None and torch.cuda.is_available():
        result = []
        for i in range(num_gpus):
            try:
                result.append(torch.cuda.get_device_properties(i).total_mem // (1024 * 1024))
            except Exception:
                result.append(8192)
        return result

    return [8192] * num_gpus


# ------------------------------------------------------------------
# Model splitting
# ------------------------------------------------------------------

def split_model_into_blocks(model_name: str, num_gpus: int,
                            model_path: Optional[str] = None,
                            use_free_vram: bool = True,
                            use_profiler: bool = True) -> List[Any]:
    """Load a HuggingFace model and split it into blocks for multi-GPU.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path.
    num_gpus : int
        Number of GPUs to split across.
    model_path : str, optional
        Path to save the loaded model.
    use_free_vram : bool
        If True, split proportionally to FREE VRAM (not total).
    use_profiler : bool
        If True and LayerProfiler is available, use DP-optimal placement
        based on real latency/FLOPS measurements instead of VRAM ratios.

    Returns
    -------
    list of nn.Sequential
        Model blocks, one per GPU.
    """
    if torch is None or AutoModel is None:
        raise ImportError(
            "model_splitter requires torch and transformers. "
            "Install with: pip install torch transformers"
        )

    _logger.info("Loading model: %s", model_name)
    model = AutoModel.from_pretrained(model_name)

    if model_path:
        torch.save(model, model_path)

    layers = _extract_layers(model)
    if layers is None:
        raise ValueError(
            f"Cannot detect layers for model '{model_name}'. "
            f"Tried patterns: {['/'.join(p) for p in _LAYER_CANDIDATES]}"
        )

    n_layers = len(layers)
    _logger.info("Detected %d layers in %s", n_layers, model_name)

    backend = detect_backend()
    if num_gpus <= 1 or backend == "cpu":
        return [nn.Sequential(*layers)]

    # Try profiler-based optimal placement
    if use_profiler and _PROFILER_AVAILABLE and not _MINIMAL:
        try:
            return _split_by_profiler(model, layers, num_gpus)
        except Exception as exc:
            _logger.warning("Profiler-based split failed (%s), falling back to VRAM", exc)

    # Fallback: VRAM-proportional
    if use_free_vram:
        vram_per_gpu = _get_free_vram_per_gpu(num_gpus)
    else:
        vram_per_gpu = _get_total_vram_per_gpu(num_gpus)

    blocks = _split_by_vram(layers, vram_per_gpu)
    _logger.info("Split into %d blocks: %s",
                 len(blocks), [len(list(b.children())) for b in blocks])
    return blocks


def _split_by_profiler(model: Any, layers: list, num_gpus: int) -> List[Any]:
    """Split using LayerProfiler + DP-optimal placement."""
    profiler = LayerProfiler()

    _logger.info("Profiling %d layers for optimal placement...", len(layers))
    layer_profiles = profiler.profile_model(model)

    _logger.info("Benchmarking %d GPUs...", num_gpus)
    gpu_profiles = profiler.profile_gpus()[:num_gpus]

    plan = compute_optimal_placement(layer_profiles, gpu_profiles)

    _logger.info(
        "Optimal placement: estimated latency=%.1fms, transfer overhead=%.1fms",
        plan.estimated_latency_ms,
        plan.estimated_transfer_overhead_ms,
    )

    # Group contiguous layers assigned to the same GPU
    blocks = []
    current_gpu = None
    current_layers = []

    for layer_idx, gpu_idx in plan.assignments:
        if gpu_idx != current_gpu:
            if current_layers:
                blocks.append(nn.Sequential(*current_layers))
            current_layers = [layers[layer_idx]]
            current_gpu = gpu_idx
        else:
            current_layers.append(layers[layer_idx])

    if current_layers:
        blocks.append(nn.Sequential(*current_layers))

    _logger.info(
        "Profiler split into %d blocks: %s",
        len(blocks),
        [len(list(b.children())) for b in blocks],
    )
    return blocks


def _split_by_vram(layers: list, vram_per_gpu: List[int]) -> List[Any]:
    """Split layers proportionally to VRAM."""
    total_vram = sum(vram_per_gpu)
    n_layers = len(layers)

    if total_vram == 0:
        per_gpu = max(1, n_layers // len(vram_per_gpu))
        counts = [per_gpu] * len(vram_per_gpu)
    else:
        ratios = [v / total_vram for v in vram_per_gpu]
        counts = [max(1, round(r * n_layers)) for r in ratios]

    while sum(counts) < n_layers:
        idx = max(range(len(vram_per_gpu)), key=lambda i: vram_per_gpu[i])
        counts[idx] += 1
    while sum(counts) > n_layers:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] > 1:
            counts[idx] -= 1

    blocks = []
    start = 0
    for c in counts:
        block_layers = layers[start:start + c]
        if block_layers:
            blocks.append(nn.Sequential(*block_layers))
        start += c

    return blocks


def assign_blocks_to_gpus(blocks: List[Any]) -> List[Any]:
    """Move each block to its corresponding GPU device."""
    if torch is None:
        return blocks

    for idx, block in enumerate(blocks):
        try:
            device = get_device_type(idx)
            block.to(device)
        except Exception as exc:
            _logger.warning("Failed to move block %d: %s", idx, exc)

    return blocks

