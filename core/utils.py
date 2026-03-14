# core/utils.py
"""
Utility helpers for GPU‑aware inference.
"""

from __future__ import annotations

import os, sys, re
from typing import Iterable, Sequence, Optional, List, Dict, Any

if os.environ.get('VRM_MINIMAL_TEST'):
    # Stub torch ultra léger pour tests rapides sans dépendances lourdes
    class _CudaStub:
        def is_available(self): return False
        def device_count(self): return 0
        def get_device_properties(self, i): raise RuntimeError('no cuda')
    class _BackendsStub:
        class _MPS:
            def is_available(self): return False
        mps = _MPS()
    class _TorchStub:
        cuda = _CudaStub()
        backends = _BackendsStub()
        class version: hip = None
        class nn:
            class Module: ...
        def device(self, name): return name
    torch = _TorchStub()  # type: ignore
    sys.modules.setdefault('torch', torch)  # assure importlib spec fallback
    def get_tokenizer(model_name: str):  # pragma: no cover - stub
        return None
    def is_tokenizers_available():  # pragma: no cover
        return False
else:  # mode normal mais on protège chaque import lourd + fallback tokenizer pur Python
    FORCE_BASIC = os.environ.get('VRM_FORCE_BASIC_TOKENIZER') in {'1','true','TRUE'} or \
                  os.environ.get('USE_SLOW_TOKENIZER') in {'1','true','TRUE'}

    try:  # Torch (tolérant)
        import torch  # type: ignore
    except Exception:
        class _TorchLite:
            class cuda:
                @staticmethod
                def is_available(): return False
                @staticmethod
                def device_count(): return 0
            class backends:
                class mps:
                    @staticmethod
                    def is_available(): return False
            class nn:
                class Module: ...
            class version: hip=None
            @staticmethod
            def device(name): return name
        torch = _TorchLite()  # type: ignore

    class BasicTokenizer:
        """Tokenizer pur Python très simple (fallback) – non performant.

        - Normalise en lower + supprime espaces multiples
        - Découpe sur espaces
        - encode() -> ids basés sur hash stable tronqué
        - decode() -> reconstruction approximative (perte possible)
        """
        fallback = True
        def __init__(self):
            self._vocab: dict[str,int] = {}
            self._rev: dict[int,str] = {}
            self._next_id = 5  # réserver 0..4 pour spéciaux
        def tokenize(self, text: str):
            text = re.sub(r"\s+"," ", text.strip().lower())
            return text.split(' ') if text else []
        def _assign(self, tok: str):
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._rev[self._next_id] = tok
                self._next_id += 1
            return self._vocab[tok]
        def encode(self, text: str):
            return [self._assign(t) for t in self.tokenize(text)]
        def decode(self, ids):
            return ' '.join(self._rev.get(i,'?') for i in ids)
    _BASIC_SINGLETON: BasicTokenizer | None = None
    def _basic():  # pas de nonlocal (module scope variable)
        global _BASIC_SINGLETON
        if _BASIC_SINGLETON is None:
            _BASIC_SINGLETON = BasicTokenizer()
            print("[Tokenizer] Utilisation du BasicTokenizer fallback (transformers indisponible ou forcé)")
        return _BASIC_SINGLETON

    if not FORCE_BASIC:
        try:
            from transformers import AutoTokenizer  # type: ignore
            from transformers.utils import is_tokenizers_available  # type: ignore
            def get_tokenizer(model_name: str):  # noqa
                if os.environ.get('VRM_FORCE_BASIC_TOKENIZER') in {'1','true','TRUE'}:
                    return _basic()
                try:
                    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
                except Exception as e:  # pragma: no cover - fallback slow ou basic
                    if os.environ.get('USE_SLOW_TOKENIZER') in {'1','true','TRUE'}:
                        print(f"[Tokenizer] Fast indisponible ({e}) -> slow")
                        try:
                            return AutoTokenizer.from_pretrained(model_name, use_fast=False)
                        except Exception:
                            return _basic()
                    return _basic()
        except Exception:
            FORCE_BASIC = True
    if FORCE_BASIC:
        def get_tokenizer(model_name: str):  # pragma: no cover - trivial
            return _basic()
        def is_tokenizers_available():  # pragma: no cover
            return False


# --------------------------------------------------------------
# 1️⃣  Helpers
# --------------------------------------------------------------
if not os.environ.get('VRM_MINIMAL_TEST'):
    # Docstring préservée pour mode normal
    def _doc():
        """Helper interne pour conserver la documentation de get_tokenizer dans mode normal."""
        pass


def detect_backend() -> str:
    """Détecte le backend principal disponible.

    Ordre logique:
      1. ROCm (hip) – si build ROCm (torch.version.hip non nul)
      2. CUDA      – sinon si CUDA disponible
      3. XPU       – Intel GPUs (IPEX)
      4. NPU       – Huawei Ascend NPUs
      5. TPU       – Google TPUs (XLA)
      6. MPS       – Apple Silicon
      7. CPU       – fallback
    """
    try:
        # Vérification ROCm en premier (AMD GPUs avec support HIP)
        if hasattr(torch.version, 'hip') and torch.version.hip:  # build ROCm
            return 'rocm'
        
        # CUDA standard (NVIDIA)
        if torch.cuda.is_available():
            # Double-check pour AMD GPUs sur build CUDA+ROCm hybride
            try:
                device_name = torch.cuda.get_device_name(0).upper()
                if 'AMD' in device_name or 'RADEON' in device_name or 'INSTINCT' in device_name:
                    return 'rocm'  # AMD GPU détectée
            except Exception:
                pass
            return 'cuda'

        # Intel XPU (IPEX)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return 'xpu'

        # Huawei NPU
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return 'npu'
            
        # Google TPU (Torch XLA)
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device():
                return 'tpu'
        except ImportError:
            pass
        
        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def detect_device_backend(device_index: int) -> str:
    """Détecte le backend d'un GPU spécifique (per-device, pas global).

    Contrairement à detect_backend() qui retourne un seul backend global,
    cette fonction identifie le vendor/backend de chaque GPU individuellement.
    Essentiel pour les setups mixtes AMD + NVIDIA.

    Returns:
        'cuda' (NVIDIA), 'rocm' (AMD), 'mps' (Apple), ou 'cpu'
    """
    try:
        if not torch.cuda.is_available():
            return 'cpu'
        if device_index >= torch.cuda.device_count():
            return 'cpu'

        name = torch.cuda.get_device_name(device_index).upper()

        # AMD patterns
        amd_patterns = ('AMD', 'RADEON', 'INSTINCT', 'MI100', 'MI200',
                        'MI250', 'MI300', 'NAVI', 'VEGA', 'CDNA')
        if any(p in name for p in amd_patterns):
            return 'rocm'

        # NVIDIA patterns (explicit match vs default)
        nvidia_patterns = ('NVIDIA', 'GEFORCE', 'QUADRO', 'TESLA', 'RTX',
                           'GTX', 'TITAN', 'A100', 'H100', 'L40')
        if any(p in name for p in nvidia_patterns):
            return 'cuda'

        # Fallback to global detection
        if hasattr(torch.version, 'hip') and torch.version.hip:
            return 'rocm'
        return 'cuda'
    except Exception:
        return 'cpu'


_LOGICAL_MAPPING: Optional[Dict[int, int]] = None

def _get_logical_mapping() -> Dict[int, int]:
    global _LOGICAL_MAPPING
    if _LOGICAL_MAPPING is not None:
        return _LOGICAL_MAPPING
    
    mapping = {}
    backend = detect_backend()
    if backend in ('cuda', 'rocm') and torch.cuda.is_available():
        count = torch.cuda.device_count()
        devices_info = []
        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
                cap = getattr(props, 'major', 0) * 10 + getattr(props, 'minor', 0)
                vram = getattr(props, 'total_memory', 0)
                name = getattr(props, 'name', '').lower()
                # Priorité absolue pour les architectures 5070 ou Blackwell ou NVFP4
                tier = 2 if ('5070' in name or 'blackwell' in name or 'nvfp4' in name) else 1
                devices_info.append((i, tier, cap, vram))
            except Exception:
                devices_info.append((i, 0, 0, 0))
        
        # Tri décroissant par (tier, capability, VRAM)
        sorted_info = sorted(devices_info, key=lambda x: (x[1], x[2], x[3]), reverse=True)
        for logical_idx, info in enumerate(sorted_info):
            mapping[logical_idx] = info[0]
            
    _LOGICAL_MAPPING = mapping
    return mapping

def get_device_type(idx: int) -> torch.device:
    """Retourne un objet torch.device cohérent pour un index.

    Note: sous ROCm, l'API PyTorch reste `cuda:X`; on choisit donc de retourner
    `cuda:{idx}` tout en exposant le backend logique "rocm" ailleurs.
    """
    backend = detect_backend()
    if backend in ('cuda', 'rocm') and torch.cuda.is_available():
        mapping = _get_logical_mapping()
        phys_idx = mapping.get(idx, idx)
        if torch.cuda.device_count() > phys_idx:
            return torch.device(f"cuda:{phys_idx}")
    if backend == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.device_count() > idx:
        return torch.device(f"xpu:{idx}")
    if backend == 'npu' and hasattr(torch, 'npu') and torch.npu.device_count() > idx:
        return torch.device(f"npu:{idx}")
    if backend == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            pass
    if backend == 'mps':
        return torch.device('mps')
    return torch.device(f"cpu:{idx}")


def enumerate_devices() -> List[Dict[str, Any]]:
    """Fournit une liste unifiée des devices disponibles.

    Structure par entrée:
      {"id": <str>, "backend": <cuda|rocm|mps|cpu>, "index": <int|str>,
       "name": str, "total_memory": Optional[int], "vendor": str}

    Supporte les setups mixtes AMD + NVIDIA en détectant le vendor
    par GPU individuellement via detect_device_backend().
    """
    devices: List[Dict[str, Any]] = []
    # CUDA / ROCm partagent torch.cuda
    try:
        if torch.cuda.is_available():
            mapping = _get_logical_mapping()
            # Inverser le mapping pour obtenir phys->logical
            phys_to_logical = {v: k for k, v in mapping.items()}
            # Créer une liste temporaire pour CUDA
            cuda_devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                dev_backend = detect_device_backend(i)
                cuda_devices.append({
                    'id': f"{dev_backend}:{i}",
                    'backend': dev_backend,
                    'index': i,
                    'logical_index': phys_to_logical.get(i, i),
                    'name': props.name,
                    'total_memory': props.total_memory,
                    'vendor': 'amd' if dev_backend == 'rocm' else (
                        'nvidia' if dev_backend == 'cuda' else 'unknown'),
                })
            # Trier selon l'index logique pour que L0 soit le puissant GPU
            cuda_devices.sort(key=lambda d: d.get('logical_index', d['index']))
            devices.extend(cuda_devices)
    except Exception:
        pass
    # Intel XPU
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i) if hasattr(torch.xpu, 'get_device_properties') else None
                devices.append({
                    'id': f"xpu:{i}",
                    'backend': 'xpu',
                    'index': i,
                    'name': getattr(props, 'name', f"Intel XPU {i}") if props else f"Intel XPU {i}",
                    'total_memory': getattr(props, 'total_memory', None) if props else None,
                    'vendor': 'intel',
                })
    except Exception:
        pass
    # Huawei NPU
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            for i in range(torch.npu.device_count()):
                props = torch.npu.get_device_properties(i) if hasattr(torch.npu, 'get_device_properties') else None
                devices.append({
                    'id': f"npu:{i}",
                    'backend': 'npu',
                    'index': i,
                    'name': getattr(props, 'name', f"Huawei NPU {i}") if props else f"Huawei NPU {i}",
                    'total_memory': getattr(props, 'total_memory', None) if props else None,
                    'vendor': 'huawei',
                })
    except Exception:
        pass
    # Google TPU
    try:
        import torch_xla.core.xla_model as xm
        if xm.xla_device():
            # TPUs generally work as a single logical cluster device in standard PyTorch mappings
            devices.append({
                'id': 'tpu:0',
                'backend': 'tpu',
                'index': 0,
                'name': 'Google TPU',
                'total_memory': None,
                'vendor': 'google',
            })
    except ImportError:
        pass
    # MPS
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append({
                'id': 'mps:0',
                'backend': 'mps',
                'index': 0,
                'name': 'Apple MPS',
                'total_memory': None,
                'vendor': 'apple',
            })
    except Exception:
        pass
    if not devices:  # CPU fallback explicite
        devices.append({
            'id': 'cpu:0',
            'backend': 'cpu',
            'index': 0,
            'name': 'CPU',
            'total_memory': None,
            'vendor': 'generic',
        })
    return devices


def assign_block_to_device(block: torch.nn.Module, idx: int) -> torch.nn.Module:
    """
    Move a sub‑module (`block`) to the device corresponding to GPU index `idx`.
    The function returns the same module reference, but moved in‑place.

    Parameters
    ----------
    block : torch.nn.Module
        Sub‑module to be moved.
    idx : int
        Target GPU index.

    Returns
    -------
    torch.nn.Module
        The module moved to the desired device.
    """
    device = get_device_type(idx)
    moved = block.to(device)
    # Certains modules nn.Linear n'exposent pas attribut device directement; l'ajouter pour tests
    if not hasattr(moved, 'device'):
        try:
            setattr(moved, 'device', device)
        except Exception:
            pass
    return moved

# ------------------------------------------------------------------
# Tensor Serialization (Moved from legacy utils/helpers.py)
# ------------------------------------------------------------------
def serialize_tensors(tensors):
    """Serialize a list of tensors into bytes."""
    import torch
    import numpy as np
    return b"".join([t.contiguous().cpu().numpy().tobytes() for t in tensors])

def deserialize_tensors(data, shapes, dtypes):
    """Deserialize a list of bytes back into tensors."""
    import torch
    import numpy as np
    tensors = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape))
        arr = np.frombuffer(data[offset:offset+size*dtype.itemsize], dtype=dtype)
        tensors.append(torch.from_numpy(arr.copy()).reshape(shape))
        offset += size * dtype.itemsize
    return tensors

__all__ = [
    'get_tokenizer',
    'get_device_type',
    'detect_backend',
    'detect_device_backend',
    'enumerate_devices',
    'assign_block_to_device',
    'serialize_tensors',
    'deserialize_tensors',
]
