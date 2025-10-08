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
      3. MPS       – Apple Silicon
      4. CPU       – fallback
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
        
        # Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def get_device_type(idx: int) -> torch.device:
    """Retourne un objet torch.device cohérent pour un index.

    Note: sous ROCm, l'API PyTorch reste `cuda:X`; on choisit donc de retourner
    `cuda:{idx}` tout en exposant le backend logique "rocm" ailleurs.
    """
    backend = detect_backend()
    if backend in ('cuda', 'rocm') and torch.cuda.device_count() > idx:
        return torch.device(f"cuda:{idx}")
    if backend == 'mps':
        return torch.device('mps')
    return torch.device(f"cpu:{idx}")


def enumerate_devices() -> List[Dict[str, Any]]:
    """Fournit une liste unifiée des devices disponibles.

    Structure par entrée:
      {"id": <str>, "backend": <cuda|rocm|mps|cpu>, "index": <int|str>, "name": str, "total_memory": Optional[int]}
    """
    devices: List[Dict[str, Any]] = []
    backend = detect_backend()
    # CUDA / ROCm partagent torch.cuda
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                dev_backend = 'rocm' if (backend == 'rocm' and 'AMD' in props.name.upper()) else 'cuda'
                devices.append({
                    'id': f"{dev_backend}:{i}",
                    'backend': dev_backend,
                    'index': i,
                    'name': props.name,
                    'total_memory': props.total_memory,
                })
    except Exception:
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

__all__ = [
    'get_tokenizer',
    'get_device_type',
    'detect_backend',
    'enumerate_devices',
    'assign_block_to_device'
]
