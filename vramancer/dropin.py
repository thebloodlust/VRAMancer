"""S1 — drop-in patch : une ligne pour activer les optimisations VRAMancer prouvées.

    import vramancer; vramancer.patch()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B")  # multi-GPU, anti-OOM

**Honnête sur ce que ça fait** : le split multi-GPU, c'est `accelerate device_map="auto"`.
`patch()` n'invente PAS un moteur. Il **package en une ligne** ce qu'on a mesuré :
  - `device_map="auto"` + `max_memory` compute-aware → évite l'OOM au chargement
    (le piège upcast fp32 sur modèle juste-trop-gros qu'on a diagnostiqué).
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fix mémoire CUDA).
  - prompt-lookup decoding par défaut dans `.generate` (optim mesurée, native HF).

Deux modes (à comparer — `benchmarks/test_s1_patch.py`) :
  - ``patch()`` / ``patch("light")`` : monkeypatch fin. L'objet reste un modèle HF
    standard. ~zéro surprise, zéro machinerie. **Recommandé.**
  - ``patch("heavy")`` : route via ``InferencePipeline`` (turbo, fault-tolerance,
    lending…). Plus puissant mais lourd et comportement non-HF.

``unpatch()`` restaure ``from_pretrained`` d'origine.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

_log = logging.getLogger("vramancer.patch")

_ORIG_FROM_PRETRAINED = None  # type: ignore
_PATCH_MODE: Optional[str] = None
_DEFAULT_PROMPT_LOOKUP = int(os.environ.get("VRM_PROMPT_LOOKUP_NGRAM", "10"))


# --------------------------------------------------------------------------- #
# Helper partagé : placement max_memory anti-OOM (réutilise hetero_config)
# --------------------------------------------------------------------------- #
def compute_max_memory(reserve: float = 0.95) -> Optional[dict]:
    """Construit un dict ``max_memory`` compute-aware pour accelerate.

    Reprend la logique prouvée de ``core.backends._build_compute_aware_memory_map``
    sous forme autonome : budget par GPU = VRAM libre × reserve, + overflow CPU.
    Renvoie ``None`` si < 2 GPU (laisser accelerate décider).
    """
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            return None
        from core.hetero_config import auto_configure
        config = auto_configure(strategy="balanced")
        if len(config.gpus) < 2:
            return None
        mm = {}
        for gpu in config.gpus:
            base = gpu.free_vram_gb if gpu.free_vram_gb > 0 else gpu.total_vram_gb
            mm[gpu.index] = f"{max(2.0, base * reserve):.1f}GiB"
        mm["cpu"] = "48GiB"
        return mm
    except Exception as e:  # pragma: no cover - best effort
        _log.debug("compute_max_memory a échoué (%s) — accelerate décidera", e)
        return None


def _ensure_expandable_segments() -> None:
    """Active expandable_segments si pas déjà fixé (fix mémoire CUDA)."""
    cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in cur:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            (cur + ",") if cur else "") + "expandable_segments:True"


# --------------------------------------------------------------------------- #
# Variante LÉGÈRE : monkeypatch fin de from_pretrained
# --------------------------------------------------------------------------- #
def _wrap_generate_prompt_lookup(model):
    """Injecte prompt_lookup_num_tokens par défaut dans model.generate (optim mesurée)."""
    if getattr(model, "_vrm_generate_wrapped", False):
        return model
    orig_generate = model.generate

    def generate(*args, **kwargs):
        # N'agit que si l'utilisateur n'a rien demandé de spécifique côté assist.
        if (_DEFAULT_PROMPT_LOOKUP > 0
                and "prompt_lookup_num_tokens" not in kwargs
                and "assistant_model" not in kwargs):
            kwargs["prompt_lookup_num_tokens"] = _DEFAULT_PROMPT_LOOKUP
        try:
            return orig_generate(*args, **kwargs)
        except (TypeError, ValueError):
            # Modèle/config qui n'aime pas prompt-lookup → repli propre.
            kwargs.pop("prompt_lookup_num_tokens", None)
            return orig_generate(*args, **kwargs)

    model.generate = generate
    model._vrm_generate_wrapped = True
    return model


def _make_light_wrapper(orig):
    def from_pretrained(cls, *args, **kwargs):
        _ensure_expandable_segments()
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            ngpu = torch.cuda.device_count() if has_cuda else 0
        except Exception:
            has_cuda, ngpu = False, 0

        if has_cuda and "device_map" not in kwargs:
            if ngpu >= 2:
                kwargs["device_map"] = "auto"
                mm = compute_max_memory()
                if mm:
                    kwargs.setdefault("max_memory", mm)
                _log.info("[vramancer.patch] device_map=auto + max_memory=%s", mm)
            else:
                kwargs["device_map"] = {"": 0}
                _log.info("[vramancer.patch] mono-GPU → device_map={'':0}")

        model = orig.__func__(cls, *args, **kwargs)
        try:
            _wrap_generate_prompt_lookup(model)
        except Exception as e:  # pragma: no cover
            _log.debug("wrap generate ignoré (%s)", e)
        return model

    return classmethod(from_pretrained)


# --------------------------------------------------------------------------- #
# Variante LOURDE : route via InferencePipeline
# --------------------------------------------------------------------------- #
class _HeavyModel:
    """Objet renvoyé par patch('heavy') : enveloppe une InferencePipeline.

    API : ``.generate(prompt: str, max_new_tokens=...) -> str`` (NON-HF — c'est
    la rançon du mode lourd). Expose ``.pipeline`` pour le reste.
    """
    def __init__(self, model_name, **kwargs):
        from core.inference_pipeline import InferencePipeline
        self.pipeline = InferencePipeline(
            backend_name=os.environ.get("VRM_HEAVY_BACKEND", "huggingface"),
            enable_metrics=False, enable_discovery=False, verbose=False,
        )
        self.pipeline.load(model_name, **kwargs)

    def generate(self, prompt, max_new_tokens: int = 128, **kw):
        return self.pipeline.generate(prompt, max_new_tokens=max_new_tokens, **kw)


def _make_heavy_wrapper(orig):
    box = {}

    def from_pretrained(cls, *args, **kwargs):
        _ensure_expandable_segments()
        model_name = args[0] if args else kwargs.pop("pretrained_model_name_or_path")
        # Le backend HF de la pipeline rappelle from_pretrained : restaurer
        # l'original pendant le load pour éviter la récursion infinie.
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained = orig
        try:
            return _HeavyModel(model_name, **kwargs)
        finally:
            AutoModelForCausalLM.from_pretrained = box["w"]

    w = classmethod(from_pretrained)
    box["w"] = w
    return w


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #
def patch(mode: str = "light") -> None:
    """Active le drop-in VRAMancer. ``mode`` = 'light' (défaut) ou 'heavy'."""
    global _ORIG_FROM_PRETRAINED, _PATCH_MODE
    if mode not in ("light", "heavy"):
        raise ValueError("mode doit être 'light' ou 'heavy'")
    from transformers import AutoModelForCausalLM
    if _ORIG_FROM_PRETRAINED is None:
        _ORIG_FROM_PRETRAINED = AutoModelForCausalLM.from_pretrained
    orig = _ORIG_FROM_PRETRAINED
    wrapper = _make_light_wrapper(orig) if mode == "light" else _make_heavy_wrapper(orig)
    AutoModelForCausalLM.from_pretrained = wrapper
    _PATCH_MODE = mode
    _log.info("[vramancer.patch] activé (mode=%s)", mode)


def unpatch() -> None:
    """Restaure transformers.AutoModelForCausalLM.from_pretrained d'origine."""
    global _PATCH_MODE
    if _ORIG_FROM_PRETRAINED is not None:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained = _ORIG_FROM_PRETRAINED
        _PATCH_MODE = None
        _log.info("[vramancer.patch] désactivé")


def is_patched() -> bool:
    return _PATCH_MODE is not None
