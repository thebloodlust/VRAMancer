"""S5 — LoRA hot-swap : charger/switcher/décharger des adaptateurs en < 1s.

Mesuré (`benchmarks/test_s5_lora_hotswap.py`) : switch ~3 ms, load disque ~0.1 s,
sans recharger le modèle de base. Utile multi-tenant / A-B testing / fine-tune incrémental.

Fin wrapper sur l'API PEFT intégrée à transformers (`load_adapter`/`set_adapter`/
`delete_adapter`/`disable_adapters`). Pas de magie : on expose juste proprement.
"""
from __future__ import annotations
import os
import time
from typing import Optional, Dict, Any


def _name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path)) or "adapter"


class LoraManager:
    """Gère les adaptateurs LoRA d'un modèle transformers chargé (avec PEFT installé)."""

    def __init__(self, model):
        if not (hasattr(model, "load_adapter") and hasattr(model, "set_adapter")):
            raise RuntimeError(
                "Le modèle n'expose pas l'API PEFT (load_adapter/set_adapter). "
                "Installe peft (pip install 'peft>=0.18.2') et charge un modèle HF."
            )
        self.model = model
        self._loaded: Dict[str, str] = {}  # name -> source path/id

    def load(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Charge un adaptateur (disque ou HF id) sous `name`. Renvoie le temps."""
        name = name or _name_from_path(path)
        t0 = time.perf_counter()
        self.model.load_adapter(path, adapter_name=name)
        self._loaded[name] = path
        return {"ok": True, "name": name, "load_s": round(time.perf_counter() - t0, 3)}

    def use(self, name: str) -> Dict[str, Any]:
        """Active un adaptateur déjà chargé (le hot-swap, ~ms)."""
        if name not in self._loaded:
            return {"ok": False, "msg": f"adaptateur '{name}' non chargé"}
        t0 = time.perf_counter()
        self.model.set_adapter(name)
        return {"ok": True, "active": name, "switch_ms": round(1000 * (time.perf_counter() - t0), 2)}

    def disable(self) -> Dict[str, Any]:
        """Revient au modèle de base (désactive tous les adaptateurs)."""
        self.model.disable_adapters()
        return {"ok": True, "active": None}

    def enable(self) -> Dict[str, Any]:
        self.model.enable_adapters()
        return {"ok": True}

    def unload(self, name: str) -> Dict[str, Any]:
        """Décharge un adaptateur (libère sa mémoire)."""
        if name not in self._loaded:
            return {"ok": False, "msg": f"adaptateur '{name}' non chargé"}
        try:
            self.model.delete_adapter(name)
        except Exception as e:  # pragma: no cover
            return {"ok": False, "msg": str(e)}
        self._loaded.pop(name, None)
        return {"ok": True, "unloaded": name}

    def list(self) -> Dict[str, Any]:
        """Liste les adaptateurs chargés + l'actif."""
        active = []
        try:
            if hasattr(self.model, "active_adapters"):
                aa = self.model.active_adapters()
                active = list(aa) if isinstance(aa, (list, tuple, set)) else [aa]
        except Exception:
            pass
        return {"loaded": list(self._loaded), "active": active}
