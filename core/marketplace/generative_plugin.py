"""
Plugin IA générative (marketplace) :
- Plugins LLM/diffusion/audio/vidéo
- Sandboxing, scoring communautaire
"""
import hashlib
_REGISTRY = {}
_SANDBOX_ALLOW = {"len","range","min","max","sum"}

def register_plugin(plugin):
    _REGISTRY[plugin.name] = plugin

def list_plugins():
    return [ {"name": p.name, "type": p.type_, "score": p.score} for p in _REGISTRY.values() ]

def compute_signature(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16]

class GenerativePlugin:
    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_
        self.score = 0

    def run(self, *args, **kwargs):
        print(f"[Plugin] Exécution {self.name} ({self.type_})")
        return "resultat"

    def rate(self, score):
        self.score = score
        return self.score

    def sandboxed_run(self, code: str, payload: dict | None = None):
        """Exécute un micro-snippet en environnement restreint (prototype).

        ATTENTION: Ce sandbox est simplifié et ne doit pas être considéré sécurisé
        en production; il empêche simplement la plupart des accès évidents.
        """
        safe_globals = {k: __builtins__[k] for k in _SANDBOX_ALLOW if k in __builtins__} if isinstance(__builtins__, dict) else {}
        safe_globals['__builtins__'] = safe_globals
        safe_locals = {'payload': payload or {}, 'result': None}
        # Interdictions minimales
        if any(forbidden in code for forbidden in ('import','open','exec','eval','os.','sys.','subprocess')):
            return {"error": "forbidden"}
        try:
            exec(code, safe_globals, safe_locals)
        except Exception as e:  # pragma: no cover - chemin erreur
            return {"error": str(e)}
        return {"result": safe_locals.get('result')}

__all__ = ["GenerativePlugin","register_plugin","list_plugins","compute_signature"]
