import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optionnel: définir variable token par défaut pour tests
os.environ.setdefault('VRM_API_TOKEN', 'testtoken')

# Mock léger de torch si non installé (réduit dépendances dans CI légère)
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    import types, math
    torch = types.SimpleNamespace(
        nn=types.SimpleNamespace(Module=object, Identity=lambda: (lambda x: x)),
        device=lambda *a, **k: 'cpu',
        randn=lambda *a, **k: __import__('random').random(),
        onnx=types.SimpleNamespace(export=lambda *a, **k: None),
    )
    sys.modules['torch'] = torch
