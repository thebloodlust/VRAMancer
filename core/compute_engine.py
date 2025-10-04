"""ComputeEngine abstrait avec fallback si ONNX ou torch features manquants.

En environnement Windows minimal (sans onnx installé), l'import ne doit pas casser
le démarrage des dashboards. `export VRM_DISABLE_ONNX=1` force la désactivation.
"""

import os, time
STRICT = os.environ.get('VRM_STRICT_IMPORT','0') in {'1','true','TRUE'}
try:
    import torch  # type: ignore
except Exception as _e:  # pragma: no cover - torch absent
    if STRICT:
        raise RuntimeError(f"[STRICT_IMPORT] torch indisponible: {_e}")
    class _TorchStub:
        class nn:
            class Module: ...
            class functional:
                @staticmethod
                def relu(x): return x
        class profiler:
            class profile:
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
            class record_function:
                def __init__(self,*a,**k): pass
                def __enter__(self): return self
                def __exit__(self,*a): pass
        def device(self,name): return name
        class cuda:
            @staticmethod
            def is_available(): return False
    torch = _TorchStub()  # type: ignore

try:  # ONNX optionnel
    if os.environ.get('VRM_DISABLE_ONNX') == '1':
        raise ImportError('ONNX disabled by VRM_DISABLE_ONNX=1')
    import onnx  # type: ignore
except Exception as _e:  # pragma: no cover - fallback silencieux
    if STRICT:
        raise RuntimeError(f"[STRICT_IMPORT] onnx indisponible: {_e}")
    onnx = None
import psutil
try:
    from torch.profiler import profile, record_function, ProfilerActivity  # type: ignore
except Exception:  # profiler peut être absent ou torch stub
    class _NullCtx:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
    profile = _NullCtx  # type: ignore
    record_function = _NullCtx  # type: ignore
    class ProfilerActivity:  # type: ignore
        CPU='cpu'; CUDA='cuda'

class ComputeEngine:
    def __init__(self, backend="auto", verbose=True):
        self.verbose = verbose
        self.backend = self._detect_backend() if backend == "auto" else backend

    def _detect_backend(self):
        from core.utils import detect_backend
        return detect_backend()

    def _get_device(self, device_id=0):
        if self.backend in ["cuda", "rocm"]:
            # Sur ROCm on reste sur l'API cuda: (PyTorch unifié)
            return torch.device(f"cuda:{device_id}")
        if self.backend == "mps":
            return torch.device("mps")
        return torch.device("cpu")

    def execute_layer(self, layer, input_tensor, device_id=0, track_gradients=False, use_compile=False, profile_gpu=False):
        device = self._get_device(device_id)
        input_tensor = input_tensor.to(device)

        if track_gradients:
            input_tensor.requires_grad = True

        if self.verbose:
            print(f"[{self.backend.upper()}] ▶️ {layer['name']} sur {device} | Batch: {input_tensor.shape[0]}")

        def layer_fn(x):
            weight = torch.randn(x.shape[-1], x.shape[-1], device=device, requires_grad=track_gradients)
            bias = torch.randn(x.shape[-1], device=device, requires_grad=track_gradients)
            return torch.nn.functional.relu(torch.matmul(x, weight) + bias)

        if use_compile and hasattr(torch, "compile"):
            layer_fn = torch.compile(layer_fn)

        if profile_gpu and self.backend in ["cuda", "rocm"]:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("layer_execution"):
                    output = layer_fn(input_tensor)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        else:
            start = time.time()
            output = layer_fn(input_tensor)
            if self.verbose:
                print(f"⏱️ Temps d'exécution : {time.time() - start:.4f}s")

        if track_gradients:
            loss = output.sum()
            loss.backward()
            if self.verbose:
                print(f"🧠 Gradients calculés pour {layer['name']}")

        return output

    def export_onnx(self, model, dummy_input, filename="model.onnx"):
        if onnx is None:
            if self.verbose:
                print("[WARN] ONNX indisponible (ou désactivé), export ignoré")
            return False
        try:
            torch.onnx.export(model, dummy_input, filename, export_params=True, opset_version=17)
            if self.verbose:
                print(f"📦 Modèle exporté en ONNX : {filename}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Export ONNX échoué: {e}")
            return False

    def get_ram_status(self):
        mem = psutil.virtual_memory()
        return mem.available, mem.total
