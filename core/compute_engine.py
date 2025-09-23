# core/compute_engine.py

import torch
import time
import onnx
import psutil
from torch.profiler import profile, record_function, ProfilerActivity

class ComputeEngine:
    def __init__(self, backend="auto", verbose=True):
        self.verbose = verbose
        self.backend = self._detect_backend() if backend == "auto" else backend

    def _detect_backend(self):
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch.version, "hip") and torch.version.hip:
            return "rocm"
        else:
            return "cpu"

    def _get_device(self, device_id=0):
        if self.backend in ["cuda", "rocm"]:
            return torch.device(f"{self.backend}:{device_id}")
        elif self.backend == "mps":
            return torch.device("mps")
        else:
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
        torch.onnx.export(model, dummy_input, filename, export_params=True, opset_version=17)
        if self.verbose:
            print(f"📦 Modèle exporté en ONNX : {filename}")

    def get_ram_status(self):
        mem = psutil.virtual_memory()
        return mem.available, mem.total
