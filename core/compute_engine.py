import torch

class ComputeEngine:
    def __init__(self, backend="auto", verbose=True):
        self.verbose = verbose
        self.backend = self._detect_backend() if backend == "auto" else backend

    def _detect_backend(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Metal
        else:
            return "cpu"

    def execute_layer(self, layer, input_tensor, device_id=0):
        """
        Exécute une couche simulée sur le GPU ou backend spécifié.
        """
        device = self._get_device(device_id)
        input_tensor = input_tensor.to(device)

        if self.verbose:
            print(f"[{self.backend.upper()}] Exécution de {layer['name']} sur {device}")

        # Simulation d'une opération (matmul fictif)
        output = input_tensor @ torch.randn(input_tensor.shape[-1], input_tensor.shape[-1], device=device)

        return output

    def _get_device(self, device_id):
        if self.backend == "cuda":
            return torch.device(f"cuda:{device_id}")
        elif self.backend == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")
