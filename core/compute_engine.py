class ComputeEngine:
    def __init__(self, backend="cuda"):
        self.backend = backend

    def execute_layer(self, layer, input_tensor, device_id):
        print(f"Exécution de {layer['name']} sur GPU {device_id} via {self.backend}")
        # Simulation : renvoie un tensor fictif
        return input_tensor * 1.0  # placeholder
