class TransferManager:
    def __init__(self, protocol="vramancer", secure=False):
        self.protocol = protocol
        self.secure = secure

    def send_activation(self, source_gpu, target_gpu, tensor):
        # Simule un transfert de données
        print(f"Transfert de {tensor.shape} de GPU {source_gpu} vers GPU {target_gpu} via {self.protocol}")

    def sync(self, activations_map):
        # activations_map = {layer_name: (source_gpu, target_gpu, tensor)}
        for name, (src, tgt, tensor) in activations_map.items():
            self.send_activation(src, tgt, tensor)
