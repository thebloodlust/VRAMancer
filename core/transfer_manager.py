import time

class TransferManager:
    def __init__(self, protocol="vramancer", secure=False, verbose=True):
        self.protocol = protocol
        self.secure = secure
        self.verbose = verbose

    def send_activation(self, source_gpu, target_gpu, tensor):
        """
        Simule le transfert d'un tensor entre deux GPU.
        """
        if self.verbose:
            print(f"[{self.protocol.upper()}] Transfert de {tensor.shape} de GPU {source_gpu} → GPU {target_gpu}")
        # Simulation de latence réseau
        time.sleep(0.005)  # 5 ms fictifs
        return True

    def sync_activations(self, activations_map):
        """
        active_map = {
            "layer_1": (source_gpu, target_gpu, tensor),
            ...
        }
        """
        for layer_name, (src, tgt, tensor) in activations_map.items():
            success = self.send_activation(src, tgt, tensor)
            if self.verbose:
                print(f"✔️ Synchronisation {layer_name} : {'OK' if success else 'Échec'}")
