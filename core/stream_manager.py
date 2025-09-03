class StreamManager:
    def __init__(self, scheduler, verbose=True):
        self.scheduler = scheduler
        self.verbose = verbose

    def preload_layer(self, layer):
        block_size = layer["size_mb"]
        block = self.scheduler.allocate_block(block_size)
        if self.verbose:
            print(f"Préchargement de {layer['name']} ({block_size}MB) sur GPU {block.gpu_id}")
        return block

    def release_layer(self, layer):
        if self.verbose:
            print(f"Libération de {layer['name']} de la VRAM")
