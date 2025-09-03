
import time
import random
from core.scheduler import Scheduler
from core.logger import Logger
from core.monitor import GPUMonitor
from core.compressor import Compressor

class StreamManager:
    def __init__(self, scheduler: Scheduler, logger: Logger = None, verbose=True):
        self.scheduler = scheduler
        self.logger = logger or Logger()
        self.verbose = verbose
        self.loaded_layers = {}
        self.monitor = GPUMonitor()
        self.compressor = Compressor()

    def preload_layer(self, layer):
        """
        Charge une couche en VRAM avec compression et priorisation.
        """
        block_size = layer["size_mb"]
        priority = 1 if layer["type"] == "core" else 2

        # Compression adaptative si VRAM saturée
        if self.monitor.vram_usage() > 85:
            block_size = self.compressor.compress(layer["name"], block_size)
            if self.verbose:
                print(f"🗜️ Compression {layer['name']} → {block_size}MB")

        block = self.scheduler.allocate_block(block_size, priority=priority)
        self.loaded_layers[layer["name"]] = block

        if self.verbose:
            print(f"📥 Chargement {layer['name']} ({block_size}MB) sur GPU {block.gpu_id} [prio {priority}]")
        self.logger.log(f"Layer {layer['name']} loaded on GPU {block.gpu_id}")
        return block

    def release_layer(self, layer_name):
        """
        Libère la couche de la VRAM.
        """
        block = self.loaded_layers.get(layer_name)
        if block:
            self.scheduler.release_block(block)
            if self.verbose:
                print(f"🧹 Libération {layer_name} de la VRAM (GPU {block.gpu_id})")
            self.logger.log(f"Layer {layer_name} released from GPU {block.gpu_id}")
            del self.loaded_layers[layer_name]

    def prefetch_layers(self, layers, lookahead=2):
        """
        Précharge les couches à venir selon prédiction du scheduler.
        """
        predicted = self.scheduler.predict_next_layers(layers, lookahead)
        for layer in predicted:
            if layer["name"] not in self.loaded_layers:
                self.preload_layer(layer)
                time.sleep(0.002)

    def unload_unused(self, active_layers):
        """
        Libère les couches non utilisées dans le batch courant.
        """
        active_names = set(layer["name"] for layer in active_layers)
        for name in list(self.loaded_layers.keys()):
            if name not in active_names:
                self.release_layer(name)

    def swap_if_needed(self):
        """
        Déclenche un swap inter-GPU si surcharge détectée.
        """
        overloaded_gpu = self.monitor.detect_overload()
        if overloaded_gpu:
            for name, block in self.loaded_layers.items():
                if block.gpu_id == overloaded_gpu:
                    new_gpu = self.scheduler.find_alternate_gpu()
                    self.scheduler.migrate_block(block, new_gpu)
                    if self.verbose:
                        print(f"🔄 Swap {name} → GPU {new_gpu}")
                    self.logger.log(f"Layer {name} migrated from GPU {overloaded_gpu} to GPU {new_gpu}")

    def live_monitoring(self):
        """
        Affiche l’état live de la VRAM.
        """
        status = self.monitor.status()
        print(f"📊 VRAM Status: {status}")
