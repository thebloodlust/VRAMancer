import random
from core.scheduler import Scheduler
from core.logger import Logger
from core.monitor import GPUMonitor
from core.compressor import Compressor

class MemoryBalancer:
    def __init__(self, scheduler: Scheduler, logger: Logger = None, verbose=True):
        self.scheduler = scheduler
        self.logger = logger or Logger()
        self.monitor = GPUMonitor()
        self.compressor = Compressor()
        self.verbose = verbose

    def balance(self, active_layers):
        """
        Rééquilibre la mémoire entre GPU selon usage et priorité.
        """
        for layer in active_layers:
            gpu_id = self.scheduler.get_gpu_for_layer(layer["name"])
            usage = self.monitor.vram_usage(gpu_id)

            if usage > 90:
                if self.verbose:
                    print(f"🔥 GPU {gpu_id} saturé ({usage}%) → rééquilibrage {layer['name']}")

                # Compression d'urgence
                new_size = self.compressor.compress(layer["name"], layer["size_mb"])
                self.scheduler.update_block_size(layer["name"], new_size)
                self.logger.log(f"Layer {layer['name']} compressé à {new_size}MB")

                # Swap vers GPU secondaire
                alt_gpu = self.scheduler.find_alternate_gpu(exclude=gpu_id)
                if alt_gpu is not None:
                    self.scheduler.migrate_layer(layer["name"], alt_gpu)
                    self.logger.log(f"Layer {layer['name']} migré vers GPU {alt_gpu}")
                    if self.verbose:
                        print(f"🔄 Swap {layer['name']} → GPU {alt_gpu}")

    def emergency_release(self):
        """
        Libère les couches les moins prioritaires en cas de crash imminent.
        """
        overloaded = self.monitor.detect_overload()
        if overloaded:
            low_priority = self.scheduler.get_low_priority_layers(overloaded)
            for layer_name in low_priority:
                self.scheduler.release_layer(layer_name)
                self.logger.log(f"Layer {layer_name} libéré en urgence (GPU {overloaded})")
                if self.verbose:
                    print(f"💣 Libération d’urgence : {layer_name} (GPU {overloaded})")

    def predictive_balance(self, history):
        """
        Utilise l’historique pour anticiper les pics de charge.
        """
        predicted_spikes = self._analyze_history(history)
        for gpu_id in predicted_spikes:
            if self.verbose:
                print(f"📈 Pic anticipé sur GPU {gpu_id} → pré-rééquilibrage")
            self.scheduler.shift_load(gpu_id)

    def _analyze_history(self, history):
        """
        Analyse simple : détecte les GPU avec usage > 85% sur 3 cycles consécutifs.
        """
        spikes = []
        for gpu_id, usage_list in history.items():
            if len(usage_list) >= 3 and all(u > 85 for u in usage_list[-3:]):
                spikes.append(gpu_id)
        return spikes
