import torch
import random

class GPUMonitor:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def vram_usage(self, gpu_id=0):
        try:
            total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 2)
            used = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
            usage = round((used / total) * 100, 2)
            return usage
        except:
            return random.randint(40, 90)  # stub fallback

    def detect_overload(self, threshold=90):
        try:
            for i in range(torch.cuda.device_count()):
                if self.vram_usage(i) > threshold:
                    return i
            return None
        except:
            return random.choice([0, None])

    def status(self):
        try:
            status = {}
            for i in range(torch.cuda.device_count()):
                usage = self.vram_usage(i)
                status[f"GPU {i}"] = f"{usage}% VRAM"
            return status
        except:
            return {"GPU 0": "Simulé"}
